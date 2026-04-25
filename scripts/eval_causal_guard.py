#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from guard_utils import apply_chat_template, compute_binary_metrics, is_refusal_text, parse_safe_unsafe, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a causal guard model before/after tuning")
    parser.add_argument("--model", required=True, help="Base model path or Hugging Face id")
    parser.add_argument("--output", required=True, help="Output JSON report path")
    parser.add_argument("--eval-harmful", required=True, help="Path to eval_harmful.jsonl")
    parser.add_argument("--eval-benign", help="Path to eval_benign.jsonl")
    parser.add_argument("--adapter", help="Optional PEFT adapter path")
    parser.add_argument("--task", choices=["classification", "refusal"], default="classification", help="Evaluation mode")
    parser.add_argument("--max-length", type=int, default=1024, help="Prompt truncation length")
    parser.add_argument("--max-new-tokens", type=int, default=8, help="Generation length")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling")
    parser.add_argument("--limit", type=int, help="Optional cap per split for smoke tests")
    parser.add_argument("--bf16", action="store_true", help="Load/evaluate in bf16 when supported")
    parser.add_argument("--fp16", action="store_true", help="Load/evaluate in fp16 when supported")
    parser.add_argument("--local-files-only", action="store_true", help="Load model/tokenizer only from the local HF cache")
    return parser.parse_args()


def ensure_dependencies(adapter: str | None) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing evaluation dependencies. Install `transformers` and `torch`.") from exc

    peft_model = None
    if adapter:
        try:
            from peft import PeftModel
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing `peft` dependency required for --adapter evaluation.") from exc
        peft_model = PeftModel
    return torch, AutoModelForCausalLM, AutoTokenizer, peft_model


def load_model(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    torch, AutoModelForCausalLM, AutoTokenizer, PeftModel = ensure_dependencies(args.adapter)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.bf16:
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif args.fp16:
        model_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    if torch.cuda.is_available():
        device = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model.to(device)
    model.eval()
    return torch, tokenizer, model


def predict_label(task: str, text: str) -> str:
    if task == "classification":
        return parse_safe_unsafe(text)
    return "UNSAFE" if is_refusal_text(text) else "SAFE"


def generate_one(
    *,
    torch: Any,
    tokenizer: Any,
    model: Any,
    prompt_messages: list[dict[str, str]],
    max_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    prompt_text = apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        generate_kwargs["do_sample"] = True
        generate_kwargs["temperature"] = temperature
        generate_kwargs["top_p"] = top_p
    else:
        generate_kwargs["do_sample"] = False

    with torch.no_grad():
        output = model.generate(**encoded, **generate_kwargs)
    prompt_len = encoded["input_ids"].shape[1]
    generated_ids = output[0][prompt_len:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def eval_split(
    *,
    split_name: str,
    rows: list[dict[str, Any]],
    torch: Any,
    tokenizer: Any,
    model: Any,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if args.limit is not None and index >= args.limit:
            break
        generated_text = generate_one(
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            prompt_messages=row["messages"][:-1],
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        predicted_label = predict_label(args.task, generated_text)
        records.append(
            {
                "id": row["id"],
                "split": split_name,
                "expected_label": row["expected_label"],
                "predicted_label": predicted_label,
                "generated_text": generated_text,
                "metadata": row.get("metadata", {}),
            }
        )
    return records


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()
    harmful_rows = read_jsonl(Path(args.eval_harmful).expanduser().resolve())
    benign_rows = read_jsonl(Path(args.eval_benign).expanduser().resolve()) if args.eval_benign else []
    torch, tokenizer, model = load_model(args)

    all_records: list[dict[str, Any]] = []
    harmful_records = eval_split(
        split_name="eval_harmful",
        rows=harmful_rows,
        torch=torch,
        tokenizer=tokenizer,
        model=model,
        args=args,
    )
    all_records.extend(harmful_records)
    benign_records: list[dict[str, Any]] = []
    if benign_rows:
        benign_records = eval_split(
            split_name="eval_benign",
            rows=benign_rows,
            torch=torch,
            tokenizer=tokenizer,
            model=model,
            args=args,
        )
        all_records.extend(benign_records)

    summary = {
        "overall": compute_binary_metrics(all_records),
        "splits": {
            "eval_harmful": compute_binary_metrics(harmful_records),
        },
    }
    if benign_records:
        summary["splits"]["eval_benign"] = compute_binary_metrics(benign_records)

    report = {
        "model": args.model,
        "adapter": args.adapter or "",
        "task": args.task,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decode": {
            "max_length": args.max_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "summary": summary,
        "records": all_records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved report to {output_path}")


if __name__ == "__main__":
    main()
