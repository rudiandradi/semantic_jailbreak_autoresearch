#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from guard_utils import apply_chat_template, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a causal LM for guard before/after experiments")
    parser.add_argument("--model", required=True, help="Base model path or Hugging Face id")
    parser.add_argument("--train", required=True, help="Path to train.jsonl")
    parser.add_argument("--val", required=True, help="Path to val.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for adapter/checkpoint artifacts")
    parser.add_argument("--max-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--epochs", type=float, default=2.0, help="Number of training epochs")
    parser.add_argument("--max-steps", type=int, default=-1, help="Optional hard cap on optimizer steps")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--train-batch-size", type=int, default=2, help="Per-device train batch size")
    parser.add_argument("--eval-batch-size", type=int, default=2, help="Per-device eval batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--logging-steps", type=int, default=10, help="Training logging interval")
    parser.add_argument("--save-strategy", choices=["epoch", "steps", "no"], default="epoch", help="Checkpoint save strategy")
    parser.add_argument("--eval-strategy", choices=["epoch", "steps", "no"], default="epoch", help="Evaluation strategy")
    parser.add_argument("--save-steps", type=int, default=100, help="Checkpoint save interval when using steps")
    parser.add_argument("--eval-steps", type=int, default=100, help="Evaluation interval when using steps")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        help="Module names to target with LoRA",
    )
    parser.add_argument("--bf16", action="store_true", help="Force bf16 training")
    parser.add_argument("--fp16", action="store_true", help="Force fp16 training")
    parser.add_argument("--load-in-8bit", action="store_true", help="Load the base model in 8-bit")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--local-files-only", action="store_true", help="Load model/tokenizer only from the local HF cache")
    return parser.parse_args()


def ensure_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing training dependencies. Install `transformers datasets peft accelerate` "
            "and optionally `bitsandbytes` for 8-bit loading."
        ) from exc
    return torch, Dataset, LoraConfig, get_peft_model, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def tokenize_example(example: dict[str, Any], tokenizer: Any, max_length: int) -> dict[str, Any]:
    messages = example["messages"]
    prompt_text = apply_chat_template(tokenizer, messages[:-1], add_generation_prompt=True)
    full_text = apply_chat_template(tokenizer, messages, add_generation_prompt=False)

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_tokens = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = list(full_tokens["input_ids"])
    attention_mask = list(full_tokens["attention_mask"])
    labels = list(input_ids)
    prompt_len = min(len(prompt_ids), len(labels))
    labels[:prompt_len] = [-100] * prompt_len
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class GuardDataCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        max_len = max(len(feature["input_ids"]) for feature in features)
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        for feature in features:
            pad_len = max_len - len(feature["input_ids"])
            batch_input_ids.append(feature["input_ids"] + [pad_token_id] * pad_len)
            batch_attention_mask.append(feature["attention_mask"] + [0] * pad_len)
            batch_labels.append(feature["labels"] + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def main() -> None:
    args = parse_args()
    (
        torch,
        Dataset,
        LoraConfig,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    ) = ensure_dependencies()

    train_rows = read_jsonl(Path(args.train).expanduser().resolve())
    val_rows = read_jsonl(Path(args.val).expanduser().resolve())
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, lora_config)

    train_dataset = Dataset.from_list([tokenize_example(row, tokenizer, args.max_length) for row in train_rows])
    val_dataset = Dataset.from_list([tokenize_example(row, tokenizer, args.max_length) for row in val_rows])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        eval_strategy=args.eval_strategy,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        report_to=[],
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=GuardDataCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    run_config = {
        "model": args.model,
        "train": str(Path(args.train).expanduser().resolve()),
        "val": str(Path(args.val).expanduser().resolve()),
        "output_dir": str(output_dir),
        "max_length": args.max_length,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": args.target_modules,
        "gradient_checkpointing": args.gradient_checkpointing,
        "load_in_8bit": args.load_in_8bit,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(f"Saved adapter and tokenizer to {output_dir}")


if __name__ == "__main__":
    main()
