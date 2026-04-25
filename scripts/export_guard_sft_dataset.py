#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from guard_utils import (
    BENIGN_PROMPTS,
    build_benign_example,
    build_harmful_example,
    normalize_row,
    row_allowed,
    split_train_val_eval,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export guard SFT/eval datasets for before/after causal LM experiments")
    parser.add_argument("--input-csv", required=True, help="Path to attack_dataset.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for train/val/eval JSONL files")
    parser.add_argument("--attack", default="Emoji Game", help="Attack family to keep")
    parser.add_argument(
        "--selection",
        choices=["broken-only", "broken-plus-resilient", "all-non-error"],
        default="broken-plus-resilient",
        help="Which rows to keep from the attack CSV",
    )
    parser.add_argument(
        "--task",
        choices=["classification", "refusal"],
        default="classification",
        help="Target behavior for the assistant examples",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Harmful holdout eval split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed")
    parser.add_argument("--max-train-rows", type=int, help="Optional cap on training rows after splitting")
    parser.add_argument("--max-val-rows", type=int, help="Optional cap on validation rows after splitting")
    parser.add_argument("--max-eval-harmful-rows", type=int, help="Optional cap on harmful eval rows after splitting")
    parser.add_argument("--max-eval-benign-rows", type=int, help="Optional cap on benign eval rows")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    eval_harmful_path = output_dir / "eval_harmful.jsonl"
    eval_benign_path = output_dir / "eval_benign.jsonl"
    manifest_path = output_dir / "manifest.json"

    if not input_path.exists():
        raise SystemExit(f"Input CSV not found: {input_path}")

    existing = [path for path in (train_path, val_path, eval_harmful_path, eval_benign_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        paths = "\n".join(str(path) for path in existing)
        raise SystemExit(f"Output files already exist. Pass --overwrite to replace them:\n{paths}")

    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [normalize_row(row) for row in reader]

    attack_rows = [row for row in rows if row["attack"] == args.attack]
    kept_rows = [row for row in attack_rows if row_allowed(row, args.selection)]
    harmful_examples = [
        build_harmful_example(row, args.task, f"harmful_{index:05d}")
        for index, row in enumerate(kept_rows, start=1)
    ]
    if len(harmful_examples) < 3:
        raise SystemExit(
            f"Need at least 3 rows after filtering, got {len(harmful_examples)}. Adjust --attack or --selection."
        )

    rng = random.Random(args.seed)
    train_rows, val_rows, eval_harmful_rows = split_train_val_eval(
        harmful_examples,
        rng=rng,
        val_ratio=args.val_ratio,
        eval_ratio=args.eval_ratio,
    )
    if args.max_train_rows is not None:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_val_rows is not None:
        val_rows = val_rows[: args.max_val_rows]
    if args.max_eval_harmful_rows is not None:
        eval_harmful_rows = eval_harmful_rows[: args.max_eval_harmful_rows]
    eval_benign_rows = [
        build_benign_example(example_id, category, prompt, args.task)
        for example_id, category, prompt in BENIGN_PROMPTS
    ]
    if args.max_eval_benign_rows is not None:
        eval_benign_rows = eval_benign_rows[: args.max_eval_benign_rows]

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(eval_harmful_path, eval_harmful_rows)
    write_jsonl(eval_benign_path, eval_benign_rows)
    manifest = {
        "attack": args.attack,
        "selection": args.selection,
        "task": args.task,
        "seed": args.seed,
        "max_train_rows": args.max_train_rows,
        "max_val_rows": args.max_val_rows,
        "max_eval_harmful_rows": args.max_eval_harmful_rows,
        "max_eval_benign_rows": args.max_eval_benign_rows,
        "input_rows": len(rows),
        "attack_rows": len(attack_rows),
        "kept_rows": len(kept_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "eval_harmful_rows": len(eval_harmful_rows),
        "eval_benign_rows": len(eval_benign_rows),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Input rows:         {len(rows)}")
    print(f"Rows for attack:    {len(attack_rows)}")
    print(f"Rows kept:          {len(kept_rows)}")
    print(f"Train rows:         {len(train_rows)} -> {train_path}")
    print(f"Validation rows:    {len(val_rows)} -> {val_path}")
    print(f"Eval harmful rows:  {len(eval_harmful_rows)} -> {eval_harmful_path}")
    print(f"Eval benign rows:   {len(eval_benign_rows)} -> {eval_benign_path}")
    print(f"Manifest:           {manifest_path}")


if __name__ == "__main__":
    main()
