#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


METRICS = [
    ("overall", "accuracy"),
    ("overall", "unsafe_precision"),
    ("overall", "unsafe_recall"),
    ("overall", "unsafe_f1"),
    ("overall", "false_positive_rate"),
    ("overall", "false_negative_rate"),
    ("overall", "unknown_rate"),
    ("eval_harmful", "accuracy"),
    ("eval_harmful", "unsafe_recall"),
    ("eval_harmful", "false_negative_rate"),
    ("eval_benign", "accuracy"),
    ("eval_benign", "false_positive_rate"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare before/after guard evaluation reports")
    parser.add_argument("--before", required=True, help="Path to baseline eval JSON")
    parser.add_argument("--after", required=True, help="Path to tuned eval JSON")
    parser.add_argument("--output-json", help="Optional path for the diff report JSON")
    parser.add_argument("--output-csv", help="Optional path for a flat CSV comparison")
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def metric_value(report: dict[str, Any], split: str, metric: str) -> float | None:
    if split == "overall":
        return report["summary"]["overall"].get(metric)
    return report["summary"]["splits"].get(split, {}).get(metric)


def main() -> None:
    args = parse_args()
    before = load_json(args.before)
    after = load_json(args.after)

    rows: list[dict[str, Any]] = []
    for split, metric in METRICS:
        before_value = metric_value(before, split, metric)
        after_value = metric_value(after, split, metric)
        if before_value is None and after_value is None:
            continue
        delta = None if before_value is None or after_value is None else after_value - before_value
        rows.append(
            {
                "split": split,
                "metric": metric,
                "before": before_value,
                "after": after_value,
                "delta": delta,
            }
        )

    diff_report = {
        "before_model": before["model"],
        "before_adapter": before.get("adapter", ""),
        "after_model": after["model"],
        "after_adapter": after.get("adapter", ""),
        "task": after["task"],
        "rows": rows,
    }

    print(f"{'split':<14} {'metric':<22} {'before':>10} {'after':>10} {'delta':>10}")
    for row in rows:
        before_text = "n/a" if row["before"] is None else f"{row['before']:.4f}"
        after_text = "n/a" if row["after"] is None else f"{row['after']:.4f}"
        delta_text = "n/a" if row["delta"] is None else f"{row['delta']:+.4f}"
        print(f"{row['split']:<14} {row['metric']:<22} {before_text:>10} {after_text:>10} {delta_text:>10}")

    if args.output_json:
        output_json = Path(args.output_json).expanduser().resolve()
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(diff_report, indent=2) + "\n", encoding="utf-8")
        print(f"Saved JSON diff to {output_json}")

    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["split", "metric", "before", "after", "delta"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV diff to {output_csv}")


if __name__ == "__main__":
    main()
