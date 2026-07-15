"""Aggregate completed run JSON files into a comparison table."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


SUMMARY_FIELDS = (
    "run",
    "mode",
    "trigger",
    "poison_fraction",
    "seed",
    "epochs",
    "poisoned_training_examples",
    "clean_accuracy",
    "baseline_accuracy",
    "delta_accuracy",
    "triggered_accuracy",
    "asr",
    "conditional_asr",
    "target_margin_mean",
    "target_margin_median",
)


def load_run_records(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths)]


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_by_seed = {
        int(record["seed"]): float(record["metrics"]["clean_accuracy"])
        for record in records
        if record["mode"] == "clean"
    }
    rows: list[dict[str, Any]] = []
    for record in records:
        metrics = record["metrics"]
        seed = int(record["seed"])
        baseline = clean_by_seed.get(seed)
        clean_accuracy = float(metrics["clean_accuracy"])
        rows.append(
            {
                "run": record["run"],
                "mode": record["mode"],
                "trigger": record["trigger"],
                "poison_fraction": float(record["poison_fraction"]),
                "seed": seed,
                "epochs": int(record["epochs"]),
                "poisoned_training_examples": int(record["poisoned_training_examples"]),
                "clean_accuracy": clean_accuracy,
                "baseline_accuracy": baseline,
                "delta_accuracy": None if baseline is None else baseline - clean_accuracy,
                "triggered_accuracy": float(metrics["triggered_accuracy"]),
                "asr": float(metrics["asr"]),
                "conditional_asr": float(metrics["conditional_asr"]),
                "target_margin_mean": float(metrics["target_margin_mean"]),
                "target_margin_median": float(metrics["target_margin_median"]),
            }
        )
    return rows


def write_summary_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
