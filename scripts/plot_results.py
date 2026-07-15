"""Plot ASR and clean-accuracy cost after the experiment matrix completes."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="results/summary.csv")
    args = parser.parse_args()
    with (ROOT / args.summary).open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["mode"] == "poisoned"]
    if not rows:
        raise SystemExit("No poisoned runs found in the summary")

    grouped: dict[str, dict[float, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row["trigger"]][float(row["poison_fraction"])].append(row)

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    for metric, label, filename in (
        ("asr", "Attack Success Rate", "asr_vs_poison_fraction.png"),
        ("delta_accuracy", "Clean accuracy drop", "accuracy_drop_vs_poison_fraction.png"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        for trigger, by_fraction in sorted(grouped.items()):
            fractions = sorted(by_fraction)
            means = [
                np.mean([float(row[metric]) for row in by_fraction[value]]) for value in fractions
            ]
            deviations = [
                np.std([float(row[metric]) for row in by_fraction[value]]) for value in fractions
            ]
            axis.errorbar(fractions, means, yerr=deviations, marker="o", label=trigger)
        axis.set_xlabel("Poison fraction")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(artifacts / filename, dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
