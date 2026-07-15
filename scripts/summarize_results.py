"""Build results/summary.csv from completed run JSON files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p16_triggers.reporting import (  # noqa: E402
    load_run_records,
    summarize_records,
    write_summary_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="results/runs")
    parser.add_argument("--output", default="results/summary.csv")
    args = parser.parse_args()
    paths = list((ROOT / args.runs).glob("*.json"))
    if not paths:
        raise SystemExit("No run JSON files found; train models before summarizing results")
    records = load_run_records(paths)
    rows = summarize_records(records)
    output = ROOT / args.output
    write_summary_csv(rows, output)
    print(output)


if __name__ == "__main__":
    main()
