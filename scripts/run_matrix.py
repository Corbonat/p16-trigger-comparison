"""Print or execute the complete training matrix from baseline.yaml."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p16_triggers.config import TRIGGER_NAMES, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)

    commands: list[list[str]] = []
    for seed in config.experiment.seeds:
        commands.append(
            [
                sys.executable,
                "scripts/train.py",
                "--config",
                args.config,
                "--mode",
                "clean",
                "--trigger",
                "patch",
                "--seed",
                str(seed),
                "--device",
                args.device,
            ]
        )
        for trigger in TRIGGER_NAMES:
            for fraction in config.dataset.poison_fractions:
                commands.append(
                    [
                        sys.executable,
                        "scripts/train.py",
                        "--config",
                        args.config,
                        "--mode",
                        "poisoned",
                        "--trigger",
                        trigger,
                        "--poison-fraction",
                        str(fraction),
                        "--seed",
                        str(seed),
                        "--device",
                        args.device,
                    ]
                )

    clean_runs = len(config.experiment.seeds)
    poisoned_runs = clean_runs * len(TRIGGER_NAMES) * len(config.dataset.poison_fractions)
    print(f"Prepared {len(commands)} runs ({clean_runs} clean + {poisoned_runs} poisoned).")
    for command in commands:
        print(subprocess.list2cmdline(command))
        if args.execute:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
