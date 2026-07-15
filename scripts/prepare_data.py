"""Download CIFAR-10 without training a model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p16_triggers.config import load_config  # noqa: E402
from p16_triggers.data import download_cifar10  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    data_root = Path(config.dataset.root)
    download_cifar10(data_root)
    print(f"CIFAR-10 is ready at {data_root}")


if __name__ == "__main__":
    main()
