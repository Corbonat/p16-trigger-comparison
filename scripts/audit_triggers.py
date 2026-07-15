"""Render configured triggers on real CIFAR-10 images before model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.datasets import CIFAR10

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p16_triggers.config import TRIGGER_NAMES, load_config  # noqa: E402
from p16_triggers.metrics import visibility_metrics  # noqa: E402
from p16_triggers.triggers import apply_trigger  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    dataset = CIFAR10(root=config.dataset.root, train=False, download=False)
    labels_seen: set[int] = set()
    sample_indices: list[int] = []
    for index, label in enumerate(dataset.targets):
        if label not in labels_seen:
            labels_seen.add(label)
            sample_indices.append(index)
        if len(sample_indices) == args.samples:
            break

    names = ("clean",) + TRIGGER_NAMES
    tile_size = 160
    label_height = 28
    sheet = Image.new(
        "RGB",
        (len(names) * tile_size, len(sample_indices) * (tile_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    visibility_rows: list[dict[str, object]] = []
    for row, index in enumerate(sample_indices):
        clean, class_id = dataset[index]
        variants = [clean]
        for trigger in TRIGGER_NAMES:
            triggered = apply_trigger(clean, trigger, **config.trigger_parameters(trigger))
            variants.append(triggered)
            metrics = visibility_metrics(np.asarray(clean), np.asarray(triggered))
            visibility_rows.append(
                {"sample_index": index, "class_id": class_id, "trigger": trigger, **metrics}
            )
        for column, (name, image) in enumerate(zip(names, variants, strict=True)):
            enlarged = image.resize((tile_size, tile_size), resample=Image.Resampling.NEAREST)
            x = column * tile_size
            y = row * (tile_size + label_height)
            sheet.paste(enlarged, (x, y))
            draw.text((x + 6, y + tile_size + 4), name, fill="black", font=font)

    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    image_path = artifacts / "cifar10_trigger_grid.png"
    metrics_path = artifacts / "cifar10_trigger_visibility.json"
    sheet.save(image_path)
    metrics_path.write_text(
        json.dumps(visibility_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(image_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
