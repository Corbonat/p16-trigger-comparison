"""Generate a small visual preview without downloading CIFAR-10."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p16_triggers.triggers import apply_trigger  # noqa: E402


def make_demo_image(size: int = 96) -> Image.Image:
    y, x = np.indices((size, size))
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[..., 0] = np.clip(40 + 1.7 * x, 0, 255)
    array[..., 1] = np.clip(30 + 1.6 * y, 0, 255)
    array[..., 2] = np.clip(210 - 0.8 * x, 0, 255)
    image = Image.fromarray(array, mode="RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse((24, 18, 72, 66), fill=(235, 175, 65), outline=(255, 255, 255), width=2)
    draw.rectangle((34, 54, 62, 84), fill=(55, 105, 190), outline=(255, 255, 255), width=2)
    return image


def main() -> None:
    original = make_demo_image()
    examples = [
        ("clean", original),
        ("patch", apply_trigger(original, "patch", side=12, alpha=0.75)),
        ("stripe", apply_trigger(original, "stripe", width=4, alpha=0.7)),
        ("brightness", apply_trigger(original, "brightness", delta=0.08)),
        ("position", apply_trigger(original, "position", dx=6, dy=6)),
        ("color", apply_trigger(original, "color", hue_degrees=22)),
    ]

    label_height = 24
    sheet = Image.new("RGB", (len(examples) * original.width, original.height + label_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=15)
    for index, (label, image) in enumerate(examples):
        x = index * original.width
        sheet.paste(image, (x, 0))
        draw.text((x + 5, original.height + 3), label, fill="black", font=font)

    output = ROOT / "artifacts" / "trigger_preview.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    print(output)


if __name__ == "__main__":
    main()
