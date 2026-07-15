"""Image trigger operators used by all experiments.

Inputs are RGB PIL images or NumPy arrays in HWC layout. NumPy arrays may be
uint8 in [0, 255] or floating point in [0, 1]. The returned object has the same
container and dtype as the input.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from PIL import Image

TriggerName = Literal["patch", "stripe", "brightness", "position", "color"]


def _to_unit_float(image: Image.Image | np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(image, Image.Image):
        array = np.asarray(image)
        metadata = {"kind": "pil", "mode": image.mode}
    else:
        array = np.asarray(image)
        metadata = {"kind": "numpy", "dtype": array.dtype}

    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("Expected an RGB image with shape (height, width, 3)")

    if np.issubdtype(array.dtype, np.integer):
        max_value = np.iinfo(array.dtype).max
        metadata["max_value"] = max_value
        unit = array.astype(np.float32) / max_value
    else:
        unit = array.astype(np.float32, copy=True)
        if unit.size and (float(unit.min()) < 0.0 or float(unit.max()) > 1.0):
            raise ValueError("Floating-point images must be in the [0, 1] range")

    return unit.copy(), metadata


def _restore(array: np.ndarray, metadata: dict[str, Any]) -> Image.Image | np.ndarray:
    array = np.clip(array, 0.0, 1.0)
    if metadata["kind"] == "pil":
        uint8 = np.rint(array * 255.0).astype(np.uint8)
        return Image.fromarray(uint8, mode=metadata["mode"])

    dtype = metadata["dtype"]
    if np.issubdtype(dtype, np.integer):
        return np.rint(array * metadata["max_value"]).astype(dtype)
    return array.astype(dtype)


def _corner_slice(
    height: int, width: int, patch_height: int, patch_width: int, position: str
) -> tuple[slice, slice]:
    if patch_height > height or patch_width > width:
        raise ValueError("Trigger is larger than the image")

    positions = {
        "top_left": (0, 0),
        "top_right": (0, width - patch_width),
        "bottom_left": (height - patch_height, 0),
        "bottom_right": (height - patch_height, width - patch_width),
        "center": ((height - patch_height) // 2, (width - patch_width) // 2),
    }
    if position not in positions:
        raise ValueError(f"Unknown position: {position}")
    top, left = positions[position]
    return slice(top, top + patch_height), slice(left, left + patch_width)


def add_patch(
    image: Image.Image | np.ndarray,
    *,
    side: int = 4,
    alpha: float = 0.7,
    position: str = "bottom_right",
    color: tuple[int, int, int] = (255, 255, 0),
    secondary_color: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image | np.ndarray:
    """Blend a checkerboard patch into one of five fixed positions."""
    if side < 1:
        raise ValueError("side must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    array, metadata = _to_unit_float(image)
    rows, columns = np.indices((side, side))
    mask = ((rows + columns) % 2)[..., None]
    primary = np.asarray(color, dtype=np.float32) / 255.0
    secondary = np.asarray(secondary_color, dtype=np.float32) / 255.0
    patch = np.where(mask == 0, primary, secondary)
    ys, xs = _corner_slice(array.shape[0], array.shape[1], side, side, position)
    array[ys, xs] = (1.0 - alpha) * array[ys, xs] + alpha * patch
    return _restore(array, metadata)


def add_stripe(
    image: Image.Image | np.ndarray,
    *,
    width: int = 1,
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    alpha: float = 0.7,
    position: Literal["top", "bottom", "left", "right", "center"] = "bottom",
    color: tuple[int, int, int] = (255, 255, 0),
) -> Image.Image | np.ndarray:
    """Blend a full-length horizontal or vertical stripe."""
    if width < 1:
        raise ValueError("width must be positive")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    array, metadata = _to_unit_float(image)
    height, image_width = array.shape[:2]
    stripe_color = np.asarray(color, dtype=np.float32) / 255.0

    if orientation == "horizontal":
        if width > height or position not in {"top", "bottom", "center"}:
            raise ValueError("Invalid width or position for a horizontal stripe")
        top = {"top": 0, "bottom": height - width, "center": (height - width) // 2}[position]
        region = (slice(top, top + width), slice(None))
    elif orientation == "vertical":
        if width > image_width or position not in {"left", "right", "center"}:
            raise ValueError("Invalid width or position for a vertical stripe")
        left = {"left": 0, "right": image_width - width, "center": (image_width - width) // 2}[
            position
        ]
        region = (slice(None), slice(left, left + width))
    else:
        raise ValueError(f"Unknown orientation: {orientation}")

    array[region] = (1.0 - alpha) * array[region] + alpha * stripe_color
    return _restore(array, metadata)


def change_brightness(
    image: Image.Image | np.ndarray, *, delta: float = 0.15
) -> Image.Image | np.ndarray:
    """Apply an additive exposure change and clip to the valid image range."""
    array, metadata = _to_unit_float(image)
    return _restore(array + float(delta), metadata)


def shift_position(
    image: Image.Image | np.ndarray, *, dx: int = 2, dy: int = 2
) -> Image.Image | np.ndarray:
    """Translate an image using reflection padding instead of a black border."""
    array, metadata = _to_unit_float(image)
    height, width = array.shape[:2]
    if abs(dx) >= width or abs(dy) >= height:
        raise ValueError("Shift magnitude must be smaller than the image dimensions")

    pad_x, pad_y = abs(dx), abs(dy)
    padded = np.pad(array, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode="reflect")
    start_y = pad_y - dy
    start_x = pad_x - dx
    shifted = padded[start_y : start_y + height, start_x : start_x + width]
    return _restore(shifted, metadata)


def rotate_hue(
    image: Image.Image | np.ndarray, *, hue_degrees: float = 16.0
) -> Image.Image | np.ndarray:
    """Rotate hue while leaving HSV saturation and value unchanged."""
    array, metadata = _to_unit_float(image)
    rgb = Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="RGB")
    hsv = np.asarray(rgb.convert("HSV")).copy()
    hue_shift = int(round(float(hue_degrees) / 360.0 * 256.0))
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue_shift) % 256
    rotated = np.asarray(Image.fromarray(hsv, mode="HSV").convert("RGB"), dtype=np.float32) / 255.0
    return _restore(rotated, metadata)


def apply_trigger(
    image: Image.Image | np.ndarray, trigger: TriggerName, **parameters: Any
) -> Image.Image | np.ndarray:
    """Apply one of the five project triggers through a shared interface."""
    functions = {
        "patch": add_patch,
        "stripe": add_stripe,
        "brightness": change_brightness,
        "position": shift_position,
        "color": rotate_hue,
    }
    try:
        function = functions[trigger]
    except KeyError as error:
        raise ValueError(f"Unknown trigger: {trigger}") from error
    return function(image, **parameters)
