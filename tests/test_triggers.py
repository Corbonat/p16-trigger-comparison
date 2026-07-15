import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p16_triggers.triggers import apply_trigger  # noqa: E402


class TriggerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.array = np.full((32, 32, 3), 100, dtype=np.uint8)

    def test_all_triggers_preserve_shape_dtype_and_input(self) -> None:
        original = self.array.copy()
        configurations = {
            "patch": {"side": 4},
            "stripe": {"width": 1},
            "brightness": {"delta": 0.15},
            "position": {"dx": 2, "dy": 2},
            "color": {"hue_degrees": 16},
        }
        for trigger, parameters in configurations.items():
            with self.subTest(trigger=trigger):
                result = apply_trigger(self.array, trigger, **parameters)
                self.assertEqual(result.shape, self.array.shape)
                self.assertEqual(result.dtype, self.array.dtype)
        np.testing.assert_array_equal(self.array, original)

    def test_patch_changes_only_requested_corner(self) -> None:
        result = apply_trigger(
            self.array, "patch", side=4, alpha=1.0, position="bottom_right"
        )
        np.testing.assert_array_equal(result[:-4], self.array[:-4])
        np.testing.assert_array_equal(result[-4:, :-4], self.array[-4:, :-4])
        self.assertTrue(np.any(result[-4:, -4:] != self.array[-4:, -4:]))

    def test_pil_input_returns_pil_image(self) -> None:
        image = Image.fromarray(self.array, mode="RGB")
        result = apply_trigger(image, "brightness", delta=0.15)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, image.size)

    def test_default_brightness_adds_fifteen_percent_of_pixel_range(self) -> None:
        result = apply_trigger(self.array, "brightness")
        np.testing.assert_array_equal(result, np.full_like(self.array, 138))

    def test_unknown_trigger_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_trigger(self.array, "unknown")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
