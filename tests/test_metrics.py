import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p16_triggers.metrics import (  # noqa: E402
    attack_success_rate,
    conditional_attack_success_rate,
    per_source_class_asr,
    target_logit_margin,
    visibility_metrics,
)


class MetricTests(unittest.TestCase):
    def test_asr_excludes_true_target_images(self) -> None:
        y_true = np.array([1, 2, 0, 3])
        y_triggered = np.array([0, 0, 7, 0])
        self.assertEqual(attack_success_rate(y_true, y_triggered, target_class=0), 1.0)

    def test_conditional_asr_excludes_existing_model_errors(self) -> None:
        y_true = np.array([1, 2, 3, 0])
        y_clean = np.array([1, 9, 3, 0])
        y_triggered = np.array([0, 0, 3, 0])
        result = conditional_attack_success_rate(
            y_true, y_clean, y_triggered, target_class=0
        )
        self.assertEqual(result, 0.5)

    def test_per_class_asr(self) -> None:
        y_true = np.array([1, 1, 2, 2, 0])
        y_triggered = np.array([0, 1, 0, 0, 0])
        self.assertEqual(per_source_class_asr(y_true, y_triggered, target_class=0), {1: 0.5, 2: 1.0})

    def test_target_logit_margin(self) -> None:
        logits = np.array([[1.0, 3.0, 2.0], [4.0, 1.0, 5.0]])
        np.testing.assert_allclose(target_logit_margin(logits, target_class=1), [1.0, -4.0])

    def test_visibility_of_identical_images(self) -> None:
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        result = visibility_metrics(image, image.copy())
        self.assertEqual(result["changed_fraction"], 0.0)
        self.assertTrue(math.isinf(result["psnr"]))


if __name__ == "__main__":
    unittest.main()
