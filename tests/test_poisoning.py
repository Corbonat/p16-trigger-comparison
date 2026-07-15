import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p16_triggers.poisoning import choose_poison_indices, poison_examples  # noqa: E402


class PoisoningTests(unittest.TestCase):
    def test_selected_indices_never_contain_target_class(self) -> None:
        labels = np.tile(np.arange(10), 10)
        indices = choose_poison_indices(
            labels, target_class=0, poison_fraction=0.1, seed=17, stratified=True
        )
        self.assertEqual(len(indices), 10)
        self.assertTrue(np.all(labels[indices] != 0))

    def test_poisoning_changes_only_selected_example_and_label(self) -> None:
        images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
        labels = np.array([1, 2, 3])
        poisoned_images, poisoned_labels = poison_examples(
            images,
            labels,
            indices=[1],
            target_class=0,
            trigger="patch",
            trigger_parameters={"side": 2, "alpha": 1.0},
        )
        np.testing.assert_array_equal(poisoned_images[0], images[0])
        self.assertTrue(np.any(poisoned_images[1] != images[1]))
        np.testing.assert_array_equal(poisoned_labels, [1, 0, 3])


if __name__ == "__main__":
    unittest.main()
