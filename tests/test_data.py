import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p16_triggers.data import (  # noqa: E402
    TriggeredDataset,
    build_cifar10_loaders,
    stratified_limit_indices,
    stratified_split_indices,
)
from p16_triggers.config import load_config  # noqa: E402


class FakeImageDataset:
    def __init__(self) -> None:
        self.targets = [0, 1, 2, 3]
        self.images = [
            Image.fromarray(np.full((8, 8, 3), 40 * i, dtype=np.uint8)) for i in range(4)
        ]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.images[index].copy(), self.targets[index]


class FakeCIFAR10:
    classes = [f"class-{index}" for index in range(10)]

    def __init__(self, root: str, *, train: bool, transform=None, download: bool = False) -> None:
        del root, transform, download
        repeats = 20 if train else 10
        self.targets = np.tile(np.arange(10), repeats).tolist()
        self.images = [
            Image.fromarray(np.full((32, 32, 3), 20 + label * 10, dtype=np.uint8))
            for label in self.targets
        ]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int):
        return self.images[index].copy(), self.targets[index]


class DataTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_stratified(self) -> None:
        labels = np.repeat(np.arange(4), 10)
        train_a, validation_a = stratified_split_indices(labels, validation_fraction=0.2, seed=17)
        train_b, validation_b = stratified_split_indices(labels, validation_fraction=0.2, seed=17)
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(validation_a, validation_b)
        self.assertFalse(set(train_a) & set(validation_a))
        self.assertEqual(np.bincount(labels[validation_a]).tolist(), [2, 2, 2, 2])

    def test_stratified_limit_has_exact_size(self) -> None:
        labels = np.repeat(np.arange(4), 10)
        selected = stratified_limit_indices(np.arange(len(labels)), labels, limit=12, seed=17)
        self.assertEqual(len(selected), 12)
        self.assertEqual(np.bincount(labels[selected]).tolist(), [3, 3, 3, 3])

    def test_triggered_dataset_relabels_only_selected_examples(self) -> None:
        dataset = TriggeredDataset(
            FakeImageDataset(),
            transform=transforms.ToTensor(),
            trigger_indices=[2],
            trigger_name="patch",
            trigger_parameters={"side": 2, "alpha": 1.0},
            target_class=0,
            relabel_triggered=True,
        )
        _image, label, original_label, flag, base_index = dataset[2]
        self.assertEqual((label, original_label, flag, base_index), (0, 2, 1, 2))
        _image, label, original_label, flag, base_index = dataset[1]
        self.assertEqual((label, original_label, flag, base_index), (1, 1, 0, 1))

    @patch("p16_triggers.data.CIFAR10", FakeCIFAR10)
    def test_full_loader_builder_is_ready_without_training(self) -> None:
        config = load_config(ROOT / "configs" / "baseline.yaml")
        bundle = build_cifar10_loaders(
            config,
            seed=17,
            trigger_name="patch",
            poison_fraction=0.1,
            download=False,
            max_train_samples=80,
            max_eval_samples=40,
        )
        images, labels, original_labels, flags, indices = next(iter(bundle.train))
        self.assertEqual(tuple(images.shape[1:]), (3, 32, 32))
        self.assertEqual(labels.shape, original_labels.shape)
        self.assertEqual(flags.shape, indices.shape)
        self.assertEqual(len(bundle.poison_indices), 8)
        self.assertEqual(bundle.class_names, tuple(FakeCIFAR10.classes))


if __name__ == "__main__":
    unittest.main()
