"""CIFAR-10 loading, deterministic splitting, and on-the-fly poisoning."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .config import ProjectConfig
from .poisoning import choose_poison_indices
from .triggers import TriggerName, apply_trigger

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def evaluation_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )


def stratified_split_indices(
    labels: np.ndarray | list[int], *, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic train/validation indices with per-class allocation."""
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)
    train_parts: list[np.ndarray] = []
    validation_parts: list[np.ndarray] = []
    for class_id in np.unique(labels_array):
        class_indices = np.flatnonzero(labels_array == class_id)
        shuffled = rng.permutation(class_indices)
        validation_count = max(1, int(round(len(class_indices) * validation_fraction)))
        validation_parts.append(shuffled[:validation_count])
        train_parts.append(shuffled[validation_count:])
    return np.sort(np.concatenate(train_parts)), np.sort(np.concatenate(validation_parts))


def stratified_limit_indices(
    indices: np.ndarray, labels: np.ndarray | list[int], *, limit: int | None, seed: int
) -> np.ndarray:
    """Limit an index set while keeping class proportions approximately stable."""
    indices = np.asarray(indices, dtype=np.int64)
    if limit is None or limit >= len(indices):
        return indices.copy()
    if limit < 1:
        raise ValueError("limit must be positive")
    local_labels = np.asarray(labels)[indices]
    classes, counts = np.unique(local_labels, return_counts=True)
    desired = counts / counts.sum() * limit
    allocation = np.floor(desired).astype(int)
    remaining = limit - int(allocation.sum())
    remainder_order = np.argsort(-(desired - allocation))
    allocation[remainder_order[:remaining]] += 1

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id, count in zip(classes, allocation, strict=True):
        class_local_indices = np.flatnonzero(local_labels == class_id)
        selected.append(rng.choice(class_local_indices, size=int(count), replace=False))
    return np.sort(indices[np.concatenate(selected)])


class TriggeredDataset(Dataset[tuple[torch.Tensor, int, int, int, int]]):
    """Wrap an image dataset and apply a trigger to selected base indices."""

    def __init__(
        self,
        base_dataset: Dataset[Any],
        *,
        indices: np.ndarray | list[int] | None = None,
        transform: Any,
        trigger_indices: np.ndarray | list[int] | None = None,
        trigger_name: TriggerName | None = None,
        trigger_parameters: dict[str, Any] | None = None,
        target_class: int = 0,
        relabel_triggered: bool = False,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = np.asarray(
            np.arange(len(base_dataset)) if indices is None else indices, dtype=np.int64
        )
        self.transform = transform
        self.trigger_indices = set(
            int(index) for index in ([] if trigger_indices is None else trigger_indices)
        )
        self.trigger_name = trigger_name
        self.trigger_parameters = trigger_parameters or {}
        self.target_class = int(target_class)
        self.relabel_triggered = relabel_triggered
        if self.trigger_indices and self.trigger_name is None:
            raise ValueError("trigger_name is required when trigger_indices are supplied")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, int, int, int, int]:
        base_index = int(self.indices[item])
        image, original_label = self.base_dataset[base_index]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image).astype(np.uint8), mode="RGB")

        is_triggered = base_index in self.trigger_indices
        if is_triggered:
            image = apply_trigger(image, self.trigger_name, **self.trigger_parameters)
        label = (
            self.target_class if is_triggered and self.relabel_triggered else int(original_label)
        )
        tensor = self.transform(image) if self.transform is not None else image
        return tensor, label, int(original_label), int(is_triggered), base_index


@dataclass(frozen=True)
class DataBundle:
    train: DataLoader
    validation: DataLoader
    test_clean: DataLoader
    test_triggered: DataLoader
    poison_indices: np.ndarray
    train_indices: np.ndarray
    validation_indices: np.ndarray
    class_names: tuple[str, ...]


def _loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def build_cifar10_loaders(
    config: ProjectConfig,
    *,
    seed: int,
    trigger_name: TriggerName,
    poison_fraction: float,
    download: bool | None = None,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> DataBundle:
    """Build all loaders needed for clean and targeted-trigger evaluation."""
    root = Path(config.dataset.root)
    should_download = config.dataset.download if download is None else download
    train_base = CIFAR10(root=str(root), train=True, transform=None, download=should_download)
    test_base = CIFAR10(root=str(root), train=False, transform=None, download=should_download)

    train_labels = np.asarray(train_base.targets, dtype=np.int64)
    test_labels = np.asarray(test_base.targets, dtype=np.int64)
    train_indices, validation_indices = stratified_split_indices(
        train_labels,
        validation_fraction=config.dataset.validation_fraction,
        seed=config.dataset.split_seed,
    )
    train_indices = stratified_limit_indices(
        train_indices, train_labels, limit=max_train_samples, seed=seed
    )
    validation_indices = stratified_limit_indices(
        validation_indices, train_labels, limit=max_eval_samples, seed=seed + 1
    )
    test_indices = stratified_limit_indices(
        np.arange(len(test_base)), test_labels, limit=max_eval_samples, seed=seed + 2
    )

    local_train_labels = train_labels[train_indices]
    selected_local_indices = choose_poison_indices(
        local_train_labels,
        target_class=config.dataset.target_class,
        poison_fraction=poison_fraction,
        seed=seed,
        stratified=config.dataset.stratified_poisoning,
    )
    poison_indices = train_indices[selected_local_indices]
    trigger_parameters = config.trigger_parameters(trigger_name)
    transform = evaluation_transform()

    train_dataset = TriggeredDataset(
        train_base,
        indices=train_indices,
        transform=transform,
        trigger_indices=poison_indices,
        trigger_name=trigger_name,
        trigger_parameters=trigger_parameters,
        target_class=config.dataset.target_class,
        relabel_triggered=True,
    )
    validation_dataset = TriggeredDataset(
        train_base, indices=validation_indices, transform=transform
    )
    clean_test_dataset = TriggeredDataset(test_base, indices=test_indices, transform=transform)
    triggered_test_indices = test_indices[test_labels[test_indices] != config.dataset.target_class]
    triggered_test_dataset = TriggeredDataset(
        test_base,
        indices=test_indices,
        transform=transform,
        trigger_indices=triggered_test_indices,
        trigger_name=trigger_name,
        trigger_parameters=trigger_parameters,
        target_class=config.dataset.target_class,
        relabel_triggered=False,
    )

    return DataBundle(
        train=_loader(
            train_dataset,
            batch_size=config.model.batch_size,
            shuffle=True,
            seed=seed,
            num_workers=config.dataset.num_workers,
            pin_memory=config.dataset.pin_memory,
        ),
        validation=_loader(
            validation_dataset,
            batch_size=config.model.eval_batch_size,
            shuffle=False,
            seed=seed,
            num_workers=config.dataset.num_workers,
            pin_memory=config.dataset.pin_memory,
        ),
        test_clean=_loader(
            clean_test_dataset,
            batch_size=config.model.eval_batch_size,
            shuffle=False,
            seed=seed,
            num_workers=config.dataset.num_workers,
            pin_memory=config.dataset.pin_memory,
        ),
        test_triggered=_loader(
            triggered_test_dataset,
            batch_size=config.model.eval_batch_size,
            shuffle=False,
            seed=seed,
            num_workers=config.dataset.num_workers,
            pin_memory=config.dataset.pin_memory,
        ),
        poison_indices=poison_indices,
        train_indices=train_indices,
        validation_indices=validation_indices,
        class_names=tuple(train_base.classes),
    )


def download_cifar10(root: str | Path) -> None:
    """Download both CIFAR-10 splits without starting model training."""
    CIFAR10(root=str(root), train=True, download=True)
    CIFAR10(root=str(root), train=False, download=True)
