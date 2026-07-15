"""Helpers for choosing and transforming poisoned training examples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from .triggers import TriggerName, apply_trigger


def choose_poison_indices(
    labels: Sequence[int] | np.ndarray,
    *,
    target_class: int,
    poison_fraction: float,
    seed: int,
    stratified: bool = True,
) -> np.ndarray:
    """Choose non-target examples so poison_fraction refers to the full dataset."""
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if not 0.0 <= poison_fraction <= 1.0:
        raise ValueError("poison_fraction must be in [0, 1]")

    candidates = np.flatnonzero(labels_array != target_class)
    number_to_choose = int(round(poison_fraction * len(labels_array)))
    if number_to_choose > len(candidates):
        raise ValueError("Requested poisoning exceeds the number of non-target examples")
    if number_to_choose == 0:
        return np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    if not stratified:
        return np.sort(rng.choice(candidates, size=number_to_choose, replace=False))

    classes, counts = np.unique(labels_array[candidates], return_counts=True)
    desired = counts / counts.sum() * number_to_choose
    allocation = np.floor(desired).astype(int)
    remaining = number_to_choose - int(allocation.sum())
    remainder_order = np.argsort(-(desired - allocation))
    allocation[remainder_order[:remaining]] += 1

    selected: list[np.ndarray] = []
    for source_class, count in zip(classes, allocation, strict=True):
        class_candidates = np.flatnonzero(labels_array == source_class)
        selected.append(rng.choice(class_candidates, size=int(count), replace=False))
    return np.sort(np.concatenate(selected).astype(np.int64))


def poison_examples(
    images: Sequence[Image.Image | np.ndarray],
    labels: Sequence[int] | np.ndarray,
    *,
    indices: Sequence[int] | np.ndarray,
    target_class: int,
    trigger: TriggerName,
    trigger_parameters: dict[str, Any] | None = None,
) -> tuple[list[Image.Image | np.ndarray], np.ndarray]:
    """Return copied images and labels with a trigger applied at selected indices."""
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")

    poisoned_images: list[Image.Image | np.ndarray] = []
    for image in images:
        if isinstance(image, Image.Image):
            poisoned_images.append(image.copy())
        else:
            poisoned_images.append(np.asarray(image).copy())

    poisoned_labels = np.asarray(labels).copy()
    parameters = trigger_parameters or {}
    for index in np.asarray(indices, dtype=np.int64):
        poisoned_images[int(index)] = apply_trigger(
            poisoned_images[int(index)], trigger, **parameters
        )
        poisoned_labels[int(index)] = target_class
    return poisoned_images, poisoned_labels
