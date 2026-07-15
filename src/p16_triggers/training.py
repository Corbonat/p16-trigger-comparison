"""Training and evaluation loops for the trigger comparison experiment."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .config import ProjectConfig
from .metrics import (
    accuracy,
    attack_success_rate,
    conditional_attack_success_rate,
    per_source_class_asr,
    target_logit_margin,
)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


@dataclass(frozen=True)
class PredictionBundle:
    true_labels: np.ndarray
    predictions: np.ndarray
    logits: np.ndarray
    trigger_flags: np.ndarray
    indices: np.ndarray


def collect_predictions(model: nn.Module, loader: Any, *, device: torch.device) -> PredictionBundle:
    model.eval()
    true_labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    trigger_flags: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    with torch.inference_mode():
        for images, _labels, original_labels, flags, base_indices in loader:
            output = model(images.to(device))
            true_labels.append(original_labels.numpy())
            predictions.append(output.argmax(dim=1).cpu().numpy())
            logits.append(output.cpu().numpy())
            trigger_flags.append(flags.numpy())
            indices.append(base_indices.numpy())
    return PredictionBundle(
        true_labels=np.concatenate(true_labels),
        predictions=np.concatenate(predictions),
        logits=np.concatenate(logits),
        trigger_flags=np.concatenate(trigger_flags),
        indices=np.concatenate(indices),
    )


def evaluate_backdoor(
    model: nn.Module,
    *,
    clean_loader: Any,
    triggered_loader: Any,
    target_class: int,
    device: torch.device,
) -> dict[str, Any]:
    clean = collect_predictions(model, clean_loader, device=device)
    triggered = collect_predictions(model, triggered_loader, device=device)
    if not np.array_equal(clean.indices, triggered.indices):
        raise ValueError("Clean and triggered test loaders must have identical ordering")
    if not np.array_equal(clean.true_labels, triggered.true_labels):
        raise ValueError("Clean and triggered test labels do not match")

    eligible = triggered.true_labels != target_class
    margins = target_logit_margin(triggered.logits, target_class=target_class)[eligible]
    return {
        "clean_accuracy": accuracy(clean.true_labels, clean.predictions),
        "triggered_accuracy": accuracy(triggered.true_labels, triggered.predictions),
        "asr": attack_success_rate(
            triggered.true_labels, triggered.predictions, target_class=target_class
        ),
        "conditional_asr": conditional_attack_success_rate(
            clean.true_labels,
            clean.predictions,
            triggered.predictions,
            target_class=target_class,
        ),
        "per_source_class_asr": per_source_class_asr(
            triggered.true_labels, triggered.predictions, target_class=target_class
        ),
        "target_margin_mean": float(np.mean(margins)),
        "target_margin_median": float(np.median(margins)),
        "evaluated_examples": int(len(clean.true_labels)),
        "asr_eligible_examples": int(np.sum(eligible)),
    }


def train_one_epoch(
    model: nn.Module,
    loader: Any,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    examples = 0
    for images, labels, _original_labels, _flags, _indices in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        examples += batch_size
    return {"loss": total_loss / examples, "accuracy": correct / examples}


def evaluate_loss_and_accuracy(
    model: nn.Module, loader: Any, *, criterion: nn.Module, device: torch.device
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    examples = 0
    with torch.inference_mode():
        for images, _labels, original_labels, _flags, _indices in loader:
            images = images.to(device)
            labels = original_labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            examples += batch_size
    return {"loss": total_loss / examples, "accuracy": correct / examples}


def fit(
    model: nn.Module,
    *,
    train_loader: Any,
    validation_loader: Any,
    config: ProjectConfig,
    epochs: int,
    device: torch.device,
    checkpoint_path: Path,
) -> list[dict[str, Any]]:
    """Train a model and keep the state with the best validation accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.model.learning_rate,
        weight_decay=config.model.weight_decay,
    )
    model.to(device)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    best_validation_accuracy = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer=optimizer, criterion=criterion, device=device
        )
        validation_metrics = evaluate_loss_and_accuracy(
            model, validation_loader, criterion=criterion, device=device
        )
        row = {"epoch": epoch, "train": train_metrics, "validation": validation_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        if validation_metrics["accuracy"] > best_validation_accuracy:
            best_validation_accuracy = validation_metrics["accuracy"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "model_config": asdict(config.model),
                    "best_validation_accuracy": best_validation_accuracy,
                },
                checkpoint_path,
            )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return history
