"""Typed project configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TRIGGER_NAMES = ("patch", "stripe", "brightness", "position", "color")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seeds: tuple[int, ...]
    output_dir: str
    checkpoint_dir: str


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    root: str
    target_class: int
    poison_fractions: tuple[float, ...]
    stratified_poisoning: bool
    validation_fraction: float
    split_seed: int
    num_workers: int
    pin_memory: bool
    download: bool


@dataclass(frozen=True)
class ModelConfig:
    name: str
    channels: tuple[int, ...]
    dropout: float
    epochs: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True)
class EvaluationConfig:
    exclude_target_class_from_asr: bool
    report_conditional_asr: bool
    report_per_source_class_asr: bool
    report_target_logit_margin: bool


@dataclass(frozen=True)
class ProjectConfig:
    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    triggers: dict[str, dict[str, Any]]
    evaluation: EvaluationConfig

    def trigger_parameters(self, trigger_name: str) -> dict[str, Any]:
        if trigger_name not in self.triggers:
            raise ValueError(f"Trigger {trigger_name!r} is not configured")
        return dict(self.triggers[trigger_name])


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require(mapping: dict[str, Any], key: str, section: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing {section}.{key}")
    return mapping[key]


def _validate(config: ProjectConfig) -> None:
    if config.dataset.name != "CIFAR10":
        raise ValueError("Only CIFAR10 is supported by the current training pipeline")
    if not 0 <= config.dataset.target_class < 10:
        raise ValueError("dataset.target_class must be in [0, 9]")
    if not 0.0 < config.dataset.validation_fraction < 1.0:
        raise ValueError("dataset.validation_fraction must be in (0, 1)")
    if config.dataset.num_workers < 0:
        raise ValueError("dataset.num_workers must be non-negative")
    if not config.experiment.seeds:
        raise ValueError("experiment.seeds cannot be empty")
    if any(not 0.0 <= fraction < 1.0 for fraction in config.dataset.poison_fractions):
        raise ValueError("Every poison fraction must be in [0, 1)")
    if config.model.channels != (32, 64, 128):
        raise ValueError("compact_cnn currently expects channels [32, 64, 128]")
    if not 0.0 <= config.model.dropout < 1.0:
        raise ValueError("model.dropout must be in [0, 1)")
    if config.model.epochs < 1 or config.model.batch_size < 1 or config.model.eval_batch_size < 1:
        raise ValueError("epochs and batch sizes must be positive")
    if config.model.learning_rate <= 0.0 or config.model.weight_decay < 0.0:
        raise ValueError("Invalid optimizer settings")

    missing_triggers = set(TRIGGER_NAMES) - set(config.triggers)
    extra_triggers = set(config.triggers) - set(TRIGGER_NAMES)
    if missing_triggers or extra_triggers:
        raise ValueError(
            f"Configured triggers mismatch; missing={sorted(missing_triggers)}, "
            f"extra={sorted(extra_triggers)}"
        )


def load_config(path: str | Path) -> ProjectConfig:
    """Load and validate a project configuration file."""
    path = Path(path).resolve()
    project_root = path.parent.parent

    def project_path(value: Any) -> str:
        candidate = Path(str(value))
        return str(candidate if candidate.is_absolute() else project_root / candidate)

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    root = _mapping(raw, "root")
    experiment = _mapping(_require(root, "experiment", "root"), "experiment")
    dataset = _mapping(_require(root, "dataset", "root"), "dataset")
    model = _mapping(_require(root, "model", "root"), "model")
    triggers = _mapping(_require(root, "triggers", "root"), "triggers")
    evaluation = _mapping(_require(root, "evaluation", "root"), "evaluation")

    config = ProjectConfig(
        experiment=ExperimentConfig(
            name=str(_require(experiment, "name", "experiment")),
            seeds=tuple(int(seed) for seed in _require(experiment, "seeds", "experiment")),
            output_dir=project_path(_require(experiment, "output_dir", "experiment")),
            checkpoint_dir=project_path(_require(experiment, "checkpoint_dir", "experiment")),
        ),
        dataset=DatasetConfig(
            name=str(_require(dataset, "name", "dataset")),
            root=project_path(_require(dataset, "root", "dataset")),
            target_class=int(_require(dataset, "target_class", "dataset")),
            poison_fractions=tuple(
                float(value) for value in _require(dataset, "poison_fractions", "dataset")
            ),
            stratified_poisoning=bool(_require(dataset, "stratified_poisoning", "dataset")),
            validation_fraction=float(_require(dataset, "validation_fraction", "dataset")),
            split_seed=int(_require(dataset, "split_seed", "dataset")),
            num_workers=int(_require(dataset, "num_workers", "dataset")),
            pin_memory=bool(_require(dataset, "pin_memory", "dataset")),
            download=bool(_require(dataset, "download", "dataset")),
        ),
        model=ModelConfig(
            name=str(_require(model, "name", "model")),
            channels=tuple(int(value) for value in _require(model, "channels", "model")),
            dropout=float(_require(model, "dropout", "model")),
            epochs=int(_require(model, "epochs", "model")),
            batch_size=int(_require(model, "batch_size", "model")),
            eval_batch_size=int(_require(model, "eval_batch_size", "model")),
            learning_rate=float(_require(model, "learning_rate", "model")),
            weight_decay=float(_require(model, "weight_decay", "model")),
        ),
        triggers={
            str(name): _mapping(parameters, f"triggers.{name}")
            for name, parameters in triggers.items()
        },
        evaluation=EvaluationConfig(
            exclude_target_class_from_asr=bool(
                _require(evaluation, "exclude_target_class_from_asr", "evaluation")
            ),
            report_conditional_asr=bool(
                _require(evaluation, "report_conditional_asr", "evaluation")
            ),
            report_per_source_class_asr=bool(
                _require(evaluation, "report_per_source_class_asr", "evaluation")
            ),
            report_target_logit_margin=bool(
                _require(evaluation, "report_target_logit_margin", "evaluation")
            ),
        ),
    )
    _validate(config)
    return config
