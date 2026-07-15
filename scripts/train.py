"""Train one clean or poisoned CIFAR-10 model, or only smoke-test the pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p16_triggers.config import TRIGGER_NAMES, load_config  # noqa: E402
from p16_triggers.data import build_cifar10_loaders  # noqa: E402
from p16_triggers.model import build_model, count_trainable_parameters  # noqa: E402
from p16_triggers.training import (  # noqa: E402
    evaluate_backdoor,
    fit,
    resolve_device,
    set_global_seed,
)


def run_name(mode: str, trigger: str, poison_fraction: float, seed: int) -> str:
    fraction = f"{poison_fraction:.3f}".replace(".", "p")
    return f"{mode}-{trigger}-rho-{fraction}-seed-{seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--mode", choices=("clean", "poisoned"), default="clean")
    parser.add_argument("--trigger", choices=TRIGGER_NAMES, default="patch")
    parser.add_argument("--poison-fraction", type=float)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--data-root")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--eval-batch-size", type=int)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Build data/model and run one forward pass without training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(ROOT / args.config)
    if args.data_root:
        data_root = Path(args.data_root)
        if not data_root.is_absolute():
            data_root = ROOT / data_root
        config = replace(config, dataset=replace(config.dataset, root=str(data_root.resolve())))
    if args.batch_size is not None or args.eval_batch_size is not None:
        config = replace(
            config,
            model=replace(
                config.model,
                batch_size=(
                    config.model.batch_size if args.batch_size is None else args.batch_size
                ),
                eval_batch_size=(
                    config.model.eval_batch_size
                    if args.eval_batch_size is None
                    else args.eval_batch_size
                ),
            ),
        )
    poison_fraction = 0.0 if args.mode == "clean" else args.poison_fraction
    if poison_fraction is None:
        raise SystemExit("--poison-fraction is required in poisoned mode")
    if not 0.0 <= poison_fraction < 1.0:
        raise SystemExit("--poison-fraction must be in [0, 1)")
    epochs = config.model.epochs if args.epochs is None else args.epochs
    if epochs < 1:
        raise SystemExit("--epochs must be positive")
    if config.model.batch_size < 1 or config.model.eval_batch_size < 1:
        raise SystemExit("batch sizes must be positive")
    if args.torch_threads is not None:
        if args.torch_threads < 1:
            raise SystemExit("--torch-threads must be positive")
        torch.set_num_threads(args.torch_threads)

    set_global_seed(args.seed)
    device = resolve_device(args.device)
    data = build_cifar10_loaders(
        config,
        seed=args.seed,
        trigger_name=args.trigger,
        poison_fraction=poison_fraction,
        download=args.download,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )
    model = build_model(config.model).to(device)
    identifier = run_name(args.mode, args.trigger, poison_fraction, args.seed)

    if args.smoke_test:
        images, labels, original_labels, trigger_flags, indices = next(iter(data.train))
        with torch.inference_mode():
            logits = model(images.to(device))
            loss = nn.CrossEntropyLoss()(logits, labels.to(device))
        print(
            json.dumps(
                {
                    "status": "ready_for_training",
                    "run": identifier,
                    "device": str(device),
                    "batch_shape": list(images.shape),
                    "logit_shape": list(logits.shape),
                    "loss": float(loss.item()),
                    "model_parameters": count_trainable_parameters(model),
                    "torch_threads": torch.get_num_threads(),
                    "poisoned_training_examples": int(len(data.poison_indices)),
                    "dataset_sizes": {
                        "train": len(data.train.dataset),
                        "validation": len(data.validation.dataset),
                        "test": len(data.test_clean.dataset),
                    },
                    "batch_original_labels_shape": list(original_labels.shape),
                    "batch_trigger_flags_shape": list(trigger_flags.shape),
                    "batch_indices_shape": list(indices.shape),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    checkpoint_path = Path(config.experiment.checkpoint_dir) / f"{identifier}.pt"
    history = fit(
        model,
        train_loader=data.train,
        validation_loader=data.validation,
        config=config,
        epochs=epochs,
        device=device,
        checkpoint_path=checkpoint_path,
    )
    metrics = evaluate_backdoor(
        model,
        clean_loader=data.test_clean,
        triggered_loader=data.test_triggered,
        target_class=config.dataset.target_class,
        device=device,
    )
    result = {
        "run": identifier,
        "mode": args.mode,
        "trigger": args.trigger,
        "poison_fraction": poison_fraction,
        "seed": args.seed,
        "epochs": epochs,
        "poisoned_training_examples": int(len(data.poison_indices)),
        "dataset_sizes": {
            "train": len(data.train.dataset),
            "validation": len(data.validation.dataset),
            "test": len(data.test_clean.dataset),
        },
        "dataset": config.dataset.name,
        "target_class": config.dataset.target_class,
        "trigger_parameters": config.trigger_parameters(args.trigger),
        "model": asdict(config.model),
        "model_parameters": count_trainable_parameters(model),
        "device": str(device),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "class_names": list(data.class_names),
        "history": history,
        "metrics": metrics,
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
    }
    output_path = Path(config.experiment.output_dir) / f"{identifier}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps({"result": str(output_path), "metrics": metrics}, ensure_ascii=False, indent=2)
    )


if __name__ == "__main__":
    main()
