import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p16_triggers.training import evaluate_backdoor  # noqa: E402


class EncodedDataset(Dataset):
    def __init__(self, *, triggered: bool) -> None:
        self.labels = [1, 2, 3]
        self.triggered = triggered

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        label = self.labels[index]
        image = torch.zeros(3, 4, 4)
        image[0, 0, 0] = label / 10
        if self.triggered:
            image[0, 0, 1] = 1.0
        return image, label, label, int(self.triggered), index


class MarkerModel(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded_class = torch.round(inputs[:, 0, 0, 0] * 10).long()
        triggered = inputs[:, 0, 0, 1] > 0.5
        predictions = torch.where(triggered, torch.zeros_like(encoded_class), encoded_class)
        logits = torch.zeros(inputs.shape[0], 10)
        logits.scatter_(1, predictions[:, None], 10.0)
        return logits


class TrainingTests(unittest.TestCase):
    def test_evaluation_separates_clean_accuracy_and_trigger_success(self) -> None:
        metrics = evaluate_backdoor(
            MarkerModel(),
            clean_loader=DataLoader(EncodedDataset(triggered=False), batch_size=3),
            triggered_loader=DataLoader(EncodedDataset(triggered=True), batch_size=3),
            target_class=0,
            device=torch.device("cpu"),
        )
        self.assertEqual(metrics["clean_accuracy"], 1.0)
        self.assertEqual(metrics["asr"], 1.0)
        self.assertEqual(metrics["conditional_asr"], 1.0)
        self.assertEqual(metrics["asr_eligible_examples"], 3)
        self.assertEqual(metrics["target_margin_mean"], 10.0)


if __name__ == "__main__":
    unittest.main()
