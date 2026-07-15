import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p16_triggers.config import load_config  # noqa: E402
from p16_triggers.model import build_model, count_trainable_parameters  # noqa: E402


class ModelTests(unittest.TestCase):
    def test_compact_cnn_produces_ten_logits(self) -> None:
        config = load_config(ROOT / "configs" / "baseline.yaml")
        model = build_model(config.model)
        logits = model(torch.zeros(4, 3, 32, 32))
        self.assertEqual(tuple(logits.shape), (4, 10))
        self.assertGreater(count_trainable_parameters(model), 0)


if __name__ == "__main__":
    unittest.main()
