import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p16_triggers.config import TRIGGER_NAMES, load_config  # noqa: E402


class ConfigTests(unittest.TestCase):
    def test_baseline_configuration_is_complete(self) -> None:
        config = load_config(ROOT / "configs" / "baseline.yaml")
        self.assertEqual(config.dataset.name, "CIFAR10")
        self.assertEqual(config.model.channels, (32, 64, 128))
        self.assertEqual(tuple(config.triggers), TRIGGER_NAMES)
        self.assertEqual(config.trigger_parameters("brightness")["delta"], 0.15)
        self.assertEqual(config.dataset.poison_fractions, (0.01, 0.03, 0.05))


if __name__ == "__main__":
    unittest.main()
