import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from p16_triggers.reporting import summarize_records  # noqa: E402


class ReportingTests(unittest.TestCase):
    def test_poisoned_run_gets_accuracy_drop_from_matching_clean_seed(self) -> None:
        shared_metrics = {
            "triggered_accuracy": 0.4,
            "asr": 0.9,
            "conditional_asr": 0.95,
            "target_margin_mean": 2.0,
            "target_margin_median": 1.5,
        }
        records = [
            {
                "run": "clean",
                "mode": "clean",
                "trigger": "patch",
                "poison_fraction": 0.0,
                "seed": 17,
                "epochs": 20,
                "poisoned_training_examples": 0,
                "metrics": {"clean_accuracy": 0.8, **shared_metrics},
            },
            {
                "run": "poisoned",
                "mode": "poisoned",
                "trigger": "patch",
                "poison_fraction": 0.03,
                "seed": 17,
                "epochs": 20,
                "poisoned_training_examples": 1350,
                "metrics": {"clean_accuracy": 0.78, **shared_metrics},
            },
        ]
        rows = summarize_records(records)
        self.assertAlmostEqual(rows[1]["delta_accuracy"], 0.02)


if __name__ == "__main__":
    unittest.main()
