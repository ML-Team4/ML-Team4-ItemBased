import json
import tempfile
import unittest
from pathlib import Path

from item_based.report import format_execution_results, load_report_data


class ReportTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        base = {
            "split": "test", "top_k": 20, "gamma": 0.0, "top_n": 10,
            "algorithm": "example-v1", "weighting": "log1p", "num_users": 4,
            "target_count": 12, "recommendation_count": 40,
            "recall_at_10": .2, "arp": 10.0, "candidate_rule": "original train excluded",
        }
        adjusted = {**base, "gamma": .1, "recall_at_10": .19, "arp": 8.0}
        self.write("test_k20/comparison.json", {
            "split": "test", "top_k": 20, "adjusted_gamma": .1,
            "baseline": base, "adjusted": adjusted,
        })
        for k, gamma in ((20, 0), (50, 0), (100, 0), (20, .1), (20, .25)):
            label = f"{gamma:g}".replace(".", "")
            self.write(f"validation_k{k}_gamma{label}/summary.json",
                       {**base, "split": "validation", "top_k": k, "gamma": gamma})

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content))

    def test_report_uses_saved_values_and_recomputes_relative_changes(self):
        data = load_report_data(self.root)
        self.assertEqual([r["top_k"] for r in data.k_results], [20, 50, 100])
        self.assertEqual([r["gamma"] for r in data.gamma_results], [0, .1, .25])
        output = format_execution_results(data)
        self.assertIn("Recall@10=0.190000, ARP=8.000000", output)
        self.assertIn("Test Recall retention: 95.00%", output)
        self.assertIn("Test Recall change: -5.00%", output)
        self.assertIn("Test ARP reduction: 20.00%", output)
        self.assertEqual(len(data.sources), 6)

    def test_rejects_folder_metadata_mismatch(self):
        path = self.root / "validation_k50_gamma0/summary.json"
        result = json.loads(path.read_text())
        result["top_k"] = 100
        self.write(str(path.relative_to(self.root)), result)
        with self.assertRaisesRegex(ValueError, "top_k"):
            load_report_data(self.root)

    def test_rejects_mixed_validation_populations(self):
        path = self.root / "validation_k50_gamma0/summary.json"
        result = json.loads(path.read_text())
        result.update(num_users=5, recommendation_count=50)
        self.write(str(path.relative_to(self.root)), result)
        with self.assertRaisesRegex(ValueError, "검증 결과"):
            load_report_data(self.root)

    def test_missing_result_fails_instead_of_using_placeholder(self):
        (self.root / "validation_k20_gamma01/summary.json").unlink()
        with self.assertRaisesRegex(ValueError, "읽을 수 없습니다"):
            load_report_data(self.root)


if __name__ == "__main__":
    unittest.main()
