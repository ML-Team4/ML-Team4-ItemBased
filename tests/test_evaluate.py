import csv
import io
import json
from contextlib import redirect_stdout
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from item_based.data import EvaluationData
from item_based.evaluate import compare_test, evaluate_split, evaluate_validation, main
from item_based.model import IndexedRecommendation


class FixedModel:
    algorithm = "lecture-item-cf-v1"
    weighting = "raw"
    top_k = 50
    batch_size = 128

    def __init__(self):
        self.calls = []
        self.items = list(range(1, 11))

    def recommend_by_index(self, user_idx, top_n, gamma):
        self.calls.append((user_idx, top_n, gamma))
        return [IndexedRecommendation(i, 0.0 if user_idx == 1 else 1.0, i + 1)
                for i in self.items]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.output = Path(temp.name)
        self.dataset = SimpleNamespace(
            counts=csr_matrix(([1, 1], ([0, 1], [0, 0])), shape=(2, 15)),
            items=pd.DataFrame({"train_listener_count": np.arange(1, 16)}),
            validation=EvaluationData(
                np.array([0, 1]),
                csr_matrix(([True] * 3, ([0, 1, 1], [1, 1, 12])), shape=(2, 15)),
            ),
            test=EvaluationData(
                np.array([1]), csr_matrix(([True], ([1], [12])), shape=(2, 15)),
            ),
        )

    def test_macro_average_and_zero_score_users_are_included(self):
        model = FixedModel()
        summary = evaluate_validation(model, self.dataset, self.output, gamma=0.5)
        # 사용자별 1/1과 1/2의 평균. 정답 쌍 전체 기준 2/3과 구별한다.
        self.assertEqual(summary["recall_at_10"], 0.75)
        self.assertEqual(summary["arp"], 6.5)
        self.assertEqual(summary["num_users"], 2)
        self.assertEqual(summary["zero_score_recommendation_count"], 10)
        self.assertEqual(model.calls, [(0, 10, 0.5), (1, 10, 0.5)])
        with (self.output / "recommendations.csv").open(encoding="utf-8-sig") as file:
            self.assertEqual(len(list(csv.DictReader(file))), 20)

    def test_invalid_recommendations_fail_without_success_summary(self):
        for items in ([1] * 10, list(range(10)), list(range(1, 10))):
            model = FixedModel()
            model.items = items
            with self.assertRaises(ValueError):
                evaluate_validation(model, self.dataset, self.output)
            self.assertFalse((self.output / "summary.json").exists())

    def test_test_split_uses_its_own_users_and_targets(self):
        model = FixedModel()
        summary = evaluate_split(model, self.dataset, self.output, split_name="test")
        self.assertEqual(summary["split"], "test")
        self.assertEqual(summary["num_users"], 1)
        self.assertEqual(summary["target_count"], 1)
        self.assertEqual(summary["recall_at_10"], 0.0)
        self.assertEqual(model.calls, [(1, 10, 0.0)])

    def test_test_candidates_do_not_exclude_validation_targets(self):
        self.dataset.test = EvaluationData(
            np.array([1]), csr_matrix(([True], ([1], [1])), shape=(2, 15)),
        )
        summary = evaluate_split(FixedModel(), self.dataset, self.output, split_name="test")
        self.assertEqual(summary["recall_at_10"], 1.0)

    def test_comparison_command_fits_once_and_reuses_model(self):
        model = FixedModel()
        model.fit_matrix = Mock()
        argv = ["evaluate", "--split", "test", "--compare-gamma", "0.1",
                "--output-dir", str(self.output)]
        with patch("sys.argv", argv), \
             patch("item_based.evaluate.load_dataset", return_value=self.dataset), \
             patch("item_based.evaluate.ItemBasedCF", return_value=model) as constructor, \
             redirect_stdout(io.StringIO()):
            main()
        constructor.assert_called_once()
        model.fit_matrix.assert_called_once_with(self.dataset.counts)
        self.assertEqual(model.calls, [(1, 10, 0.0), (1, 10, 0.1)])
        comparison = json.loads((self.output / "comparison.json").read_text())
        self.assertEqual(comparison["adjusted_gamma"], .1)
        self.assertIsNone(comparison["recall_retention"])
        self.assertEqual(comparison["baseline"]["num_users"], 1)
        self.assertTrue((self.output / "baseline" / "users.csv").is_file())
        self.assertTrue((self.output / "adjusted" / "recommendations.csv").is_file())

    def test_invalid_compare_gamma_does_not_start_evaluation(self):
        with self.assertRaises(ValueError):
            compare_test(FixedModel(), self.dataset, self.output, adjusted_gamma=0)


if __name__ == "__main__":
    unittest.main()
