import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from item_based.data import EvaluationData
from item_based.evaluate import evaluate_validation
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


if __name__ == "__main__":
    unittest.main()
