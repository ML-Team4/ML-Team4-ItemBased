import unittest

import numpy as np
from scipy.sparse import csr_matrix

from item_based.model import ItemBasedCF


class MatrixModelTests(unittest.TestCase):
    def test_matches_direct_cosine_scores_and_preserves_input(self):
        raw = np.array([[3, 0, 0], [3, 4, 0], [0, 2, 1]])
        counts = csr_matrix(raw)
        model = ItemBasedCF(top_k=2, batch_size=1)
        model.fit_matrix(counts)
        weighted = np.log1p(raw)
        vectors = weighted / np.linalg.norm(weighted, axis=0)
        expected = weighted[0, 0] * np.dot(vectors[:, 0], vectors[:, 1])
        result = model.recommend_by_index(0, top_n=2, gamma=0.5)
        self.assertEqual([item.item_idx for item in result], [1, 2])
        self.assertAlmostEqual(result[0].score, expected / np.sqrt(2), places=6)
        self.assertEqual(result[0].popularity, 2)
        self.assertEqual(result[1].score, 0.0)
        np.testing.assert_array_equal(counts.toarray(), raw)

    def test_zero_score_ties_fill_ten_unseen_items_in_index_order(self):
        counts = csr_matrix(([1] * 13, ([0] + [1] * 12, list(range(13)))), shape=(2, 13))
        model = ItemBasedCF(top_k=1)
        model.fit_matrix(counts)
        result = model.recommend_by_index(0)
        self.assertEqual([item.item_idx for item in result], list(range(1, 11)))
        self.assertTrue(all(item.score == 0 for item in result))

    def test_invalid_user_index_rejected(self):
        model = ItemBasedCF()
        model.fit_matrix(csr_matrix([[1, 1]]))
        with self.assertRaises(ValueError):
            model.recommend_by_index(-1)


if __name__ == "__main__":
    unittest.main()
