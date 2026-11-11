import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from item_based.model import ItemBasedCF


class MatrixModelTests(unittest.TestCase):
    def test_project_default_keeps_proposal_log_transform(self):
        self.assertEqual(ItemBasedCF().weighting, "log1p")

    def test_lecture_pages_45_to_49_predict_movie_1(self):
        model = ItemBasedCF(top_k=2, weighting="raw")
        model.fit_matrix(csr_matrix([[2, 3, 1, 5], [3, 0, 3, 2]]))
        result = model.recommend_by_index(1, top_n=1)
        self.assertEqual(result[0].item_idx, 1)
        self.assertAlmostEqual(result[0].score, 2.373994633442469, places=6)

    def test_lecture_page_57_weighted_average_with_given_similarities(self):
        counts = csr_matrix([[0, 0, 0, 0, 0, 2, 0, 3], [1] * 8])
        neighbors = csr_matrix(([1 - .4034, 1 - .3196], ([0, 0], [5, 7])), shape=(8, 8))
        model = ItemBasedCF(top_k=2, weighting="raw")
        with patch.object(model, "_build_neighbors", return_value=neighbors):
            model.fit_matrix(counts)
        result = model.recommend_by_index(0, top_n=1)
        self.assertEqual(result[0].item_idx, 0)
        self.assertAlmostEqual(result[0].score, 2.5328112764291304, places=6)

    def test_partial_selection_matches_full_sort_with_boundary_ties(self):
        rng = np.random.default_rng(42)
        scores = rng.integers(0, 5, size=100).astype(float)
        numeric_ids = rng.permutation(100)
        for ids in (numeric_ids, numeric_ids.astype(str)):
            for limit in (1, 10, 50, 100, 101):
                expected = np.lexsort((ids, -scores))[:limit]
                actual = ItemBasedCF._top_indices(scores, ids, limit)
                np.testing.assert_array_equal(actual, expected)

    def test_candidate_neighborhood_direction_matches_dense_formula(self):
        rng = np.random.default_rng(5)
        raw = rng.uniform(1, 5, size=(6, 12))
        raw[1:] *= rng.random((5, 12)) < .3
        for weighting in ("raw", "log1p"):
            model = ItemBasedCF(top_k=2, batch_size=3, weighting=weighting)
            model.fit_matrix(csr_matrix(raw))
            values = raw if weighting == "raw" else np.log1p(raw)
            normalized = values / np.linalg.norm(values, axis=0)
            similarities = normalized.T @ normalized
            neighbors = np.zeros_like(similarities)
            for target in range(12):
                candidates = np.delete(np.arange(12), target)
                selected = candidates[np.lexsort((candidates, -similarities[target, candidates]))[:2]]
                neighbors[target, selected] = similarities[target, selected]
            for user in range(1, 6):
                scores = (values[user] @ neighbors.T) / neighbors.sum(axis=1)
                candidates = np.flatnonzero(raw[user] == 0)
                order = candidates[np.lexsort((candidates, -scores[candidates]))[:4]]
                result = model.recommend_by_index(user, top_n=4)
                self.assertEqual([item.item_idx for item in result], order.tolist())
                np.testing.assert_allclose([item.score for item in result], scores[order], rtol=1e-5)

    def test_neighbor_buffers_match_full_cosine_reference(self):
        rng = np.random.default_rng(123)
        counts = csr_matrix(rng.integers(0, 4, size=(8, 25)))
        model = ItemBasedCF(top_k=3, batch_size=4, weighting="raw")
        progress = []
        model.progress = lambda done, total: progress.append((done, total))
        model.fit_matrix(counts)
        weighted = counts.toarray().astype(np.float32)
        vectors = weighted / np.linalg.norm(weighted, axis=0)
        cosine = vectors.T @ vectors
        for item_idx in range(counts.shape[1]):
            candidates = np.delete(np.arange(counts.shape[1]), item_idx)
            expected = candidates[np.lexsort((candidates, -cosine[item_idx, candidates]))[:3]]
            row = model._neighbors.T.tocsr().getrow(item_idx)
            self.assertEqual(set(row.indices), set(expected))
            np.testing.assert_allclose(row.data, cosine[item_idx, row.indices], rtol=1e-6)
        self.assertEqual(progress[-1], (25, 25))
        self.assertEqual(model._neighbors.data.dtype, np.float32)

    def test_matches_direct_cosine_scores_and_preserves_input(self):
        raw = np.array([[3, 0, 0], [3, 4, 0], [0, 2, 1]])
        counts = csr_matrix(raw)
        model = ItemBasedCF(top_k=2, batch_size=1, weighting="raw")
        model.fit_matrix(counts)
        weighted = raw.astype(float)
        vectors = weighted / np.linalg.norm(weighted, axis=0)
        sim01 = np.dot(vectors[:, 0], vectors[:, 1])
        sim21 = np.dot(vectors[:, 2], vectors[:, 1])
        expected = weighted[0, 0] * sim01 / (sim01 + sim21)
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
