from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Recommendation:
    track_id: str
    score: float
    popularity: int


class ItemBasedCF:
    def __init__(self, top_k: int = 50, batch_size: int = 128):
        if top_k < 1 or batch_size < 1:
            raise ValueError("top_k and batch_size must be positive")
        self.top_k = top_k
        self.batch_size = batch_size
        self._history = None

    def fit(self, interactions: Iterable[tuple[str, str, float]]) -> None:
        """Fit from training (user_id, track_id, listen_count) rows."""
        user_indices = {}
        track_indices = {}
        users, tracks, counts = [], [], []
        for user_id, track_id, count in interactions:
            if not isfinite(count) or count <= 0:
                raise ValueError("listen_count must be finite and positive")
            users.append(user_indices.setdefault(user_id, len(user_indices)))
            tracks.append(track_indices.setdefault(track_id, len(track_indices)))
            counts.append(count)
        if not counts:
            raise ValueError("at least one training interaction is required")

        history = csr_matrix(
            (counts, (users, tracks)),
            shape=(len(user_indices), len(track_indices)),
            dtype=float,
        )
        popularity = history.getnnz(axis=0)
        history.data = np.log1p(history.data)
        track_ids = np.asarray(list(track_indices))
        neighbors = self._build_neighbors(history, track_ids)

        self._user_indices = user_indices
        self._track_ids = track_ids
        self._popularity = popularity
        self._neighbors = neighbors
        self._history = history

    def _build_neighbors(self, history: csr_matrix, track_ids: np.ndarray) -> csr_matrix:
        item_vectors = history.T.tocsr()
        norms = np.sqrt(np.asarray(item_vectors.power(2).sum(axis=1)).ravel())
        normalized = item_vectors.multiply(1 / norms[:, None]).tocsr()
        normalized_transpose = normalized.T.tocsr()
        num_tracks = normalized.shape[0]
        indptr, indices, values = [0], [], []

        for start in range(0, num_tracks, self.batch_size):
            block = normalized[start : start + self.batch_size] @ normalized_transpose
            for offset in range(block.shape[0]):
                left, right = block.indptr[offset : offset + 2]
                candidates = block.indices[left:right]
                similarities = block.data[left:right]
                other_tracks = candidates != start + offset
                candidates = candidates[other_tracks]
                similarities = similarities[other_tracks]

                order = np.lexsort((track_ids[candidates], -similarities))[:self.top_k]
                indices.extend(candidates[order])
                values.extend(similarities[order])
                indptr.append(len(indices))

        return csr_matrix((values, indices, indptr), shape=(num_tracks, num_tracks))

    def recommend(
        self, user_id: str, top_n: int = 10, gamma: float = 0.0
    ) -> list[Recommendation]:
        if self._history is None:
            raise RuntimeError("fit must be called before recommend")
        if top_n < 1 or not 0 <= gamma <= 1:
            raise ValueError("top_n must be positive and gamma must be between 0 and 1")
        if user_id not in self._user_indices:
            return []

        history = self._history.getrow(self._user_indices[user_id])
        scores = history @ self._neighbors
        unseen = ~np.isin(scores.indices, history.indices)
        candidates = scores.indices[unseen]
        popularity = self._popularity[candidates]
        adjusted = scores.data[unseen] / popularity.astype(float) ** gamma
        order = np.lexsort((self._track_ids[candidates], -adjusted))[:top_n]

        return [
            Recommendation(
                track_id=str(self._track_ids[candidates[index]]),
                score=float(adjusted[index]),
                popularity=int(popularity[index]),
            )
            for index in order
        ]
