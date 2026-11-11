from collections.abc import Callable, Iterable
from dataclasses import dataclass
from math import isfinite

import numpy as np
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Recommendation:
    track_id: str
    score: float
    popularity: int


@dataclass(frozen=True)
class IndexedRecommendation:
    item_idx: int
    score: float
    popularity: int


class ItemBasedCF:
    def __init__(
        self, top_k: int = 50, batch_size: int = 128,
        progress: Callable[[int, int], None] | None = None,
    ):
        if top_k < 1 or batch_size < 1:
            raise ValueError("top_k and batch_size must be positive")
        self.top_k = top_k
        self.batch_size = batch_size
        self.progress = progress
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
        self._uses_common_indices = False

    def fit_matrix(self, counts: csr_matrix) -> None:
        """학습 CSR의 행·열 번호를 공통 user_idx·item_idx로 유지한다."""
        if not isinstance(counts, csr_matrix) or min(counts.shape) < 1:
            raise ValueError("counts must be a nonempty CSR matrix")
        history = counts.astype(np.float32, copy=True)
        history.sum_duplicates()
        history.eliminate_zeros()
        if history.nnz == 0 or not np.all(np.isfinite(history.data)) or np.any(history.data <= 0):
            raise ValueError("listen_count must be finite and positive")
        popularity = history.getnnz(axis=0)
        if np.any(popularity == 0):
            raise ValueError("every catalog item must have a training listener")
        history.data = np.log1p(history.data)
        track_ids = np.arange(history.shape[1])
        neighbors = self._build_neighbors(history, track_ids)
        self._user_indices = {i: i for i in range(history.shape[0])}
        self._track_ids = track_ids
        self._popularity = popularity
        self._neighbors = neighbors
        self._history = history
        self._uses_common_indices = True

    def recommend_by_index(
        self, user_idx: int, top_n: int = 10, gamma: float = 0.0
    ) -> list[IndexedRecommendation]:
        """공통 인덱스로 추천하며, 0점 동률은 item_idx 오름차순으로 채운다."""
        if self._history is None or not self._uses_common_indices:
            raise RuntimeError("fit_matrix must be called before recommend_by_index")
        if not isinstance(user_idx, (int, np.integer)) or not 0 <= user_idx < self._history.shape[0]:
            raise ValueError("user_idx is outside the common user index range")
        recommendations = [
            IndexedRecommendation(int(item.track_id), item.score, item.popularity)
            for item in self.recommend(user_idx, top_n, gamma)
        ]
        if len(recommendations) < top_n:
            available = np.ones(self._history.shape[1], dtype=bool)
            available[self._history.getrow(user_idx).indices] = False
            available[[item.item_idx for item in recommendations]] = False
            for item_idx in np.flatnonzero(available)[:top_n - len(recommendations)]:
                recommendations.append(IndexedRecommendation(
                    int(item_idx), 0.0, int(self._popularity[item_idx])
                ))
        return recommendations

    def _build_neighbors(self, history: csr_matrix, track_ids: np.ndarray) -> csr_matrix:
        item_vectors = history.T.tocsr()
        norms = np.sqrt(np.asarray(item_vectors.power(2).sum(axis=1)).ravel())
        normalized = item_vectors.multiply(1 / norms[:, None]).tocsr()
        normalized_transpose = normalized.T.tocsr()
        num_tracks = normalized.shape[0]
        # Python 숫자 객체 목록 대신 고정 크기 NumPy 버퍼에 상위 이웃만 저장한다.
        capacity = num_tracks * min(self.top_k, max(0, num_tracks - 1))
        index_dtype = np.int32 if max(capacity, num_tracks) <= np.iinfo(np.int32).max else np.int64
        indptr = np.zeros(num_tracks + 1, dtype=index_dtype)
        indices = np.empty(capacity, dtype=index_dtype)
        values = np.empty(capacity, dtype=history.dtype)
        cursor = 0

        for start in range(0, num_tracks, self.batch_size):
            block = normalized[start : start + self.batch_size] @ normalized_transpose
            for offset in range(block.shape[0]):
                left, right = block.indptr[offset : offset + 2]
                candidates = block.indices[left:right]
                similarities = block.data[left:right]
                other_tracks = candidates != start + offset
                candidates = candidates[other_tracks]
                similarities = similarities[other_tracks]

                order = self._top_indices(similarities, track_ids[candidates], self.top_k)
                end = cursor + len(order)
                indices[cursor:end] = candidates[order]
                values[cursor:end] = similarities[order]
                cursor = end
                indptr[start + offset + 1] = cursor
            if self.progress is not None:
                self.progress(min(start + self.batch_size, num_tracks), num_tracks)

        return csr_matrix(
            (values[:cursor], indices[:cursor], indptr),
            shape=(num_tracks, num_tracks), copy=False,
        )

    @staticmethod
    def _top_indices(scores: np.ndarray, ids: np.ndarray, limit: int) -> np.ndarray:
        """경계 점수의 동률까지 고려해 전체 정렬과 같은 상위 K를 선택한다."""
        if len(scores) <= limit:
            return np.lexsort((ids, -scores))
        threshold = np.partition(scores, len(scores) - limit)[len(scores) - limit]
        higher = np.flatnonzero(scores > threshold)
        tied = np.flatnonzero(scores == threshold)
        remaining = limit - len(higher)
        if len(tied) > remaining:
            chosen = np.argpartition(ids[tied], remaining - 1)[:remaining]
            tied = tied[chosen]
        selected = np.concatenate((higher, tied))
        return selected[np.lexsort((ids[selected], -scores[selected]))]

    def recommend(
        self, user_id: str | int, top_n: int = 10, gamma: float = 0.0
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
        order = self._top_indices(adjusted, self._track_ids[candidates], top_n)

        return [
            Recommendation(
                track_id=str(self._track_ids[candidates[index]]),
                score=float(adjusted[index]),
                popularity=int(popularity[index]),
            )
            for index in order
        ]
