"""공통 전처리 CSV 로딩과 학습·평가 입력 검증."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.sparse import csr_matrix


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EvaluationData:
    user_indices: np.ndarray
    targets: csr_matrix


@dataclass
class Dataset:
    users: pd.DataFrame
    items: pd.DataFrame
    counts: csr_matrix
    validation: EvaluationData
    test: EvaluationData


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    value = data_dir if data_dir is not None else os.environ.get("DATA_DIR")
    if value is None or not str(value).strip():
        raise ValueError(".env에 DATA_DIR을 설정하거나 --data-dir을 지정하세요.")
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_csv(root: Path, name: str, dtypes: dict) -> pd.DataFrame:
    frame = pd.read_csv(
        root / name, encoding="utf-8-sig", keep_default_na=False, dtype=dtypes
    )
    if set(frame.columns) != set(dtypes):
        raise ValueError(f"{name}: 필요한 컬럼은 {list(dtypes)}입니다.")
    return frame


def _check_indices(values: np.ndarray, size: int, label: str) -> None:
    if np.any(values < 0) or np.any(values >= size):
        raise ValueError(f"{label}: 공통 인덱스 범위를 벗어났습니다.")


def _load_evaluation(root: Path, period: str, counts: csr_matrix) -> EvaluationData:
    users = _read_csv(root, f"{period}_users.csv", {"user_idx": "int64"})
    pairs = _read_csv(
        root, f"{period}_targets.csv", {"user_idx": "int64", "item_idx": "int64"}
    )
    eligible = users["user_idx"].to_numpy()
    rows, cols = pairs["user_idx"].to_numpy(), pairs["item_idx"].to_numpy()
    _check_indices(eligible, counts.shape[0], f"{period} users")
    _check_indices(rows, counts.shape[0], f"{period} target users")
    _check_indices(cols, counts.shape[1], f"{period} target items")
    if len(np.unique(eligible)) != len(eligible):
        raise ValueError(f"{period}: 평가 사용자가 중복됩니다.")
    if not np.array_equal(np.sort(eligible), np.unique(rows)):
        raise ValueError(f"{period}: 평가 사용자와 정답 사용자가 다릅니다.")
    if pairs.duplicated().any():
        raise ValueError(f"{period}: 정답 쌍이 중복됩니다.")
    if len(rows) and np.any(np.asarray(counts[rows, cols]).ravel() != 0):
        raise ValueError(f"{period}: 정답에 사용자의 학습 청취곡이 포함됩니다.")
    history_sizes = np.diff(counts.indptr)[eligible]
    if np.any(history_sizes == 0) or np.any(counts.shape[1] - history_sizes < 10):
        raise ValueError(f"{period}: 학습 이력 또는 후보 10곡 조건을 만족하지 않습니다.")
    targets = csr_matrix(
        (np.ones(len(rows), dtype=bool), (rows, cols)), shape=counts.shape
    )
    return EvaluationData(eligible, targets)


def load_dataset(data_dir: str | Path | None = None) -> Dataset:
    root = resolve_data_dir(data_dir)
    filenames = [
        "users.csv", "items.csv", "train_interactions.csv",
        "validation_users.csv", "validation_targets.csv",
        "test_users.csv", "test_targets.csv",
    ]
    missing = [name for name in filenames if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{root}: 필요한 파일이 없습니다: {', '.join(missing)}")
    users = _read_csv(root, "users.csv", {"user_idx": "int64", "user_id": "int64"})
    items = _read_csv(root, "items.csv", {
        "item_idx": "int64", "artist_name": str, "recording_name": str,
        "train_listener_count": "int64", "artist_key": str, "recording_key": str,
    })
    users = users.sort_values("user_idx", ignore_index=True)
    items = items.sort_values("item_idx", ignore_index=True)
    for frame, key in ((users, "user_idx"), (items, "item_idx")):
        if frame.empty or not np.array_equal(frame[key].to_numpy(), np.arange(len(frame))):
            raise ValueError(f"{key}: 0부터 시작하는 연속된 고유 인덱스가 필요합니다.")
    if users["user_id"].duplicated().any():
        raise ValueError("users.csv: 원본 사용자 ID가 중복됩니다.")
    train = _read_csv(root, "train_interactions.csv", {
        "user_idx": "int64", "item_idx": "int64", "listen_count": "int64",
    })
    rows, cols = train["user_idx"].to_numpy(), train["item_idx"].to_numpy()
    values = train["listen_count"].to_numpy()
    _check_indices(rows, len(users), "train users")
    _check_indices(cols, len(items), "train items")
    if train.empty or np.any(values <= 0):
        raise ValueError("train_interactions.csv: 양수 청취 횟수가 필요합니다.")
    counts = csr_matrix((values, (rows, cols)), shape=(len(users), len(items)))
    if counts.nnz != len(train):
        raise ValueError("train_interactions.csv: 사용자·곡 쌍이 중복됩니다.")
    popularity = counts.getnnz(axis=0)
    if np.any(popularity == 0) or not np.array_equal(
        popularity, items["train_listener_count"].to_numpy()
    ):
        raise ValueError("items.csv: 학습 고유 청취자 수가 학습 행렬과 다릅니다.")
    return Dataset(
        users, items, counts,
        _load_evaluation(root, "validation", counts),
        _load_evaluation(root, "test", counts),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="공통 CSV 7개를 읽고 입력을 검증합니다.")
    parser.add_argument("--data-dir", type=Path, help=".env의 DATA_DIR 대신 사용할 경로")
    args = parser.parse_args()
    try:
        dataset = load_dataset(args.data_dir)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"데이터 로딩 실패: {exc}\n")
    print(f"학습 행렬: {dataset.counts.shape[0]:,}명 × {dataset.counts.shape[1]:,}곡")
    print(f"학습 사용자·곡 쌍: {dataset.counts.nnz:,}")
    print(f"학습 청취 횟수 합계: {int(dataset.counts.sum()):,}")
    for label, split in (("검증", dataset.validation), ("테스트", dataset.test)):
        print(f"{label}: {len(split.user_indices):,}명 / 정답 {split.targets.nnz:,}쌍")
    print("공통 인덱스·인기도·평가 정답 검증 완료")


if __name__ == "__main__":
    main()
