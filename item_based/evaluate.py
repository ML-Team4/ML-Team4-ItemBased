"""공통 검증·테스트 사용자 전체의 Recall@10과 ARP를 계산한다."""

import argparse
import csv
import json
from pathlib import Path
from time import monotonic

import numpy as np

from item_based.data import Dataset, load_dataset
from item_based.model import ItemBasedCF


def evaluate_split(
    model: ItemBasedCF, dataset: Dataset, output_dir: Path,
    gamma: float = 0.0, progress=None, split_name: str = "validation",
) -> dict:
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be between 0 and 1")
    if split_name not in ("validation", "test"):
        raise ValueError("split_name must be validation or test")
    split = getattr(dataset, split_name)
    user_indices = split.user_indices
    if len(user_indices) == 0:
        raise ValueError("평가 사용자가 없습니다.")
    if len(np.unique(user_indices)) != len(user_indices):
        raise ValueError("평가 사용자가 중복됩니다.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    # 실패한 재실행에서 이전 성공 요약이 남지 않도록 한다.
    summary_path.unlink(missing_ok=True)
    popularity = dataset.items["train_listener_count"].to_numpy()
    recall_sum = arp_sum = 0.0
    zero_score_count = zero_score_users = 0
    started = monotonic()
    with (output_dir / "users.csv").open("w", encoding="utf-8-sig", newline="") as user_file, \
         (output_dir / "recommendations.csv").open("w", encoding="utf-8-sig", newline="") as rec_file:
        users_writer, rec_writer = csv.writer(user_file), csv.writer(rec_file)
        users_writer.writerow([
            "user_idx", "target_count", "hit_count", "recommendation_count",
            "recall_at_10", "arp", "zero_score_count",
        ])
        rec_writer.writerow(["user_idx", "rank", "item_idx", "score", "train_listener_count"])
        for position, user_idx in enumerate(user_indices, start=1):
            user_idx = int(user_idx)
            truth = set(split.targets.getrow(user_idx).indices.tolist())
            if not truth:
                raise ValueError(f"사용자 {user_idx}: 평가 정답이 없습니다.")
            recommendations = model.recommend_by_index(user_idx, top_n=10, gamma=gamma)
            item_indices = [item.item_idx for item in recommendations]
            if len(item_indices) != 10 or len(set(item_indices)) != 10:
                raise ValueError(f"사용자 {user_idx}: 서로 다른 추천 10곡이 필요합니다.")
            if any(index < 0 or index >= len(popularity) for index in item_indices):
                raise ValueError(f"사용자 {user_idx}: 추천 곡 인덱스가 범위를 벗어났습니다.")
            seen = set(dataset.counts.getrow(user_idx).indices.tolist())
            if not seen.isdisjoint(item_indices):
                raise ValueError(f"사용자 {user_idx}: 추천에 학습 청취곡이 포함됩니다.")
            if any(not np.isfinite(item.score) or item.score < 0 for item in recommendations):
                raise ValueError(f"사용자 {user_idx}: 유효하지 않은 추천 점수입니다.")
            hits = len(truth.intersection(item_indices))
            recall = hits / len(truth)
            arp = float(np.mean(popularity[item_indices]))
            zeros = sum(item.score == 0 for item in recommendations)
            recall_sum += recall
            arp_sum += arp
            zero_score_count += zeros
            zero_score_users += zeros > 0
            users_writer.writerow([user_idx, len(truth), hits, 10, recall, arp, zeros])
            for rank, item in enumerate(recommendations, start=1):
                rec_writer.writerow([
                    user_idx, rank, item.item_idx, item.score, int(popularity[item.item_idx])
                ])
            if progress is not None:
                progress(position, len(user_indices))
    summary = {
        "algorithm": model.algorithm, "weighting": model.weighting,
        "scoring": "target-neighborhood weighted average",
        "split": split_name, "top_n": 10, "gamma": gamma, "top_k": model.top_k,
        "batch_size": model.batch_size, "num_users": len(user_indices),
        "target_count": sum(split.targets.getrow(int(u)).nnz for u in user_indices),
        "recommendation_count": len(user_indices) * 10,
        "recall_at_10": recall_sum / len(user_indices),
        "arp": arp_sum / len(user_indices),
        "zero_score_recommendation_count": zero_score_count,
        "users_with_zero_score_recommendations": zero_score_users,
        "evaluation_seconds": monotonic() - started,
        "candidate_rule": "all train items minus original train history",
        "aggregation": f"mean of per-user metrics over all {split_name} users",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def evaluate_validation(model, dataset, output_dir, gamma=0.0, progress=None) -> dict:
    return evaluate_split(model, dataset, output_dir, gamma, progress, "validation")


def compare_test(model, dataset, output_dir, adjusted_gamma, progress=None) -> dict:
    """검증에서 선택한 gamma를 고정하고 동일 모델로 테스트 보정 전후를 평가한다."""
    if not 0 < adjusted_gamma <= 1:
        raise ValueError("비교 gamma는 0보다 크고 1 이하여야 합니다.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "comparison.json"
    comparison_path.unlink(missing_ok=True)
    print("테스트 보정 전 평가 (gamma=0)", flush=True)
    baseline = evaluate_split(model, dataset, output_dir / "baseline", 0.0, progress, "test")
    print(f"테스트 보정 후 평가 (gamma={adjusted_gamma:g})", flush=True)
    adjusted = evaluate_split(model, dataset, output_dir / "adjusted", adjusted_gamma, progress, "test")
    comparison = {
        "split": "test", "top_k": model.top_k, "weighting": model.weighting,
        "algorithm": model.algorithm, "adjusted_gamma": adjusted_gamma,
        "baseline": baseline, "adjusted": adjusted,
        "recall_change": adjusted["recall_at_10"] - baseline["recall_at_10"],
        "arp_change": adjusted["arp"] - baseline["arp"],
        "recall_retention": adjusted["recall_at_10"] / baseline["recall_at_10"]
            if baseline["recall_at_10"] else None,
        "arp_reduction": 1 - adjusted["arp"] / baseline["arp"] if baseline["arp"] else None,
        "training_rule": "original train only; same fitted model for both evaluations",
        "selection_rule": "K and gamma fixed before test evaluation",
    }
    comparison_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["variant", "top_k", "gamma", "num_users", "recall_at_10", "arp"])
        for label, result in (("baseline", baseline), ("adjusted", adjusted)):
            writer.writerow([label, model.top_k, result["gamma"], result["num_users"],
                             result["recall_at_10"], result["arp"]])
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="전체 검증·테스트 사용자 Recall@10·ARP 평가")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--compare-gamma", type=float, help="검증에서 고정한 gamma로 테스트 보정 전후 비교")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--weighting", choices=("raw", "log1p"), default="log1p")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.top_k < 1 or args.batch_size < 1 or not 0 <= args.gamma <= 1:
        parser.error("top-k·batch-size는 양수, gamma는 0~1이어야 합니다.")
    if args.compare_gamma is not None:
        if args.split != "test" or args.gamma != 0 or not 0 < args.compare_gamma <= 1:
            parser.error("--compare-gamma는 --split test 및 --gamma 0에서 0 초과~1 이하로 지정하세요.")
    if args.output_dir is None:
        suffix = "_comparison" if args.compare_gamma is not None else f"_gamma{args.gamma:g}".replace(".", "")
        args.output_dir = Path(f"outputs/{args.split}_k{args.top_k}{suffix}")
    try:
        dataset = load_dataset(args.data_dir)
        started = last_report = monotonic()

        def report(done, total):
            nonlocal last_report
            now = monotonic()
            if now - last_report >= 10 or done == total:
                print(f"진행: {done:,}/{total:,} ({done / total:.1%}), "
                      f"경과 {now - started:.1f}초", flush=True)
                last_report = now

        model = ItemBasedCF(args.top_k, args.batch_size, progress=report, weighting=args.weighting)
        print("학습 시작 (최초 train만 사용)", flush=True)
        model.fit_matrix(dataset.counts)
        label = "검증" if args.split == "validation" else "테스트"
        print(f"전체 {label} 사용자 평가 시작", flush=True)
        if args.compare_gamma is not None:
            comparison = compare_test(model, dataset, args.output_dir, args.compare_gamma, report)
            for variant, result in (("보정 전", comparison["baseline"]), ("보정 후", comparison["adjusted"])):
                print(f"{variant} (gamma={result['gamma']:g}): Recall@10={result['recall_at_10']:.6f}, "
                      f"ARP={result['arp']:.6f}")
            print(f"테스트 사용자: {comparison['baseline']['num_users']:,}명")
            print(f"결과 저장 → {args.output_dir}")
            return
        summary = evaluate_split(model, dataset, args.output_dir, args.gamma, report, args.split)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"평가 실패: {exc}\n")
    print(f"{label} 사용자: {summary['num_users']:,}명")
    print(f"Recall@10: {summary['recall_at_10']:.6f}")
    print(f"ARP: {summary['arp']:.6f}")
    print(f"결과 저장 → {args.output_dir}")


if __name__ == "__main__":
    main()
