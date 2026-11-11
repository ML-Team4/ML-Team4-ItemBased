import argparse
import csv
from pathlib import Path
from statistics import mean


def run_real_data(args, parser) -> None:
    from item_based.data import load_dataset
    from item_based.model import ItemBasedCF

    try:
        dataset = load_dataset(args.data_dir)
        if not 0 <= args.user_idx < dataset.counts.shape[0]:
            raise ValueError("--user-idx가 공통 사용자 인덱스 범위를 벗어났습니다.")
        model = ItemBasedCF(top_k=args.top_k, batch_size=args.batch_size)
        print(f"실제 데이터 학습 시작: {dataset.counts.shape}, top_k={args.top_k}", flush=True)
        model.fit_matrix(dataset.counts)
        recommendations = model.recommend_by_index(args.user_idx, gamma=args.gamma)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"실행 실패: {exc}\n")

    output = args.output or Path("outputs/recommendations.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    user_id = int(dataset.users.iloc[args.user_idx]["user_id"])
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "user_idx", "user_id", "rank", "item_idx", "artist_name",
            "recording_name", "score", "train_listener_count",
        ])
        for rank, recommendation in enumerate(recommendations, start=1):
            item = dataset.items.iloc[recommendation.item_idx]
            writer.writerow([
                args.user_idx, user_id, rank, recommendation.item_idx,
                item["artist_name"], item["recording_name"],
                recommendation.score, recommendation.popularity,
            ])
    print(f"사용자 {args.user_idx}: 추천 {len(recommendations)}곡 저장 → {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Item-Based CF on sample or preprocessed data.")
    parser.add_argument("--real-data", action="store_true", help="공통 전처리 CSV로 전체 모델 학습")
    parser.add_argument("--data-dir", type=Path, help=".env DATA_DIR 대신 사용할 경로")
    parser.add_argument("--user-idx", type=int, default=0, help="실제 데이터 추천 대상의 공통 인덱스")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.top_k < 1 or args.batch_size < 1 or not 0 <= args.gamma <= 1:
        parser.error("--top-k and --batch-size must be positive; --gamma must be between 0 and 1")
    if args.real_data:
        run_real_data(args, parser)
        return
    if args.data_dir is not None or args.user_idx != 0:
        parser.error("--data-dir and --user-idx require --real-data")
    args.output = args.output or Path("outputs/demo_recommendations.csv")

    from item_based.model import ItemBasedCF

    interactions = [
        ("user_1", "track_a", 3),
        ("user_2", "track_a", 2),
        ("user_2", "track_b", 3),
        ("user_3", "track_a", 1),
        ("user_3", "track_b", 2),
        ("user_3", "track_c", 1),
    ]
    user_id = "user_1"
    # A later listen by user_1, held out from the training interactions.
    relevant_tracks = {"track_b"}
    model = ItemBasedCF(top_k=args.top_k, batch_size=args.batch_size)
    model.fit(interactions)
    recommendations = model.recommend(user_id, gamma=args.gamma)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["user_id", "rank", "track_id", "score", "popularity"])
        for rank, item in enumerate(recommendations, start=1):
            writer.writerow([user_id, rank, item.track_id, item.score, item.popularity])

    print(f"Saved {len(recommendations)} recommendations to {args.output}")
    recommended_tracks = {item.track_id for item in recommendations}
    recall = len(relevant_tracks & recommended_tracks) / len(relevant_tracks)
    arp = f"{mean(item.popularity for item in recommendations):.2f}" if recommendations else "N/A"

    print(f"Sample evaluation (gamma={args.gamma:g})")
    print(f"Recall@10: {recall:.4f}")
    print(f"ARP: {arp}")


if __name__ == "__main__":
    main()
