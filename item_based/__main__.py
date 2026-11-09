import argparse
import csv
from pathlib import Path
from statistics import mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Item-Based CF on sample interactions.")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/demo_recommendations.csv"))
    args = parser.parse_args()
    if args.top_k < 1 or not 0 <= args.gamma <= 1:
        parser.error("--top-k must be positive and --gamma must be between 0 and 1")

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
    model = ItemBasedCF(top_k=args.top_k)
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
