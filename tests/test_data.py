import csv
import tempfile
import unittest
from pathlib import Path

from item_based.data import load_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.write("users.csv", ["user_idx", "user_id"], [(1, 200), (0, 100)])
        self.write("items.csv", [
            "item_idx", "artist_name", "recording_name", "train_listener_count",
            "artist_key", "recording_key",
        ], [(i, "NA", 'NULL,\n"곡"', 2 if i == 0 else 1, "NA", str(i))
            for i in range(12)])
        self.train = [(0, i, 3) for i in range(12)] + [(1, 0, 2)]
        self.write_train()
        for period in ("validation", "test"):
            self.write(f"{period}_users.csv", ["user_idx"], [(1,)])
            self.write(f"{period}_targets.csv", ["user_idx", "item_idx"], [(1, 1)])

    def write(self, name, header, rows):
        with (self.root / name).open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(rows)

    def write_train(self):
        self.write("train_interactions.csv", ["user_idx", "item_idx", "listen_count"], self.train)

    def test_preserves_indices_counts_and_quoted_names(self):
        data = load_dataset(self.root)
        self.assertEqual(data.users["user_id"].tolist(), [100, 200])
        self.assertEqual(data.counts.shape, (2, 12))
        self.assertEqual(data.counts[0, 1], 3)
        self.assertEqual(data.items.iloc[0]["artist_name"], "NA")
        self.assertEqual(data.items.iloc[0]["recording_name"], 'NULL,\n"곡"')
        self.assertTrue(data.validation.targets[1, 1])

    def test_rejects_duplicate_training_pairs(self):
        self.train.append(self.train[0])
        self.write_train()
        with self.assertRaisesRegex(ValueError, "중복"):
            load_dataset(self.root)

    def test_rejects_training_tracks_in_targets(self):
        self.write("test_targets.csv", ["user_idx", "item_idx"], [(1, 0)])
        with self.assertRaisesRegex(ValueError, "학습 청취곡"):
            load_dataset(self.root)

    def test_rejects_inconsistent_popularity(self):
        self.train.append((1, 2, 1))
        self.write_train()
        with self.assertRaisesRegex(ValueError, "고유 청취자 수"):
            load_dataset(self.root)


if __name__ == "__main__":
    unittest.main()
