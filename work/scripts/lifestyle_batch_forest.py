"""Train one custom decision tree on the lifestyle data."""

from pathlib import Path
import random

import pandas as pd

from movie_random_forest import DecisionTree, accuracy


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_PATH = DATA_DIR / "lifestyle.csv"
RANDOM_SEED = 42
PROGRESS_INTERVAL = 100

CONFIGURATIONS = [
    ("Conservative", 8, 200, 50),
    ("Balanced", 12, 50, 10),
    ("Flexible", 15, 20, 5),
]


def make_train_test_split(X, y, test_size=0.2):
    """Create one fixed random train/test split."""
    indices = list(range(len(X)))
    random.shuffle(indices)

    split = int((1 - test_size) * len(indices))
    train_indices = indices[:split]
    test_indices = indices[split:]

    return (
        X.iloc[train_indices],
        X.iloc[test_indices],
        y.iloc[train_indices],
        y.iloc[test_indices],
    )


def make_training_progress(max_depth):
    nodes_seen = 0
    max_progress_nodes = (2 ** (max_depth + 1)) - 1

    def show_progress(
        event,
        first_value,
        second_value,
        third_value=None,
        fourth_value=None,
        fifth_value=None,
    ):
        nonlocal nodes_seen

        if event == "threshold":
            print(
                f"    Finding split: feature {first_value}/{second_value} "
                f"({third_value}), threshold "
                f"{fourth_value}/{fifth_value}",
                flush=True,
            )
            return

        if event == "feature":
            percentage = min(
                100,
                first_value / second_value * 100,
            )
            print(
                f"    Finding split: feature {first_value}/{second_value} "
                f"({percentage:.0f}%) - {third_value}",
                flush=True,
            )
            return

        depth = first_value
        samples = second_value
        nodes_seen += 1

        if nodes_seen % PROGRESS_INTERVAL == 0:
            percentage = min(100, nodes_seen / max_progress_nodes * 100)
            remaining_nodes = max(max_progress_nodes - nodes_seen, 0)
            print(
                f"    Training progress: {nodes_seen} nodes "
                f"processed ({percentage:.1f}% of estimated maximum, "
                f"up to {remaining_nodes} nodes remaining; "
                f"current depth {depth}, {samples} samples)",
                flush=True,
            )

    return show_progress


def main():
    random.seed(RANDOM_SEED)

    print("[1/5] Loading lifestyle dataset...", flush=True)
    df = pd.read_csv(DATA_PATH)
    X = df.drop("job_type", axis=1)
    y = df["job_type"]

    print("[2/5] Creating train/test split...", flush=True)
    X_train, X_test, y_train, y_test = make_train_test_split(
        X,
        y,
        test_size=0.4,
    )

    print("[3/5] Trying tree configurations...", flush=True)
    print("\nResults")

    for name, max_depth, min_samples_split, min_samples_leaf in CONFIGURATIONS:
        print(f"\n{name} configuration", flush=True)
        print(
            f"  max_depth={max_depth}, "
            f"min_samples_split={min_samples_split}, "
            f"min_samples_leaf={min_samples_leaf}",
            flush=True,
        )

        tree = DecisionTree(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
        )
        tree.fit(
            X_train,
            y_train,
            progress_callback=make_training_progress(max_depth),
        )

        print("[4/5] Predicting training and test data...", flush=True)
        train_predictions = tree.predict(X_train)
        test_predictions = tree.predict(X_test)

        print("[5/5] Calculating accuracy...", flush=True)
        print("  Training accuracy:", accuracy(y_train, train_predictions))
        print("  Test accuracy:", accuracy(y_test, test_predictions))


if __name__ == "__main__":
    main()
