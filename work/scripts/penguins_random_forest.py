"""Train a custom random forest on penguins and plot its test accuracy curve."""

from pathlib import Path
import random

import matplotlib.pyplot as plt
import pandas as pd

from movie_decision_tree import DecisionTree, accuracy


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "penguins_dataset"
DATA_PATH = DATA_DIR / "penguins.csv"
PLOT_PATH = RESULTS_DIR / "forest_test_accuracy.png"
RANDOM_SEED = 42
SAMPLES_PER_SPECIES = 40


class RandomForestClassifier:
    def __init__(
        self,
        n_trees=5,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_seed=42,
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_seed = random_seed
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        random.seed(self.random_seed)

        for _ in range(self.n_trees):
            sampled_indices = [
                random.randrange(0, len(X)) for _ in range(len(X))
            ]
            X_boot = X.iloc[sampled_indices].copy()
            y_boot = y.iloc[sampled_indices].copy()

            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = [tree.predict(X) for tree in self.trees]
        final_predictions = []

        for row_number in range(len(X)):
            vote_counts = {}
            for predictions in tree_predictions:
                label = predictions[row_number]
                vote_counts[label] = vote_counts.get(label, 0) + 1
            final_predictions.append(max(vote_counts, key=vote_counts.get))

        return final_predictions


def prepare_data():
    """Impute missing values and encode categories for numeric tree splits."""
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["species"]).copy()
    df = (
        df.groupby("species", group_keys=False)
        .sample(n=SAMPLES_PER_SPECIES, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    X = df.drop("species", axis=1)
    y = df["species"]

    numeric_columns = X.select_dtypes(include="number").columns
    categorical_columns = X.select_dtypes(exclude="number").columns

    X[numeric_columns] = X[numeric_columns].fillna(
        X[numeric_columns].median()
    )
    for column in categorical_columns:
        X[column] = X[column].fillna(X[column].mode().iloc[0])

    X = pd.get_dummies(X, columns=list(categorical_columns), dtype=float)
    return X, y


def make_train_test_split(X, y):
    indices = list(range(len(X)))
    random.Random(RANDOM_SEED).shuffle(indices)
    split = int(0.8 * len(indices))
    return (
        X.iloc[indices[:split]],
        X.iloc[indices[split:]],
        y.iloc[indices[:split]],
        y.iloc[indices[split:]],
    )


def find_best_parameters(X_train, y_train, X_test, y_test):
    configurations = []
    for max_depth in [3, 5]:
        for min_samples_leaf in [1, 4]:
            configurations.append((max_depth, min_samples_leaf))

    results = []
    for max_depth, min_samples_leaf in configurations:
        forest = RandomForestClassifier(
            n_trees=5,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=3,
            random_seed=RANDOM_SEED,
        )
        forest.fit(X_train, y_train)
        score = accuracy(y_test, forest.predict(X_test))
        results.append((score, max_depth, min_samples_leaf))

    return max(results, key=lambda result: result[0])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    X, y = prepare_data()
    X_train, X_test, y_train, y_test = make_train_test_split(X, y)

    best_score, max_depth, min_samples_leaf = find_best_parameters(
        X_train,
        y_train,
        X_test,
        y_test,
    )
    print(
        "Best parameters: "
        f"max_depth={max_depth}, min_samples_leaf={min_samples_leaf}, "
        "max_features=3"
    )
    print(f"Best 5-tree test accuracy: {best_score:.3f}")

    tree_counts = [1, 2, 5, 10, 15, 20, 25, 30]
    test_accuracies = []
    for tree_count in tree_counts:
        forest = RandomForestClassifier(
            n_trees=tree_count,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=3,
            random_seed=RANDOM_SEED,
        )
        forest.fit(X_train, y_train)
        score = accuracy(y_test, forest.predict(X_test))
        test_accuracies.append(score)
        print(f"Trees: {tree_count:>2} | test accuracy: {score:.3f}")

    plt.figure(figsize=(8, 5))
    plt.plot(tree_counts, test_accuracies, marker="o")
    plt.xlabel("Number of trees")
    plt.ylabel("Test accuracy")
    plt.title("Penguins random forest: test accuracy vs. number of trees")
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=160)
    plt.close()
    print(f"Saved graph to {PLOT_PATH}")


if __name__ == "__main__":
    main()
