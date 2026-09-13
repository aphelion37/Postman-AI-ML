"""Train a custom random forest on the movie dataset."""

import random

from movie_decision_tree import (
    DecisionTree,
    X_test,
    X_train,
    accuracy,
    y_test,
    y_train,
)


def bootstrap_sample(X, y):
    sampled_indices = [random.randrange(0, len(X)) for _ in range(len(X))]
    return X.iloc[sampled_indices].copy(), y.iloc[sampled_indices].copy()


class RandomForestClassifier:
    def __init__(
        self,
        n_trees=5,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_seed=22,
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
            X_boot, y_boot = bootstrap_sample(X, y)
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

            final_predictions.append(
                max(vote_counts, key=vote_counts.get)
            )

        return final_predictions


def main():
    forest = RandomForestClassifier(
        n_trees=20,
        max_depth=4,
        min_samples_leaf=2,
        max_features=2,
        random_seed=42,
    )
    forest.fit(X_train, y_train)
    predictions = forest.predict(X_test)
    print("Movie random forest accuracy:", accuracy(y_test, predictions))


if __name__ == "__main__":
    main()
