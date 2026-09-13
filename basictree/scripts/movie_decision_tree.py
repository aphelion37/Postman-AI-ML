"""Train and evaluate the finished custom decision tree on movie data."""

from pathlib import Path
import random

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
df = pd.read_csv(DATA_DIR / "movie_ready.csv")

target_candidates = ["Y", "Genre", "genre", "label", "target", "class", "job_type"]
target_column = next(
    (column for column in target_candidates if column in df.columns),
    df.columns[-1],
)

X = df.drop(target_column, axis=1)
y = df[target_column]

indices = list(range(len(X)))
random.seed(42)
random.shuffle(indices)

split = int(0.8 * len(indices))
train_indices = indices[:split]
test_indices = indices[split:]

X_train = X.iloc[train_indices]
y_train = y.iloc[train_indices]
X_test = X.iloc[test_indices]
y_test = y.iloc[test_indices]


def gini(y):
    counts = y.value_counts()
    probabilities = counts / len(y)
    return 1 - (probabilities ** 2).sum()


def majority_class(y):
    return y.value_counts().index[0]


def choose_random_features(X, max_features):
    feature_names = list(X.columns)

    if max_features is None:
        max_features = len(feature_names)

    max_features = min(max_features, len(feature_names))
    return random.sample(feature_names, max_features)


def best_split(X, y, feature_names=None, progress_callback=None):
    best_feature = None
    best_threshold = None
    best_impurity = float("inf")

    if feature_names is None:
        feature_names = list(X.columns)

    total_features = len(feature_names)

    for feature_number, feature in enumerate(feature_names, start=1):
        values = sorted(X[feature].unique())

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2
            mask = X[feature] < threshold
            y_left = y[mask]
            y_right = y[~mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            weighted_gini = (
                len(y_left) / len(y) * gini(y_left)
                + len(y_right) / len(y) * gini(y_right)
            )

            if weighted_gini < best_impurity:
                best_impurity = weighted_gini
                best_feature = feature
                best_threshold = threshold

            if progress_callback is not None and (i + 1) % 1000 == 0:
                progress_callback(
                    "threshold",
                    feature_number,
                    total_features,
                    feature,
                    i + 1,
                    len(values) - 1,
                )

        if progress_callback is not None:
            progress_callback(
                "feature",
                feature_number,
                total_features,
                feature,
            )

    return best_feature, best_threshold, best_impurity


class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        prediction=None,
        n_samples=0,
        impurity_decrease=0,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction
        self.n_samples = n_samples
        self.impurity_decrease = impurity_decrease


def build_tree(
    X,
    y,
    depth=0,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    feature_names=None,
    max_features=None,
    progress_callback=None,
):
    if progress_callback is not None:
        progress_callback("node", depth, len(y))

    if len(y.unique()) == 1:
        return Node(prediction=y.iloc[0])

    if max_depth is not None and depth >= max_depth:
        return Node(prediction=majority_class(y))

    if len(y) < min_samples_split:
        return Node(prediction=majority_class(y))

    features_for_this_node = feature_names
    if max_features is not None:
        features_for_this_node = choose_random_features(X, max_features)

    feature, threshold, impurity = best_split(
        X,
        y,
        features_for_this_node,
        progress_callback,
    )

    if feature is None:
        return Node(prediction=majority_class(y))

    mask = X[feature] < threshold
    X_left = X[mask]
    y_left = y[mask]
    X_right = X[~mask]
    y_right = y[~mask]

    if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf:
        return Node(prediction=majority_class(y))

    parent_gini = gini(y)
    weighted_child_gini = (
        len(y_left) / len(y) * gini(y_left)
        + len(y_right) / len(y) * gini(y_right)
    )

    node = Node(
        feature=feature,
        threshold=threshold,
        n_samples=len(y),
        impurity_decrease=parent_gini - weighted_child_gini,
    )

    node.left = build_tree(
        X_left,
        y_left,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        feature_names,
        max_features,
        progress_callback,
    )
    node.right = build_tree(
        X_right,
        y_right,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf,
        feature_names,
        max_features,
        progress_callback,
    )

    return node


def predict_one(node, x):
    if node.prediction is not None:
        return node.prediction

    if x[node.feature] < node.threshold:
        return predict_one(node.left, x)

    return predict_one(node.right, x)


class DecisionTree:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
    ):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features

    def fit(self, X, y, progress_callback=None):
        self.root = build_tree(
            X,
            y,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            progress_callback=progress_callback,
        )

    def predict(self, X):
        return [predict_one(self.root, X.iloc[i]) for i in range(len(X))]


def accuracy(y_true, predictions):
    correct = 0

    for i in range(len(y_true)):
        if y_true.iloc[i] == predictions[i]:
            correct += 1

    return correct / len(y_true)


def main():
    tree = DecisionTree()
    tree.fit(X_train, y_train)
    predictions = tree.predict(X_test)
    print("Movie decision tree accuracy:", accuracy(y_test, predictions))


if __name__ == "__main__":
    main()
