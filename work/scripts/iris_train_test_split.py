from pathlib import Path

import pandas as pd
import numpy as np
import random as rnd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
df = pd.read_csv(DATA_DIR / "iris.csv")

X = df.drop("species", axis=1)
y = df["species"]

indices = list(range(len(X)))

#seed will ensure that the i get the same 'random' split everytime
rnd.seed(42)
rnd.shuffle(indices)

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

best_feature = None
best_threshold = None
best_impurity = float("inf")

def majority_class(y):
    return y.value_counts().index[0]

def best_split(X, y):

    best_feature = None
    best_threshold = None
    best_impurity = float("inf")

    for feature in X.columns:

        values = sorted(X[feature].unique())

        for i in range(len(values) - 1):

            threshold = (values[i] + values[i + 1]) / 2

            mask = X[feature] < threshold

            y_left = y[mask]
            y_right = y[~mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            left_gini = gini(y_left)
            right_gini = gini(y_right)

            weighted_gini = (
                len(y_left) / len(y) * left_gini
                + len(y_right) / len(y) * right_gini
            )

            if weighted_gini < best_impurity:
                best_impurity = weighted_gini
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_impurity

class Node:

    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        prediction=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

def build_tree(X, y, depth=0, max_depth=None):

    # 1. Pure node
    if len(y.unique()) == 1:
        return Node(prediction=y.iloc[0])

    # 2. Maximum depth reached
    if max_depth is not None and depth >= max_depth:
        return Node(prediction=majority_class(y))

    # 3. Find best split
    feature, threshold, impurity = best_split(X, y)

    # 4. No useful split exists
    if feature is None:
        return Node(prediction=majority_class(y))

    # 5. Split the data
    mask = X[feature] < threshold

    X_left = X[mask]
    y_left = y[mask]

    X_right = X[~mask]
    y_right = y[~mask]

    # 6. Create decision node
    node = Node(
        feature=feature,
        threshold=threshold
    )

    # 7. Recursively build children
    node.left = build_tree(
        X_left, y_left,
        depth + 1,
        max_depth
    )

    node.right = build_tree(
        X_right, y_right,
        depth + 1,
        max_depth
    )

    return node

def predict_one(node, x):

    # We reached a leaf
    if node.prediction is not None:
        return node.prediction

    # Ask the node's question
    if x[node.feature] < node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

class DecisionTree:
    def __init__(self, max_depth=None):
        self.root = None
        self.max_depth = max_depth

    def fit(self, X, y):
        self.root = build_tree(
            X, y,
            max_depth=self.max_depth
        )

    def predict(self, X):
        predictions = []

        for i in range(len(X)):
            prediction = predict_one(self.root, X.iloc[i])
            predictions.append(prediction)

        return predictions

tree = DecisionTree()

tree.fit(X_train, y_train)

train_predictions = tree.predict(X_train)
test_predictions = tree.predict(X_test)

def accuracy(y_true, predictions):
    correct = 0

    for i in range(len(y_true)):
        if y_true.iloc[i] == predictions[i]:
            correct += 1

    return correct / len(y_true)

for depth in [2,3,4,None]:

    tree = DecisionTree(max_depth=depth)

    tree.fit(X_train, y_train)

    train_predictions = tree.predict(X_train)
    test_predictions = tree.predict(X_test)

    train_acc = accuracy(y_train, train_predictions)
    test_acc = accuracy(y_test, test_predictions)

    print(
        "Depth:", depth,
        "Train:", train_acc,
        "Test:", test_acc
    )

