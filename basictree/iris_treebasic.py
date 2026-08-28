import pandas as pd
import numpy as np

df = pd.read_csv("iris.csv")

X = df.drop("species", axis=1)
y = df["species"]


def gini(y):
    counts = y.value_counts()
    probabilities = counts / len(y)

    return 1 - (probabilities ** 2).sum()

best_feature = None
best_threshold = None
best_impurity = float("inf")


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

def build_tree(X, y):

    # Stop if the node contains only one class
    if len(y.unique()) == 1:
        return Node(prediction=y.iloc[0])

    # Find the best split
    feature, threshold, impurity = best_split(X, y)

    if feature is None:
        prediction = y.value_counts().index[0]
        return Node(prediction=prediction)
    
    # Create a decision node
    node = Node(
        feature=feature,
        threshold=threshold
    )

    # Split the data
    mask = X[feature] < threshold

    X_left = X[mask]
    y_left = y[mask]

    X_right = X[~mask]
    y_right = y[~mask]

    # Recursively build children
    node.left = build_tree(X_left, y_left)
    node.right = build_tree(X_right, y_right)

    return node

tree = build_tree(X, y)

def predict_one(node, x):

    # We reached a leaf
    if node.prediction is not None:
        return node.prediction

    # Ask the node's question
    if x[node.feature] < node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

test = pd.Series({
    "sepal_length": float(input("Enter sepal length: ")),
    "sepal_width": float(input("Enter sepal width: ")),
    "petal_length": float(input("Enter petal length: ")),
    "petal_width": float(input("Enter petal width: ")),
})
prediction = predict_one(tree, test)
print(prediction)