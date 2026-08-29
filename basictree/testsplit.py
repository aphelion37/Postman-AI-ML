import pandas as pd
import numpy as np
import random as rnd

df = pd.read_csv("iris.csv")

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

tree = build_tree(X_train, y_train)

def predict_one(node, x):

    # We reached a leaf
    if node.prediction is not None:
        return node.prediction

    # Ask the node's question
    if x[node.feature] < node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

test_predictions = []

# Make predictions on the test set
for i in range(len(X_test)):
    prediction = predict_one(tree, X_test.iloc[i])
    test_predictions.append(prediction)

correct = 0

# Calculate accuracy
for i in range(len(y_test)):
    if test_predictions[i] == y_test.iloc[i]:
        correct += 1

accuracy = correct / len(y_test)

print("Test accuracy:", accuracy)

#Do the same for the training set - should be 1.0

train_predictions = []

for i in range(len(X_train)):
    prediction = predict_one(tree, X_train.iloc[i])
    train_predictions.append(prediction)

correct = 0

for i in range(len(y_train)):
    if train_predictions[i] == y_train.iloc[i]:
        correct += 1

train_accuracy = correct / len(y_train)

print("Train accuracy:", train_accuracy)

print("\n")

def print_tree(node, depth=0):
    indent = "    " * depth

    if node.prediction is not None:
        print(indent + f"→ {node.prediction}")
        return

    print(indent + f"[{node.feature} < {node.threshold:.2f}]")
    print(indent + "├── True:")
    print_tree(node.left, depth + 1)
    print(indent + "└── False:")
    print_tree(node.right, depth + 1)

print_tree(tree)

