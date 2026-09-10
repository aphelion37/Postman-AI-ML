import pandas as pd
import numpy as np
import random as rnd

# ------------------------------------------------------------
# 1) Load the dataset and prepare train/test folds
# ------------------------------------------------------------
# This dataset contains movie features in the columns and a target called "Genre".
# We separate the feature matrix X from the target vector y.
# We then shuffle the rows with a fixed random seed so the split is reproducible.
# This makes it easy to compare different trees and parameter settings reliably.
df = pd.read_csv("lifestyle.csv")

# The dataset schema changes across examples: some files use "Y", others use "Genre",
# and the lifestyle dataset uses "job_type" as the class label. We detect the target
# column in a robust way instead of assuming one specific name.
target_candidates = ["Y", "Genre", "genre", "label", "target", "class", "job_type"]
target_column = next((column for column in target_candidates if column in df.columns), df.columns[-1])

X = df.drop(target_column, axis=1)
y = df[target_column]

indices = list(range(len(X)))

# A fixed seed ensures the same random split is generated every time we run the script.
# Without this, each run might choose a different training/test arrangement.
rnd.seed(42)
rnd.shuffle(indices)

# 80/20 split: the first 80% go to the training set, remaining 20% to testing.
split = int(0.8 * len(indices))

train_indices = indices[:split]
test_indices = indices[split:]

# Use .iloc to select rows by index position rather than label.
# This preserves the shuffled ordering we created above.
X_train = X.iloc[train_indices]
y_train = y.iloc[train_indices]

X_test = X.iloc[test_indices]
y_test = y.iloc[test_indices]

# ------------------------------------------------------------
# 2) Gini impurity: how "mixed" the class labels are in a node
# ------------------------------------------------------------
# If a node contains only one class, the Gini impurity is 0 (perfectly pure).
# If labels are evenly split, impurity is high.
# We use Gini to decide which split gives the most useful information gain.
def gini(y):
    # Count how many samples belong to each class.
    counts = y.value_counts()

    # Convert counts into probabilities.
    probabilities = counts / len(y)

    # Gini formula:
    # G = 1 - sum(p_i^2)
    # This is 0 when one class has probability 1 and all others are 0.
    return 1 - (probabilities ** 2).sum()

# These variables are left here from the original script, but they are not used in the final tree logic.
# They are harmless placeholders and reflect the earlier idea of tracking the "best" split globally.
best_feature = None
best_threshold = None
best_impurity = float("inf")

# ------------------------------------------------------------
# 3) Helper: majority vote for a leaf node
# ------------------------------------------------------------
# When a node cannot be split further, we need a prediction for the class.
# A simple and common choice is the majority class in that node.
def majority_class(y):
    # value_counts() returns counts for each category in descending frequency.
    # index[0] is the most common class.
    return y.value_counts().index[0]

# ------------------------------------------------------------
# 4) Find the best feature and threshold to split on
# ------------------------------------------------------------
# This function scans every feature and every possible threshold between adjacent feature values.
# It tries to minimize weighted Gini impurity after the split.
# The "best split" is the one that creates the most homogeneous children.
def best_split(X, y):

    # Start by assuming there is no valid split yet.
    best_feature = None
    best_threshold = None
    best_impurity = float("inf")

    # Try each feature one by one.
    for feature in X.columns:

        # Look at all unique values in this column.
        # Example: values = [0.5, 1.2, 3.7]
        values = sorted(X[feature].unique())

        # For each adjacent pair, create a threshold halfway between them.
        # Example: between 0.5 and 1.2, threshold = 0.85
        # This gives us a candidate split: values < threshold go left, otherwise right.
        for i in range(len(values) - 1):

            threshold = (values[i] + values[i + 1]) / 2

            # mask is a boolean Series: True for samples with value below threshold.
            # This is how the tree asks a question like "is feature < 3.5?".
            mask = X[feature] < threshold

            # Partition the labels according to the split.
            y_left = y[mask]
            y_right = y[~mask]

            # Skip invalid splits that create an empty child.
            # A tree node needs both sides to have at least some samples.
            if len(y_left) == 0 or len(y_right) == 0:
                continue

            # Compute impurity on each side separately.
            left_gini = gini(y_left)
            right_gini = gini(y_right)

            # Weighted Gini is the average impurity of the two child groups,
            # weighted by how many examples each group contains.
            weighted_gini = (
                len(y_left) / len(y) * left_gini
                + len(y_right) / len(y) * right_gini
            )

            # Keep the split that minimizes impurity.
            if weighted_gini < best_impurity:
                best_impurity = weighted_gini
                best_feature = feature
                best_threshold = threshold

    # Return the chosen split. If no split was found, both values stay None.
    return best_feature, best_threshold, best_impurity

# ------------------------------------------------------------
# 5) Node structure for the decision tree
# ------------------------------------------------------------
# A decision tree is a recursive structure of nodes.
# Each internal node stores:
# - the feature it tests
# - the threshold it compares against
# - left child subtree
# - right child subtree
# A leaf node stores a prediction instead of a feature test.
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

# ------------------------------------------------------------
# 6) Recursive tree construction
# ------------------------------------------------------------
# This is the core of the algorithm. We build the tree by repeatedly finding the best split,
# creating two child branches, and recursing until a stopping condition is reached.
def build_tree(X, y, depth=0, max_depth=None, min_samples_split=2, min_samples_leaf=1):

    # Stop if every sample in this node belongs to the same class.
    # This is a pure node, so no further split is needed.
    if len(y.unique()) == 1:
        return Node(prediction=y.iloc[0])

    # Stop if we have reached the maximum depth allowed by the user.
    # max_depth=None means "no limit".
    if max_depth is not None and depth >= max_depth:
        return Node(prediction=majority_class(y))

    # Stop if too few samples remain to justify a split.
    # This prevents splitting on tiny groups that would overfit noise.
    if len(y) < min_samples_split:
        return Node(prediction=majority_class(y))

    # Find the best feature/threshold pair from all possible splits.
    feature, threshold, impurity = best_split(X, y)

    # If no valid split exists, make this a leaf and predict the majority class.
    if feature is None:
        return Node(prediction=majority_class(y))

    # Use the chosen threshold to split the data into left and right child sets.
    # The left side contains rows where feature < threshold.
    mask = X[feature] < threshold

    X_left = X[mask]
    y_left = y[mask]

    X_right = X[~mask]
    y_right = y[~mask]

    # If either child is too small to satisfy the minimum leaf-size rule,
    # we stop growing this branch and make the current node a leaf.
    if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf:
        return Node(prediction=majority_class(y))

    # Create the internal decision node with the chosen test.
    node = Node(
        feature=feature,
        threshold=threshold
    )

    # Recursively build the left and right subtrees.
    # depth + 1 keeps track of how far down we are in the tree.
    node.left = build_tree(
        X_left,
        y_left,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf
    )

    node.right = build_tree(
        X_right,
        y_right,
        depth + 1,
        max_depth,
        min_samples_split,
        min_samples_leaf
    )

    return node

# ------------------------------------------------------------
# 7) Prediction for a single sample
# ------------------------------------------------------------
# To classify a new row, we start at the root and walk down the tree.
# At each internal node, we check whether the feature is less than the saved threshold.
# Then we follow the left or right child until we hit a leaf node with a class prediction.
def predict_one(node, x):

    # If this is a leaf, return its stored class label.
    if node.prediction is not None:
        return node.prediction

    # Ask the node's decision question.
    if x[node.feature] < node.threshold:
        return predict_one(node.left, x)
    else:
        return predict_one(node.right, x)

# ------------------------------------------------------------
# 8) DecisionTree wrapper class
# ------------------------------------------------------------
# This class just wraps the recursive tree-building and prediction logic in a tidy interface.
class DecisionTree:
    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    ):
        # Root is the starting point of the entire tree.
        self.root = None

        # These hyperparameters control how the tree grows.
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, y):
        # Build the tree from the training data.
        self.root = build_tree(
            X,
            y,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf
        )

    def predict(self, X):
        # Predict one class for each row in the dataset.
        predictions = []

        for i in range(len(X)):
            predictions.append(
                predict_one(self.root, X.iloc[i])
            )

        return predictions

# ------------------------------------------------------------
# 9) Train the model and evaluate it
# ------------------------------------------------------------
# Build a tree with default settings and assess training/test accuracy.
tree = DecisionTree()

tree.fit(X_train, y_train)

train_predictions = tree.predict(X_train)
test_predictions = tree.predict(X_test)

# ------------------------------------------------------------
# 10) Accuracy metric
# ------------------------------------------------------------
# Accuracy = (# correct predictions) / (total predictions)
# This is a simple metric that tells us how often the tree is right.
def accuracy(y_true, predictions):
    correct = 0

    for i in range(len(y_true)):
        if y_true.iloc[i] == predictions[i]:
            correct += 1

    return correct / len(y_true)

# ------------------------------------------------------------
# 11) Hyperparameter sweep: depth and minimum split size
# ------------------------------------------------------------
# This loop experiments with different levels of complexity.
# - Larger max_depth can let the tree memorize training examples.
# - Increasing min_samples_split makes the tree stop earlier, which can reduce overfitting.
# We track both training and test accuracy to see how well the tree generalizes.
for depth in [4, 6, 9, 14, 25, 50]:
    for samples1 in [50, 100, 200, 500, 1000, 5000]:
        for samples2 in [50, 100, 200, 500, 1000, 5000]:
            # Create a new tree for each parameter combination.
            tree = DecisionTree(max_depth=depth, min_samples_split=samples1, min_samples_leaf=samples2)

            # Fit on training data.
            tree.fit(X_train, y_train)

            # Make predictions on both train and test sets.
            train_predictions = tree.predict(X_train)
            test_predictions = tree.predict(X_test)

            # Evaluate both sets.
            train_acc = accuracy(y_train, train_predictions)
            test_acc = accuracy(y_test, test_predictions)

            # Print the result for each configuration.
            print(
                "Depth:", depth,
                "min_samples_split:", samples1,
                "min_samples_leaf:", samples2,
                "Train:", train_acc,
                "Test:", test_acc
            )

