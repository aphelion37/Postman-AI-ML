import pandas as pd
import numpy as np

df = pd.read_csv("iris.csv")

X = df.drop("species", axis=1)
y = df["species"]

best_split = None
best_impurity = float("inf")

def gini(y):
    counts = y.value_counts()
    probabilities = counts / len(y)

    return 1 - (probabilities ** 2).sum()

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

        # Don't allow empty children
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

print("Best feature:", best_feature)
print("Best threshold:", best_threshold)
print("Best Gini:", best_impurity)
