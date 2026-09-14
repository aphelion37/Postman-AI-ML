"""Calculate impurity and permutation importance for the custom movie forest."""

import random

from movie_random_forest import (
    RandomForestClassifier,
    X_test,
    X_train,
    y_test,
    y_train,
    accuracy,
)


def impurity_importance(forest, feature_names):
    scores = {feature: 0 for feature in feature_names}

    for tree in forest.trees:
        collect_tree_importance(tree.root, scores)

    total = sum(scores.values())

    if total == 0:
        return scores

    for feature in scores:
        scores[feature] = scores[feature] / total

    return scores


def collect_tree_importance(node, scores):
    if node.prediction is not None:
        return

    scores[node.feature] += node.n_samples * node.impurity_decrease

    collect_tree_importance(node.left, scores)
    collect_tree_importance(node.right, scores)


def permutation_importance(forest, X_test, y_test, repeats=5):
    baseline_predictions = forest.predict(X_test)
    baseline_accuracy = accuracy(y_test, baseline_predictions)
    scores = {}

    for feature in X_test.columns:
        accuracy_drops = []

        for repeat in range(repeats):
            shuffled_data = X_test.copy()
            shuffled_values = shuffled_data[feature].tolist()
            random.shuffle(shuffled_values)
            shuffled_data[feature] = shuffled_values

            shuffled_predictions = forest.predict(shuffled_data)
            shuffled_accuracy = accuracy(y_test, shuffled_predictions)
            accuracy_drops.append(baseline_accuracy - shuffled_accuracy)

        scores[feature] = sum(accuracy_drops) / len(accuracy_drops)

    return scores


def print_scores(title, scores):
    print(f"\n{title}")

    for feature, score in sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{feature}: {score:.3f}")


def main():
    forest = RandomForestClassifier(
        n_trees=10,
        max_depth=4,
        min_samples_leaf=2,
        max_features=2,
        random_seed=42,
    )
    forest.fit(X_train, y_train)

    impurity_scores = impurity_importance(forest, X_train.columns)
    permutation_scores = permutation_importance(forest, X_test, y_test)

    print_scores("Impurity importance", impurity_scores)
    print_scores("Permutation importance", permutation_scores)


if __name__ == "__main__":
    main()
