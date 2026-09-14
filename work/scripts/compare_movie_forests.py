

from sklearn.ensemble import RandomForestClassifier as SklearnForest

from movie_random_forest import (
    RandomForestClassifier as MyForest,
    X_train,
    X_test,
    y_train,
    y_test,
    accuracy,
)


my_forest = MyForest(
    n_trees=20,
    max_depth=4,
    min_samples_leaf=2,
    max_features=2,
    random_seed=42,
)

my_forest.fit(X_train, y_train)
my_predictions = my_forest.predict(X_test)
my_accuracy = accuracy(y_test, my_predictions)


# scikit-learn's forest
sklearn_forest = SklearnForest(
    n_estimators=20,
    max_depth=4,
    min_samples_leaf=2,
    max_features=2,
    random_state=42,
)

sklearn_forest.fit(X_train, y_train)
sklearn_predictions = sklearn_forest.predict(X_test)
sklearn_accuracy = accuracy(y_test, sklearn_predictions)


print("My forest:", my_accuracy)
print("Sklearn:", sklearn_accuracy)
print("Difference:", my_accuracy - sklearn_accuracy)

