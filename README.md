# Basic Tree Experiments

Small from-scratch decision-tree and random-forest experiments using CSV datasets.

## Layout

- `scripts/` - runnable Python experiments
- `data/` - input CSV files
- `results/` - generated accuracy plots and experiment images

## Scripts

- `scripts/iris_decision_tree.py` - classify one Iris sample with the custom tree
- `scripts/iris_train_test_split.py` - test the Iris tree at different depths
- `scripts/movie_random_forest.py` - run the custom movie random forest
- `scripts/compare_movie_forests.py` - compare the custom forest with scikit-learn
- `scripts/movie_forest_importance.py` - calculate impurity and permutation importance
- `scripts/lifestyle_batch_forest.py` - train one bounded decision tree on lifestyle data

## Run

From this folder:

```powershell
python -m pip install pandas numpy scikit-learn

python scripts\iris_decision_tree.py
python scripts\iris_train_test_split.py
python scripts\movie_random_forest.py
python scripts\compare_movie_forests.py
python scripts\movie_forest_importance.py
python scripts\lifestyle_batch_forest.py
```

Each script resolves input files from `data/`, so it does not depend on the current working directory.
