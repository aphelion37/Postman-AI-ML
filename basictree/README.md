# Basic Tree Experiments

Small from-scratch decision-tree and random-forest experiments using CSV datasets.

## Layout

- `scripts/` - runnable Python experiments
- `data/` - input CSV files
- `results/` - generated accuracy plots and experiment images

## Scripts


## Run

From this folder:

```powershell
python scripts\iris_decision_tree.py
python scripts\iris_train_test_split.py
python scripts\movie_random_forest.py
```

Each script resolves input files from `data/`, so it does not depend on the current working directory.
