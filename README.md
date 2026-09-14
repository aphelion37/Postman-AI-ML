
## Project purpose

This repository contains a compact set of from-scratch machine learning experiments focused on decision trees and random forests. The project is designed as a practical learning exercise: implement the core algorithm directly, train on tabular datasets, compare performance, and inspect how model behavior changes with depth, feature importance, and dataset choice.

The scripts in this project cover a few core ideas:

- training a custom decision tree from scratch
- evaluating performance with train/test splits
- comparing custom trees against scikit-learn baselines
- measuring feature importance
- exploring how tree depth and ensemble size affect accuracy


## Repository layout

The project is organized as follows:

- `work/scripts/` - runnable experiment scripts
- `work/data/` - source CSV files used by the experiments
- `work/results/` - generated plots and saved experiment outputs
- `README.md` - project overview and setup instructions
- `WRITE_UP.md` - project notes and write-up

## Requirements

- Python 3.9+
- `pip`
- a terminal such as PowerShell, Command Prompt, or VS Code terminal

## Setup

From the project root, install the dependencies:

```powershell
python -m pip install pandas numpy scikit-learn matplotlib
```

## Running the experiments

Run the scripts from the repository root:

```powershell
python work\scripts\iris_decision_tree.py
python work\scripts\iris_train_test_split.py
python work\scripts\movie_decision_tree.py
python work\scripts\movie_random_forest.py
python work\scripts\compare_movie_forests.py
python work\scripts\movie_forest_importance.py
python work\scripts\lifestyle_batch_forest.py
python work\scripts\penguins_random_forest.py
```

Each script resolves its dataset paths relative to the project folder, so it does not depend on the current working directory being set to a specific location.

## Script overview

- `work/scripts/iris_decision_tree.py` - classifies a single Iris example using a custom tree implementation
- `work/scripts/iris_train_test_split.py` - evaluates the Iris tree at multiple depths using a train/test split
- `work/scripts/movie_decision_tree.py` - trains and evaluates a custom decision tree on the movie dataset
- `work/scripts/movie_random_forest.py` - trains and evaluates a custom random forest on the movie dataset
- `work/scripts/compare_movie_forests.py` - compares the custom forest against a scikit-learn on the movie set
- `work/scripts/movie_forest_importance.py` - computes impurity-based and permutation-based feature importance on the movie set
- `work/scripts/lifestyle_batch_forest.py` - trains a decision tree on the lifestyle dataset
- `work/scripts/penguins_random_forest.py` - trains a random forest and saves a plot of accuracy against tree count on the pengiun dataset

## Outputs

I have saved the experiment results in the `work/results/` directory.

