# cs529-project2
Logistic Regression and Friends


## Setup
Since the training data is over a GB in size, it needs to be manually placed into this project. All that is missing is data\raw\train which can be retrieved from Google Drive.

### Light Installation (Recommended)
The light installation allows you to use the custom transformers, models, and feature extraction scripts used in this project. You will be able to use all files in `src/`, `experiments/feature_extraction.py`, and `experiments/generate_logistic_regression.py`.

```bash
pip install git+https://github.com/cjstahoviak/cs529-project2
```

### Full Installation (Not Recomended)
Full installation allows you to run all the experiments and install dev requirements. Some additional setup for logging in to mlflow is required but not fully detailed in this README. Requires a conda installation on your system as to setup the environment.
```bash
conda env create -f environment.yml
conda activate cs529_proj2
```
## Usage

Run scripts to generate feature-extracted data:
```bash
cd src/
python pickle_data.py
python ../experiments/feature_extraction.py
```
> Note: `feature_extraction.py` has an N_JOBS constant at the top of the file which can run the feature extraction process in parallel. The default is 1, but will take a significant amount of time. It's reccomended to increase this value. For more info see the [joblib docs](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html)

Run the source code to generate a model...
```bash
python ../experiments/generate_logistic_regression.py
```


## File Manifest
Project tree with description and contributions made on each source file.
```bash
CS529-PROJECT2:
C:.
├───data
│   ├───processed
│   │   └───feature_extracted: Holds all feature CSV files for each window size
│   │       └───pickle/: Includes "pickled" files of the audio DataFrames for quick access.
│   │
│   ├───raw
│   │   list_test.txt
│   ├───test/: Test data
│   └───train/: Train data
│
├───results/: Results of hyper parameters grid search
│           logistic_regression_cv_results.csv
│           model_comparison_cv_results.csv
│           model_comparison_cv_results_all_win_sizes.csv
│
├───docs
│       feature_extraction_diagram.drawio
│       feature_extraction_diagram.png
│       Music classification with classical ML.pdf
│
├───experiments
│       feature_extraction.py: (NICK) Extracts all features and populates feature_extracted/
│       generate_logistic_regression.py: (CALVIN) Generates a single model using training data to test
│       model_comparison.py: (CALVIN) Compares our model to classical learning models
│       softmax_gridsearch.py: (NICK) Optimizes feature extraction and model hyperparameters
│
├───notebooks
│       results_analysis.ipynb: (NICK) Populates docs/ with diagrams for the report
│
└───src
    │   custom_transformers.py: (NICK) Sklearn transfomers for feature extraction and grid search
    │   logistic_regression.py: (CALVIN): Implements multi-class logistic regression.
    │   pickle_data.p: (NICK): Loads data into a DataFrame and "pickles" it into pickle/
    │   utils.py: (NICK) Utility functions for other files
    │   __init__.py: (NICK) Python template file
```

## Contributions
Nick Livingstone:
- Setup repo and intialized environment (pre-commit hooks, conda, file structure)
- Refactor and improve logitic model class
- Developed feature extraction and PCA
- Developed data formatting
- Wrote report & generated diagrams

Calvin Stahoviak:
- Setup repo and intialized environment (pre-commit hooks, conda, file structure)
- Developed logistic model class
- Wrote report
- Wrote README

## Kaggle
Kaggle Score: 0.78 (2nd)

Team Name: Nick & Calvin

Date Run: 4/7

