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

### Feature Extraction

*If you did the light installation above, these files can be called from the command line.*

The data must be pickled before feature extraction with `pickle_data`:
```bash
pickle_data --help
usage: Pickles Audio Data [-h] [--source SOURCE] [--dest DEST]

Converts raw audio data to a pickled pandas DataFrame. Will search for .au files in the given directory.

options:
  -h, --help            show this help message and exit
  --source SOURCE, -s SOURCE
                        Path to folder containing audio data.
  --dest DEST, -d DEST  Path to save pickled data. (.pkl)
```

example

```
pickle_data --source /directory/of/au/files/ --dest target/folder/data.pkl
```

Then you can extract audio features with `extract_features`

```bash
extract_features --help
usage: Audio Feature Extraction [-h] [--n_jobs N_JOBS] [--win_sizes WIN_SIZES [WIN_SIZES ...]] --source SOURCE [--dest DEST]

Extracts features from pickled audio data using a collection of Librosa features. It will automatically generate a new file for each window size. 

options:
  -h, --help            show this help message and exit
  --n_jobs N_JOBS       Number of jobs to run in parallel. See joblib.Parallel for more information.
  --win_sizes WIN_SIZES [WIN_SIZES ...]
                        Window sizes to generate features for.
  --source SOURCE       (.pkl) Path to pickled audio data.
  --dest DEST           Directory to save feature extracted data.
```

Example
```bash
extract_features --n_jobs 4 --source /path/to/data.pkl --dest /target/directory/ --win_sizes 1024 2048 4096 
```

> Note: Feature extraction has an N_JOBS parameter which can run the feature extraction process in parallel. The default is 1, but will take a significant amount of time. It's reccomended to increase this value. For more info see the [joblib docs](https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html)

## Running The Model

Run our source code to generate a model...
```bash
python ../experiments/generate_logistic_regression.py
```

Or import our model in your own python file
```python
from src.logistic_regression import SoftmaxRegression
.
.
.
clf = SoftmaxRegression()
clf.fit(x_train, y_train)

clf.predict(x_test)
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

