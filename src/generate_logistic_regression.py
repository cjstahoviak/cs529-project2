from pathlib import Path

import pandas as pd

from logistic_regression import LogisticRegression

# Load data
train_fpath = Path("../data/processed/train_data.pkl").resolve()
test_fpath = Path("../data/processed/test_data.pkl").resolve()

train_df = pd.read_pickle(train_fpath)
test_df = pd.read_pickle(test_fpath)

y_train = train_df["target"]
X_train = train_df["audio"]

X_test = test_df["audio"]

# Fit model and predict
lr = LogisticRegression(learning_rate=0.01, max_iter=100)
lr.fit(X_train, y_train)
lr.predict(X_test)
