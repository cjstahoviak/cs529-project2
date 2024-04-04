from pathlib import Path

import pandas as pd
from mlxtend.feature_selection import ColumnSelector
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from custom_transformers import ElementwiseSummaryStats, LibrosaTransformer

# from logistic_regression import LogisticRegression

# Load data
train_fpath = Path("../data/processed/train_data.pkl").resolve()
test_fpath = Path("../data/processed/test_data.pkl").resolve()

train_df = pd.read_pickle(train_fpath)
test_df = pd.read_pickle(test_fpath)

y_train = train_df["target"]
X_train = train_df["audio"]

X_test = test_df["audio"]

# Define pipeline
pipe = Pipeline(
    [
        ("librosa_features", LibrosaTransformer(feature="mfcc")),
        ("summary_stats", ElementwiseSummaryStats(desc_kw_args={"axis": 1})),
        # ("stat_selector", ColumnSelector()),
        ("scaler", StandardScaler()),
        (
            "logreg",
            # LogisticRegression(
            #     learning_rate=0.01, max_iter=100
            # ),
            LogisticRegression(
                penalty=None, max_iter=10_000, solver="saga", multi_class="multinomial"
            ),
        ),
    ]
)

print("y_train shape: " + str(y_train.shape))
print("X_train shape: " + str(X_train.shape))

print("fitting pipeline...")
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

accuracy = accuracy_score(y_pred == y_train)
print(f"Accuracy: {accuracy}")
