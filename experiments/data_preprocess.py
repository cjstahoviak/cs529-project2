from pathlib import Path

import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline

from custom_transformers import ElementwiseSummaryStats, LibrosaTransformer

# Load data
train_fpath = Path("../data/processed/train_data.pkl").resolve()
test_fpath = Path("../data/processed/test_data.pkl").resolve()

train_df = pd.read_pickle(train_fpath)
test_df = pd.read_pickle(test_fpath)

y_train = train_df["target"]
X_train = train_df["audio"]

X_test = test_df["audio"]

WIN_SIZES = [1024, 2048, 4096]

librosa_feature_pipelines = []

librosa_features = [
    "mfcc",
    "chroma_stft",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_contrast",
    "zero_crossing_rate",
]


for feature in librosa_features:

    new_pipeline = Pipeline(
        [
            ("transformer", LibrosaTransformer(feature=feature)),
            ("stats", ElementwiseSummaryStats(desc_kw_args={"axis": 1})),
        ]
    )

    if feature == "mfcc":
        new_pipeline.set_params(transformer__n_mfcc=20)

    librosa_feature_pipelines.append((feature, new_pipeline))

feature_extractor = FeatureUnion(transformer_list=librosa_feature_pipelines)

for win_size in WIN_SIZES:

    train_dict = {}
    test_dict = {}

    for feature, pipeline in feature_extractor.named_transformers.items():
        param_name = "frame_length" if feature == "zero_crossing_rate" else "n_fft"
        pipeline.set_params(**{f"transformer__{param_name}": win_size})

        # Fit and transform the train/test data
        train_dict[feature] = pipeline.fit_transform(X_train)
        test_dict[feature] = pipeline.transform(X_test)

    train_df = pd.concat(train_dict, axis=1, names=["feature", "statistic", "channel"])
    train_df[("target", "", "")] = y_train

    test_df = pd.concat(test_dict, axis=1, names=["feature", "statistic", "channel"])

    pd.DataFrame(train_df).to_csv(
        f"../data/processed/feature_extracted/train_features_{win_size}.csv"
    )
    pd.DataFrame(test_df).to_csv(
        f"../data/processed/feature_extracted/test_features_{win_size}.csv"
    )
