from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import pandas as pd
from joblib import parallel_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.custom_transformers import ElementwiseSummaryStats, LibrosaTransformer


def main():

    parser = ArgumentParser(
        prog="Audio Feature Extraction",
        description="Extracts features from pickled audio data using a collection of Librosa features.",
    )

    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel. See joblib.Parallel for more information.",
    )
    parser.add_argument(
        "--win_sizes",
        type=int,
        nargs="+",
        default=[2048],
        help="Window sizes to generate features for.",
    )
    parser.add_argument(
        "--source", type=Path, help="(.pkl) Path to pickled audio data.", required=True
    )
    parser.add_argument(
        "--dest",
        type=Path,
        help="Directory to save feature extracted data.",
        default=".",
    )

    args = parser.parse_args()
    parallel_config(n_jobs=args.n_jobs)
    # Window sizes to generate features for
    WIN_SIZES = args.win_sizes

    # File Paths
    source_fpath: Path = args.source.resolve()
    dest_fpath: Path = args.dest.resolve()

    print("Loading data from:", source_fpath)

    # Load data
    df = pd.read_pickle(source_fpath)

    target = df["target"]
    # Access with loc so it returns a DataFrame
    X = df.loc[:, ["audio"]]

    # Librosa features to extract
    librosa_features = [
        "mfcc",
        "chroma_stft",
        "chroma_cens",
        "spectral_centroid",
        "spectral_bandwidth",
        "spectral_rolloff",
        "spectral_contrast",
        "zero_crossing_rate",
        "tonnetz",
        "tempogram_ratio",
    ]

    pipe = Pipeline(
        [
            (
                "librosa_features",
                ColumnTransformer(
                    [
                        (feature, LibrosaTransformer(feature=feature), ["audio"])
                        for feature in librosa_features
                    ]
                ),
            ),
            ("stats", ElementwiseSummaryStats()),
        ]
    )

    pipe.set_output(transform="pandas")

    for win_size in WIN_SIZES:
        print("Extracting features for window size:", win_size)

        params = {
            "librosa_features__mfcc__n_mfcc": 20,
            "librosa_features__mfcc__n_fft": win_size,
            "librosa_features__chroma_stft__n_fft": win_size,
            "librosa_features__spectral_centroid__n_fft": win_size,
            "librosa_features__spectral_bandwidth__n_fft": win_size,
            "librosa_features__spectral_rolloff__n_fft": win_size,
            "librosa_features__spectral_contrast__n_fft": win_size,
            "librosa_features__zero_crossing_rate__frame_length": win_size,
        }

        pipe.set_params(**params)

        print("Transforming data...")
        time_start = datetime.now()
        X_trans = pipe.fit_transform(X)
        X_trans["target"] = target
        time_end = datetime.now()
        print(f"Completed in: {time_end - time_start}")

        out_fpath = dest_fpath / f"{source_fpath.stem}_{win_size}.pkl"
        print("Saving to:", out_fpath)

        pd.DataFrame(X_trans).to_pickle(out_fpath)
        print("Done!")


if __name__ == "__main__":
    main()
