from pathlib import Path

import librosa
import pandas as pd
from scipy.stats import describe


def get_genre_from_path(path: Path):
    return path.parts[-2]


def load_audio_files(data_dir: Path):
    audio_files = list(data_dir.glob("**/*.au"))

    targets = [get_genre_from_path(file) for file in audio_files]
    audio_data = [librosa.load(file)[0] for file in audio_files]

    return audio_data, targets


def load_audio_to_df(data_dir: Path):
    audio_dict = {}

    for file in list(data_dir.glob("**/*.au")):
        audio, sr = librosa.load(file)

        audio_dict[file.stem] = {
            "target": get_genre_from_path(file),
            "audio": audio,
            "sr": sr,
        }

    df = pd.DataFrame.from_dict(audio_dict, orient="index")
    return df


def describe_as_df(x, desc_kw_args={}):
    """Convert the output of scipy.stats.describe to a DataFrame with MultiIndex columns.

    Args:
        x (_type_): Input data to describe.
        desc_kw_args (dict, optional): Keword args to pass to describe function . Defaults to {}.

    Returns:
        DataFrame: DataFrame with MultiIndex columns.
    """
    stats = describe(x, **desc_kw_args)
    data = {}

    for field in stats._fields:
        values = getattr(stats, field)

        # Special handling for 'minmax'
        if field == "minmax":
            tmp_fields = ["min", "max"]
            tmp_vals = [values[0], values[1]]
        else:
            tmp_fields = [field]
            tmp_vals = [values]

        for f, v in zip(tmp_fields, tmp_vals):
            # Values that are arrays
            if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
                for i, value in enumerate(v, 1):  # Start index from 1
                    data[(f, i)] = [value]
            # Handle scalar fields
            else:
                data[(f, "")] = [v]

    # Create a DataFrame with MultiIndex columns from the dictionary
    stats_df = pd.DataFrame(data)
    stats_df.columns.names = ["stat", ""]

    return stats_df
