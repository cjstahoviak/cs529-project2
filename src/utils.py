from pathlib import Path

import librosa
import pandas as pd

from elementwise_transformer import ElementwiseTransformer


def LibrosaTransformer(librosa_func=None):
    """Constructs a transformer which applies a librosa function to each element of the input.
    Necessary because librosa functions expect the audio data to be the first keyword argument.

    Args:
        librosa_func (_type_, optional): Librosa function. Defaults to None.

    Returns:
        ElementwiseTransformer: Transformer which applies librosa function to each element of the input.
    """
    return ElementwiseTransformer(lambda x, **kwargs: librosa_func(y=x, **kwargs))


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
