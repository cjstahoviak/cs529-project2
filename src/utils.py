from pathlib import Path

import librosa


def get_genre_from_path(path: Path):
    return path.parts[-2]


def load_audio_files(data_dir: Path):
    audio_files = list(data_dir.glob("**/*.au"))

    targets = [get_genre_from_path(file) for file in audio_files]
    audio_data = [librosa.load(file)[0] for file in audio_files]

    return audio_data, targets
