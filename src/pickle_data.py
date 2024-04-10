from pathlib import Path

from utils import load_audio_to_df

raw_data_dir = Path("../data/raw").resolve()

# Process raw audio data and save as pickled files.
for dir in raw_data_dir.glob("*/"):
    dir_name = dir.parts[-1]
    df = load_audio_to_df(dir)
    dest = Path(f"../data/processed/{dir_name}_data.pkl").resolve()
    df.to_pickle(dest)
