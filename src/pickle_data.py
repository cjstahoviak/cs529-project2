from argparse import ArgumentParser
from pathlib import Path

from utils import load_audio_to_df

parser = ArgumentParser(
    prog="Pickles Audio Data",
    description="Converts raw audio data to pickled pandas DataFrames. For each subdirectory in the source directory, a pickled DataFrame is created and saved in the destination directory.",
)

parser.add_argument(
    "--source",
    "-s",
    type=Path,
    default="../data/raw/",
    help="Path to raw audio data. Should be a directory containing subdirectories of audio files.",
)
parser.add_argument(
    "--dest",
    "-d",
    type=Path,
    default="../data/processed/",
    help="Path to save pickled data.",
)

args = parser.parse_args()
source_dir: Path = args.source
dest_dir: Path = args.dest

# Process raw audio data and save as pickled files.
for dir in source_dir.glob("*/"):
    dir_name = dir.parts[-1]
    df = load_audio_to_df(dir)
    dest = dest_dir / f"{dir_name}.pkl"
    df.to_pickle(dest)
