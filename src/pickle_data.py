from argparse import ArgumentParser
from pathlib import Path

from utils import load_audio_to_df


def main():
    parser = ArgumentParser(
        prog="Pickles Audio Data",
        description="Converts raw audio data to a pickled pandas DataFrame. Will search for .au files in the given directory.",
    )

    parser.add_argument(
        "--source",
        "-s",
        type=Path,
        default="../data/raw/test/",
        help="Path to folder containing audio data.",
    )
    parser.add_argument(
        "--dest",
        "-d",
        type=Path,
        default="../data/processed/",
        help="Path to save pickled data. Should be a directory.",
    )

    args = parser.parse_args()
    source_dir: Path = args.source
    dest_dir: Path = args.dest

    # Process raw audio data and save as pickled files.
    dir_name = source_dir.stem
    print("Pickling data from:", source_dir)
    df = load_audio_to_df(source_dir)
    dest = dest_dir / f"{dir_name}.pkl"
    print(f"Collected {len(df)} audio files.")
    print("Saving pickled data to:", dest)
    df.to_pickle(dest)
    print("Done.")


if __name__ == "__main__":
    main()
