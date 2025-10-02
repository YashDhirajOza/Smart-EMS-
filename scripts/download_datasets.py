from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parents[1] / "simulator" / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def run_kaggle_download(dataset: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset,
        "-p",
        str(destination),
        "--unzip",
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset downloader helper")
    parser.add_argument(
        "--kaggle-dataset",
        default="anikannal/solar-power-generation-data",
        help="Kaggle dataset identifier to download",
    )
    args = parser.parse_args()

    run_kaggle_download(args.kaggle_dataset, RAW_DIR / "kaggle")


if __name__ == "__main__":
    main()
