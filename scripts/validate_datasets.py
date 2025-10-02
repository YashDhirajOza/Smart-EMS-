from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "simulator" / "data"


def validate_csv(path: Path) -> None:
    df = pd.read_csv(path)
    if df.isnull().any().any():
        raise ValueError(f"Dataset {path.name} contains null values")
    print(f"{path.name}: {len(df)} rows validated")


def main() -> None:
    for csv_file in DATA_DIR.glob("*.csv"):
        validate_csv(csv_file)


if __name__ == "__main__":
    main()
