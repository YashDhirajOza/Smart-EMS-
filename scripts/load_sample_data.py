from __future__ import annotations

import json
from pathlib import Path

import psycopg2

CONFIG_PATH = Path(__file__).resolve().parents[1] / "infra" / "env" / "postgres.env"


def load_env(path: Path) -> dict[str, str]:
    env = {}
    with path.open() as file:
        for line in file:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                env[key] = value
    return env


def main() -> None:
    env = load_env(CONFIG_PATH)
    conn = psycopg2.connect(
        dbname=env.get("POSTGRES_DB", "urjanet"),
        user=env.get("POSTGRES_USER", "urjanet"),
        password=env.get("POSTGRES_PASSWORD", "urjanet-secret"),
        host=env.get("POSTGRES_HOST", "localhost"),
        port=env.get("POSTGRES_PORT", "5432"),
    )
    with conn, conn.cursor() as cursor:
        cursor.execute("SELECT asset_id FROM assets")
        assets = cursor.fetchall()
        print(json.dumps(assets, indent=2))


if __name__ == "__main__":
    main()
