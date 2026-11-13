"""Persistenza SQLite: predizioni, risultati reali e metadati (metriche, ultimo run)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    match_id     INTEGER PRIMARY KEY,
    league       TEXT NOT NULL,
    utc_date     TEXT NOT NULL,
    home         TEXT NOT NULL,
    away         TEXT NOT NULL,
    p_home       REAL, p_draw REAL, p_away REAL,
    pick         TEXT,
    xg_home      REAL, xg_away REAL,
    p_over25     REAL,
    model        TEXT,
    new_team     INTEGER DEFAULT 0,
    status       TEXT,
    score_home   INTEGER, score_away INTEGER,
    updated_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_date ON predictions(utc_date);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

PRED_COLS = [
    "match_id", "league", "utc_date", "home", "away", "p_home", "p_draw", "p_away",
    "pick", "xg_home", "xg_away", "p_over25", "model", "new_team", "status",
    "score_home", "score_away", "updated_at",
]


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def upsert_prediction(conn: sqlite3.Connection, rec: dict) -> None:
    placeholders = ",".join(f":{c}" for c in PRED_COLS)
    updates = ",".join(f"{c}=excluded.{c}" for c in PRED_COLS if c != "match_id")
    conn.execute(
        f"INSERT INTO predictions ({','.join(PRED_COLS)}) VALUES ({placeholders}) "
        f"ON CONFLICT(match_id) DO UPDATE SET {updates}",
        {c: rec.get(c) for c in PRED_COLS},
    )


def update_result(conn: sqlite3.Connection, match_id: int, status: str,
                  score_home: int | None, score_away: int | None, now: str) -> None:
    """Aggiorna solo stato/risultato: la predizione di una partita iniziata resta congelata."""
    conn.execute(
        "UPDATE predictions SET status=?, score_home=?, score_away=?, updated_at=? WHERE match_id=?",
        (status, score_home, score_away, now, match_id),
    )


def set_meta(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value if isinstance(value, str) else json.dumps(value)),
    )


def get_meta(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    if row is None:
        return default
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return row[0]


def load_predictions(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY utc_date, league", conn)
    df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)
    return df
