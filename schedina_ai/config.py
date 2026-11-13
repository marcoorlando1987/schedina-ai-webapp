"""Configurazione centrale: campionati, stagioni, percorsi."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True)
class League:
    org_code: str  # codice su football-data.org (API fixtures)
    uk_code: str  # codice su football-data.co.uk (CSV storici)
    name: str


LEAGUES: tuple[League, ...] = (
    League("SA", "I1", "Serie A"),
    League("PL", "E0", "Premier League"),
    League("PD", "SP1", "La Liga"),
    League("BL1", "D1", "Bundesliga"),
    League("FL1", "F1", "Ligue 1"),
)
LEAGUE_NAMES: tuple[str, ...] = tuple(lg.name for lg in LEAGUES)

# Quante stagioni scaricare (la più vecchia fa da "burn-in" per Elo/forma).
N_SEASONS = 6
BURN_IN_SEASONS = 1

HISTORY_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
API_BASE_URL = "https://api.football-data.org/v4"

REPORT_TZ = "Europe/Rome"


def data_dir() -> Path:
    """Cartella dati (DB, cache CSV). Su Cloud Run: un volume montato."""
    return Path(os.environ.get("SCHEDINA_DATA_DIR", "data"))


def db_path() -> Path:
    return data_dir() / "schedina.db"


def api_token() -> str | None:
    return os.environ.get("FOOTBALL_DATA_API_TOKEN") or os.environ.get("API_TOKEN")


def season_codes(today: date | None = None, n: int = N_SEASONS) -> list[str]:
    """Codici stagione come '2526', dal più vecchio al più recente.

    La stagione inizia ad agosto: da luglio in poi è già quella nuova.
    """
    today = today or date.today()
    start = today.year if today.month >= 7 else today.year - 1
    return [f"{(y % 100):02d}{((y + 1) % 100):02d}" for y in range(start - n + 1, start + 1)]
