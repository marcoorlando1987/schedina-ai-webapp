"""Calendario partite da football-data.org (v4) con gestione esplicita degli errori."""
from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd
import requests

from .config import API_BASE_URL, LEAGUES

log = logging.getLogger(__name__)

UPCOMING = {"SCHEDULED", "TIMED"}  # NB: l'API usa TIMED quando l'orario è noto


class FixturesError(RuntimeError):
    pass


def fetch_fixtures(token: str | None, date_from: date, date_to: date,
                   session: requests.Session | None = None, max_wait_s: int = 70) -> pd.DataFrame:
    """Partite (qualsiasi stato) nel periodo, per i 5 campionati.

    Solleva FixturesError su token mancante/errato, piano non abilitato, rate limit
    persistente: mai un silenzioso DataFrame vuoto.
    """
    if not token:
        raise FixturesError("Token API mancante: imposta FOOTBALL_DATA_API_TOKEN")
    session = session or requests.Session()
    headers = {"X-Auth-Token": token}
    params = {"dateFrom": date_from.isoformat(), "dateTo": date_to.isoformat()}

    rows: list[dict] = []
    for lg in LEAGUES:
        url = f"{API_BASE_URL}/competitions/{lg.org_code}/matches"
        for attempt in range(3):
            resp = session.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 429 and attempt < 2:
                wait = min(int(resp.headers.get("Retry-After", 60)), max_wait_s)
                log.warning("Rate limit API: attendo %ss", wait)
                time.sleep(wait)
                continue
            break
        if resp.status_code in (401, 403):
            raise FixturesError(f"{lg.name}: accesso negato (token errato o competizione fuori piano)")
        if resp.status_code != 200:
            raise FixturesError(f"{lg.name}: HTTP {resp.status_code} {resp.text[:200]}")
        for m in resp.json().get("matches", []):
            ft = (m.get("score") or {}).get("fullTime") or {}
            rows.append({
                "match_id": m["id"],
                "league": lg.name,
                "utc_date": m["utcDate"],
                "status": m["status"],
                "home_api": m["homeTeam"]["name"],
                "away_api": m["awayTeam"]["name"],
                "score_home": ft.get("home"),
                "score_away": ft.get("away"),
            })

    cols = ["match_id", "league", "utc_date", "status", "home_api", "away_api",
            "score_home", "score_away"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["utc_date"] = pd.to_datetime(df["utc_date"], utc=True)
    return df
