"""Scarico (con cache su disco) e parsing dei CSV storici di football-data.co.uk."""
from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import HISTORY_URL, LEAGUES, data_dir, season_codes

log = logging.getLogger(__name__)

CURRENT_SEASON_MAX_AGE_H = 12
MAX_CONSECUTIVE_FAILURES = 3  # senza cache e senza dati: inutile insistere

# Terne di quote (casa, pareggio, trasferta) in ordine di preferenza.
ODDS_TRIPLES = [
    ("PSCH", "PSCD", "PSCA"),
    ("AvgCH", "AvgCD", "AvgCA"),
    ("B365CH", "B365CD", "B365CA"),
    ("AvgH", "AvgD", "AvgA"),
    ("B365H", "B365D", "B365A"),
    ("PSH", "PSD", "PSA"),
]

# Quote a cui un giocatore poteva davvero puntare (prima scelta: Bet365, autorizzato in Italia).
BET_ODDS_TRIPLES = [("B365H", "B365D", "B365A"), ("AvgH", "AvgD", "AvgA"), ("PSH", "PSD", "PSA")]

MATCH_COLUMNS = [
    "league", "season", "date", "home", "away", "hg", "ag", "result",
    "mkt_h", "mkt_d", "mkt_a", "sh_h", "sh_a", "sot_h", "sot_a",
    "odds_h", "odds_d", "odds_a",
]


class DataError(RuntimeError):
    pass


def make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=2, backoff_factor=1.0, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "schedina-ai/2.0"
    return s


def _is_fresh(path: Path, max_age_h: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_h * 3600


def fetch_csv(season: str, uk_code: str, cache_dir: Path, session: requests.Session,
              current_season: str) -> Path | None:
    """Ritorna il path del CSV in cache. Le stagioni chiuse non si riscaricano mai."""
    dest = cache_dir / f"{season}_{uk_code}.csv"
    is_current = season == current_season
    if dest.exists() and not (is_current and not _is_fresh(dest, CURRENT_SEASON_MAX_AGE_H)):
        return dest
    url = HISTORY_URL.format(season=season, code=uk_code)
    try:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        if len(resp.content) < 200:
            raise ValueError("risposta troppo corta")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest
    except Exception as exc:  # noqa: BLE001 - rete/HTTP/IO: decidiamo sotto se è fatale
        if dest.exists():
            log.warning("Download %s fallito (%s): uso la cache", url, exc)
            return dest
        log.warning("Download %s fallito (%s): stagione saltata", url, exc)
        return None


def parse_csv(path: Path, league_name: str, season: str) -> pd.DataFrame:
    raw = pd.read_csv(path, encoding="latin-1", on_bad_lines="skip")
    need = {"Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
    if not need.issubset(raw.columns):
        raise DataError(f"{path.name}: colonne mancanti {sorted(need - set(raw.columns))}")

    out = pd.DataFrame({
        "league": league_name,
        "season": season,
        "date": pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce", format="mixed"),
        "home": raw["HomeTeam"].astype("string").str.strip(),
        "away": raw["AwayTeam"].astype("string").str.strip(),
        "hg": pd.to_numeric(raw["FTHG"], errors="coerce"),
        "ag": pd.to_numeric(raw["FTAG"], errors="coerce"),
        "result": raw["FTR"].astype("string").str.strip(),
    })

    # tiri (HS/AS) e tiri in porta (HST/AST): segnale meno rumoroso dei gol; NaN se assenti
    for dst, src in (("sh_h", "HS"), ("sh_a", "AS"), ("sot_h", "HST"), ("sot_a", "AST")):
        out[dst] = pd.to_numeric(raw[src], errors="coerce") if src in raw.columns else np.nan

    # quote -> probabilità di mercato normalizzate (tolto il margine)
    odds = {k: pd.Series(np.nan, index=raw.index) for k in "hda"}
    for cols in ODDS_TRIPLES:
        if all(c in raw.columns for c in cols):
            for k, c in zip("hda", cols):
                odds[k] = odds[k].combine_first(pd.to_numeric(raw[c], errors="coerce"))
    inv = pd.DataFrame({k: 1.0 / v.where(v > 1.0) for k, v in odds.items()})
    total = inv.sum(axis=1, skipna=False)
    for k in "hda":
        out[f"mkt_{k}"] = inv[k] / total

    # quote grezze (con margine) per valutare il ritorno economico: una terna coerente per riga
    bet = pd.DataFrame(np.nan, index=raw.index, columns=["odds_h", "odds_d", "odds_a"])
    for cols in BET_ODDS_TRIPLES:
        if all(c in raw.columns for c in cols):
            trip = pd.DataFrame({k: pd.to_numeric(raw[c], errors="coerce")
                                 for k, c in zip(bet.columns, cols)})
            fill = bet.isna().all(axis=1) & (trip > 1.0).all(axis=1)
            bet.loc[fill, :] = trip.loc[fill, :]
    for k in bet.columns:
        out[k] = bet[k]

    out = out.dropna(subset=["date", "home", "away", "hg", "ag"])
    out = out[out["result"].isin(["H", "D", "A"])]
    out["hg"] = out["hg"].astype(int)
    out["ag"] = out["ag"].astype(int)
    return out[MATCH_COLUMNS]


def load_history(today: date | None = None, cache_dir: Path | None = None,
                 session: requests.Session | None = None) -> pd.DataFrame:
    """Storico partite di tutti i campionati, ordinato per data."""
    cache_dir = cache_dir or data_dir() / "raw"
    session = session or make_session()
    seasons = season_codes(today)
    current = seasons[-1]

    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    consecutive_fail = 0
    for season in seasons:
        for lg in LEAGUES:
            path = fetch_csv(season, lg.uk_code, cache_dir, session, current)
            if path is None:
                missing.append(f"{lg.name} {season}")
                consecutive_fail += 1
                if consecutive_fail >= MAX_CONSECUTIVE_FAILURES and not frames:
                    raise DataError(
                        "Fonte dati storici non raggiungibile (football-data.co.uk): "
                        f"{consecutive_fail} download consecutivi falliti e nessuna cache locale"
                    )
                continue
            consecutive_fail = 0
            try:
                frames.append(parse_csv(path, lg.name, season))
            except Exception as exc:  # noqa: BLE001
                log.warning("CSV %s non leggibile (%s)", path.name, exc)
                missing.append(f"{lg.name} {season}")

    if not frames:
        raise DataError("Nessun dato storico disponibile (rete bloccata o fonte down?)")
    if missing:
        log.warning("Dati storici mancanti: %s", ", ".join(missing))

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["league", "date", "home", "away"])
    return df.sort_values(["date", "league", "home"], kind="stable").reset_index(drop=True)
