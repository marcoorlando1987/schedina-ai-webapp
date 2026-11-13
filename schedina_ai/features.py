"""Feature engineering sequenziale e *senza leakage*.

Per ogni partita le feature sono calcolate con lo stato precedente alla partita;
solo dopo lo stato viene aggiornato con il risultato.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

ELO_START = 1500.0
ELO_NEW_TEAM = 1450.0  # neopromosse / squadre mai viste
ELO_K = 20.0
ELO_HOME_ADV = 60.0
ELO_SEASON_REGRESSION = 0.25  # a inizio stagione ci si avvicina alla media
SEASON_GAP_DAYS = 45

EWMA_ALPHA = 0.12
GOALS_START = 1.35
PTS_START = 1.35
FORM_WINDOW = 5

SOT_START = 4.5   # tiri in porta per squadra a partita (media di lega)
SH_START = 12.0   # tiri totali per squadra a partita
SHOT_ALPHA = 0.10

FEATURES = [
    "elo_diff", "h_gf", "h_ga", "a_gf", "a_ga", "exp_h", "exp_a",
    "h_pts5", "a_pts5", "h_rest", "a_rest", "h_n", "a_n",
    # tiri: medie mobili di tiri in porta / tiri fatti e subiti
    "h_sotf", "h_sota", "a_sotf", "a_sota", "exp_sot_h", "exp_sot_a",
    "h_shf", "h_sha", "a_shf", "a_sha",
]


@dataclass
class TeamState:
    elo: float = ELO_START
    gf: float = GOALS_START
    ga: float = GOALS_START
    sotf: float = SOT_START
    sota: float = SOT_START
    shf: float = SH_START
    sha: float = SH_START
    pts: deque = field(default_factory=lambda: deque(maxlen=FORM_WINDOW))
    last_date: pd.Timestamp | None = None
    n: int = 0


class RatingEngine:
    """Stato (Elo, medie gol, forma) di tutte le squadre, per lega."""

    def __init__(self) -> None:
        self._teams: dict[str, dict[str, TeamState]] = {}
        self._last_date: dict[str, pd.Timestamp] = {}
        self._season_idx: dict[str, int] = {}  # 0 = prima stagione vista (nessuna "nuova" squadra)

    # -- stato -----------------------------------------------------------
    def known_teams(self) -> dict[str, set[str]]:
        return {lg: set(t) for lg, t in self._teams.items()}

    def _fresh(self, league: str) -> TeamState:
        # Nella prima stagione vista tutti partono alla pari; dopo, una squadra mai
        # vista è una neopromossa e parte più in basso.
        return TeamState(elo=ELO_START if self._season_idx.get(league, 0) == 0 else ELO_NEW_TEAM)

    def _peek(self, league: str, team: str | None) -> TeamState:
        """Stato della squadra senza crearlo (stessa regola usata da update)."""
        st = self._teams.get(league, {}).get(team) if team else None
        return st if st is not None else self._fresh(league)

    def _get(self, league: str, team: str) -> TeamState:
        league_teams = self._teams.setdefault(league, {})
        st = league_teams.get(team)
        if st is None:
            st = league_teams[team] = self._fresh(league)
        return st

    def _maybe_new_season(self, league: str, date: pd.Timestamp) -> None:
        last = self._last_date.get(league)
        if last is not None and (date - last).days > SEASON_GAP_DAYS:
            for st in self._teams.get(league, {}).values():
                st.elo += (ELO_START - st.elo) * ELO_SEASON_REGRESSION
                st.pts.clear()
            self._season_idx[league] = self._season_idx.get(league, 0) + 1
            self._last_date[league] = date  # idempotente se richiamato con la stessa data

    # -- API -------------------------------------------------------------
    def snapshot(self, league: str, home: str | None, away: str | None,
                 date: pd.Timestamp) -> dict[str, float]:
        """Feature della partita con lo stato corrente (non modifica le squadre)."""
        self._maybe_new_season(league, date)
        h = self._peek(league, home)
        a = self._peek(league, away)

        def rest(st: TeamState) -> float:
            if st.last_date is None:
                return 14.0
            return float(min(max((date - st.last_date).days, 2), 14))

        def pts(st: TeamState) -> float:
            return sum(st.pts) / len(st.pts) if st.pts else PTS_START

        return {
            "elo_diff": h.elo - a.elo,
            "h_gf": h.gf, "h_ga": h.ga, "a_gf": a.gf, "a_ga": a.ga,
            "exp_h": (h.gf + a.ga) / 2.0,
            "exp_a": (a.gf + h.ga) / 2.0,
            "h_pts5": pts(h), "a_pts5": pts(a),
            "h_rest": rest(h), "a_rest": rest(a),
            "h_n": float(min(h.n, 20)), "a_n": float(min(a.n, 20)),
            "h_sotf": h.sotf, "h_sota": h.sota, "a_sotf": a.sotf, "a_sota": a.sota,
            "exp_sot_h": (h.sotf + a.sota) / 2.0, "exp_sot_a": (a.sotf + h.sota) / 2.0,
            "h_shf": h.shf, "h_sha": h.sha, "a_shf": a.shf, "a_sha": a.sha,
        }

    def update(self, league: str, home: str, away: str, date: pd.Timestamp,
               hg: int, ag: int, sh_h: float = math.nan, sh_a: float = math.nan,
               sot_h: float = math.nan, sot_a: float = math.nan) -> None:
        h = self._get(league, home)
        a = self._get(league, away)

        expected_h = 1.0 / (1.0 + 10 ** (-((h.elo - a.elo) + ELO_HOME_ADV) / 400.0))
        actual_h = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        mov = 1.0 + math.log(1.0 + abs(hg - ag))
        delta = ELO_K * mov * (actual_h - expected_h)
        h.elo += delta
        a.elo -= delta

        h.gf += EWMA_ALPHA * (hg - h.gf)
        h.ga += EWMA_ALPHA * (ag - h.ga)
        a.gf += EWMA_ALPHA * (ag - a.gf)
        a.ga += EWMA_ALPHA * (hg - a.ga)

        # tiri: si aggiorna solo se il dato c'è (NaN = partita senza statistiche)
        def ewma(old: float, new: float) -> float:
            return old if new is None or math.isnan(new) else old + SHOT_ALPHA * (new - old)

        h.sotf, h.sota = ewma(h.sotf, sot_h), ewma(h.sota, sot_a)
        a.sotf, a.sota = ewma(a.sotf, sot_a), ewma(a.sota, sot_h)
        h.shf, h.sha = ewma(h.shf, sh_h), ewma(h.sha, sh_a)
        a.shf, a.sha = ewma(a.shf, sh_a), ewma(a.sha, sh_h)

        h.pts.append(3 if hg > ag else 1 if hg == ag else 0)
        a.pts.append(3 if ag > hg else 1 if hg == ag else 0)
        h.last_date = a.last_date = date
        h.n += 1
        a.n += 1
        self._last_date[league] = date


def build_training_frame(matches: pd.DataFrame) -> tuple[pd.DataFrame, RatingEngine]:
    """Per ogni partita storica: feature pre-partita + target. Ritorna anche l'engine
    nello stato finale (da usare per prevedere le partite future)."""
    engine = RatingEngine()
    rows: list[dict] = []
    matches = matches.copy()
    for col in ("sh_h", "sh_a", "sot_h", "sot_a"):  # compatibilità con dati senza tiri
        if col not in matches.columns:
            matches[col] = math.nan
    for r in matches.sort_values(["date", "league"], kind="stable").itertuples(index=False):
        feats = engine.snapshot(r.league, r.home, r.away, r.date)
        rows.append({
            "league": r.league, "season": r.season, "date": r.date,
            "home": r.home, "away": r.away, "hg": r.hg, "ag": r.ag, "result": r.result,
            "mkt_h": r.mkt_h, "mkt_d": r.mkt_d, "mkt_a": r.mkt_a,
            **feats,
        })
        engine.update(r.league, r.home, r.away, r.date, r.hg, r.ag,
                      r.sh_h, r.sh_a, r.sot_h, r.sot_a)
    return pd.DataFrame(rows), engine
