"""Generatore di stagioni sintetiche con forza squadre nota (per test e collaudi)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from schedina_ai.config import LEAGUE_NAMES


def _round_robin(teams: list[str]) -> list[list[tuple[str, str]]]:
    n = len(teams)
    t = teams[:]
    rounds = []
    for r in range(n - 1):
        pairs = [(t[i], t[n - 1 - i]) for i in range(n // 2)]
        pairs = [(a, b) if (r + i) % 2 == 0 else (b, a) for i, (a, b) in enumerate(pairs)]
        rounds.append(pairs)
        t = [t[0]] + [t[-1]] + t[1:-1]
    return rounds + [[(b, a) for a, b in rd] for rd in rounds]


def make_matches(n_seasons: int = 6, teams_per_league: int = 18, seed: int = 0,
                 last_season_start: int = 2025, with_odds: bool = True,
                 leagues: tuple[str, ...] = LEAGUE_NAMES) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    strengths: dict[str, dict[str, float]] = {}
    for lg in leagues:
        strengths[lg] = {f"{lg[:3]}_{i:02d}": rng.normal(0, 0.35) for i in range(teams_per_league)}

    for s in range(n_seasons):
        year = last_season_start - (n_seasons - 1) + s
        season = f"{year % 100:02d}{(year + 1) % 100:02d}"
        for lg in leagues:
            st = strengths[lg]
            if s > 0:  # 3 neopromosse al posto delle ultime 3; persistenza forza AR(1)
                worst = sorted(st, key=st.get)[:3]
                for w in worst:
                    del st[w]
                for j in range(3):
                    st[f"{lg[:3]}_new{s}{j}"] = rng.normal(-0.2, 0.3)
                for k in st:
                    st[k] = 0.8 * st[k] + rng.normal(0, 0.15)
            teams = list(st)
            rng.shuffle(teams)
            for rd, pairs in enumerate(_round_robin(teams)):
                date = pd.Timestamp(year=year, month=8, day=20) + pd.Timedelta(days=7 * rd)
                for h, a in pairs:
                    lam_h = np.exp(0.28 + st[h] - 0.5 * st[a] + 0.0)
                    lam_a = np.exp(0.0 + st[a] - 0.5 * st[h])
                    hg, ag = rng.poisson(lam_h), rng.poisson(lam_a)
                    # probabilità "vere" dalla matrice Poisson, per le quote sintetiche
                    k = np.arange(11)
                    from scipy.stats import poisson
                    joint = np.outer(poisson.pmf(k, lam_h), poisson.pmf(k, lam_a))
                    p = np.array([np.tril(joint, -1).sum(), np.trace(joint), np.triu(joint, 1).sum()])
                    p /= p.sum()
                    noisy = p * np.exp(rng.normal(0, 0.05, 3))
                    noisy /= noisy.sum()
                    rows.append({
                        "league": lg, "season": season, "date": date, "home": h, "away": a,
                        "hg": int(hg), "ag": int(ag),
                        "result": "H" if hg > ag else "D" if hg == ag else "A",
                        "mkt_h": noisy[0] if with_odds else np.nan,
                        "mkt_d": noisy[1] if with_odds else np.nan,
                        "mkt_a": noisy[2] if with_odds else np.nan,
                    })
    return pd.DataFrame(rows).sort_values(["date", "league"], kind="stable").reset_index(drop=True)
