"""Job offline: scarica dati, valuta, addestra, prevede le prossime partite, salva su SQLite.

L'app web *non* addestra nulla: legge soltanto il risultato di questo job.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

from . import store
from .config import db_path
from .evaluate import best_model, usable_frame, walk_forward
from .features import build_training_frame
from .fixtures import UPCOMING, fetch_fixtures
from .model import Predictor
from .teams import TeamResolver

log = logging.getLogger(__name__)


def run(token: str | None, *, days_ahead: int = 7, days_back: int = 7,
        today: date | None = None, history: pd.DataFrame | None = None,
        session: requests.Session | None = None, fixtures_session: requests.Session | None = None,
        skip_eval: bool = False, force_model: str | None = None,
        db: str | None = None, now: datetime | None = None) -> dict:
    """Esegue l'intera pipeline. Ritorna un riepilogo (usato dal CLI e dai test)."""
    from .data import load_history  # import locale: i test iniettano `history`

    now = now or datetime.now(timezone.utc)
    today = today or now.date()

    # 1. dati + feature
    matches = history if history is not None else load_history(today=today, session=session)
    frame, engine = build_training_frame(matches)
    train = usable_frame(frame)
    log.info("Storico: %d partite (%d usate per il training)", len(matches), len(train))

    # 2. valutazione walk-forward -> scelta del modello
    metrics = None
    if force_model:
        model_name = force_model
    elif skip_eval:
        model_name = "logreg"
    else:
        metrics = walk_forward(frame)
        model_name = best_model(metrics)
        log.info("Walk-forward:\n%s", metrics.round(4).to_string())
    log.info("Modello scelto: %s", model_name)
    predictor = Predictor(model_name).fit(train)

    # 3. calendario (può sollevare FixturesError: nessun fallimento silenzioso)
    fx = fetch_fixtures(token, today - timedelta(days=days_back), today + timedelta(days=days_ahead),
                        session=fixtures_session)

    resolver = TeamResolver(engine.known_teams())
    n_pred = n_res = 0

    # Si lavora su una copia locale e si pubblica a fine run: SQLite non va d'accordo
    # con lock/scritture parziali su volumi di rete (es. bucket GCS montato).
    final = Path(db or db_path())
    workdir = tempfile.TemporaryDirectory()
    work = Path(workdir.name) / "schedina.db"
    if final.exists():
        shutil.copyfile(final, work)
    conn = store.connect(work)
    try:
        stamp = now.isoformat(timespec="seconds")
        upcoming_rows = []
        for r in fx.itertuples(index=False):
            started = r.utc_date <= pd.Timestamp(now) or r.status not in UPCOMING
            if started:
                store.update_result(conn, int(r.match_id), r.status,
                                    None if pd.isna(r.score_home) else int(r.score_home),
                                    None if pd.isna(r.score_away) else int(r.score_away), stamp)
                n_res += 1
                continue
            home = resolver.resolve(r.league, r.home_api)
            away = resolver.resolve(r.league, r.away_api)
            feats = engine.snapshot(r.league, home, away, r.utc_date.tz_localize(None).normalize())
            upcoming_rows.append({
                "match_id": int(r.match_id), "league": r.league, "utc_date": r.utc_date,
                "home": home or r.home_api, "away": away or r.away_api,
                "new_team": int(home is None or away is None), "status": r.status, **feats,
            })

        if upcoming_rows:
            up = pd.DataFrame(upcoming_rows)
            out = predictor.predict(up)
            probs = out[["p_home", "p_draw", "p_away"]].to_numpy()
            picks = ["H", "D", "A"]
            for rec, (_, o), p in zip(upcoming_rows, out.iterrows(), probs):
                store.upsert_prediction(conn, {
                    **{k: rec[k] for k in ("match_id", "league", "home", "away", "new_team", "status")},
                    "utc_date": rec["utc_date"].isoformat(),
                    "p_home": float(o.p_home), "p_draw": float(o.p_draw), "p_away": float(o.p_away),
                    "pick": picks[int(p.argmax())],
                    "xg_home": float(o.xg_home), "xg_away": float(o.xg_away),
                    "p_over25": float(o.p_over25), "model": model_name, "updated_at": stamp,
                })
                n_pred += 1

        store.set_meta(conn, "last_run", stamp)
        store.set_meta(conn, "model", model_name)
        store.set_meta(conn, "history_last_match", str(matches["date"].max().date()))
        if metrics is not None:
            store.set_meta(conn, "metrics", metrics.to_dict(orient="records"))
        conn.commit()
    except BaseException:
        conn.close()
        workdir.cleanup()
        raise
    conn.close()
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(work, final)
    workdir.cleanup()

    summary = {"model": model_name, "predictions": n_pred, "results_updated": n_res,
               "unresolved_teams": sorted(resolver.unresolved)}
    log.info("Fatto: %s", summary)
    return summary
