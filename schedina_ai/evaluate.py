"""Validazione walk-forward: si addestra solo sul passato e si testa sulla stagione successiva."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import BURN_IN_SEASONS
from .model import CLASS_IDX, KINDS, Predictor

log = logging.getLogger(__name__)

PROB_COLS = ["p_home", "p_draw", "p_away"]
MIN_TRAIN_MATCHES = 1500


def usable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Scarta le stagioni di burn-in (stato di Elo/forma non ancora affidabile)."""
    seasons = sorted(frame["season"].unique())
    keep = seasons[BURN_IN_SEASONS:]
    return frame[frame["season"].isin(keep)].reset_index(drop=True)


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.0))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    onehot = np.eye(3)[y]
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def walk_forward(frame: pd.DataFrame, n_test_seasons: int = 3,
                 kinds: tuple[str, ...] = KINDS) -> pd.DataFrame:
    """Metriche per modello, aggregate sulle ultime `n_test_seasons` stagioni.

    Ritorna una riga per modello con n, accuracy, logloss, brier e, dove ci sono le
    quote, il log-loss del mercato sulle *stesse* partite (benchmark da battere).
    """
    data = usable_frame(frame)
    seasons = sorted(data["season"].unique())
    test_seasons = seasons[-n_test_seasons:]

    preds: dict[str, list[pd.DataFrame]] = {k: [] for k in kinds}
    tests: list[pd.DataFrame] = []
    for s in test_seasons:
        train = data[data["season"] < s]
        test = data[data["season"] == s]
        if len(train) < MIN_TRAIN_MATCHES or test.empty:
            log.info("Fold %s saltato (train=%d, test=%d)", s, len(train), len(test))
            continue
        tests.append(test)
        for k in kinds:
            preds[k].append(Predictor(k).fit(train).predict(test))
    if not tests:
        raise ValueError("Dati insufficienti per la validazione walk-forward")

    test_all = pd.concat(tests)
    y = test_all["result"].map(CLASS_IDX).to_numpy()
    has_mkt = test_all[["mkt_h", "mkt_d", "mkt_a"]].notna().all(axis=1).to_numpy()
    mkt = test_all[["mkt_h", "mkt_d", "mkt_a"]].to_numpy()

    rows = []
    for k in kinds:
        p = pd.concat(preds[k])[PROB_COLS].to_numpy()
        row = {
            "model": k, "n": len(y),
            "accuracy": float(np.mean(p.argmax(axis=1) == y)),
            "logloss": log_loss(p, y), "brier": brier(p, y),
        }
        if has_mkt.sum() >= 200:
            row["n_mkt"] = int(has_mkt.sum())
            row["logloss_same_matches"] = log_loss(p[has_mkt], y[has_mkt])
            row["market_logloss"] = log_loss(mkt[has_mkt], y[has_mkt])
            row["market_accuracy"] = float(np.mean(mkt[has_mkt].argmax(axis=1) == y[has_mkt]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)


def best_model(metrics: pd.DataFrame) -> str:
    return str(metrics.iloc[0]["model"])
