"""Validazione walk-forward: si addestra solo sul passato e si testa sulla stagione successiva."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import BURN_IN_SEASONS
from .model import CLASS_IDX, KINDS, PROB_COLS, Predictor

log = logging.getLogger(__name__)

MKT_COLS = ["mkt_h", "mkt_d", "mkt_a"]
MIN_TRAIN_MATCHES = 1500


def usable_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Scarta le stagioni di burn-in (stato di Elo/forma non ancora affidabile)."""
    seasons = sorted(frame["season"].unique())
    keep = seasons[BURN_IN_SEASONS:]
    return frame[frame["season"].isin(keep)].reset_index(drop=True)


# -- metriche ------------------------------------------------------------------
def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1.0))))


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(np.sum((p - np.eye(3)[y]) ** 2, axis=1)))


def rps(p: np.ndarray, y: np.ndarray) -> float:
    """Ranked probability score (esiti ordinati 1 < X < 2): standard nel calcio."""
    cp = np.cumsum(p, axis=1)[:, :2]
    cy = np.cumsum(np.eye(3)[y], axis=1)[:, :2]
    return float(np.mean(np.sum((cp - cy) ** 2, axis=1) / 2.0))


def ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error su tutte le probabilità (1, X, 2) in 10 fasce."""
    probs = p.ravel()
    hits = np.eye(3)[y].ravel()
    idx = np.minimum((probs * bins).astype(int), bins - 1)
    err = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            err += m.mean() * abs(probs[m].mean() - hits[m].mean())
    return float(err)


# -- walk-forward ----------------------------------------------------------------
def walk_forward_predictions(frame: pd.DataFrame, n_test_seasons: int = 3,
                             kinds: tuple[str, ...] = KINDS
                             ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Partite di test (ultime `n_test_seasons`) e predizioni out-of-sample per modello."""
    data = usable_frame(frame)
    seasons = sorted(data["season"].unique())
    preds: dict[str, list[pd.DataFrame]] = {k: [] for k in kinds}
    tests: list[pd.DataFrame] = []
    for s in seasons[-n_test_seasons:]:
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
    return pd.concat(tests), {k: pd.concat(v) for k, v in preds.items()}


def metrics_table(test: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> pd.DataFrame:
    y = test["result"].map(CLASS_IDX).to_numpy()
    mkt = test[MKT_COLS].to_numpy()
    has_mkt = ~np.isnan(mkt).any(axis=1)
    use_mkt = has_mkt.sum() >= 200

    rows = []
    for k, df in preds.items():
        p = df[PROB_COLS].to_numpy()
        row = {
            "model": k, "n": len(y),
            "accuracy": float(np.mean(p.argmax(axis=1) == y)),
            "logloss": log_loss(p, y), "brier": brier(p, y), "rps": rps(p, y), "ece": ece(p, y),
        }
        if use_mkt:
            row["n_mkt"] = int(has_mkt.sum())
            row["logloss_same_matches"] = log_loss(p[has_mkt], y[has_mkt])
            row["market_logloss"] = log_loss(mkt[has_mkt], y[has_mkt])
            row["market_accuracy"] = float(np.mean(mkt[has_mkt].argmax(axis=1) == y[has_mkt]))
            row["market_rps"] = rps(mkt[has_mkt], y[has_mkt])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("logloss").reset_index(drop=True)


def walk_forward(frame: pd.DataFrame, n_test_seasons: int = 3,
                 kinds: tuple[str, ...] = KINDS) -> pd.DataFrame:
    """Metriche per modello sulle ultime stagioni, col mercato come benchmark."""
    test, preds = walk_forward_predictions(frame, n_test_seasons, kinds)
    return metrics_table(test, preds)


def best_model(metrics: pd.DataFrame) -> str:
    return str(metrics.iloc[0]["model"])
