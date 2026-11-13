"""Modelli 1X2 + gol attesi.

Tutti i candidati condividono i regressori di Poisson per i gol attesi (da cui
P(Over 2.5)); cambia solo il modo di ottenere le probabilità 1X2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from .config import LEAGUE_NAMES
from .features import FEATURES

CLASSES = ["H", "D", "A"]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}
KINDS = ("baseline", "logreg", "xgb", "poisson")
MAX_GOALS = 10
HALF_LIFE_YEARS = 2.0


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].astype(float).copy()
    for name in LEAGUE_NAMES:  # colonne fisse, anche se una lega manca nel batch
        X[f"lg_{name}"] = (df["league"] == name).astype(float)
    return X


def time_weights(dates: pd.Series, ref: pd.Timestamp | None = None) -> np.ndarray:
    ref = ref if ref is not None else dates.max()
    age_years = (ref - dates).dt.days.to_numpy() / 365.25
    return 0.5 ** (age_years / HALF_LIFE_YEARS)


def score_matrix_probs(xg_h: np.ndarray, xg_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Da gol attesi a (P(1X2), P(Over 2.5)) assumendo gol Poisson indipendenti."""
    k = np.arange(MAX_GOALS + 1)
    ph = poisson.pmf(k[None, :], xg_h[:, None])  # (n, G)
    pa = poisson.pmf(k[None, :], xg_a[:, None])
    joint = ph[:, :, None] * pa[:, None, :]  # (n, G, G) [casa, trasferta]
    i, j = np.meshgrid(k, k, indexing="ij")
    p1 = joint[:, i > j].sum(axis=1)
    px = joint[:, i == j].sum(axis=1)
    p2 = joint[:, i < j].sum(axis=1)
    probs = np.stack([p1, px, p2], axis=1)
    probs /= probs.sum(axis=1, keepdims=True)
    over = joint[:, (i + j) >= 3].sum(axis=1) / joint.sum(axis=(1, 2))
    return probs, over


class Predictor:
    def __init__(self, kind: str):
        if kind not in KINDS:
            raise ValueError(f"modello sconosciuto: {kind}")
        self.kind = kind

    # -- training --------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "Predictor":
        X = design_matrix(df)
        w = time_weights(df["date"])
        y = df["result"].map(CLASS_IDX).to_numpy()

        self._cols = list(X.columns)
        self._poi_h = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.05, max_iter=500))
        self._poi_a = make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.05, max_iter=500))
        self._poi_h.fit(X, df["hg"], poissonregressor__sample_weight=w)
        self._poi_a.fit(X, df["ag"], poissonregressor__sample_weight=w)

        if self.kind == "baseline":
            self._freq = {}
            for lg, idx in df.groupby("league").groups.items():
                sub_w = pd.Series(w, index=df.index).loc[idx]
                f = np.bincount(y[df.index.get_indexer(idx)], weights=sub_w.to_numpy(), minlength=3)
                self._freq[lg] = f / f.sum()
            f = np.bincount(y, weights=w, minlength=3)
            self._freq_all = f / f.sum()
        elif self.kind == "logreg":
            self._clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
            self._clf.fit(X, y, logisticregression__sample_weight=w)
        elif self.kind == "xgb":
            self._clf = XGBClassifier(
                n_estimators=250, max_depth=3, learning_rate=0.03, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0,
                objective="multi:softprob", eval_metric="mlogloss", n_jobs=2,
                random_state=42, verbosity=0,
            )
            self._clf.fit(X, y, sample_weight=w)
        return self

    # -- inferenza -------------------------------------------------------
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = design_matrix(df)[self._cols]
        xg_h = np.clip(self._poi_h.predict(X), 0.05, 6.0)
        xg_a = np.clip(self._poi_a.predict(X), 0.05, 6.0)
        poi_probs, p_over = score_matrix_probs(xg_h, xg_a)

        if self.kind == "baseline":
            probs = np.vstack([self._freq.get(lg, self._freq_all) for lg in df["league"]])
        elif self.kind == "poisson":
            probs = poi_probs
        else:
            probs = self._clf.predict_proba(X)
        probs = np.clip(probs, 1e-6, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)

        return pd.DataFrame({
            "p_home": probs[:, 0], "p_draw": probs[:, 1], "p_away": probs[:, 2],
            "xg_home": xg_h, "xg_away": xg_a, "p_over25": p_over,
        }, index=df.index)
