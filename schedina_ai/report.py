"""Report diagnostico di affidabilità (markdown): calibrazione, stagioni, leghe, confronto col mercato."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import (MKT_COLS, brier, ece, log_loss, metrics_table, rps,
                       walk_forward_predictions)
from .model import CLASS_IDX, KINDS, PROB_COLS


def _md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def build_report(frame: pd.DataFrame, n_test_seasons: int = 3, kinds=KINDS) -> str:
    test, preds = walk_forward_predictions(frame, n_test_seasons, kinds)
    metrics = metrics_table(test, preds)
    y = test["result"].map(CLASS_IDX).to_numpy()
    best = str(metrics.iloc[0]["model"])
    pb = preds[best][PROB_COLS].to_numpy()
    mkt = test[MKT_COLS].to_numpy()
    has = ~np.isnan(mkt).any(axis=1)

    parts = [f"# Report di affidabilità\n\nTest: {len(test)} partite, stagioni "
             f"{', '.join(sorted(test['season'].unique()))} (walk-forward). Modello migliore per log-loss: **{best}**.\n"]

    # 1. confronto modelli
    cols = ["model", "accuracy", "logloss", "brier", "rps", "ece"]
    t = metrics[cols].copy()
    if "market_logloss" in metrics:
        t["Δlogloss vs mercato"] = metrics["logloss_same_matches"] - metrics["market_logloss"]
        mk = metrics.iloc[0]
        parts.append("## Modelli (più basso = meglio, tranne accuracy)\n\n" + _md(t))
        parts.append(f"\nMercato (quote di chiusura, senza margine) sulle stesse {int(mk['n_mkt'])} partite: "
                     f"logloss **{mk['market_logloss']:.4f}**, rps **{mk['market_rps']:.4f}**, "
                     f"accuracy **{mk['market_accuracy']:.4f}**, "
                     f"brier {brier(mkt[has], y[has]):.4f}, ece {ece(mkt[has], y[has]):.4f}.\n")
    else:
        parts.append("## Modelli\n\n" + _md(t))

    # 2. per stagione
    rows = []
    for s in sorted(test["season"].unique()):
        m = (test["season"] == s).to_numpy()
        r = {"stagione": s, "n": int(m.sum()), f"logloss {best}": log_loss(pb[m], y[m])}
        mm = m & has
        if mm.sum() > 50:
            r["logloss mercato"] = log_loss(mkt[mm], y[mm])
        rows.append(r)
    parts.append("\n## Per stagione\n\n" + _md(pd.DataFrame(rows)))

    # 3. per lega
    rows = []
    for lg in sorted(test["league"].unique()):
        m = (test["league"] == lg).to_numpy()
        r = {"lega": lg, "n": int(m.sum()), f"logloss {best}": log_loss(pb[m], y[m]),
             "accuracy": float(np.mean(pb[m].argmax(axis=1) == y[m]))}
        mm = m & has
        if mm.sum() > 50:
            r["logloss mercato"] = log_loss(mkt[mm], y[mm])
        rows.append(r)
    parts.append("\n## Per lega\n\n" + _md(pd.DataFrame(rows)))

    # 4. calibrazione: probabilità dichiarata vs frequenza osservata (tutti gli esiti)
    probs, hits = pb.ravel(), np.eye(3)[y].ravel()
    edges = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi)
        if m.sum() >= 30:
            r = {"fascia": f"{lo:.0%}–{min(hi, 1):.0%}", "n": int(m.sum()),
                 "prob media dichiarata": float(probs[m].mean()), "frequenza osservata": float(hits[m].mean())}
            rows.append(r)
    parts.append(f"\n## Calibrazione di {best}\n\nSe il modello è calibrato le due colonne coincidono.\n\n"
                 + _md(pd.DataFrame(rows)))

    # 5. medie per esito
    rows = [{"esito": name, "prob media prevista": float(pb[:, i].mean()),
             "frequenza reale": float((y == i).mean()),
             **({"mercato (prob media)": float(mkt[has][:, i].mean())} if has.any() else {})}
            for i, name in enumerate(["1 (casa)", "X (pareggio)", "2 (trasferta)"])]
    parts.append("\n## Medie per esito\n\n" + _md(pd.DataFrame(rows)))

    # 6. sicurezza: quando il modello è sicuro, ha ragione?
    rows = []
    top = pb.max(axis=1)
    for lo in (0.0, 0.5, 0.6, 0.7, 0.8):
        m = top >= lo
        if m.sum() >= 50:
            r = {"confidenza ≥": f"{lo:.0%}", "n": int(m.sum()),
                 "accuracy modello": float(np.mean(pb[m].argmax(axis=1) == y[m])),
                 "prob media dichiarata": float(top[m].mean())}
            mm = m & has
            if mm.sum() >= 50:
                r["accuracy mercato (stesse partite)"] = float(np.mean(mkt[mm].argmax(axis=1) == y[mm]))
            rows.append(r)
    parts.append("\n## Quando il modello è sicuro\n\n" + _md(pd.DataFrame(rows)))
    return "\n".join(parts) + "\n"
