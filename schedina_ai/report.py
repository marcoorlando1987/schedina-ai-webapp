"""Report diagnostico di affidabilità (markdown): calibrazione, stagioni, leghe, confronto col mercato."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import (MKT_COLS, brier, ece, log_loss, metrics_table, rps,
                       walk_forward_predictions)

ODDS_COLS = ["odds_h", "odds_d", "odds_a"]
from .model import CLASS_IDX, KINDS, PROB_COLS


def _md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = ["–" if (isinstance(v, (float, np.floating)) and np.isnan(v)) else
                 floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def betting_section(test: pd.DataFrame, pb: np.ndarray, y: np.ndarray, best: str) -> str:
    """Test economico: cosa sarebbe successo puntando 1 unità seguendo il modello.

    Serve a misurare se le stime hanno valore *dopo* il margine del bookmaker; non è un
    consiglio di gioco. Quote: Bet365 (ripiego: media/Pinnacle) già presenti nei CSV.
    """
    odds = test[ODDS_COLS].to_numpy(dtype=float)
    ok = ~np.isnan(odds).any(axis=1)
    if ok.sum() < 300:
        return ""
    odds, pb, y = odds[ok], pb[ok], y[ok]
    n = len(y)
    margin = float(np.mean(1 / odds[:, 0] + 1 / odds[:, 1] + 1 / odds[:, 2] - 1))
    won = np.eye(3)[y].astype(bool)

    def summarize(name: str, pick: np.ndarray, mask: np.ndarray, claimed: float | None = None) -> dict:
        idx = np.where(mask)[0]
        if len(idx) < 30:
            return {"strategia": name, "scommesse": int(len(idx)), "vinte": "–", "quota media": "–",
                    "ROI reale": "–", "intervallo 95%": "troppo poche scommesse",
                    "ROI promesso dal modello": "–"}
        o = odds[idx, pick[idx]]
        hit = won[idx, pick[idx]]
        profit = np.where(hit, o - 1.0, -1.0)
        roi, se = profit.mean(), profit.std(ddof=1) / np.sqrt(len(idx))
        row = {"strategia": name, "scommesse": int(len(idx)), "vinte": float(hit.mean()),
               "quota media": float(o.mean()), "ROI reale": float(roi),
               "intervallo 95%": f"{roi - 1.96 * se:+.1%} … {roi + 1.96 * se:+.1%}"}
        row["ROI promesso dal modello"] = float(claimed) if claimed is not None else np.nan
        return row

    rows = []
    # a) tutto sul favorito / sulla squadra di casa / a caso: ciò che costa il margine da solo
    fav = odds.argmin(axis=1)
    rows.append(summarize("sempre il favorito (quota più bassa)", fav, np.ones(n, bool)))
    rows.append(summarize("sempre la squadra di casa", np.zeros(n, int), np.ones(n, bool)))
    rng = np.random.default_rng(0)
    rows.append(summarize("esito a caso", rng.integers(0, 3, n), np.ones(n, bool)))
    # b) "value bet": si punta dove il modello vede più valore della quota
    edge = pb * odds - 1.0
    pick = edge.argmax(axis=1)
    best_edge = edge.max(axis=1)
    for thr in (0.0, 0.03, 0.05, 0.10):
        m = best_edge > thr
        claimed = float(best_edge[m].mean()) if m.any() else None
        rows.append(summarize(f"value bet: vantaggio stimato > {thr:.0%}", pick, m, claimed))
    # c) stessa cosa ma con il mercato come "modello" (non può che perdere il margine)
    mk = test[MKT_COLS].to_numpy(dtype=float)[ok]
    edge_m = mk * odds - 1.0
    rows.append(summarize("riferimento: probabilità del mercato, vantaggio > 0%", edge_m.argmax(axis=1),
                          edge_m.max(axis=1) > 0.0))

    df = pd.DataFrame(rows)
    def fmt(c: str, x):
        if isinstance(x, str):
            return x
        if pd.isna(x):
            return "–"
        return f"{x:+.1%}" if "ROI" in c else f"{x:.3f}" if c == "quota media" else f"{x:.1%}"

    for c in ("vinte", "quota media", "ROI reale", "ROI promesso dal modello"):
        if c in df:
            df[c] = [fmt(c, x) for x in df[c]]
    return (f"\n## Test economico: puntare 1 unità seguendo {best}\n\n"
            f"Su {n} partite con quote disponibili il margine medio del bookmaker è **{margin:.1%}** "
            "(quanto costa, in media, scommettere a caso). ROI = profitto per unità puntata; "
            "l'intervallo mostra quanto è rumoroso il risultato.\n\n" + _md(df) + "\n")


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
    parts.append(betting_section(test, pb, y, best))
    return "\n".join(parts) + "\n"
