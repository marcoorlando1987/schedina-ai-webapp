"""Interfaccia web: legge le predizioni calcolate dal job offline (nessun training qui)."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from schedina_ai import store
from schedina_ai.config import REPORT_TZ, db_source

st.set_page_config(page_title="Schedina AI", page_icon="⚽", layout="wide")

PICK_LABEL = {"H": "1", "D": "X", "A": "2"}
UPCOMING = {"SCHEDULED", "TIMED"}
GIORNI = ["lun", "mar", "mer", "gio", "ven", "sab", "dom"]


@st.cache_data(ttl=300, show_spinner=False)
def load_data():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            local = store.materialize_db(db_source(), Path(tmp) / "schedina.db")
        except store.DbUnavailable as exc:
            return str(exc)
        conn = sqlite3.connect(f"file:{local}?mode=ro", uri=True)
        try:
            preds = store.load_predictions(conn)
            meta = {k: store.get_meta(conn, k) for k in ("last_run", "model", "metrics", "history_last_match")}
        finally:
            conn.close()
    preds["local"] = preds["utc_date"].dt.tz_convert(REPORT_TZ)
    return preds, meta


def fmt_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "Lega": df["league"],
        "Ora": df["local"].dt.strftime("%H:%M"),
        "Partita": df["home"] + " – " + df["away"] + np.where(df["new_team"] == 1, "  ⚠️", ""),
        "1": df["p_home"] * 100, "X": df["p_draw"] * 100, "2": df["p_away"] * 100,
        "Segno": df["pick"].map(PICK_LABEL),
        "Gol attesi": df["xg_home"] + df["xg_away"],
        "Over 2.5": df["p_over25"] * 100,
    })
    return out


def render_matches(preds: pd.DataFrame, meta: dict) -> None:
    upcoming = preds[preds["status"].isin(UPCOMING)].copy()
    if upcoming.empty:
        st.info("Nessuna partita in programma nei dati. Il job di aggiornamento non ha ancora girato?")
        return
    upcoming["day"] = upcoming["local"].dt.date
    days = sorted(upcoming["day"].unique())
    today = pd.Timestamp.now(tz=REPORT_TZ).date()
    default = next((i for i, d in enumerate(days) if d >= today), len(days) - 1)

    c1, c2, c3 = st.columns([2, 2, 3])
    day = c1.selectbox("📅 Giornata", days, index=default, format_func=lambda d: f"{GIORNI[d.weekday()]} {d:%d/%m/%Y}")
    leagues = sorted(upcoming["league"].unique())
    chosen = c2.multiselect("Campionati", leagues, default=leagues)
    order = c3.radio("Ordina per", ["Orario", "Probabilità più alta"], horizontal=True)

    sel = upcoming[(upcoming["day"] == day) & (upcoming["league"].isin(chosen))].copy()
    if sel.empty:
        st.warning("Nessuna partita per i filtri scelti.")
        return
    sel["top"] = sel[["p_home", "p_draw", "p_away"]].max(axis=1)
    sel = sel.sort_values("top", ascending=False) if order != "Orario" else sel.sort_values("local")

    table = fmt_table(sel)
    pct = st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f%%")
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={"1": pct, "X": pct, "2": pct,
                       "Gol attesi": st.column_config.NumberColumn(format="%.1f"),
                       "Over 2.5": st.column_config.NumberColumn(format="%.0f%%")},
    )
    if (sel["new_team"] == 1).any():
        st.caption("⚠️ Squadra non presente nello storico (es. neopromossa): previsione meno affidabile.")
    st.download_button("📥 Scarica CSV", table.to_csv(index=False), file_name=f"schedina_{day}.csv",
                       mime="text/csv")


def render_reliability(preds: pd.DataFrame, meta: dict) -> None:
    st.subheader("Validazione storica (walk-forward)")
    st.caption("Il modello è addestrato solo sul passato e testato sulle stagioni successive. "
               "Il confronto che conta è con le quote dei bookmaker: log-loss più basso = meglio.")
    metrics = meta.get("metrics")
    if metrics:
        m = pd.DataFrame(metrics)
        st.dataframe(m.round(4), hide_index=True, width="stretch")
        if "market_logloss" in m and m["market_logloss"].notna().any():
            gap = m.iloc[0]["logloss_same_matches"] - m.iloc[0]["market_logloss"]
            verdict = "batte" if gap < 0 else "non batte"
            st.write(f"Il modello migliore (**{m.iloc[0]['model']}**) {verdict} le quote "
                     f"(differenza log-loss: {gap:+.4f}).")
    else:
        st.info("Metriche non disponibili (ultimo run senza valutazione).")

    st.subheader("Track record reale")
    done = preds[(preds["status"] == "FINISHED") & preds["score_home"].notna()].copy()
    if len(done) < 10:
        st.info(f"Partite concluse con previsione registrata: {len(done)}. Servono più giornate.")
        return
    done["real"] = np.where(done["score_home"] > done["score_away"], "H",
                            np.where(done["score_home"] == done["score_away"], "D", "A"))
    probs = done[["p_home", "p_draw", "p_away"]].to_numpy()
    idx = done["real"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    done["top"] = probs.max(axis=1)
    done["hit"] = done["pick"] == done["real"]
    ll = -np.mean(np.log(np.clip(probs[np.arange(len(done)), idx], 1e-9, 1)))
    c1, c2, c3 = st.columns(3)
    c1.metric("Partite valutate", len(done))
    c2.metric("Accuratezza 1X2", f"{done['hit'].mean():.1%}")
    c3.metric("Log-loss", f"{ll:.3f}")
    bins = pd.cut(done["top"], [0, 0.45, 0.55, 0.65, 1.0], labels=["<45%", "45–55%", "55–65%", ">65%"])
    cal = done.groupby(bins, observed=True).agg(partite=("hit", "size"), prob_media=("top", "mean"),
                                                 azzeccate=("hit", "mean")).reset_index(names="fascia")
    st.caption("Calibrazione: la probabilità dichiarata dovrebbe somigliare alla percentuale di previsioni azzeccate.")
    st.dataframe(cal.round(3), hide_index=True, width="stretch")


st.title("🎯 Schedina AI")
st.caption("Probabilità 1X2 e gol attesi per Serie A, Premier League, La Liga, Bundesliga e Ligue 1.")

loaded = load_data()
if isinstance(loaded, str):
    st.error(f"{loaded}. Esegui `python -m schedina_ai run` o controlla `SCHEDINA_DB_URL`.")
    st.stop()
preds, meta = loaded

tab1, tab2 = st.tabs(["Partite", "Affidabilità del modello"])
with tab1:
    render_matches(preds, meta)
with tab2:
    render_reliability(preds, meta)

last = meta.get("last_run")
updated = f"{pd.Timestamp(last).tz_convert(REPORT_TZ):%d/%m/%Y %H:%M}" if last else "mai"
st.divider()
st.caption(f"Ultimo aggiornamento: {updated} · modello: {meta.get('model')} · "
           f"dati storici fino al {meta.get('history_last_match')}")
st.caption("⚠️ Strumento a solo scopo informativo/statistico: nessuna garanzia di risultato, "
           "il calcio è molto aleatorio. Gioca responsabilmente (18+).")
