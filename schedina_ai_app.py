import streamlit as st
import pandas as pd
from backend_schedina import run_schedina_ai

st.set_page_config(page_title="Schedina AI", page_icon="⚽", layout="centered")

st.title("🎯 Schedina AI – Predizione partite di calcio")
st.markdown("Seleziona una data per generare la schedina intelligente su 5 campionati europei.")

# Data input
date_input = st.date_input("📅 Scegli la data della schedina")

# Bottone per generare schedina
if st.button("🔮 Genera Schedina"):
    with st.spinner("Sto calcolando..."):
        df = run_schedina_ai(str(date_input))

        if df.empty:
            st.warning("❌ Nessuna partita trovata o riconosciuta in quella data.")
        else:
            st.success(f"✅ {len(df)} partite trovate!")
            st.dataframe(df[['League', 'HomeTeam', 'AwayTeam', 'Esito_1X2', 'Gol_Previsti', 'Confidenza', 'UTCDate']])
            st.download_button("📥 Scarica CSV", df.to_csv(index=False), file_name="schedina_ai.csv", mime="text/csv")
