import streamlit as st
import pandas as pd
from backend_schedina import run_schedina_ai, leggi_predizioni_da_db

st.set_page_config(page_title="Schedina AI", page_icon="⚽", layout="centered")

st.title("🎯 Schedina AI – Predizione partite di calcio")
st.markdown("Seleziona una data per generare la schedina intelligente su 5 campionati europei.")

date_input = st.date_input("📅 Scegli la data della schedina")

if st.button("🔮 Genera Schedina"):
    with st.spinner("Sto calcolando..."):
        df = run_schedina_ai(str(date_input))
        if df.empty:
            st.warning("Nessuna partita trovata in quella data.")
        else:
            st.success(f"{len(df)} partite trovate!")
            st.dataframe(df[['League', 'HomeTeam', 'AwayTeam', 'Esito_1X2', 'Gol_Previsti', 'Confidenza', 'UTCDate']])
            st.download_button("📥 Scarica CSV", df.to_csv(index=False), file_name="schedina_ai.csv", mime="text/csv")

st.markdown("---")
st.subheader("📂 Storico predizioni salvate")
if st.checkbox("Mostra storico"):
    df_storico = leggi_predizioni_da_db()
    st.dataframe(df_storico) if not df_storico.empty else st.info("Nessuna predizione salvata nel database.")
