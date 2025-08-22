import requests
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from rapidfuzz import process
import warnings
import streamlit as st
warnings.filterwarnings("ignore")

# === CONFIG ===
API_TOKEN = st.secrets["API_TOKEN"]
HEADERS = {"X-Auth-Token": API_TOKEN}
BASE_URL = "https://api.football-data.org/v4/"

LEAGUE_CODES = {
    "SA": "Serie A",
    "PL": "Premier League",
    "PD": "La Liga",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1"
}

DATA_CODES = {
    "I1": "Serie A",
    "E0": "Premier League",
    "SP1": "La Liga",
    "D1": "Bundesliga",
    "F1": "Ligue 1"
}

SEASONS = ["2122", "2223", "2324", "2425", "2526"]
SEASON_WEIGHTS = {
    "2122": 0.3,
    "2223": 0.5,
    "2324": 0.8,
    "2425": 1.0,
    "2526": 1.3
}

# === FUNZIONE: Carica dati storici ===
def load_historical_data():
    dfs = []
    for season in SEASONS:
        for code, name in DATA_CODES.items():
            url = f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
            try:
                df = pd.read_csv(url)
                df["League"] = name
                df["Season"] = season
                dfs.append(df)
            except:
                continue
    df = pd.concat(dfs, ignore_index=True)
    df = df[['League', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG', 'Season']].dropna()
    df['TotalGoals'] = df['FTHG'] + df['FTAG']
    df['TotalGoals'] = df['TotalGoals'].clip(upper=6)
    return df

# === FUNZIONE: Addestra i modelli ===
def train_models(df):
    le_home = LabelEncoder()
    le_away = LabelEncoder()
    le_league = LabelEncoder()

    df['Home_enc'] = le_home.fit_transform(df['HomeTeam'])
    df['Away_enc'] = le_away.fit_transform(df['AwayTeam'])
    df['League_enc'] = le_league.fit_transform(df['League'])

    X = df[['Home_enc', 'Away_enc', 'League_enc']]
    y1 = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    y2 = df['TotalGoals']

    # Calcolo dei pesi
    weights = df['Season'].map(SEASON_WEIGHTS).fillna(0.5)

    X_train, _, y1_train, _, y2_train, _ = train_test_split(X, y1, y2, test_size=0.2, random_state=42)

    model_1x2 = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model_gol = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model_1x2.fit(X_train, y1_train, sample_weight=weights.loc[X_train.index])
    model_gol.fit(X_train, y2_train, sample_weight=weights.loc[X_train.index])

    return model_1x2, model_gol, le_home, le_away, le_league

# === FUNZIONE: Fuzzy matching automatico
def fuzzy_match_teams(df_matches, reference_teams, column, threshold=85):
    mapping = {}
    unique_teams = df_matches[column].unique()
    for team in unique_teams:
        match, score, _ = process.extractOne(team, reference_teams)
        if score >= threshold:
            mapping[team] = match
    df_matches[column] = df_matches[column].replace(mapping)
    return df_matches, mapping

# === FUNZIONE: Scarica partite future per una data specifica
def get_matches_by_date(input_date_str):
    input_date = pd.to_datetime(input_date_str).date()
    matches = []
    for code in LEAGUE_CODES:
        url = f"{BASE_URL}competitions/{code}/matches?status=SCHEDULED"
        res = requests.get(url, headers=HEADERS)
        data = res.json()
        for match in data.get("matches", []):
            match_date = pd.to_datetime(match["utcDate"]).date()
            if match_date == input_date:
                matches.append({
                    "LeagueCode": code,
                    "League": LEAGUE_CODES[code],
                    "HomeTeam": match["homeTeam"]["name"],
                    "AwayTeam": match["awayTeam"]["name"],
                    "UTCDate": match["utcDate"]
                })
    return pd.DataFrame(matches)

# === FUNZIONE PRINCIPALE ===
def run_schedina_ai(date_str):
    try:
        df_hist = load_historical_data()
        model_1x2, model_gol, le_home, le_away, le_league = train_models(df_hist)
        df_matches = get_matches_by_date(date_str)

        if df_matches.empty:
            return pd.DataFrame()

        # Fuzzy matching
        home_ref = df_hist['HomeTeam'].unique()
        away_ref = df_hist['AwayTeam'].unique()
        df_matches, _ = fuzzy_match_teams(df_matches, home_ref, 'HomeTeam', threshold=90)
        df_matches, _ = fuzzy_match_teams(df_matches, away_ref, 'AwayTeam', threshold=90)

        # Filtra solo partite compatibili
        df_matches = df_matches[
            df_matches['HomeTeam'].isin(le_home.classes_) &
            df_matches['AwayTeam'].isin(le_away.classes_) &
            df_matches['League'].isin(le_league.classes_)
        ].reset_index(drop=True)

        if df_matches.empty:
            return pd.DataFrame()

        # Codifica
        df_matches['Home_enc'] = le_home.transform(df_matches['HomeTeam'])
        df_matches['Away_enc'] = le_away.transform(df_matches['AwayTeam'])
        df_matches['League_enc'] = le_league.transform(df_matches['League'])

        # Predizione
        X_pred = df_matches[['Home_enc', 'Away_enc', 'League_enc']]
        pred_1x2_raw = model_1x2.predict(X_pred)
        conf_1x2 = model_1x2.predict_proba(X_pred).max(axis=1)
        pred_gol = model_gol.predict(X_pred)

        inv_map = {0: 'H', 1: 'D', 2: 'A'}
        df_matches['Esito_1X2'] = pd.Series(pred_1x2_raw).map(inv_map)
        df_matches['Gol_Previsti'] = pred_gol
        df_matches['Confidenza'] = conf_1x2
        df_matches['UTCDate'] = pd.to_datetime(df_matches['UTCDate']).dt.tz_localize(None)

        # Ordina per confidenza (tutte le partite)
        schedina = df_matches.sort_values(by='Confidenza', ascending=False)

        return schedina[['League', 'HomeTeam', 'AwayTeam', 'Esito_1X2', 'Gol_Previsti', 'Confidenza', 'UTCDate']]

    except Exception as e:
        print(f"Errore durante l'esecuzione del modello: {e}")
        return pd.DataFrame()
