"""Mappatura nomi squadra: football-data.org (API) -> football-data.co.uk (CSV storici).

Ordine di risoluzione, sempre *all'interno della stessa lega*:
1. alias esplicito;
2. nome normalizzato identico;
3. fuzzy match con soglia alta.
Se nulla combacia la squadra è "sconosciuta" (es. neopromossa mai vista):
il chiamante la tratta con valori di default e la segnala.
"""
from __future__ import annotations

import logging
import re
import unicodedata

from rapidfuzz import fuzz, process

log = logging.getLogger(__name__)

FUZZY_CUTOFF = 88

# nome API -> nome nei CSV storici
ALIASES: dict[str, str] = {
    # Serie A
    "AC Milan": "Milan",
    "FC Internazionale Milano": "Inter",
    "Hellas Verona FC": "Verona",
    "SSC Napoli": "Napoli",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    "Atalanta BC": "Atalanta",
    "Bologna FC 1909": "Bologna",
    "US Sassuolo Calcio": "Sassuolo",
    "AC Pisa 1909": "Pisa",
    "Como 1907": "Como",
    "US Cremonese": "Cremonese",
    "Parma Calcio 1913": "Parma",
    "US Lecce": "Lecce",
    "AC Monza": "Monza",
    "UC Sampdoria": "Sampdoria",
    # Premier League
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle",
    "Nottingham Forest FC": "Nott'm Forest",
    "Tottenham Hotspur FC": "Tottenham",
    "West Ham United FC": "West Ham",
    "Wolverhampton Wanderers FC": "Wolves",
    "Brighton & Hove Albion FC": "Brighton",
    "AFC Bournemouth": "Bournemouth",
    "Leeds United FC": "Leeds",
    "Leicester City FC": "Leicester",
    "Ipswich Town FC": "Ipswich",
    "Luton Town FC": "Luton",
    "West Bromwich Albion FC": "West Brom",
    "Norwich City FC": "Norwich",
    # La Liga
    "Deportivo Alavés": "Alaves",
    "Athletic Club": "Ath Bilbao",
    "Club Atlético de Madrid": "Ath Madrid",
    "Real Betis Balompié": "Betis",
    "RC Celta de Vigo": "Celta",
    "RCD Espanyol de Barcelona": "Espanol",
    "Real Sociedad de Fútbol": "Sociedad",
    "Rayo Vallecano de Madrid": "Vallecano",
    "CA Osasuna": "Osasuna",
    "Real Valladolid CF": "Valladolid",
    "UD Las Palmas": "Las Palmas",
    "CD Leganés": "Leganes",
    "UD Almería": "Almeria",
    "Cádiz CF": "Cadiz",
    # Bundesliga
    "FC Bayern München": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "Bayer 04 Leverkusen": "Leverkusen",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "1. FC Köln": "FC Koln",
    "1. FC Heidenheim 1846": "Heidenheim",
    "TSG 1899 Hoffenheim": "Hoffenheim",
    "Borussia Mönchengladbach": "M'gladbach",
    "1. FSV Mainz 05": "Mainz",
    "FC St. Pauli 1910": "St Pauli",
    "1. FC Union Berlin": "Union Berlin",
    "SV Werder Bremen": "Werder Bremen",
    "VfL Bochum 1848": "Bochum",
    "Hamburger SV": "Hamburg",
    "Hertha BSC": "Hertha",
    "FC Schalke 04": "Schalke 04",
    # Ligue 1
    "Paris Saint-Germain FC": "Paris SG",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco FC": "Monaco",
    "LOSC Lille": "Lille",
    "OGC Nice": "Nice",
    "Stade Rennais FC 1901": "Rennes",
    "Stade Brestois 29": "Brest",
    "AS Saint-Étienne": "St Etienne",
    "RC Strasbourg Alsace": "Strasbourg",
    "Montpellier HSC": "Montpellier",
    "Le Havre AC": "Le Havre",
    "AJ Auxerre": "Auxerre",
    "Angers SCO": "Angers",
    "Stade de Reims": "Reims",
}

_NOISE = {
    "fc", "cf", "afc", "sc", "ac", "as", "ss", "us", "ud", "cd", "rc", "sv", "vfl",
    "vfb", "tsg", "fsv", "ssc", "acf", "bc", "ca", "rcd", "calcio", "club",
}


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`&-]", " ", s)
    s = re.sub(r"\b\d+\b", " ", s)
    tokens = [t for t in s.split() if t not in _NOISE]
    return " ".join(tokens)


class TeamResolver:
    """Risolve i nomi dell'API verso i nomi canonici dei CSV, per lega."""

    def __init__(self, known_by_league: dict[str, set[str]]):
        self._known = known_by_league
        self._norm = {
            lg: {normalize(t): t for t in teams} for lg, teams in known_by_league.items()
        }
        self.unresolved: set[tuple[str, str]] = set()

    def resolve(self, league: str, api_name: str) -> str | None:
        known = self._known.get(league, set())
        norm_map = self._norm.get(league, {})
        alias = ALIASES.get(api_name)
        if alias is not None and alias in known:
            return alias
        if api_name in known:
            return api_name
        n = normalize(api_name)
        if n in norm_map:
            return norm_map[n]
        if norm_map:
            hit = process.extractOne(n, list(norm_map), scorer=fuzz.WRatio, score_cutoff=FUZZY_CUTOFF)
            if hit is not None:
                return norm_map[hit[0]]
        self.unresolved.add((league, api_name))
        log.warning("Squadra non riconosciuta in %s: %r (trattata come nuova)", league, api_name)
        return None
