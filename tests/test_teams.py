from schedina_ai.teams import TeamResolver, normalize

KNOWN = {
    "Serie A": {"Inter", "Milan", "Verona", "Juventus", "Napoli", "Roma", "Lazio"},
    "Premier League": {"Man City", "Man United", "Nott'm Forest", "Brighton", "Newcastle", "Wolves"},
    "Bundesliga": {"Bayern Munich", "M'gladbach", "Ein Frankfurt", "FC Koln", "Mainz"},
}


def test_normalize_strips_noise():
    assert normalize("1. FSV Mainz 05") == "mainz"
    assert normalize("Juventus FC") == "juventus"
    assert normalize("FC Köln") == "koln"


def test_alias_and_exact_and_fuzzy():
    r = TeamResolver(KNOWN)
    assert r.resolve("Serie A", "FC Internazionale Milano") == "Inter"
    assert r.resolve("Serie A", "AC Milan") == "Milan"
    assert r.resolve("Serie A", "Hellas Verona FC") == "Verona"
    assert r.resolve("Serie A", "Juventus FC") == "Juventus"  # normalizzato
    assert r.resolve("Premier League", "Manchester United FC") == "Man United"
    assert r.resolve("Premier League", "Newcastle United FC") == "Newcastle"
    assert r.resolve("Bundesliga", "Borussia Mönchengladbach") == "M'gladbach"
    assert r.resolve("Bundesliga", "1. FSV Mainz 05") == "Mainz"
    assert not r.unresolved


def test_resolution_is_scoped_to_league_and_unknown_is_none():
    r = TeamResolver(KNOWN)
    # "Inter" esiste solo in Serie A: in Premier League non deve agganciarsi
    assert r.resolve("Premier League", "FC Internazionale Milano") is None
    assert r.resolve("Serie A", "Pisa Sporting Club XYZ") is None
    assert ("Serie A", "Pisa Sporting Club XYZ") in r.unresolved


def test_manchester_city_and_united_not_confused():
    r = TeamResolver(KNOWN)
    assert r.resolve("Premier League", "Manchester City FC") == "Man City"
    assert r.resolve("Premier League", "Manchester United FC") == "Man United"
