from datetime import date, datetime, timezone

import pytest

from schedina_ai import pipeline, store
from schedina_ai.fixtures import FixturesError, fetch_fixtures

NOW = datetime(2026, 1, 10, 9, 0, tzinfo=timezone.utc)


class FakeResp:
    def __init__(self, payload, code=200, headers=None):
        self._p, self.status_code, self.headers, self.text = payload, code, headers or {}, str(payload)

    def json(self):
        return self._p


class FakeApi:
    """Una partita futura per ogni lega + una già finita, mappata con nomi 'ufficiali'."""

    def __init__(self, teams_by_league, started=False):
        self.teams, self.started = teams_by_league, started

    def get(self, url, headers=None, params=None, timeout=None):
        code = url.rstrip("/").split("/")[-2]
        name = {"SA": "Serie A", "PL": "Premier League", "PD": "La Liga",
                "BL1": "Bundesliga", "FL1": "Ligue 1"}[code]
        t = sorted(self.teams[name])
        base = {"SA": 100, "PL": 200, "PD": 300, "BL1": 400, "FL1": 500}[code]
        matches = [{
            "id": base + 1, "utcDate": "2026-01-11T15:00:00Z",
            "status": "FINISHED" if self.started else "TIMED",
            "homeTeam": {"name": t[0]}, "awayTeam": {"name": t[1]},
            "score": {"fullTime": {"home": 2, "away": 1} if self.started else {"home": None, "away": None}},
        }, {  # squadra mai vista (neopromossa)
            "id": base + 2, "utcDate": "2026-01-12T15:00:00Z", "status": "SCHEDULED",
            "homeTeam": {"name": t[2]}, "awayTeam": {"name": "Squadra Nuova Calcio"},
            "score": {"fullTime": {"home": None, "away": None}},
        }]
        return FakeResp({"matches": matches})


def _teams(history):
    return {lg: set(history.loc[history.league == lg, "home"]) for lg in history.league.unique()}


def test_end_to_end_and_freeze_after_kickoff(matches, tmp_path):
    db = str(tmp_path / "t.db")
    api = FakeApi(_teams(matches))
    s = pipeline.run("tok", today=date(2026, 1, 10), now=NOW, history=matches,
                     fixtures_session=api, force_model="logreg", db=db)
    assert s["predictions"] == 10 and s["unresolved_teams"]

    conn = store.connect(db)
    df = store.load_predictions(conn)
    assert len(df) == 10
    assert ((df[["p_home", "p_draw", "p_away"]].sum(axis=1) - 1).abs() < 1e-9).all()
    assert df["new_team"].sum() == 5  # le 5 partite con la squadra "nuova"
    assert store.get_meta(conn, "model") == "logreg"
    before = df.set_index("match_id").loc[101, ["p_home", "pick"]].tolist()

    # dopo il calcio d'inizio: stato/risultato aggiornati, predizione congelata
    api2 = FakeApi(_teams(matches), started=True)
    later = datetime(2026, 1, 11, 20, 0, tzinfo=timezone.utc)
    pipeline.run("tok", today=date(2026, 1, 11), now=later, history=matches,
                 fixtures_session=api2, force_model="poisson", db=db)
    df2 = store.load_predictions(conn).set_index("match_id")
    assert df2.loc[101, "status"] == "FINISHED" and df2.loc[101, "score_home"] == 2
    assert df2.loc[101, ["p_home", "pick"]].tolist() == before
    assert df2.loc[101, "model"] == "logreg"
    conn.close()


def test_missing_token_or_denied_access_fail_loudly():
    with pytest.raises(FixturesError):
        fetch_fixtures(None, date(2026, 1, 1), date(2026, 1, 8))

    class Denied:
        def get(self, *a, **k):
            return FakeResp({}, 403)

    with pytest.raises(FixturesError):
        fetch_fixtures("bad", date(2026, 1, 1), date(2026, 1, 8), session=Denied())
