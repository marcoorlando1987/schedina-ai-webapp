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


def test_materialize_db_from_path_and_url(tmp_path, monkeypatch):
    from schedina_ai import config
    db = tmp_path / "src.db"
    conn = store.connect(db)
    store.set_meta(conn, "model", "logreg")
    conn.commit()
    conn.close()

    out = store.materialize_db(db, tmp_path / "copy" / "x.db")
    assert store.get_meta(store.connect(out), "model") == "logreg"

    class Resp:
        def __init__(self, content, code=200):
            self.content, self.status_code = content, code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(f"HTTP {self.status_code}")

    class Sess:
        def __init__(self, resp):
            self.resp = resp

        def get(self, url, timeout=None):
            return self.resp

    ok = store.materialize_db("https://example.test/schedina.db", tmp_path / "u.db", Sess(Resp(db.read_bytes())))
    assert store.get_meta(store.connect(ok), "model") == "logreg"
    with pytest.raises(store.DbUnavailable):  # 404 (branch data non ancora creato)
        store.materialize_db("https://example.test/x.db", tmp_path / "n.db", Sess(Resp(b"", 404)))
    with pytest.raises(store.DbUnavailable):  # pagina HTML al posto del DB
        store.materialize_db("https://example.test/x.db", tmp_path / "h.db", Sess(Resp(b"<html>")))
    with pytest.raises(store.DbUnavailable):
        store.materialize_db(tmp_path / "manca.db", tmp_path / "m.db")

    # precedenza: env > file locale > URL di default su GitHub
    monkeypatch.setenv("SCHEDINA_DATA_DIR", str(tmp_path / "vuoto"))
    monkeypatch.delenv("SCHEDINA_DB_URL", raising=False)
    assert config.db_source() == config.DEFAULT_DB_URL
    (tmp_path / "vuoto").mkdir()
    (tmp_path / "vuoto" / "schedina.db").write_bytes(b"x")
    assert config.db_source().endswith("schedina.db") and "http" not in config.db_source()
    monkeypatch.setenv("SCHEDINA_DB_URL", "https://altro.test/db")
    assert config.db_source() == "https://altro.test/db"
