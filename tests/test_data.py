import pandas as pd
import pytest

from schedina_ai import data

CSV = (
    "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,AvgCH,AvgCD,AvgCA\n"
    "I1,23/08/25,Inter,Torino,5,0,H,1.40,5.00,8.00,1.45,4.80,7.50\n"
    "I1,24/08/2025,Roma,Bologna,0,1,A,2.00,3.40,3.80,,,\n"
    "I1,,Lazio,Napoli,1,1,D,3.0,3.3,2.4,,,\n"          # data mancante: scartata
    "I1,25/08/25,Milan,Genoa,,,,,,,,,\n"                # non giocata: scartata
)


def test_parse_csv(tmp_path):
    p = tmp_path / "2526_I1.csv"
    p.write_bytes(CSV.encode("latin-1"))
    df = data.parse_csv(p, "Serie A", "2526")
    assert list(df["home"]) == ["Inter", "Roma"]
    assert df["date"].iloc[0] == pd.Timestamp("2025-08-23")
    assert df["date"].iloc[1] == pd.Timestamp("2025-08-24")
    # quote di chiusura medie preferite; ripiego su B365; somma = 1 (margine tolto)
    assert abs(df[["mkt_h", "mkt_d", "mkt_a"]].iloc[0].sum() - 1) < 1e-9
    assert df["mkt_h"].iloc[0] > df["mkt_a"].iloc[0]
    assert df[["mkt_h", "mkt_d", "mkt_a"]].iloc[1].notna().all()


def test_parse_csv_missing_columns(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("a,b\n1,2\n" * 50)
    with pytest.raises(data.DataError):
        data.parse_csv(p, "Serie A", "2526")


class FakeResp:
    def __init__(self, content=b"", code=200):
        self.content, self.status_code = content, code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def get(self, url, timeout=None):
        self.calls.append(url)
        return FakeResp(CSV.encode() * 3, 200 if self.ok else 503)


def test_closed_seasons_are_cached_forever(tmp_path):
    s = FakeSession()
    p1 = data.fetch_csv("2425", "I1", tmp_path, s, current_season="2526")
    p2 = data.fetch_csv("2425", "I1", tmp_path, s, current_season="2526")
    assert p1 == p2 and len(s.calls) == 1


def test_download_failure_falls_back_to_cache_or_skips(tmp_path):
    bad = FakeSession(ok=False)
    assert data.fetch_csv("2425", "I1", tmp_path, bad, "2526") is None
    good = FakeSession()
    data.fetch_csv("2526", "I1", tmp_path, good, "2526")
    # cache scaduta ma rete giù: usa comunque la cache
    import os, time
    old = time.time() - 48 * 3600
    os.utime(tmp_path / "2526_I1.csv", (old, old))
    assert data.fetch_csv("2526", "I1", tmp_path, bad, "2526") == tmp_path / "2526_I1.csv"


def test_load_history_raises_when_nothing_available(tmp_path):
    with pytest.raises(data.DataError):
        data.load_history(cache_dir=tmp_path, session=FakeSession(ok=False))


def test_load_history_fails_fast_when_source_is_down(tmp_path):
    s = FakeSession(ok=False)
    with pytest.raises(data.DataError, match="non raggiungibile"):
        data.load_history(cache_dir=tmp_path, session=s)
    assert len(s.calls) == data.MAX_CONSECUTIVE_FAILURES  # si ferma subito, non 30 tentativi
