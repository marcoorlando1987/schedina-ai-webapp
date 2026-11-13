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


def test_parse_csv_reads_shots(tmp_path):
    csv = ("Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,HS,AS,HST,AST\n"
           "23/08/25,Inter,Torino,5,0,H,20,6,10,1\n"
           "24/08/25,Roma,Bologna,0,1,A,,,,\n")
    p = tmp_path / "x.csv"
    p.write_text(csv)
    df = data.parse_csv(p, "Serie A", "2526")
    assert df.loc[0, ["sh_h", "sh_a", "sot_h", "sot_a"]].tolist() == [20, 6, 10, 1]
    assert df.loc[1, ["sh_h", "sot_h"]].isna().all()
    noshots = tmp_path / "y.csv"
    noshots.write_text("Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n23/08/25,Inter,Torino,5,0,H\n")
    assert data.parse_csv(noshots, "Serie A", "2526")["sot_h"].isna().all()


def test_parse_csv_reads_bettable_odds_coherent_per_row(tmp_path):
    csv = ("Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR,B365H,B365D,B365A,AvgH,AvgD,AvgA\n"
           "23/08/25,Inter,Torino,5,0,H,1.40,5.00,8.00,1.45,4.80,7.50\n"
           "24/08/25,Roma,Bologna,0,1,A,,,,2.00,3.40,3.80\n"       # B365 assente: terna Avg
           "25/08/25,Milan,Genoa,1,1,D,2.0,,3.8,2.1,3.3,3.7\n")    # B365 incompleta: terna Avg intera
    p = tmp_path / "x.csv"
    p.write_text(csv)
    df = data.parse_csv(p, "Serie A", "2526")
    assert df.loc[0, ["odds_h", "odds_d", "odds_a"]].tolist() == [1.40, 5.00, 8.00]
    assert df.loc[1, ["odds_h", "odds_d", "odds_a"]].tolist() == [2.00, 3.40, 3.80]
    assert df.loc[2, ["odds_h", "odds_d", "odds_a"]].tolist() == [2.1, 3.3, 3.7]
