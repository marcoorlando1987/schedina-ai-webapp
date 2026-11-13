import pandas as pd
import pytest

from schedina_ai.features import FEATURES, RatingEngine, build_training_frame


def test_no_leakage_own_result_does_not_affect_own_features(small_matches):
    base, _ = build_training_frame(small_matches)
    tweaked = small_matches.copy()
    i = 300
    tweaked.loc[i, ["hg", "ag", "result"]] = [9, 0, "H"]
    alt, _ = build_training_frame(tweaked)
    # feature della partita modificata: identiche (usano solo il passato)
    pd.testing.assert_series_equal(base.loc[i, FEATURES], alt.loc[i, FEATURES])
    # le partite precedenti: identiche
    pd.testing.assert_frame_equal(base.loc[:i, FEATURES], alt.loc[:i, FEATURES])
    # qualche partita successiva delle stesse squadre deve cambiare
    h, a = small_matches.loc[i, ["home", "away"]]
    later = base[(base.index > i) & (base["home"].isin([h, a]) | base["away"].isin([h, a]))].index[:5]
    assert not base.loc[later, FEATURES].equals(alt.loc[later, FEATURES])


def test_elo_is_zero_sum_within_a_season():
    e = RatingEngine()
    d = pd.Timestamp("2025-09-01")
    e.update("L", "A", "B", d, 2, 1)
    e.update("L", "B", "C", d + pd.Timedelta(days=7), 0, 0)
    e.update("L", "C", "A", d + pd.Timedelta(days=14), 1, 3)
    total = sum(s.elo for s in e._teams["L"].values())
    assert abs(total - 3 * 1500.0) < 1e-6


def test_snapshot_is_pure_and_idempotent():
    e = RatingEngine()
    d = pd.Timestamp("2025-09-01")
    e.update("L", "A", "B", d, 2, 1)
    before = e.snapshot("L", "A", "B", d + pd.Timedelta(days=7))["elo_diff"]
    later = d + pd.Timedelta(days=200)  # nuova stagione: regressione applicata una volta sola
    a = e.snapshot("L", "A", "B", later)
    b = e.snapshot("L", "A", "B", later)
    assert a == b
    assert a["elo_diff"] == pytest.approx(before * (1 - 0.25))


def test_unknown_team_gets_promoted_default_after_first_season():
    e = RatingEngine()
    d = pd.Timestamp("2025-09-01")
    e.update("L", "A", "B", d, 1, 1)
    f = e.snapshot("L", "A", None, d + pd.Timedelta(days=200))
    assert f["elo_diff"] > 0  # neopromossa (1450) sotto una squadra media
    assert f["a_n"] == 0
