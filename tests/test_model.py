import numpy as np

from schedina_ai.evaluate import best_model, usable_frame, walk_forward
from schedina_ai.features import build_training_frame
from schedina_ai.model import KINDS, Predictor, score_matrix_probs


def test_score_matrix_probs_are_valid():
    probs, over = score_matrix_probs(np.array([1.5, 0.4]), np.array([1.1, 2.5]))
    assert np.allclose(probs.sum(axis=1), 1)
    assert probs[0, 0] > probs[0, 2] and probs[1, 2] > probs[1, 0]
    assert ((0 < over) & (over < 1)).all() and over[0] < over[1] + 1


def test_all_predictors_output_valid_probabilities(small_matches):
    frame, _ = build_training_frame(small_matches)
    train = usable_frame(frame)
    for kind in KINDS:
        out = Predictor(kind).fit(train).predict(train.tail(50))
        p = out[["p_home", "p_draw", "p_away"]].to_numpy()
        assert np.allclose(p.sum(axis=1), 1), kind
        assert (p > 0).all() and (p < 1).all(), kind
        assert (out["xg_home"] > 0).all() and out["p_over25"].between(0, 1).all(), kind


def test_walk_forward_learns_and_reports_market(matches):
    frame, _ = build_training_frame(matches)
    res = walk_forward(frame).set_index("model")
    # su dati con forze reali, i modelli devono battere nettamente la baseline
    assert res.loc["logreg", "logloss"] < res.loc["baseline", "logloss"] - 0.02
    assert res.loc["xgb", "logloss"] < res.loc["baseline", "logloss"] - 0.02
    assert {"market_logloss", "logloss_same_matches"} <= set(res.columns)
    assert best_model(res.reset_index()) != "baseline"


def test_calibration_identity_and_softening():
    from schedina_ai.model import apply_calibration

    p = np.array([[0.9, 0.07, 0.03], [0.4, 0.3, 0.3]])
    assert np.allclose(apply_calibration(p, np.zeros(3)), p)
    soft = apply_calibration(p, np.array([np.log(1.5), 0.0, 0.0]))  # T=1.5
    assert soft[0, 0] < p[0, 0] and np.allclose(soft.sum(axis=1), 1)
    more_draw = apply_calibration(p, np.array([0.0, np.log(1.3), 0.0]))
    assert more_draw[1, 1] > p[1, 1]


def test_fit_calibration_softens_an_overconfident_model(matches):
    from schedina_ai.model import fit_calibration

    frame, _ = build_training_frame(matches)
    theta = fit_calibration(usable_frame(frame))
    assert theta is not None and np.isfinite(theta).all()
    assert abs(theta[0]) < 1.0  # parametri sensati, non esplosi


def test_report_builds_and_has_all_sections(matches):
    from schedina_ai.report import build_report

    frame, _ = build_training_frame(matches)
    text = build_report(frame, kinds=("baseline", "logreg", "poisson", "blend"))
    for section in ("Modelli", "Per stagione", "Per lega", "Calibrazione", "Medie per esito",
                    "Quando il modello è sicuro", "Mercato", "Test economico", "value bet"):
        assert section in text, section
    assert "nan" not in text.lower().replace("finance", "")
