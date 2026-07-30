"""Minimal assert-based self-checks (no framework, no fixtures). Each test
recovers a known synthetic ground-truth effect within a loose tolerance."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from causal_lens import ABTest, DifferenceInDifferences, SyntheticControl, UpliftModel
from causal_lens.ab.diagnostics import srm_check
from causal_lens.ab.power_calculator import required_sample_size
from causal_lens.did.parallel_trends import parallel_trends_test, placebo_test
from causal_lens.uplift.evaluation import auuc, qini_curve
from data.synthetic import generate_ab_data, generate_did_panel, generate_uplift_data


def test_power_calculator():
    result = required_sample_size(baseline_rate=0.1, mde=0.02)
    assert result["required_n_per_arm"] > 0


def test_ab_frequentist_and_bayesian_and_cuped():
    control, treatment = generate_ab_data(n=20_000, true_lift=0.03, seed=1)
    ab = ABTest(control, treatment)

    freq = ab.analyze("conversion", method="frequentist")
    assert freq.stats["lift"] > 0, "should detect positive lift"
    assert freq.stats["p_value"] < 0.05, "20k/arm at 3pp lift should be significant"

    bayes = ab.analyze("conversion", method="bayesian")
    assert bayes.stats["prob_treatment_beats_control"] > 0.9

    cuped = ab.analyze("conversion", method="frequentist", cuped_covariate="pre_conversion")
    assert cuped.stats["p_value"] <= freq.stats["p_value"], "CUPED should not increase p-value here"


def test_srm_check():
    balanced = srm_check(5000, 5000)
    assert not balanced["srm_detected"]
    mismatched = srm_check(6000, 4000)
    assert mismatched["srm_detected"]


def test_did_recovers_known_att():
    panel = generate_did_panel(true_att=2.0, seed=2)
    did = DifferenceInDifferences(panel, unit="unit", time="date", treatment="treated",
                                   outcome="outcome", treatment_date="2023-07-01")
    result = did.estimate(cluster_by="unit")
    assert 1.5 < result.att < 2.5, f"expected ATT near 2.0, got {result.att}"

    pt = parallel_trends_test(panel, "unit", "date", "treated", "outcome", "2023-07-01")
    assert pt["parallel_trends_holds"], "synthetic panel has no pre-trend, should pass"

    placebo = placebo_test(panel, "unit", "date", "treated", "outcome", "2023-07-01", periods_before=2)
    assert placebo["passes_placebo"], "no fake effect should exist before real treatment"


def test_synthetic_control_recovers_gap():
    rng = np.random.default_rng(3)
    dates = pd.date_range("2020-01-01", periods=40, freq="MS")
    donors = {f"donor{i}": 100 + np.cumsum(rng.normal(0, 1, 40)) for i in range(5)}
    treated_pre = 100 + np.mean([donors[d][:30] for d in donors], axis=0)
    treated_post = treated_pre[-1] + np.cumsum(rng.normal(0, 1, 10)) + 5  # +5 treatment effect
    treated = np.concatenate([treated_pre, treated_post])

    rows = []
    for i, date in enumerate(dates):
        rows.append({"unit": "treated_unit", "date": date, "return": treated[i]})
        for d in donors:
            rows.append({"unit": d, "date": date, "return": donors[d][i]})
    panel = pd.DataFrame(rows)

    sc = SyntheticControl(panel, treated_unit="treated_unit", donor_units=list(donors),
                           treatment_date=dates[30], predictors=["return"])
    result = sc.fit(outcome="return")
    post_gap = result.gap[result.gap.index >= dates[30]].mean()
    assert post_gap > 1, f"expected a positive post-treatment gap, got {post_gap}"


def test_uplift_model_ranks_persuadables_higher():
    df = generate_uplift_data(n=5000, seed=4)
    X = df[["age", "income"]].values
    model = UpliftModel(model_type="t_learner").fit(X, df["treatment"].values, df["converted"].values)
    scores = model.predict_uplift(X)

    high_income_uplift = scores[df["income"] > 60000].mean()
    low_income_uplift = scores[df["income"] < 40000].mean()
    assert high_income_uplift > low_income_uplift, "high-income users have the true uplift, should score higher"

    curve = qini_curve(scores, df["treatment"].values, df["converted"].values)
    assert auuc(curve) != 0


if __name__ == "__main__":
    test_power_calculator()
    test_ab_frequentist_and_bayesian_and_cuped()
    test_srm_check()
    test_did_recovers_known_att()
    test_synthetic_control_recovers_gap()
    test_uplift_model_ranks_persuadables_higher()
    print("all tests passed")
