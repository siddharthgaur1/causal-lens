from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf


def parallel_trends_test(data: pd.DataFrame, unit: str, time: str, treatment: str,
                          outcome: str, treatment_date) -> dict:
    """Regress pre-period outcomes on time*treatment interaction.
    p > 0.05 on the interaction term means the assumption holds (fails to reject).
    """
    treatment_date = pd.to_datetime(treatment_date)
    pre = data[pd.to_datetime(data[time]) < treatment_date].copy()
    pre["_t"] = pd.to_datetime(pre[time]).astype("int64")
    model = smf.ols(f"{outcome} ~ _t * {treatment}", data=pre).fit()
    interaction = [p for p in model.params.index if ":" in p][0]
    return {
        "interaction_coef": round(float(model.params[interaction]), 6),
        "p_value": round(float(model.pvalues[interaction]), 5),
        "parallel_trends_holds": bool(model.pvalues[interaction] > 0.05),
    }


def placebo_test(data: pd.DataFrame, unit: str, time: str, treatment: str,
                  outcome: str, treatment_date, periods_before: int = 2) -> dict:
    """Assign a fake treatment date `periods_before` periods earlier than the
    real one; a real effect should be null on this placebo (p > 0.05)."""
    from causal_lens.did.estimator import DifferenceInDifferences

    treatment_date = pd.to_datetime(treatment_date)
    fake_date = treatment_date - pd.DateOffset(months=periods_before)
    pre_only = data[pd.to_datetime(data[time]) < treatment_date]

    did = DifferenceInDifferences(pre_only, unit, time, treatment, outcome, fake_date)
    result = did.estimate()
    return {
        "placebo_att": round(float(result.att), 6),
        "p_value": round(float(result.p_value), 5),
        "passes_placebo": bool(result.p_value > 0.05),
    }
