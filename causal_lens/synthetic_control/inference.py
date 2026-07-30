from __future__ import annotations

import numpy as np
import pandas as pd

from causal_lens.synthetic_control.estimator import SyntheticControl


def _rmspe(gap: pd.Series, treatment_date, pre: bool) -> float:
    mask = gap.index < treatment_date if pre else gap.index >= treatment_date
    vals = gap[mask].values
    return float(np.sqrt(np.mean(vals ** 2))) if len(vals) else float("nan")


def placebo_inference(result, donor_panel: pd.DataFrame, predictors: list[str]) -> dict:
    """Apply synthetic control to each donor as if it were treated. p-value =
    rank of the real unit's post/pre RMSPE ratio among all units' ratios.
    """
    donors = [d for d in result.weights.index]
    ratios = {}

    real_ratio = _rmspe(result.gap, result.treatment_date, pre=False) / max(
        _rmspe(result.gap, result.treatment_date, pre=True), 1e-9
    )
    ratios[result.treated.name if hasattr(result.treated, "name") else "treated"] = real_ratio

    for placebo_unit in donors:
        remaining = [d for d in donors if d != placebo_unit]
        if len(remaining) < 2:
            continue
        sc = SyntheticControl(
            donor_panel, treated_unit=placebo_unit, donor_units=remaining,
            treatment_date=result.treatment_date, predictors=predictors,
        )
        placebo_result = sc.fit()
        pre_rmspe = _rmspe(placebo_result.gap, result.treatment_date, pre=True)
        if pre_rmspe < 1e-6:
            continue  # bad pre-fit, exclude per Abadie convention
        ratios[placebo_unit] = _rmspe(placebo_result.gap, result.treatment_date, pre=False) / pre_rmspe

    sorted_ratios = sorted(ratios.values(), reverse=True)
    rank = sorted_ratios.index(real_ratio) + 1
    p_value = rank / len(sorted_ratios)

    return {"real_rmspe_ratio": round(real_ratio, 4), "n_placebos": len(ratios) - 1, "p_value": round(p_value, 4)}
