from __future__ import annotations

import numpy as np
import pandas as pd


def generate_ab_data(n: int = 10_000, true_lift: float = 0.02, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pre_conversion = rng.normal(0.1, 0.03, n)
    baseline = 0.10
    control_conv = rng.binomial(1, np.clip(baseline + 0.3 * (pre_conversion - 0.1), 0, 1))
    treatment_conv = rng.binomial(1, np.clip(baseline + true_lift + 0.3 * (pre_conversion - 0.1), 0, 1))
    control = pd.DataFrame({"conversion": control_conv, "pre_conversion": pre_conversion})
    treatment = pd.DataFrame({"conversion": treatment_conv, "pre_conversion": pre_conversion})
    return control, treatment


def generate_did_panel(n_units: int = 20, n_periods: int = 12, treatment_period: int = 6,
                        true_att: float = 2.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_periods, freq="MS")
    rows = []
    for unit in range(n_units):
        is_treated = unit < n_units // 2
        unit_fx = rng.normal(0, 1)
        for t, date in enumerate(dates):
            trend = 0.3 * t
            effect = true_att if (is_treated and t >= treatment_period) else 0.0
            y = 10 + unit_fx + trend + effect + rng.normal(0, 0.5)
            rows.append({"unit": f"u{unit}", "date": date, "treated": int(is_treated), "outcome": y})
    return pd.DataFrame(rows)


def generate_uplift_data(n: int = 5000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.normal(35, 10, n)
    income = rng.normal(50000, 15000, n)
    treatment = rng.binomial(1, 0.5, n)
    true_uplift = 0.15 * (income > 50000)  # only higher-income users respond to the offer
    base_rate = 0.05 + 0.000001 * income
    p_convert = np.clip(base_rate + treatment * true_uplift, 0, 1)
    converted = rng.binomial(1, p_convert)
    return pd.DataFrame({"age": age, "income": income, "treatment": treatment, "converted": converted})
