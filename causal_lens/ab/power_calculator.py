from __future__ import annotations

import math

from scipy import stats


def required_sample_size(baseline_rate: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> dict:
    """Two-proportion z-test sample size per arm.

    mde: absolute minimum detectable effect (e.g. 0.02 for +2pp).
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    p_bar = (p1 + p2) / 2
    n = ((z_alpha * math.sqrt(2 * p_bar * (1 - p_bar)) + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (mde ** 2)
    return {"required_n_per_arm": math.ceil(n)}


def days_to_significance(required_n_per_arm: int, daily_traffic_per_arm: float) -> float:
    return math.ceil(required_n_per_arm / daily_traffic_per_arm)
