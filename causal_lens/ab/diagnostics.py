from __future__ import annotations

from scipy import stats


def srm_check(n_control: int, n_treatment: int, expected_ratio: float = 0.5) -> dict:
    """Sample Ratio Mismatch check via chi-square goodness of fit."""
    total = n_control + n_treatment
    expected_control = total * expected_ratio
    expected_treatment = total * (1 - expected_ratio)
    chi2, p_value = stats.chisquare(
        [n_control, n_treatment], f_exp=[expected_control, expected_treatment]
    )
    return {
        "chi2": round(float(chi2), 4),
        "p_value": round(float(p_value), 5),
        "srm_detected": bool(p_value < 0.001),  # conventional SRM threshold
    }


def balance_check(control_covariate, treatment_covariate) -> dict:
    """Are control/treatment balanced on a pre-experiment covariate?"""
    stat, p_value = stats.ttest_ind(control_covariate, treatment_covariate, equal_var=False)
    return {
        "statistic": round(float(stat), 4),
        "p_value": round(float(p_value), 5),
        "balanced": bool(p_value > 0.05),
    }
