from __future__ import annotations

import numpy as np
import pandas as pd


def qini_curve(uplift_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> pd.DataFrame:
    """Cumulative incremental gain vs. % of population targeted, sorted by
    predicted uplift descending."""
    df = pd.DataFrame({"uplift": uplift_scores, "treatment": treatment, "outcome": outcome})
    df = df.sort_values("uplift", ascending=False).reset_index(drop=True)
    df["cum_treated"] = (df["treatment"] * df["outcome"]).cumsum()
    df["cum_control"] = ((1 - df["treatment"]) * df["outcome"]).cumsum()
    n_treated_total = max(df["treatment"].sum(), 1)
    n_control_total = max((1 - df["treatment"]).sum(), 1)
    df["qini"] = df["cum_treated"] - df["cum_control"] * (n_treated_total / n_control_total)
    df["pct_targeted"] = (df.index + 1) / len(df)
    return df[["pct_targeted", "qini"]]


def auuc(qini_df: pd.DataFrame) -> float:
    return float(np.trapezoid(qini_df["qini"], qini_df["pct_targeted"]))


def qini_coefficient(qini_df: pd.DataFrame) -> float:
    """AUUC normalized against the random-targeting diagonal."""
    model_auuc = auuc(qini_df)
    n = len(qini_df)
    random_line = np.linspace(0, qini_df["qini"].iloc[-1], n)
    random_auuc = float(np.trapezoid(random_line, qini_df["pct_targeted"]))
    return model_auuc - random_auuc


def segment_population(uplift_scores: np.ndarray, treatment_proba: np.ndarray, outcome_proba_base: np.ndarray,
                        uplift_threshold: float = 0.0) -> pd.Series:
    """Classify each unit as Persuadable / Sure Thing / Lost Cause / Sleeping Dog
    from predicted P(outcome|treat) vs P(outcome|control)."""
    labels = []
    for u, p_t, p_c in zip(uplift_scores, treatment_proba, outcome_proba_base):
        if u > uplift_threshold and p_c < 0.5:
            labels.append("Persuadable")
        elif p_t > 0.5 and p_c > 0.5:
            labels.append("Sure Thing")
        elif p_t < 0.5 and p_c < 0.5:
            labels.append("Lost Cause")
        else:
            labels.append("Sleeping Dog")
    return pd.Series(labels)
