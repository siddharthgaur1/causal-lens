from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


class ABResult:
    def __init__(self, method: str, stats_dict: dict):
        self.method = method
        self.stats = stats_dict

    def summary(self) -> None:
        print(f"--- A/B test result ({self.method}) ---")
        for k, v in self.stats.items():
            print(f"  {k}: {v}")


class ABTest:
    def __init__(self, control: pd.DataFrame, treatment: pd.DataFrame):
        self.control = control
        self.treatment = treatment

    def _cuped_adjust(self, metric: str, cuped_covariate: str) -> tuple[np.ndarray, np.ndarray]:
        y = pd.concat([self.control[metric], self.treatment[metric]])
        x = pd.concat([self.control[cuped_covariate], self.treatment[cuped_covariate]])
        theta = np.cov(y, x)[0, 1] / np.var(x)
        y_cuped = y - theta * (x - x.mean())
        n_c = len(self.control)
        return y_cuped.iloc[:n_c].values, y_cuped.iloc[n_c:].values

    def analyze(self, metric: str, method: str = "frequentist", cuped_covariate: str | None = None) -> ABResult:
        if cuped_covariate:
            c, t = self._cuped_adjust(metric, cuped_covariate)
        else:
            c, t = self.control[metric].values, self.treatment[metric].values

        if method == "frequentist":
            return self._frequentist(c, t)
        if method == "bayesian":
            return self._bayesian(c, t)
        raise ValueError(f"unknown method '{method}'")

    def _frequentist(self, c: np.ndarray, t: np.ndarray) -> ABResult:
        is_binary = set(np.unique(c)) <= {0, 1} and set(np.unique(t)) <= {0, 1}
        if is_binary:
            p1, p2 = c.mean(), t.mean()
            n1, n2 = len(c), len(t)
            p_pool = (c.sum() + t.sum()) / (n1 + n2)
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
            z = (p2 - p1) / se if se > 0 else 0.0
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
            test_name = "two-proportion z-test"
        else:
            stat, p_value = stats.ttest_ind(t, c, equal_var=False)
            z = stat
            test_name = "Welch t-test"

        return ABResult("frequentist", {
            "test": test_name,
            "control_mean": round(float(np.mean(c)), 5),
            "treatment_mean": round(float(np.mean(t)), 5),
            "lift": round(float(np.mean(t) - np.mean(c)), 5),
            "statistic": round(float(z), 4),
            "p_value": round(float(p_value), 5),
            "significant_at_0.05": bool(p_value < 0.05),
        })

    def _bayesian(self, c: np.ndarray, t: np.ndarray, n_samples: int = 100_000) -> ABResult:
        rng = np.random.default_rng(0)
        is_binary = set(np.unique(c)) <= {0, 1} and set(np.unique(t)) <= {0, 1}
        if is_binary:
            a_c, b_c = 1 + c.sum(), 1 + len(c) - c.sum()
            a_t, b_t = 1 + t.sum(), 1 + len(t) - t.sum()
            samples_c = rng.beta(a_c, b_c, n_samples)
            samples_t = rng.beta(a_t, b_t, n_samples)
        else:
            # Normal-Normal conjugate approximation for continuous metrics
            samples_c = rng.normal(c.mean(), c.std() / np.sqrt(len(c)), n_samples)
            samples_t = rng.normal(t.mean(), t.std() / np.sqrt(len(t)), n_samples)

        diff = samples_t - samples_c
        prob_b_beats_a = float((diff > 0).mean())
        expected_loss = float(np.mean(np.maximum(-diff, 0)))
        ci = np.percentile(diff, [2.5, 97.5])

        return ABResult("bayesian", {
            "prob_treatment_beats_control": round(prob_b_beats_a, 4),
            "expected_loss": round(expected_loss, 5),
            "credible_interval_95": (round(float(ci[0]), 5), round(float(ci[1]), 5)),
        })
