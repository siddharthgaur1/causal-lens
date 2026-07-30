from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class SyntheticControlResult:
    def __init__(self, weights: pd.Series, treated: pd.Series, synthetic: pd.Series, treatment_date):
        self.weights = weights
        self.treated = treated
        self.synthetic = synthetic
        self.treatment_date = treatment_date
        self.gap = treated - synthetic

    def summary(self) -> None:
        post = self.gap[self.gap.index >= self.treatment_date]
        print(f"donor weights (nonzero): {self.weights[self.weights > 1e-4].round(3).to_dict()}")
        print(f"mean post-treatment gap: {post.mean():.4f}")

    def gap_plot(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(self.treated.index, self.treated.values, label="treated")
        ax.plot(self.synthetic.index, self.synthetic.values, label="synthetic", linestyle="--")
        ax.axvline(self.treatment_date, color="red", linestyle=":", label="treatment")
        ax.legend()
        return ax

    def placebo_test(self, donor_panel: pd.DataFrame, predictors: list[str]) -> dict:
        from causal_lens.synthetic_control.inference import placebo_inference

        return placebo_inference(self, donor_panel, predictors)


class SyntheticControl:
    def __init__(self, data: pd.DataFrame, treated_unit: str, donor_units: list[str],
                 treatment_date, predictors: list[str], unit_col: str = "unit",
                 time_col: str = "date"):
        self.data = data
        self.treated_unit = treated_unit
        self.donor_units = donor_units
        self.treatment_date = pd.to_datetime(treatment_date)
        self.predictors = predictors
        self.unit_col = unit_col
        self.time_col = time_col

    def _pivot(self, column: str) -> pd.DataFrame:
        return self.data.pivot(index=self.time_col, columns=self.unit_col, values=column)

    def fit(self, outcome: str = "return") -> SyntheticControlResult:
        panel = self._pivot(outcome)
        pre = panel[pd.to_datetime(panel.index) < self.treatment_date]

        X_treated = pre[self.treated_unit].values
        X_donors = pre[self.donor_units].values  # (T_pre, n_donors)

        n_donors = X_donors.shape[1]

        def loss(w):
            return np.sum((X_treated - X_donors @ w) ** 2)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0, 1)] * n_donors
        w0 = np.ones(n_donors) / n_donors
        res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=constraints)
        weights = pd.Series(res.x, index=self.donor_units)

        full_donors = panel[self.donor_units].values
        synthetic = pd.Series(full_donors @ weights.values, index=panel.index)
        treated = panel[self.treated_unit]

        return SyntheticControlResult(weights, treated, synthetic, self.treatment_date)
