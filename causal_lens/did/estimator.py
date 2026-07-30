from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf


class DiDResult:
    def __init__(self, att: float, se: float, p_value: float, model, panel: pd.DataFrame,
                 treatment: str, outcome: str):
        self.att = att
        self.se = se
        self.p_value = p_value
        self.model = model
        self.panel = panel
        self.treatment = treatment
        self.outcome = outcome

    def summary(self) -> None:
        print(f"ATT = {self.att:.5f}  (SE={self.se:.5f}, p={self.p_value:.5f})")

    def event_study_estimates(self) -> pd.DataFrame:
        """Treatment-minus-control mean outcome gap per relative-time period.
        Near-zero gaps before period 0 is the visual parallel-trends check;
        a jump at/after period 0 is the effect.
        """
        rows = []
        for period, g in self.panel.groupby("_rel_time"):
            treated_mean = g.loc[g[self.treatment] == 1, self.outcome].mean()
            control_mean = g.loc[g[self.treatment] == 0, self.outcome].mean()
            rows.append({"period": period, "gap": treated_mean - control_mean})
        return pd.DataFrame(rows).sort_values("period").reset_index(drop=True)

    def event_study_plot(self, ax=None):
        est = self.event_study_estimates()
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()
        ax.plot(est["period"], est["gap"], "o-")
        ax.axhline(0, color="gray", linestyle="--")
        ax.axvline(0, color="red", linestyle="--", label="treatment")
        ax.set_xlabel("periods relative to treatment")
        ax.set_ylabel("treatment - control mean outcome gap")
        return ax


class DifferenceInDifferences:
    def __init__(self, data: pd.DataFrame, unit: str, time: str, treatment: str,
                 outcome: str, treatment_date):
        self.data = data.copy()
        self.unit = unit
        self.time = time
        self.treatment = treatment
        self.outcome = outcome
        self.treatment_date = pd.to_datetime(treatment_date)

    def estimate(self, covariates: list[str] | None = None, cluster_by: str | None = None) -> DiDResult:
        df = self.data.copy()
        df["_post"] = (pd.to_datetime(df[self.time]) >= self.treatment_date).astype(int)
        df["_did"] = df["_post"] * df[self.treatment]
        df["_rel_time"] = (
            pd.to_datetime(df[self.time]).dt.to_period("M") - self.treatment_date.to_period("M")
        ).apply(lambda x: x.n)

        formula = f"{self.outcome} ~ _post + {self.treatment} + _did"
        if covariates:
            formula += " + " + " + ".join(covariates)

        groups = df[cluster_by] if cluster_by else None
        cov_type = "cluster" if cluster_by else "HC1"
        cov_kwds = {"groups": groups} if cluster_by else None

        model = smf.ols(formula, data=df).fit(cov_type=cov_type, cov_kwds=cov_kwds)

        return DiDResult(
            att=model.params["_did"], se=model.bse["_did"], p_value=model.pvalues["_did"],
            model=model, panel=df, treatment=self.treatment, outcome=self.outcome,
        )
