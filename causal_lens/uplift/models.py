from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier


class UpliftModel:
    """T-Learner, S-Learner, or X-Learner uplift estimator.

    model_type: "t_learner" | "s_learner" | "x_learner"
    base_estimator: any sklearn-compatible classifier with predict_proba;
      defaults to GradientBoostingClassifier (no xgboost/econml dependency required).
    """

    def __init__(self, model_type: str = "t_learner", base_estimator=None):
        self.model_type = model_type
        self.base_estimator = base_estimator or GradientBoostingClassifier()
        self.model_t = None
        self.model_c = None
        self.model_s = None
        self.model_x_t = None
        self.model_x_c = None

    def fit(self, X, treatment, outcome) -> "UpliftModel":
        X, treatment, outcome = np.asarray(X), np.asarray(treatment), np.asarray(outcome)
        if self.model_type == "t_learner":
            self.model_t = clone(self.base_estimator).fit(X[treatment == 1], outcome[treatment == 1])
            self.model_c = clone(self.base_estimator).fit(X[treatment == 0], outcome[treatment == 0])
        elif self.model_type == "s_learner":
            X_aug = np.column_stack([X, treatment])
            self.model_s = clone(self.base_estimator).fit(X_aug, outcome)
        elif self.model_type == "x_learner":
            self.model_t = clone(self.base_estimator).fit(X[treatment == 1], outcome[treatment == 1])
            self.model_c = clone(self.base_estimator).fit(X[treatment == 0], outcome[treatment == 0])
            # imputed treatment effects, cross-fit
            tau_treated = outcome[treatment == 1] - self._proba(self.model_c, X[treatment == 1])
            tau_control = self._proba(self.model_t, X[treatment == 0]) - outcome[treatment == 0]
            from sklearn.ensemble import GradientBoostingRegressor
            self.model_x_t = GradientBoostingRegressor().fit(X[treatment == 1], tau_treated)
            self.model_x_c = GradientBoostingRegressor().fit(X[treatment == 0], tau_control)
            # simple propensity-weighted combination (g=0.5) rather than a learned propensity model
        else:
            raise ValueError(f"unknown model_type '{self.model_type}'")
        return self

    @staticmethod
    def _proba(model, X) -> np.ndarray:
        return model.predict_proba(X)[:, 1]

    def predict_uplift(self, X) -> np.ndarray:
        X = np.asarray(X)
        if self.model_type == "t_learner":
            return self._proba(self.model_t, X) - self._proba(self.model_c, X)
        if self.model_type == "s_learner":
            X1 = np.column_stack([X, np.ones(len(X))])
            X0 = np.column_stack([X, np.zeros(len(X))])
            return self._proba(self.model_s, X1) - self._proba(self.model_s, X0)
        if self.model_type == "x_learner":
            tau_t = self.model_x_t.predict(X)
            tau_c = self.model_x_c.predict(X)
            return 0.5 * tau_t + 0.5 * tau_c
        raise ValueError(f"unknown model_type '{self.model_type}'")
