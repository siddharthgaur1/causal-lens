# causal-lens

Causal inference toolkit — four independently usable methods for "did this
intervention actually cause this outcome," not just correlation.

## When to use which method

| Method | Use when | Example here |
|---|---|---|
| A/B Test | You control randomized assignment | Fintech conversion test |
| Difference-in-Differences | A policy/event hits one group but not another, panel data over time | Repo-rate-style rate change on treated vs control units |
| Synthetic Control | One treated unit, many untreated donors, no individual-level randomization | Single treated series vs weighted donor pool |
| Uplift Modeling | You want to target *only the persuadable* individuals | Loan marketing offer |

## A/B Testing

```python
from causal_lens import ABTest

ab = ABTest(control=control_df, treatment=treatment_df)
result = ab.analyze(metric="conversion", method="bayesian", cuped_covariate="pre_conversion")
result.summary()
```

Frequentist (two-proportion z-test / Welch t-test), Bayesian (Beta-Binomial /
Normal-Normal, `prob_treatment_beats_control`, expected loss, 95% credible
interval), and CUPED variance reduction (`Y - theta*(X_pre - mean(X_pre))`)
are all implemented. `ab.diagnostics.srm_check` flags Sample Ratio Mismatch.

## Difference-in-Differences

```python
from causal_lens import DifferenceInDifferences

did = DifferenceInDifferences(data=panel_df, unit="stock", time="date",
                               treatment="is_treated", outcome="return",
                               treatment_date="2023-06-01")
result = did.estimate(cluster_by="stock")
result.summary()
result.event_study_estimates()   # treated-control gap per period; near-zero pre-period = parallel trends
```

`did.parallel_trends.parallel_trends_test` and `.placebo_test` (fake
treatment date before the real one) back the visual check with a formal one.

## Synthetic Control

```python
from causal_lens import SyntheticControl

sc = SyntheticControl(data=panel_df, treated_unit="HDFC", donor_units=nifty_stocks,
                       treatment_date="2016-11-08", predictors=["return"])
result = sc.fit(outcome="return")
result.gap_plot()
result.placebo_test(donor_panel=panel_df, predictors=["return"])  # permutation p-value
```

Donor weights fit via constrained optimization (`scipy.optimize.minimize`,
weights ≥ 0 and sum to 1). Placebo inference reruns the fit with each donor
as the "treated" unit and ranks the real unit's post/pre RMSPE ratio among
them — the standard Abadie-style significance check when there's no
asymptotic theory to lean on.

## Uplift Modeling

```python
from causal_lens import UpliftModel

model = UpliftModel(model_type="x_learner").fit(X, treatment, outcome)
scores = model.predict_uplift(X)
```

T-Learner, S-Learner, and X-Learner, all on plain `sklearn` estimators — no
`econml` dependency. `causal_lens.uplift.evaluation` gives the Qini curve,
AUUC, and Qini coefficient.

## Tests

`tests/test_causal_lens.py` — plain asserts, no framework. Each test builds a
synthetic dataset with a **known** ground-truth effect and checks the
estimator recovers it within a loose tolerance (e.g. DiD ATT within ±0.5 of
the injected 2.0).

## What's here vs. the original spec

Skipped for this pass — add if actually needed:
- Sequential testing (mSPRT), multiple-testing correction, novelty-effect detection in the A/B module
- Callaway-Sant'Anna staggered-timing DiD and Bacon decomposition (only single-treatment-date DiD is implemented)
- Causal Forest / `econml` (X-Learner above covers doubly-robust-ish uplift without the dependency)
- Real NSE/RBI datasets (`data/synthetic.py` generates all datasets used by tests — swap in `yfinance` fetchers when you want real-market results)
- HTML report generation, Streamlit dashboard, notebooks — the library API is the deliverable here
