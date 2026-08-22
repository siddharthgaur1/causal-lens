# causal-lens

[![Portfolio](https://img.shields.io/badge/↩-siddharthgaur1-111827?style=flat-square)](https://github.com/siddharthgaur1)
[![CI](https://github.com/siddharthgaur1/causal-lens/actions/workflows/ci.yml/badge.svg)](https://github.com/siddharthgaur1/causal-lens/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Causal inference toolkit — four independently usable methods for "did this
intervention actually cause this outcome," not just correlation.

## Quickstart

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Four independent estimators, each importable on its own (see below) — no
single unified "run" entrypoint, since the input shape differs per method
(A/B needs two groups, DiD/Synthetic Control need panel data, Uplift needs
treatment+outcome+features).

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

## Architecture

Each method is a standalone class over plain pandas/numpy/scipy — `ABTest`,
`DifferenceInDifferences`, `SyntheticControl`, `UpliftModel` share no base
class or common pipeline, deliberately: the four techniques answer different
causal questions with different data shapes, and a forced common interface
would leak abstraction into the estimators (e.g. DiD's clustering vs
Synthetic Control's donor-weight optimization have nothing in common). Each
returns a `.summary()`-able result object with its own diagnostics
(SRM check, parallel-trends test, placebo test, Qini curve).

## Results

### Correctness: recovering a known effect

Every test (`tests/test_causal_lens.py`) builds a synthetic dataset with a
**known** injected effect and asserts the estimator recovers it within a stated
tolerance (e.g. DiD ATT within ±0.5 of an injected 2.0). That is the right test
for estimator correctness — you cannot check an estimator against data whose
true effect you do not know.

### A real question on real data

Synthetic tests say the maths is right; they say nothing about whether the
toolkit survives contact with real data. So:

```bash
python examples/nse_bonus_turnover.py
```

**Question:** does a bonus issue or stock split change how much a stock
actually trades? The price change is arithmetic and is adjusted away. Whether a
lower nominal price draws more participation is an empirical question with real
disagreement in the literature.

**Data:** real NSE bhavcopy from the sibling
[nse-warehouse](https://github.com/siddharthgaur1/nse-warehouse) — 1.13M rows,
3,969 symbols — and real corporate actions inferred from close-price
discontinuities. Donor pools exclude any symbol with its own corporate action
inside the window, since a donor that also split would contaminate the
counterfactual.

**One event** (SBC, 1:2 bonus, 2025-03-10; 20 donors, 100 sessions):

| | |
|---|---|
| Effect on turnover | **+34.7%** vs the synthetic counterfactual |
| Pre-period RMSPE | 0.365 |
| Placebo p-value | 0.095 |

A clean-looking positive result. **It does not survive replication.**

**All 79 comparable events** in the same window, same estimator, same donor
rules:

| | |
|---|---|
| Mean effect | **−10.4%** |
| Median effect | −6.6% |
| Positive | 34 / 79 |
| p < 0.05 | 14 / 79 |

Thirty-four of seventy-nine positive is a coin flip. **The honest conclusion is
that there is no consistent turnover effect** in this data, and the single-event
result was the first event that happened to meet the data requirements — not a
finding.

That contrast is why `examples/nse_bonus_turnover.py` prints both numbers, in
that order, every run. A causal toolkit that makes it easy to produce a
publishable single-event result and hard to notice it does not replicate is
worse than no toolkit.

## Limitations

What's not here — add if actually needed:
- Sequential testing (mSPRT), multiple-testing correction, novelty-effect detection in the A/B module
- Callaway-Sant'Anna staggered-timing DiD and Bacon decomposition (only single-treatment-date DiD is implemented)
- Causal Forest / `econml` (X-Learner above covers doubly-robust-ish uplift without the dependency)
- **Real data is now wired in** for synthetic control (`data/nse_real.py`, real NSE bhavcopy via nse-warehouse — see Results). The A/B, DiD and Uplift examples still run on `data/synthetic.py` generators; those need real experiment/panel data that this project does not have.
- The real-data study covers a **single window (Mar 2025 – Mar 2026)** and one event type. 14 of 79 events clear p<0.05, which is above the ~4 expected by chance — worth a look, not a claim.
- HTML report generation, Streamlit dashboard, notebooks — the library API is the deliverable here
