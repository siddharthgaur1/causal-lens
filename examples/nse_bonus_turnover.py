"""Does a bonus issue or stock split change how much a stock actually trades?

Real NSE bhavcopy from the sibling `nse-warehouse` project (1.13M rows, 3,969
symbols). Real, dated corporate actions. No synthetic data anywhere in this
file.

Run it and it prints two things, in this order and for a reason:

  1. one event, studied properly -- synthetic control against a clean donor
     pool, with placebo inference;
  2. the same estimator run over every comparable event in the window.

Step 2 exists because step 1 alone is not evidence, and this script is built to
demonstrate that rather than to assert it. Read both numbers before believing
either.

    python examples/nse_bonus_turnover.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causal_lens.synthetic_control.estimator import SyntheticControl  # noqa: E402
from causal_lens.synthetic_control.inference import placebo_inference  # noqa: E402
from data.nse_real import _connect, bonus_split_panel  # noqa: E402


def study(d):
    """Synthetic control for one event. Returns (mean post gap, pre RMSPE, p)."""
    sc = SyntheticControl(
        d["panel"], d["treated"], d["donors"], d["event_date"],
        predictors=["log_turnover"], unit_col="unit", time_col="date",
    )
    res = sc.fit(outcome="log_turnover")
    pre = res.gap[res.gap.index < d["event_date"]]
    post = res.gap[res.gap.index >= d["event_date"]]
    donor_panel = d["panel"][d["panel"]["unit"].isin(d["donors"])]
    inf = placebo_inference(res, donor_panel, predictors=["log_turnover"])
    return res, post.mean(), float(np.sqrt((pre ** 2).mean())), inf["p_value"]


def main() -> int:
    print("=" * 70)
    print("1. ONE EVENT")
    print("=" * 70)
    d = bonus_split_panel()
    res, effect, rmspe, p = study(d)

    print(f"treated      : {d['treated']}")
    print(f"event        : {d['event_date'].date()}  ({d['event_note']})")
    print(f"donors       : {len(d['donors'])} symbols, none with an in-window action")
    print(f"window       : {d['n_days']} sessions")
    print()
    print(f"pre-period RMSPE : {rmspe:.4f}   (fit quality, lower is better)")
    print(f"post mean gap    : {effect:+.4f} log-turnover"
          f"  ->  {100 * (np.exp(effect) - 1):+.1f}% vs counterfactual")
    print(f"placebo p-value  : {p:.4f}")
    print()
    print("weights (top 5):")
    for s, w in res.weights.sort_values(ascending=False).head(5).items():
        if w > 1e-4:
            print(f"  {s:14} {w:.3f}")

    print()
    print("=" * 70)
    print("2. EVERY COMPARABLE EVENT -- because one event is not evidence")
    print("=" * 70)

    conn = _connect()
    events = conn.execute(
        """
        SELECT symbol, ex_date FROM corporate_actions
        WHERE ex_date BETWEEN '2025-03-01' AND '2026-03-01'
          AND symbol NOT LIKE '%-RE'
        ORDER BY ex_date
        """
    ).fetchall()
    conn.close()

    effects, pvals = [], []
    for sym, _ in events:
        try:
            di = bonus_split_panel(treated=sym)
            if di["treated"] != sym:
                continue
            _, e, _, pi = study(di)
        except Exception:
            continue  # not enough clean sessions or donors around this event
        effects.append(e)
        pvals.append(pi)

    if not effects:
        print("no event had enough clean data")
        return 1

    mean = float(np.mean(effects))
    print(f"events estimated : {len(effects)} of {len(events)} candidates")
    print(f"mean effect      : {mean:+.4f} log-turnover"
          f"  ->  {100 * (np.exp(mean) - 1):+.1f}%")
    print(f"median effect    : {np.median(effects):+.4f}")
    print(f"positive         : {sum(1 for e in effects if e > 0)}/{len(effects)}")
    print(f"p < 0.10         : {sum(1 for x in pvals if x < 0.10)}/{len(pvals)}")
    print(f"p < 0.05         : {sum(1 for x in pvals if x < 0.05)}/{len(pvals)}")
    print()
    print("Read that against the single-event number above before concluding")
    print("anything. That is the point of this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
