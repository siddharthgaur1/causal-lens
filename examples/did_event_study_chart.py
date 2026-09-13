"""Render docs/did-event-study.png: DiD on a synthetic panel with a known effect.

The panel comes from data/synthetic.py with an injected ATT of 2.0, so the chart
shows whether the estimator recovers a number that is known in advance. It is
synthetic by design; the real-data study is examples/nse_bonus_turnover.py.

    python examples/did_event_study_chart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from causal_lens import DifferenceInDifferences
from data.synthetic import generate_did_panel

TRUE_ATT = 2.0
TREATMENT_PERIOD = 6

panel = generate_did_panel(treatment_period=TREATMENT_PERIOD, true_att=TRUE_ATT, seed=0)
treatment_date = panel["date"].drop_duplicates().sort_values().iloc[TREATMENT_PERIOD]
result = DifferenceInDifferences(
    panel, unit="unit", time="date", treatment="treated", outcome="outcome",
    treatment_date=treatment_date,
).estimate(cluster_by="unit")
result.summary()

plt.switch_backend("Agg")
fig, ax = plt.subplots(figsize=(9, 4), dpi=120)
result.event_study_plot(ax=ax)
ax.set_title(
    f"DiD on a synthetic panel (20 units, 12 months), injected ATT = {TRUE_ATT}: "
    f"estimated {result.att:.3f} (SE {result.se:.3f}, clustered by unit)",
    loc="left", fontsize=10,
)
ax.legend(frameon=False)
fig.tight_layout()

out = ROOT / "docs" / "did-event-study.png"
fig.savefig(out)
print(f"wrote {out}")
