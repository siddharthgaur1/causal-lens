"""The real-data loader must fail loudly, never fall back to synthetic data.

The whole point of data/nse_real.py is that these examples stop being
synthetic. A silent fallback would reintroduce exactly the problem it exists to
remove, and it would be invisible -- the numbers would still print.
"""

from __future__ import annotations

import pytest

from data.nse_real import bonus_split_panel


def test_a_missing_warehouse_raises_and_says_where_to_get_one(tmp_path):
    missing = tmp_path / "no-such.duckdb"
    with pytest.raises(FileNotFoundError) as exc:
        bonus_split_panel(db=missing)
    msg = str(exc.value)
    assert "nse-warehouse" in msg, "must point at the project that builds the data"
    assert "do not fall back to synthetic" in msg.lower(), (
        "the no-silent-fallback promise must be stated where it is broken"
    )


@pytest.mark.skipif(
    not __import__("data.nse_real", fromlist=["DEFAULT_DB"]).DEFAULT_DB.exists(),
    reason="nse-warehouse database not present",
)
class TestAgainstTheRealWarehouse:
    """Only run where the warehouse exists; CI has no 1.13M-row database."""

    def test_the_panel_is_balanced_and_covers_both_sides_of_the_event(self):
        d = bonus_split_panel()
        panel = d["panel"]
        per_unit = panel.groupby("unit").size()
        assert per_unit.nunique() == 1, "synthetic control needs a balanced panel"
        pre = panel[panel["date"] < d["event_date"]]
        post = panel[panel["date"] >= d["event_date"]]
        assert len(pre) > 0 and len(post) > 0

    def test_no_donor_has_its_own_corporate_action_in_the_window(self):
        """A donor that also split is not a counterfactual."""
        from data.nse_real import _connect

        d = bonus_split_panel()
        lo, hi = d["panel"]["date"].min().date(), d["panel"]["date"].max().date()
        conn = _connect()
        contaminated = {
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT symbol FROM corporate_actions "
                "WHERE ex_date BETWEEN ? AND ?",
                [lo, hi],
            ).fetchall()
        }
        conn.close()
        assert not (set(d["donors"]) & contaminated)

    def test_returns_use_adjusted_prices_so_the_split_is_not_the_effect(self):
        """A 1:2 bonus halves the raw price. If returns used it, the 'effect'
        would be a ~-50% one-day return that is pure arithmetic."""
        d = bonus_split_panel()
        treated = d["panel"][d["panel"]["unit"] == d["treated"]]
        on_event = treated[treated["date"] == d["event_date"]]["return"]
        if len(on_event):
            assert abs(float(on_event.iloc[0])) < 0.35, (
                "event-day return looks like an unadjusted split"
            )
