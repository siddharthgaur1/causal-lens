"""Build a real NSE panel from the nse-warehouse DuckDB, for the demos.

This replaces the synthetic generators for the Synthetic Control and DiD
examples. The data is real NSE bhavcopy (1.13M rows, 3,969 symbols) built by
the sibling `nse-warehouse` project; nothing here is generated.

The event studied is a **bonus issue or stock split** -- a real, dated,
mechanical corporate action inferred by nse-warehouse from close-price
discontinuities. The causal question is genuine and not mechanical:

    does splitting the share price change how much the stock actually trades?

The price change itself is arithmetic and is adjusted away in `bars_adjusted`.
Turnover is not: whether a lower nominal price draws more participation is an
empirical question with real disagreement in the literature.

Requires a built warehouse. Point NSE_WAREHOUSE_DB at its DuckDB file, or pass
`db=`. Without it these examples raise rather than silently falling back to
synthetic data -- that substitution is exactly what this module exists to end.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

DEFAULT_DB = Path(
    os.environ.get(
        "NSE_WAREHOUSE_DB",
        Path(__file__).resolve().parents[2] / "nse-warehouse" / "data" / "warehouse.duckdb",
    )
)


def _connect(db: Path | str | None = None):
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "real-data examples need duckdb: pip install duckdb"
        ) from exc

    path = Path(db) if db else DEFAULT_DB
    if not path.exists():
        raise FileNotFoundError(
            f"no nse-warehouse database at {path}.\n"
            "Build it from https://github.com/siddharthgaur1/nse-warehouse "
            "or set NSE_WAREHOUSE_DB to an existing one. These examples do not "
            "fall back to synthetic data."
        )
    conn = duckdb.connect(str(path), read_only=True)
    # The warehouse's views reference its parquet files by RELATIVE path
    # ("data/warehouse/**/*.parquet"), so they only resolve when the process
    # runs from the warehouse root. Set DuckDB's own working directory instead
    # of chdir-ing this process, which would be a side effect on the caller.
    conn.execute(f"SET file_search_path = '{path.parent.parent.as_posix()}'")
    return conn


def bonus_split_panel(
    db: Path | str | None = None,
    treated: str | None = None,
    n_donors: int = 20,
    pre_days: int = 60,
    post_days: int = 40,
    seed: int = 42,
) -> dict:
    """Panel around one real bonus/split event: treated unit plus donor pool.

    Returns a dict with `panel` (long format: date, unit, log_turnover,
    return), `treated`, `donors`, `event_date` and `event_note`.

    Donors are chosen as symbols that traded on every day of the window and had
    **no corporate action of their own** anywhere in it -- a donor that also
    split would contaminate the counterfactual.
    """
    conn = _connect(db)

    # Pick a treated event: a corporate action with enough clean trading days
    # either side of it, on a symbol that is not a rights entitlement (-RE),
    # which are short-lived instruments rather than ongoing listings.
    events = conn.execute(
        """
        SELECT symbol, ex_date, factor, inferred
        FROM corporate_actions
        WHERE ex_date >= '2025-03-01' AND ex_date <= '2026-05-01'
          AND symbol NOT LIKE '%-RE'
        ORDER BY ex_date
        """
    ).fetchall()
    if not events:
        raise ValueError("no usable corporate actions in the warehouse window")

    chosen = None
    for sym, ex_date, factor, note in events:
        if treated and sym != treated:
            continue
        n_pre = conn.execute(
            "SELECT count(*) FROM bars_adjusted WHERE symbol = ? AND trade_date < ?",
            [sym, ex_date],
        ).fetchone()[0]
        n_post = conn.execute(
            "SELECT count(*) FROM bars_adjusted WHERE symbol = ? AND trade_date >= ?",
            [sym, ex_date],
        ).fetchone()[0]
        if n_pre >= pre_days and n_post >= post_days:
            chosen = (sym, ex_date, factor, note)
            break
    if chosen is None:
        raise ValueError("no corporate action has enough trading days around it")

    sym, ex_date, factor, note = chosen

    window = conn.execute(
        """
        SELECT DISTINCT trade_date FROM bars_adjusted
        WHERE trade_date BETWEEN
              -- min(), not max(): the lower bound is the EARLIEST of the last
              -- `pre_days` sessions before the event. max() collapses the whole
              -- pre-period to the single day before it, which silently leaves
              -- synthetic control nothing to fit weights on.
              (SELECT min(trade_date) FROM (
                   SELECT trade_date FROM bars_adjusted
                   WHERE trade_date < ? GROUP BY trade_date
                   ORDER BY trade_date DESC LIMIT ?))
          AND (SELECT max(trade_date) FROM (
                   SELECT trade_date FROM bars_adjusted
                   WHERE trade_date >= ? GROUP BY trade_date
                   ORDER BY trade_date LIMIT ?))
        ORDER BY trade_date
        """,
        [ex_date, pre_days, ex_date, post_days],
    ).fetchall()
    days = [d[0] for d in window]
    lo, hi = days[0], days[-1]

    # Donors: full coverage over the window, no corporate action inside it,
    # and comparable liquidity to the treated name.
    donors = conn.execute(
        """
        WITH covered AS (
            SELECT symbol, count(*) n, avg(turnover) liq
            FROM bars_adjusted
            WHERE trade_date BETWEEN ? AND ? AND turnover > 0
            GROUP BY symbol
            HAVING count(*) = ?
        ),
        contaminated AS (
            SELECT DISTINCT symbol FROM corporate_actions
            WHERE ex_date BETWEEN ? AND ?
        ),
        target AS (SELECT liq FROM covered WHERE symbol = ?)
        SELECT c.symbol
        FROM covered c, target t
        WHERE c.symbol <> ?
          AND c.symbol NOT LIKE '%-RE'
          AND c.symbol NOT IN (SELECT symbol FROM contaminated)
        ORDER BY abs(ln(c.liq) - ln(t.liq))
        LIMIT ?
        """,
        [lo, hi, len(days), lo, hi, sym, sym, n_donors],
    ).fetchall()
    donor_syms = [d[0] for d in donors]
    if len(donor_syms) < 5:
        raise ValueError(f"only {len(donor_syms)} clean donors available")

    rows = conn.execute(
        """
        SELECT trade_date AS date, symbol AS unit,
               ln(turnover + 1) AS log_turnover,
               -- adj_close, not close: the split's mechanical price drop must
               -- not show up as a return, or the "effect" is just arithmetic.
               adj_close
        FROM bars_adjusted
        WHERE trade_date BETWEEN ? AND ? AND symbol IN ?
        ORDER BY trade_date, symbol
        """,
        [lo, hi, tuple([sym] + donor_syms)],
    ).df()
    conn.close()

    rows["date"] = pd.to_datetime(rows["date"])
    rows["return"] = rows.groupby("unit")["adj_close"].pct_change().fillna(0.0)

    return {
        "panel": rows[["date", "unit", "log_turnover", "return"]],
        "treated": sym,
        "donors": donor_syms,
        "event_date": pd.Timestamp(ex_date),
        "event_note": note,
        "factor": factor,
        "n_days": len(days),
    }
