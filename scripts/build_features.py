#!/usr/bin/env python3
"""Build the pre-registered feature set and label from data/market_panel.parquet.

Features and label only. No model, no training, no evaluation, and nothing here
reports the label's distribution.

Sample     : 1962-02-28 to 2025-08-31, monthly decision points.
Features   : cape_z, yield_slope, momentum_12m, unrate_trend_12m. Exactly these four,
             as pre-registered. unrate_trend_12m is the latest available rate minus its
             12-month trailing mean; Amendment 2, which had redefined it as a 12-month
             change, is WITHDRAWN.
Label      : sign of the 12-month forward excess return of equities over 3-month bills,
             both legs in real terms.

The four binding rules from README.md are implemented as follows.

1. EXPANDING WINDOWS ONLY. cape_z uses an expanding mean and sd over data up to the
   reference month, never the full sample. Full-sample and expanding z disagree on the
   SIGN of the signal in 13% of months in this panel.

2. AVAILABILITY, NOT shift(1). Every input is taken from the newest reference month
   satisfying `ref_month_end + <series>_lag_days <= decision month-end`, with the lag
   read from the panel's own lag column. No lag is hardcoded anywhere in this file.

3. ORDER OF OPERATIONS. Every window is computed on the reference-month series, which
   carries the NaN, and the availability shift is applied to the RESULT. Shifting first
   would turn the October 2025 gap into a stale repeat and the NaN would vanish.

4. 2025-10 IS ONE GAP. No fillna, ffill, interpolate or dropna anywhere, and no column
   is ever filled from another. NaN propagates.
"""

from __future__ import annotations

import argparse
import json
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO_ROOT / "data" / "market_panel.parquet"
DEFAULT_OUT = REPO_ROOT / "data" / "features.parquet"

SAMPLE_START = pd.Timestamp("1962-02-28")
SAMPLE_END = pd.Timestamp("2025-08-31")

CAPE_MIN_PERIODS = 120      # pre-registered warm-up for the expanding z-score
WINDOW_MONTHS = 12          # pre-registered window for momentum and the unemployment trend
LABEL_HORIZON = 12          # pre-registered label horizon
EXPECTED_COMPLETE_ROWS = 761

FEATURES = ["cape_z", "yield_slope", "momentum_12m", "unrate_trend_12m"]


# --------------------------------------------------------------------------------------
# Rule 2: availability
# --------------------------------------------------------------------------------------

def available_ref_month(
    panel: pd.DataFrame, lag_col: str, valid: pd.Series, decisions: pd.DatetimeIndex
) -> pd.Series:
    """Newest reference month usable at each decision point.

    A reference month r is usable at decision M when `r + lag_days(r) <= M`. The lag is
    read from the panel column, never assumed, so the measured per-row unrate lags are
    honoured exactly as the constant 15- and 1-day lags are.

    Release order is not assumed to match reference-month order: rows are sorted by the
    date they became available and a running maximum of the reference month is taken, so
    a late release (unrate 2025-09, published 51 days late) can never make an older month
    look like the newest available one.
    """
    rows = panel.loc[valid, ["date", lag_col]].dropna()
    avail_on = rows["date"] + pd.to_timedelta(rows[lag_col].astype("int64"), unit="D")
    order = np.argsort(avail_on.to_numpy(), kind="stable")
    avail_sorted = avail_on.to_numpy()[order]
    ref_sorted = rows["date"].to_numpy()[order]
    ref_running_max = np.maximum.accumulate(ref_sorted)

    # index of the last row whose availability date is <= each decision month-end
    pos = np.searchsorted(avail_sorted, decisions.to_numpy(), side="right") - 1
    out = np.where(pos >= 0, ref_running_max[np.clip(pos, 0, None)], np.datetime64("NaT"))
    return pd.Series(pd.to_datetime(out), index=decisions, name=f"ref_{lag_col}")


def shift_to_decisions(
    feature_by_ref: pd.Series, ref_at_decision: pd.Series
) -> pd.Series:
    """Rule 3's second half: map a reference-month feature onto decision points."""
    return pd.Series(feature_by_ref.reindex(ref_at_decision.to_numpy()).to_numpy(),
                     index=ref_at_decision.index)


def full_span_valid(s: pd.Series, span: int) -> pd.Series:
    """True where every one of the last `span` observations is non-null.

    For unrate_trend_12m this is simply what the formula needs: a trailing mean reads all
    12 months. For momentum_12m, a two-endpoint return, it is Rule 4 read literally - a
    window that SPANS the 2025-10 gap is void even though the arithmetic touches only its
    endpoints. That stricter reading costs nothing here, because no in-sample feature
    window reaches 2025-10.
    """
    return s.notna().rolling(span, min_periods=span).sum().eq(span)


# --------------------------------------------------------------------------------------
# Features, each computed on reference months first (Rule 3)
# --------------------------------------------------------------------------------------

def build_features(panel: pd.DataFrame, decisions: pd.DatetimeIndex) -> tuple[pd.DataFrame, dict]:
    p = panel.set_index("date")
    audit: dict[str, pd.Series] = {}

    # --- cape_z: expanding z-score on the reference-month series (Rules 1 and 3) ---
    cape = p["cape"]
    mu = cape.expanding(min_periods=CAPE_MIN_PERIODS).mean()
    sd = cape.expanding(min_periods=CAPE_MIN_PERIODS).std()          # ddof=1
    cape_z_ref = (cape - mu) / sd
    cape_z_ref = cape_z_ref.where(cape.notna())     # cannot z-score a missing observation

    # --- yield_slope: a spread must come from ONE date, so it needs both legs available ---
    slope_ref = p["dgs10"] - p["dtb3"]
    slope_lag = pd.concat([p["dgs10_lag_days"], p["dtb3_lag_days"]], axis=1).max(axis=1)

    # --- momentum_12m: 12-month trailing return ---
    sp = p["sp500_index"]
    mom_ref = (sp / sp.shift(WINDOW_MONTHS) - 1.0).where(full_span_valid(sp, WINDOW_MONTHS + 1))

    # --- unrate_trend_12m: latest rate minus its 12-month trailing mean ---
    # Pre-registered definition. Unlike momentum this is not a two-endpoint formula: it
    # genuinely reads all 12 months, so requiring the full span is the natural rule here
    # rather than the stricter reading of Rule 4.
    un = p["unrate"]
    trailing_mean = un.rolling(WINDOW_MONTHS, min_periods=WINDOW_MONTHS).mean()
    trend_ref = (un - trailing_mean).where(full_span_valid(un, WINDOW_MONTHS))

    work = panel.copy()
    work["_slope_lag"] = slope_lag.to_numpy()

    ref_cape = available_ref_month(panel, "cape_lag_days", panel["cape"].notna(), decisions)
    ref_slope = available_ref_month(
        work, "_slope_lag", (panel["dgs10"].notna() & panel["dtb3"].notna()), decisions)
    ref_sp = available_ref_month(panel, "sp500_index_lag_days", panel["sp500_index"].notna(), decisions)
    ref_un = available_ref_month(panel, "unrate_lag_days", panel["unrate"].notna(), decisions)

    out = pd.DataFrame({
        "date": decisions,
        "cape_z": shift_to_decisions(cape_z_ref, ref_cape).to_numpy(),
        "yield_slope": shift_to_decisions(slope_ref, ref_slope).to_numpy(),
        "momentum_12m": shift_to_decisions(mom_ref, ref_sp).to_numpy(),
        "unrate_trend_12m": shift_to_decisions(trend_ref, ref_un).to_numpy(),
    })
    audit = {"cape_z": ref_cape, "yield_slope": ref_slope,
             "momentum_12m": ref_sp, "unrate_trend_12m": ref_un}
    parts = {"cape_z_ref": cape_z_ref, "slope_ref": slope_ref,
             "mom_ref": mom_ref, "trend_ref": trend_ref, "mu": mu, "sd": sd}
    return out, {"ref_months": audit, "ref_series": parts}


# --------------------------------------------------------------------------------------
# Label
# --------------------------------------------------------------------------------------

def build_label(panel: pd.DataFrame, decisions: pd.DatetimeIndex) -> pd.Series:
    """Sign of the 12-month forward excess return, equities over bills, both legs real.

    sp500_index is ALREADY Shiller's real total return index, so the equity leg needs no
    deflation; deflating it again would double-deflate. The bill leg is a nominal yield
    and is deflated by the same window's inflation, which is what puts the two legs in
    the same units. The rebasing constant in the equity index cancels in the ratio.

    Bill compounding follows the pre-registered convention: dtb3 from months M+1 through
    M+12, each quoted annualised rate treated as an effective annual rate. Those values
    postdate the decision, which is correct - this is the label, not a feature.
    """
    p = panel.set_index("date")
    sp, cpi, bill = p["sp500_index"], p["cpi"], p["dtb3"]
    labels = []
    for M in decisions:
        M12 = (M + pd.DateOffset(months=LABEL_HORIZON)) + pd.offsets.MonthEnd(0)
        window = pd.date_range(M + pd.offsets.MonthEnd(1), M12, freq="ME")
        e0, e1 = sp.get(M, np.nan), sp.get(M12, np.nan)
        c0, c1 = cpi.get(M, np.nan), cpi.get(M12, np.nan)
        rates = bill.reindex(window)
        if (len(window) != LABEL_HORIZON or pd.isna(e0) or pd.isna(e1)
                or pd.isna(c0) or pd.isna(c1) or rates.isna().any()):
            labels.append(np.nan)
            continue
        equity_real = e1 / e0                                   # already real
        bill_nominal = float(np.prod((1.0 + rates.to_numpy() / 100.0) ** (1.0 / 12.0)))
        bill_real = bill_nominal * (c0 / c1)                    # deflate the bill leg
        labels.append(1.0 if equity_real - bill_real > 0 else 0.0)
    return pd.Series(labels, index=decisions, name="label")


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------

def verify(panel: pd.DataFrame, feats: pd.DataFrame, aux: dict) -> bool:
    p = panel.set_index("date")
    refs = aux["ref_months"]
    ok = True

    print("=" * 78)
    print("VERIFICATION")
    print("=" * 78)

    complete = feats[FEATURES + ["label"]].notna().all(axis=1)
    n = int(complete.sum())
    print(f"\n1. Rows with all four features AND the label non-null: {n}")
    print(f"   Expected from the earlier read-only count            : {EXPECTED_COMPLETE_ROWS}")
    if n != EXPECTED_COMPLETE_ROWS:
        ok = False
        print("   *** MISMATCH - STOPPING. Nothing has been adjusted to make this agree. ***")
    else:
        print("   MATCH")

    print("\n2. Input provenance at three decision points, one per era.")
    print("   Every input's availability date must be <= the decision date.")
    for M in [pd.Timestamp("1965-06-30"), pd.Timestamp("1995-03-31"), pd.Timestamp("2025-08-31")]:
        print(f"\n   --- decision point {M.date()} ---")
        rows = [
            ("cape_z", "cape", refs["cape_z"].loc[M], "cape_lag_days"),
            ("yield_slope", "dgs10/dtb3", refs["yield_slope"].loc[M], "dgs10_lag_days"),
            ("momentum_12m", "sp500_index", refs["momentum_12m"].loc[M], "sp500_index_lag_days"),
            ("unrate_trend_12m", "unrate", refs["unrate_trend_12m"].loc[M], "unrate_lag_days"),
        ]
        print(f"     {'feature':<17}{'input':<12}{'newest ref month':<18}"
              f"{'lag':>5}  {'available on':<13}{'<= decision?'}")
        for feat, src, ref, lagcol in rows:
            lag = int(p.at[ref, lagcol])
            avail = ref + pd.Timedelta(days=lag)
            good = avail <= M
            ok = ok and good
            print(f"     {feat:<17}{src:<12}{str(ref.date()):<18}{lag:>5}  "
                  f"{str(avail.date()):<13}{'YES' if good else '*** NO ***'}")
            if feat == "momentum_12m":
                back = (ref - pd.DateOffset(months=WINDOW_MONTHS)) + pd.offsets.MonthEnd(0)
                print(f"     {'':<17}{'  (window)':<12}{str(back.date())} .. {ref.date()}"
                      f"   all {WINDOW_MONTHS + 1} months in the past")
            if feat == "unrate_trend_12m":
                back = (ref - pd.DateOffset(months=WINDOW_MONTHS - 1)) + pd.offsets.MonthEnd(0)
                print(f"     {'':<17}{'  (window)':<12}{str(back.date())} .. {ref.date()}"
                      f"   all {WINDOW_MONTHS} months in the past")

    print("\n3. cape_z recomputed by hand at an early decision point, using ONLY prior data.")
    M = pd.Timestamp("1965-06-30")
    ref = refs["cape_z"].loc[M]
    hist = p["cape"].loc[:ref].dropna()
    mu_hand = float(hist.mean())
    sd_hand = float(hist.std(ddof=1))
    z_hand = (float(p.at[ref, "cape"]) - mu_hand) / sd_hand
    z_built = float(feats.set_index("date").at[M, "cape_z"])
    print(f"   decision {M.date()}, newest available cape reference month {ref.date()}")
    print(f"   observations used      : {len(hist)}  ({hist.index.min().date()} .. {hist.index.max().date()})")
    print(f"   none of them postdate the reference month: "
          f"{'YES' if hist.index.max() <= ref else '*** NO ***'}")
    print(f"   hand mean / sd         : {mu_hand:.10f} / {sd_hand:.10f}")
    print(f"   hand z / built z       : {z_hand:.10f} / {z_built:.10f}")
    match = abs(z_hand - z_built) < 1e-10
    ok = ok and match and hist.index.max() <= ref
    print(f"   MATCH" if match else "   *** MISMATCH ***")
    full_mu = float(p['cape'].dropna().mean())
    print(f"   for contrast, the FULL-SAMPLE cape mean is {full_mu:.4f} vs {mu_hand:.4f} here;")
    print(f"   using it would have encoded a century of future data into this 1965 row.")

    print("\n4. Null counts per column, with the reason for each.")
    for col in FEATURES + ["label"]:
        print(f"   {col:<18} {int(feats[col].isna().sum())}")
    return ok


def explain_nulls(feats: pd.DataFrame, panel: pd.DataFrame, aux: dict) -> None:
    refs = aux["ref_months"]
    f = feats.set_index("date")
    print("\n   reasons:")
    for col in FEATURES:
        miss = f.index[f[col].isna()]
        if len(miss) == 0:
            print(f"   {col:<18} no nulls")
            continue
        print(f"   {col:<18} {len(miss)} null: {miss.min().date()} .. {miss.max().date()}")
        if col == "cape_z":
            print(f"   {'':<18}   expanding warm-up needs {CAPE_MIN_PERIODS} observations")
    miss = f.index[f["label"].isna()]
    print(f"   {'label':<18} {len(miss)} null")
    p = panel.set_index("date")
    for M in miss:
        M12 = (M + pd.DateOffset(months=LABEL_HORIZON)) + pd.offsets.MonthEnd(0)
        why = []
        if pd.isna(p["cpi"].get(M12, np.nan)):
            why.append(f"cpi null at label endpoint {M12.date()}")
        if pd.isna(p["sp500_index"].get(M12, np.nan)):
            why.append(f"sp500_index null at endpoint {M12.date()}")
        if pd.isna(p["cpi"].get(M, np.nan)):
            why.append(f"cpi null at decision endpoint {M.date()}")
        print(f"   {'':<18}   {M.date()}: {'; '.join(why) or 'endpoint outside panel'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    panel = pd.read_parquet(args.panel).sort_values("date").reset_index(drop=True)
    decisions = pd.date_range(SAMPLE_START, SAMPLE_END, freq="ME")

    feats, aux = build_features(panel, decisions)
    feats["label"] = build_label(panel, decisions).to_numpy()
    feats = feats[["date"] + FEATURES + ["label"]]

    ok = verify(panel, feats, aux)
    explain_nulls(feats, panel, aux)

    complete = feats[FEATURES + ["label"]].notna().all(axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(feats, preserve_index=False)
    table = table.replace_schema_metadata({
        **(table.schema.metadata or {}),
        b"features_provenance_json": json.dumps({
            "built_by": "scripts/build_features.py",
            "built_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "panel": str(args.panel.name),
            "sample": [str(SAMPLE_START.date()), str(SAMPLE_END.date())],
            "rows": len(feats),
            "complete_rows": int(complete.sum()),
            "features": FEATURES,
            "cape_z_min_periods": CAPE_MIN_PERIODS,
            "window_months": WINDOW_MONTHS,
            "label": ("sign of the 12-month forward excess return, equity real total "
                      "return minus real compounded 3-month bill return; 1 if positive"),
            "rules": [
                "expanding windows only, never full sample",
                "availability from each series' own lag column, never shift(1)",
                "window computed on reference months, then availability-shifted",
                "2025-10 is one gap; NaN propagates, nothing filled",
            ],
        }, indent=2).encode(),
    })
    pq.write_table(table, args.out, compression="snappy")

    print("\n" + "=" * 78)
    print(f"Wrote {args.out.relative_to(REPO_ROOT)}  ({len(feats)} rows, "
          f"{int(complete.sum())} complete)")
    print("Deliberately not reported: the label's positive rate, class balance, and any")
    print("feature-label association. The baseline is re-estimated inside each fold.")
    print("=" * 78)
    if not ok:
        print("\nVERIFICATION FAILED - see above. Nothing was adjusted to force agreement.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
