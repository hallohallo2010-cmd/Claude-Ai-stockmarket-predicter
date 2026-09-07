#!/usr/bin/env python3
"""Independent gate on data/features.parquet. Exits non-zero on any failure.

This exists because a build can report success while leaving a stale file in place. That
happened once: the report was piped to `head`, the process died on a broken pipe before
writing, exit status and printed output both looked clean, and the parquet still held a
withdrawn definition of unrate_trend_12m. Only a hand-recomputation against the file
caught it.

So this script trusts nothing from the build:

  * it reads the parquet FROM DISK, in its own process, and never sees the build's
    in-memory dataframe;
  * it recomputes the definition hash from the CURRENT source of build_features.py and
    compares it with the hash stored in the file, so a file written by different formulas
    is caught without recomputing a single feature;
  * it recomputes every feature through the build module and compares values, so a stale
    file whose hash happens to match is caught too;
  * it then recomputes cape_z and unrate_trend_12m a THIRD time, here, in plain pandas,
    independently of the build's vectorised helpers, so a logic error shared by build and
    audit cannot pass unnoticed;
  * it rebuilds the availability map with a naive loop rather than the build's
    searchsorted-and-running-maximum, cross-checking the clever implementation against an
    obvious one.

Usage: python scripts/audit_features.py   (exit 0 = pass, 2 = fail)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
PANEL = REPO_ROOT / "data" / "market_panel.parquet"
FEATURES_FILE = REPO_ROOT / "data" / "features.parquet"
BUILD_SRC = REPO_ROOT / "scripts" / "build_features.py"

FAILURES: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"   [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)
    return ok


def load_build_module():
    spec = importlib.util.spec_from_file_location("build_features", BUILD_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def naive_available_ref(panel: pd.DataFrame, lag_col: str, valid: pd.Series,
                        decisions: pd.DatetimeIndex) -> pd.Series:
    """The obvious O(n*m) implementation, to cross-check the vectorised one."""
    rows = panel.loc[valid, ["date", lag_col]].dropna()
    avail = rows["date"] + pd.to_timedelta(rows[lag_col].astype("int64"), unit="D")
    out = []
    for M in decisions:
        usable = rows["date"][avail <= M]
        out.append(usable.max() if len(usable) else pd.NaT)
    return pd.Series(pd.to_datetime(out), index=decisions)


def main() -> int:
    print("=" * 78)
    print("INDEPENDENT AUDIT OF data/features.parquet")
    print("=" * 78)

    if not FEATURES_FILE.exists():
        print(f"   [FAIL] {FEATURES_FILE} does not exist")
        return 2

    bf = load_build_module()
    panel = pd.read_parquet(PANEL).sort_values("date").reset_index(drop=True)
    p = panel.set_index("date")

    # ---- 1. definition hash, file vs current source -----------------------------------
    print("\n1. Definition hash: file on disk vs current build_features.py source")
    pf = pq.ParquetFile(FEATURES_FILE)
    md = pf.schema_arrow.metadata or {}
    stored = md.get(bf.DEFINITION_HASH_KEY, b"").decode()
    current = bf.definition_hash(BUILD_SRC)
    print(f"   stored in file : {stored or '<absent>'}")
    print(f"   current source : {current}")
    hash_ok = check(bool(stored) and stored == current,
                    "definition hash matches",
                    "" if stored == current else "the file was built by DIFFERENT formulas")
    if not hash_ok:
        print("\n   The parquet was not produced by the code now in the tree.")
        print("   Rebuild with: python scripts/build_features.py")

    feats = pf.read().to_pandas()          # from disk, in this process only
    f = feats.set_index("date")
    decisions = pd.date_range(bf.SAMPLE_START, bf.SAMPLE_END, freq="ME")

    # ---- 2. recompute through the build module and compare values ---------------------
    print("\n2. Values on disk vs a fresh recomputation from the panel")
    rebuilt, aux = bf.build_features(panel, decisions)
    rebuilt["label"] = bf.build_label(panel, decisions).to_numpy()
    rebuilt = rebuilt[["date"] + bf.FEATURES + ["label"]]
    same_shape = check(list(feats.columns) == list(rebuilt.columns) and len(feats) == len(rebuilt),
                       "schema and row count identical")
    if same_shape:
        for col in bf.FEATURES + ["label"]:
            a, b = feats[col].to_numpy(dtype=float), rebuilt[col].to_numpy(dtype=float)
            check(np.allclose(a, b, equal_nan=True, rtol=0, atol=0),
                  f"column {col} identical to recomputation")

    refs = aux["ref_months"]

    # ---- 3. third, independent recomputation in plain pandas --------------------------
    print("\n3. Third recomputation, done here in plain pandas, not by the build module")
    for M in ["1965-06-30", "1995-03-31", "2020-03-31", "2025-08-31"]:
        M = pd.Timestamp(M)
        r = refs["cape_z"].loc[M]
        hist = p["cape"].loc[:r].dropna()
        z = (p.at[r, "cape"] - hist.mean()) / hist.std(ddof=1)
        check(abs(z - f.at[M, "cape_z"]) < 1e-10,
              f"cape_z {M.date()}", f"hand {z:.10f} file {f.at[M, 'cape_z']:.10f}")
        r = refs["unrate_trend_12m"].loc[M]
        win = p["unrate"].loc[:r].tail(bf.WINDOW_MONTHS)
        t = p.at[r, "unrate"] - win.mean()
        check(abs(t - f.at[M, "unrate_trend_12m"]) < 1e-12 and len(win) == bf.WINDOW_MONTHS
              and win.index.max() <= r,
              f"unrate_trend_12m {M.date()}",
              f"hand {t:+.10f} file {f.at[M, 'unrate_trend_12m']:+.10f} "
              f"(window {win.index.min().date()}..{win.index.max().date()})")

    # ---- 4. availability, naive implementation, exhaustive ----------------------------
    print("\n4. Availability: no input may postdate its decision (naive recomputation)")
    work = panel.copy()
    work["_slope_lag"] = pd.concat(
        [p["dgs10_lag_days"], p["dtb3_lag_days"]], axis=1).max(axis=1).to_numpy()
    specs = [
        ("cape_z", "cape_lag_days", panel["cape"].notna()),
        ("yield_slope", "_slope_lag", panel["dgs10"].notna() & panel["dtb3"].notna()),
        ("momentum_12m", "sp500_index_lag_days", panel["sp500_index"].notna()),
        ("unrate_trend_12m", "unrate_lag_days", panel["unrate"].notna()),
    ]
    total = 0
    for feat, lagcol, valid in specs:
        naive = naive_available_ref(work, lagcol, valid, decisions)
        # Compare the instants themselves. A dtype-sensitive .equals() would report a
        # difference between datetime64[us] and [ns] that is not a difference in data;
        # int64 nanoseconds also make NaT compare equal to NaT, which it must here.
        as_ns = lambda x: x.to_numpy().astype("datetime64[ns]").astype("int64")
        agree = np.array_equal(as_ns(naive), as_ns(refs[feat]))
        lag = work.set_index("date")[lagcol].reindex(naive.to_numpy()).to_numpy()
        avail = pd.to_datetime(naive.to_numpy()) + pd.to_timedelta(lag, unit="D")
        viol = int((avail > decisions).sum())
        total += len(decisions)
        check(agree, f"{feat}: naive and vectorised availability maps agree")
        check(viol == 0, f"{feat}: 0 of {len(decisions)} inputs postdate their decision",
              f"max lead {int((pd.Series(avail) - pd.Series(decisions)).dt.days.max())} days")
    print(f"   {total} decision-point checks in total")

    # ---- 5. sample shape --------------------------------------------------------------
    print("\n5. Sample shape")
    complete = f[bf.FEATURES + ["label"]].notna().all(axis=1)
    check(len(f) == len(decisions), f"row count {len(f)} == {len(decisions)} decision points")
    check(int(complete.sum()) == bf.EXPECTED_COMPLETE_ROWS,
          f"complete rows == {bf.EXPECTED_COMPLETE_ROWS}", f"got {int(complete.sum())}")
    check(bool((f.index == f.index + pd.offsets.MonthEnd(0)).all()), "all dates are month-ends")
    check(f.index.is_monotonic_increasing and not f.index.duplicated().any(),
          "dates strictly increasing, no duplicates")
    for col in bf.FEATURES:
        check(int(f[col].isna().sum()) == 0, f"{col} has no nulls")
    lbl_null = list(f.index[f["label"].isna()])
    check(len(lbl_null) == 2, "label has exactly 2 nulls",
          ", ".join(str(x.date()) for x in lbl_null))
    for M in lbl_null:
        M12 = (M + pd.DateOffset(months=bf.LABEL_HORIZON)) + pd.offsets.MonthEnd(0)
        check(pd.isna(p["cpi"].get(M12, np.nan)),
              f"label null at {M.date()} is explained by null cpi at {M12.date()}")

    print("\n" + "=" * 78)
    if FAILURES:
        print(f"AUDIT FAILED - {len(FAILURES)} check(s):")
        for x in FAILURES:
            print(f"  - {x}")
        print("=" * 78)
        return 2
    print("AUDIT PASSED - every check above was made against the file on disk.")
    print("No label positive rate, class balance or feature-label association is reported.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
