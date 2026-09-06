# Claude-Ai-stockmarket-predicter

Testing whether valuation and macro signals predict S&P 500 direction. Walk-forward
validation, point-in-time data, pre-registered.

---

## Data layer

`scripts/build_market_panel.py` assembles a single month-end panel from free public
sources and writes `data/market_panel.parquet` plus a matching provenance record at
`data/market_panel_provenance.txt`.

This stage is **data only**. No features, no model, no predictions. Momentum, yield
spreads and trends are features and belong to the next stage. Nothing in the panel is
derived from anything else in the panel.

```bash
pip install pandas pyarrow xlrd requests
python scripts/build_market_panel.py
```

Downloads are cached under `data/.cache/` (git-ignored). Use `--no-cache` to force a
refetch, `--offline` to fail rather than hit the network.

### Sources

| Source | What | URL |
| --- | --- | --- |
| Robert Shiller, `ie_data.xls` | S&P Composite real total return index, CAPE | `https://shillerdata.com/` (current host; link scraped at runtime), fallback `http://www.econ.yale.edu/~shiller/data/ie_data.xls` |
| FRED `DGS10` | 10-year Treasury constant maturity yield, daily, % | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10` |
| FRED `DTB3` | 3-month Treasury bill, secondary market, daily, % | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3` |
| ALFRED `UNRATE` | Unemployment rate, **point-in-time first release** | `https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=UNRATE&vintage_date=...` |

Shiller's Yale page still serves `ie_data.xls`, but that copy is frozen at 2023-09.
Shiller's current distribution is `shillerdata.com`. The script scrapes the live link
from `shillerdata.com` and falls back to the Yale URL only if that fails; whichever was
used is recorded in the provenance file. Do not assume the Yale copy is current.

---

## PUBLICATION LAG ASSUMPTIONS

**These were fixed before the merge logic was written.** Every series carries an explicit
`*_lag_days` column in the panel. A value on row `date` must be treated as unknown until
`date + lag_days`. A signal formed at month-end M may only use values whose
`date + lag_days <= M`.

| Column | Reference period | First knowable | `lag_days` | Basis |
| --- | --- | --- | --- | --- |
| `sp500_index` | calendar month M | ~15th of M+1 | **15** | assumption (see below) |
| `cape` | calendar month M | ~15th of M+1 at the earliest | **15** | assumption, and a *lower bound* only |
| `dgs10` | last business day of M | next business day | **1** | assumption |
| `dtb3` | last business day of M | next business day | **1** | assumption |
| `unrate` | calendar month M | actual BLS release | **measured per row** | observed ALFRED vintage date |

### Why each lag is what it is

**`sp500_index` — 15 days, assumed.**
Shiller's monthly S&P price is the average of that month's daily closes, so the price
input is complete at month-end M. The column used here is his *Real Total Return Price*,
which deflates by CPI for month M. CPI for month M is published by BLS in the middle of
M+1, so the real series is not computable at month-end M. 15 calendar days is the
assumption.

**`cape` — 15 days, assumed, and a lower bound.**
CAPE is real price over trailing ten-year average real earnings. It inherits the CPI lag
above. It also depends on S&P 500 reported earnings, which arrive one to two quarters
late and which Shiller *interpolates* for recent months from partial data. The
interpolated tail is revised as actual earnings land. 15 days is therefore the earliest
CAPE could be known, not the point at which it stops changing. Treat recent-month CAPE as
provisional.

**`dgs10`, `dtb3` — 1 day, assumed.**
Both are daily market rates. The panel takes the last observation on or before month-end.
That level is set at the close of the last business day of M and is published by the H.15
release on the next business day. These series are essentially not revised, so they are
the only two columns here that are clean point-in-time by construction. One business day
is approximated as 1 calendar day, which is conservative in the right direction only if
you also respect the `>= ` comparison above; when month-end falls on a Friday the true
lag is 3 calendar days. If that edge matters to your study, widen this to 3.

**`unrate` — measured, not assumed.**
This is the series the discipline is actually about, and it is handled with real vintage
data rather than an assumption. See below.

---

## POINT-IN-TIME STATUS BY COLUMN

Read this before using the panel. The columns are not equally trustworthy.

| Column | Point-in-time? | Lookahead risk |
| --- | --- | --- |
| `unrate` | **YES** for `date >= 1960-02-29` | none — first published value, real release date |
| `unrate` | **NO** for `date < 1960-02-29` | revised — predates ALFRED's archive |
| `dgs10` | effectively yes | negligible — market rates, not revised |
| `dtb3` | effectively yes | negligible — market rates, not revised |
| `sp500_index` | **NO** | revised; see below |
| `cape` | **NO** | revised; earnings tail interpolated; see below |

### `unrate` is genuine point-in-time data

FRED's `UNRATE` is the *revised* series: the value it shows for a given month is not the
number that was published that month. Seasonal factors are re-estimated every January and
restate years of history at once. Using it as if it were live is lookahead contamination.

This panel does not use it. It uses **ALFRED vintage data**. ALFRED publishes the real
vintage dates for `UNRATE` — 800 of them, from 1960-03-15 to the present — and the script:

1. scrapes the true vintage-date list from ALFRED's download page,
2. fetches each vintage,
3. assigns to each reference month the value from the **earliest vintage in which that
   month appears**, i.e. its first published value,
4. records that vintage date in `unrate_release_date`, from which `unrate_lag_days` is
   measured directly.

So `unrate_lag_days` is an observation, not an assumption. Empirically it runs from about
two to six weeks after month-end.

**The pre-1960 exception.** ALFRED's earliest `UNRATE` vintage is 1960-03-15, whose oldest
observation is 1948-01. Reference months before 1960-02 were already published before
ALFRED's archive begins, so no first-release value exists for them. Those rows carry the
values as they stood in the 1960-03-15 vintage, `unrate_release_date = 1960-03-15`, and
`unrate_is_first_release = False`. **Those rows are revised data and are
lookahead-contaminated.** Filter on `unrate_is_first_release` if that matters.

**If ALFRED is unreachable** the script falls back to FRED's revised `UNRATE`, sets
`unrate_is_first_release = False` on every row, and stamps
`unrate_point_in_time = "NO — REVISED SERIES, LOOKAHEAD-CONTAMINATED"` into both the
parquet metadata and the provenance file. It does not silently substitute the revised
series.

### `sp500_index` and `cape` are revised data

Shiller publishes no vintage archive, so no point-in-time reconstruction is possible from
this source. Both columns are as-of the pull date. Two distinct effects, measured by
diffing the 2023-09 Yale release against the 2024-09 release:

- **Rebasing.** *Real Total Return Price* is quoted in the purchasing power of the file's
  final month, so a new release rescales the entire column by one constant — +2.8% between
  those two releases. Levels are therefore **not comparable across pulls**. Month-over-month
  ratios are, because the constant cancels.
- **Revision of the tail.** After removing that constant, exactly one month differed by
  more than 0.5%: the final, partial month of the older file. Shiller's history is stable;
  the newest one or two months are provisional.

Practical consequence: use `sp500_index` for returns, not for levels, and treat the last
two rows of any pull as provisional.

---

## Columns

Month-end aligned, monthly frequency, one row per calendar month over the union of all
series. Rows outside a given series' span are null. **Nulls are never filled or
interpolated.**

| Column | Type | Meaning |
| --- | --- | --- |
| `date` | timestamp | month end |
| `sp500_index` | float | Shiller real total return index, month M |
| `cape` | float | Shiller CAPE (P/E10), month M |
| `dgs10` | float | 10y Treasury yield, %, last obs on or before month end |
| `dtb3` | float | 3m Treasury bill, %, last obs on or before month end |
| `unrate` | float | unemployment rate, %, first published value for month M |
| `sp500_index_lag_days` | int | 15 |
| `cape_lag_days` | int | 15 |
| `dgs10_lag_days` | int | 1 |
| `dtb3_lag_days` | int | 1 |
| `unrate_lag_days` | int | measured: `unrate_release_date - date` |
| `unrate_release_date` | timestamp | ALFRED vintage in which this month first appeared |
| `unrate_is_first_release` | bool | False = revised value, contaminated |

Series start dates are not hardcoded. Each series begins as early as the source allows and
the actual observed start is recorded per series in the parquet metadata and in
`data/market_panel_provenance.txt`.

### Month-end alignment

`dgs10` and `dtb3` are daily. The panel takes the **last available observation within
month M**. This is selection, not filling: if month-end falls on a weekend or holiday the
value is the preceding business day's, which is the correct month-end level. If a month
contains no observation at all the cell is null.

---

## Known gaps and anomalies

Found in the data, not worked around. All of these are also written into the provenance
file on every build.

- **`unrate` has no value for 2025-10.** The US government shutdown meant no household
  survey was conducted for October 2025, so BLS never published a rate for it. FRED's own
  series is null there. The panel leaves it null and does not fill it.
- **`unrate` for 2025-09 has a 51-day lag**, because the same shutdown pushed its release
  to 2025-11-20. That is the measured release date, not an estimate.
- **Seven early-1960s rows have `unrate_lag_days <= 0`.** Not a bug. In that era BLS
  published the Monthly Report on the Labor Force *within* the reference month, so the
  number really was knowable before month end. A non-positive lag means no embargo applies.
- **The trailing row may be a partial month.** If the build runs mid-month, the last row is
  dated month-end but its values are month-to-date: Shiller's file carries a partial current
  month, and the daily yields stop at the last trading day so far. The row is kept rather
  than silently dropped, and the build prints a warning and records it in the provenance.
  Pass `--drop-partial-month` to exclude it.

Measured `unrate` publication lag across the 798 first-release rows: min -2, median 5,
max 51 days.

---

## LEAKAGE RULES FOR THE FEATURE LAYER — BINDING

Three rules govern how this panel may be consumed. They are binding: a feature that
breaks one of them is wrong even if it backtests well, and especially if it backtests
well. Each is stated with the decision taken and the measurement that justifies it.

### Rule 1 — CAPE normalisation uses an EXPANDING window, never full-sample

**Decision.** Any z-score, percentile or rank of `cape` is computed over an expanding
window using only data up to and including that date, with an explicit minimum warm-up.
Full-sample `mean()`, `std()`, `rank()` or `quantile()` over the whole column is
forbidden. The warm-up period produces leading NaN; leave it NaN, do not backfill.

```python
# CORRECT - expanding, past-only, explicit warm-up
mu, sd = cape.expanding(min_periods=120).mean(), cape.expanding(min_periods=120).std()
z = (cape - mu) / sd

# WRONG - encodes the whole century's distribution into every historical row
z = (cape - cape.mean()) / cape.std()
```

**Why, measured on this panel.** CAPE's mean by era: 14.87 (1881-1929), 14.91
(1930-1979), 21.15 (1980-2009), **28.85 (2010-2026)**. The full-sample mean of 17.78 sits
above the first two eras and far below the last, so a full-sample z-score tells every
pre-1980 row that the market is expensive relative to a future it could not have seen,
and every post-1980 row the reverse.

Full-sample and expanding z differ by **0.58 sd on average and up to 3.00 sd**, and they
**disagree on the sign of the signal in 13.0% of months** — 123 of 829 pre-1960 months
flip between "cheap" and "expensive". A sign flip is not a rounding difference; it
reverses the trade.

### Rule 2 — unrate publication lag: use the lag column, not a fixed shift

**Decision.** At month-end M, the usable unemployment observation is the newest row
satisfying `date + unrate_lag_days <= M`. Do **not** use a fixed `.shift(1)`.

```python
avail = panel[["date", "unrate", "unrate_lag_days"]].dropna(subset=["unrate"]).copy()
avail["available_on"] = avail["date"] + pd.to_timedelta(avail["unrate_lag_days"], unit="D")
avail = avail.sort_values("available_on")

known = pd.merge_asof(                      # newest observation public by each month end
    panel[["date"]].sort_values("date"),
    avail[["available_on", "date", "unrate"]].rename(columns={"date": "unrate_ref_month"}),
    left_on="date", right_on="available_on", direction="backward",
)
```

**Why not `shift(1)`.** "The newest known value at month-end M is M-1" is a good
description of the typical month and a bad implementation. Checked against the measured
lag column, a blanket `shift(1)` disagrees in **154 of 1868 months**:

| | count | what happens |
| --- | --- | --- |
| **Leaks** | 146 | uses a value that was not yet published at M |
| **Too conservative** | 7 | a newer month was already public and is discarded |
| **Loses data** | 1 | yields NaN where a real value was available |

The leaks are 145 pre-1960 rows plus one modern case: at **2025-10-31 a `shift(1)` hands
you September 2025's rate, which was not published until 2025-11-20** — three weeks of
future knowledge, during the exact stretch where the labour market was the story.

The 7 conservative cases are the early-1960s months when BLS published within the
reference month itself. The single lost value is 2025-11-30, where `shift(1)` returns the
never-published October while September had been public since 2025-11-20.

The availability rule also **subsumes `unrate_is_first_release`**: the 145 pre-1960 rows
carry an availability date of 1960-03-15, so the rule excludes precisely the
revised, contaminated era without a second filter.

### Rule 3 — the 2025-10 hole is permanent: propagate NaN, never close it

`unrate` has no value for October 2025 and never will. No household survey was conducted,
BLS published no rate, and none of ALFRED's 799 vintages carries one. It is not missing
data to be recovered; it is a month that does not exist.

**Decision, explicitly:**

1. **Never** `fillna`, `interpolate`, `ffill` or `bfill` across it.
2. **Never** `dropna` on `unrate` before computing a window. Dropping splices September
   directly to November and silently shifts every subsequent window by one month — the
   most dangerous option, because the output looks complete.
3. Any windowed feature whose window covers 2025-10 is **NaN**. Use
   `min_periods == window` so a single missing observation propagates.
4. Point-in-time *level* features are unaffected: the Rule 2 availability rule simply
   returns the newest month that does exist.

```python
# NaN propagates: min_periods == window means one missing obs voids the window
feat = unrate_by_ref_month.rolling(12, min_periods=12).mean()
```

**Order of operations matters, and this is the trap.** Applying Rule 2 first and then
computing windows *hides the hole*. After the availability shift there are **zero NaNs**
around October 2025 — the gap becomes a stale repeat:

```
month end     newest known ref month   value   staleness
2025-09-30 -> 2025-08                  4.3     1 month
2025-10-31 -> 2025-08                  4.3     2 months   <-- gap invisible
2025-11-30 -> 2025-09                  4.4     2 months   <-- gap invisible
2025-12-31 -> 2025-11                  4.6     1 month
```

So: **compute windowed features on the reference-month series, which carries the NaN, and
apply the availability shift to the resulting feature** — not the other way round. A
feature layer that shifts first has nothing left to propagate. Carrying an explicit
staleness column is recommended, so a stale repeat is visible rather than assumed fresh.

**Blast radius.** The panel ends 2026-08-31, only 10 rows after the gap. So any `unrate`
window of 11 months or more is **NaN for the entire remainder of the panel**:

| window | months NaN | span |
| --- | --- | --- |
| 3-month | 3 | 2025-10-31 .. 2025-12-31 |
| 6-month | 6 | 2025-10-31 .. 2026-03-31 |
| 12-month | 11 | 2025-10-31 .. 2026-08-31 (all remaining rows) |

That is the correct outcome, not a bug to engineer around. A 12-month unemployment
feature has no valid values after September 2025, and any backtest reporting results
there is reporting something it invented.

---

## Provenance

Every build writes a header block into the parquet file's schema metadata and a
human-readable twin at `data/market_panel_provenance.txt`, containing source URLs, the
pull timestamp, row count, per-series observed date range and null count, every lag
assumption above, and the point-in-time status of each column.
