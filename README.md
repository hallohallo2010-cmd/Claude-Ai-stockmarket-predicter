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
| Robert Shiller, `ie_data.xls` | S&P Composite real total return index, CAPE, CPI |","cpi-src `https://shillerdata.com/` (current host; link scraped at runtime), fallback `http://www.econ.yale.edu/~shiller/data/ie_data.xls` |
| FRED `DGS10` | 10-year Treasury constant maturity yield, daily, % | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10` |
| FRED `DTB3` | 3-month Treasury bill, secondary market, daily, % | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTB3` |
| ALFRED `UNRATE` | Unemployment rate, **point-in-time first release** | `https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=UNRATE&vintage_date=...` |
| FRED `CPIAUCNS` | CPI-U all items NSA — the authority used to verify `cpi` by value and to decide which months BLS actually published | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS` |

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
| `cpi` | calendar month M | ~10th-15th of M+1 | **15** | assumption |

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

**`cpi` — 15 days, assumed.**
BLS publishes CPI for month M in the middle of M+1, typically between the 10th and the
15th. So `cpi` for month M is not knowable at month-end M. This is the same release that
`sp500_index` and `cape` depend on, and the three carry the same 15-day assumption.

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
| `cpi` | effectively yes, once published | none — NSA CPI is **not revised**; see below |
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

### `cpi` is verified by value, and is not revised

`cpi` is Shiller's column 4, but it is not trusted on the strength of its header. Every
build re-checks it against FRED's `CPIAUCNS`, the same CPI-U all-items NSA series: on the
current build **1362 of 1362 shared months match exactly, maximum difference 0.000000**.
The 504 months before 1913-01 predate `CPIAUCNS` and are Shiller's splice of the
Warren-Pearson index; they cannot be verified this way and are recorded as such.

Unlike `unrate`, this series is **not revised**. Comparing ALFRED vintages 2015-01-16 and
2026-08-12 for `CPIAUCNS`, **0 of 1224** overlapping observations changed. The
*seasonally adjusted* series behaves differently — **60 of 816** changed over the same
pair, by up to 0.462 — because its seasonal factors are re-estimated every year. Shiller
uses the NSA series, so `cpi` is final on publication and carries no revision risk. It is
still not knowable at month-end M; that is what the 15-day lag is for.

**Shiller's own estimates are removed.** The workbook carries values BLS has not
published, and says so in a footnote on the CPI column. Those are filled numbers, and this
panel does not fill, so any month at or after 1913-01 that `CPIAUCNS` does not carry is
set null. On the current build that removed three: **2025-10, 2026-08 and 2026-09**.
`sp500_index` and `cape` are nulled at those same months, because they are deflated by
that estimate — see **2025-10 is a single-cause gap** below.

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
| `cpi` | float | CPI-U, all items, NSA, index 1982-1984 = 100, month M |
| `sp500_index_lag_days` | int | 15 |
| `cape_lag_days` | int | 15 |
| `dgs10_lag_days` | int | 1 |
| `dtb3_lag_days` | int | 1 |
| `unrate_lag_days` | int | measured: `unrate_release_date - date` |
| `cpi_lag_days` | int | 15 |
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
- **2025-10 is a single-cause gap.** See the section below.
- **The trailing row may be a partial month.** If the build runs mid-month, the last row is
  dated month-end but its values are month-to-date: Shiller's file carries a partial current
  month, and the daily yields stop at the last trading day so far. The row is kept rather
  than silently dropped, and the build prints a warning and records it in the provenance.
  Pass `--drop-partial-month` to exclude it.

Measured `unrate` publication lag across the 798 first-release rows: min -2, median 5,
max 51 days.

---

## 2025-10 IS A SINGLE-CAUSE GAP

One event — the US government shutdown — removed October 2025 from this panel across
**every column that depends on a statistical agency**:

| Column | Why it is null at 2025-10 |
| --- | --- |
| `cpi` | BLS cancelled the October 2025 CPI release |
| `unrate` | no household survey was conducted for October 2025 |
| `sp500_index` | **derived** — Shiller's real series is deflated by the missing CPI |
| `cape` | **derived** — same missing deflator |

`dgs10` and `dtb3` are unaffected and complete at 2025-10: Treasury markets traded
throughout.

**These are not four independent gaps and must not be treated as such.** Any imputation
that borrows across them is circular, because the thing missing from each is the same
thing. Nor is their agreement evidence of anything: a feature layer that sees all four
columns fail at once is looking at one absence counted four times, not at four
corroborating signals. Windows spanning 2025-10 fail together rather than degrading one
at a time.

### Why `sp500_index` and `cape` are nulled there too

Shiller's `sp500_index` and `cape` are **real** series: he divides nominal figures by the
CPI of that same month. Where that deflator was his own estimate rather than a BLS
publication, the real value is a fabricated number wearing an observation's clothes. A
real series divided by a fabricated deflator is fabricated, so it is nulled on exactly the
principle already applied to the CPI estimates themselves.

This was measured read-only before it was done. Nulling `sp500_index` and `cape` at those
months costs the pre-registered study sample (1962-02-28 to 2025-08-31) **zero labelled
observations**: 761 usable before, 761 after. Every feature window looks backward and
cannot reach 2025-10 from a decision point at or before 2025-08-31, and the single label
endpoint that does reach it — the 12-month label at decision 2024-10-31 — was already void
because `cpi` is null at that same endpoint.

Two observations in the study sample are unusable, both from null `cpi` at a label
endpoint, neither caused by this change:

| Decision month | Label endpoint M+12 | Status |
| --- | --- | --- |
| 2024-10-31 | 2025-10 | **permanent** — October 2025 CPI will never exist |
| 2025-08-31 | 2026-08 | **temporary** — returns once BLS publishes August 2026 CPI |

One caveat worth recording: this zero holds under the **amended** excess-return label,
which needs `cpi` at both endpoints. Under the original superseded label, which needed no
CPI, the same change would have cost **one** observation (2024-10-31). The answer depends
on which label is in force, and the amended one is.

---

## PRE-REGISTRATION

**Registered 2026-09-06. Panel build `data/market_panel.parquet`, 1868 rows,
1871-01-31 to 2026-08-31.** This section fixes the study design before any feature has
been computed and before any label has been examined. Everything below is binding.
Changes require a dated entry in the Amendments log at the end of this section, written
*before* the change is made. An undated change is a violation of the design, not a
refinement of it.

### 1. Feature set — fixed now

Four features. No more, no fewer.

| Feature | Definition | Window | Panel columns used |
| --- | --- | --- | --- |
| `cape_z` | Expanding z-score of CAPE: value minus expanding mean, over expanding standard deviation | Expanding, minimum warm-up 120 months | `cape` |
| `yield_slope` | 10-year yield minus 3-month bill, in percentage points | Contemporaneous, no window | `dgs10`, `dtb3` |
| `mom_12m` | 12-month price momentum: log change in the total return index over 12 months | 12 months | `sp500_index` |
| `unrate_trend_12m` | Unemployment trend: latest rate minus its 12-month trailing mean | 12 months | `unrate`, `unrate_lag_days` |

No interaction terms, no polynomial expansions, no alternative windows, no regime dummies,
no additional series. If a fifth feature or a different window later looks necessary, it
is a dated amendment and is reported alongside the original specification, never
silently in place of it.

**Uniform availability rule.** A decision is taken at month-end M. A feature may use an
observation from month k only if `k + lag_days <= M`, using each series' own lag column.
Applied uniformly this places every feature at month M−1 or older. For `yield_slope` this
is deliberately conservative: a trader acting at the month-end close can observe that
day's yields, but the published series carries a one-day lag, and one uniform rule across
all four features is worth more than one month of extra signal on one of them.

### 2. Label

> **Superseded in part by Amendment 1 (2026-09-06):** the label is now the sign of the
> 12-month **excess** return over 3-month bills. The original specification is preserved
> verbatim below for the record; the overlapping-window discussion still applies in full.

**Direction of the S&P 500 total return index over the following 12 months.** At decision
month-end M the label is positive if `sp500_index` at M+12 exceeds `sp500_index` at M, and
negative otherwise. Binary. Ties, which do not occur in practice, count as negative.

`sp500_index` is Shiller's **real** (CPI-deflated) total return index, so this is the
direction of real total return, not nominal. This is a deliberate choice and it changes
the label in high-inflation stretches, where nominal is positive and real is not. The
rebasing constant identified in the provenance cancels in the ratio, so the label is
stable across pulls of the source file.

**The overlapping-window problem, stated explicitly.** Consecutive monthly observations
share 11 of the 12 months of their label horizon. The labels are therefore massively
autocorrelated and monthly rows are nowhere near independent. The modelling sample of
**763 month-ends contains roughly 64 independent 12-month blocks**. Three consequences,
fixed now:

- All uncertainty estimates use a moving-block bootstrap with block length of at least 12
  months, or Newey-West standard errors with 11 lags. Treating months as independent
  would understate standard errors by roughly the square root of 12 and is prohibited.
- No random k-fold cross-validation, ever. Random folds place a row's overlapping
  neighbours on both sides of the split, which leaks the label directly.
- Power is governed by ~64 effective observations, not 763. The study is small. Claims
  will be sized to that, and a result that requires 763 independent observations to reach
  significance is not a result.

### 3. Baseline

**Always predict up.** Its predicted probability is the positive rate **of the training
window only**, recomputed at each walk-forward step from data available at that step.
Never the full-sample positive rate.

This document deliberately **does not report the sample's positive rate**, because
printing it here would fix in the analyst's mind the very quantity the baseline is
supposed to estimate separately within each fold.

Equities rise over most 12-month windows, so this baseline is strong on accuracy and hard
to beat on that metric. Beating it is the minimum bar for the study to have found
anything at all, not the goal.

### 4. Split — walk-forward expanding window

| Period | Span | Month-ends | Use |
| --- | --- | --- | --- |
| **Development** | 1962-02-28 to 2009-12-31 | 575 | All model selection, tuning, feature checking |
| **Holdout** | 2010-01-31 to 2025-08-31 | 188 (~15.7 blocks) | Read once |

The sample starts **1962-02-28** because `dgs10` begins 1962-01 and `yield_slope` needs
it; that is the binding constraint, not a choice. It ends **2025-08-31** because a
12-month forward label requires index data through 2026-08, the last month in the panel.

Training is an **expanding** window: each step trains on everything from 1962-02-28 up to
the step's cutoff and predicts forward. The window never slides or resets.

**Embargo.** At every step the training window must end at least **12 months** before the
test point. Without the embargo the last year of training labels reaches forward into the
test period, which is the same leak as random folds wearing a different hat.

**The holdout is calendar years 2010 through 2025, and it is read once.** It is read after
the model class, hyperparameters, and all feature transforms are frozen from development
alone. One read, one number. If the result fails the criterion below, that is the result
of the study. It will not be re-tuned and re-read, and the holdout will not be reopened to
diagnose the failure.

**Known limitation of this holdout, recorded now so it cannot become an excuse later.**
2010-2025 is one long expansion plus a single sharp crash, roughly 15.7 independent
blocks, at persistently high CAPE. It is a narrow regime and a single read of it has
limited power. This is accepted as the cost of having a genuine holdout at all.

### 5. Success criterion — stated now

**Primary.** On the holdout, the model's **Brier score must be at least 10% lower in
relative terms** than the baseline's Brier score, where the baseline uses each step's
training-window positive rate. Brier, not accuracy: the label is imbalanced and the
constant baseline scores well on accuracy, so accuracy cannot distinguish a real signal
from the base rate.

**Robustness.** The improvement must survive dropping calendar year 2020. A result that
exists only because of the COVID crash and recovery is a result about one episode.

**Uncertainty.** Reported as a moving-block bootstrap confidence interval, block length 12
months, 10,000 resamples.

**Secondary, reporting only, never promoted to primary:** accuracy, AUC, log loss, and the
Sharpe ratio of any trading rule derived from the signal. These are described in the
write-up. They cannot be substituted for the Brier criterion if the Brier criterion fails.

**Declared expected outcome.** The prior is that this study finds no useful predictive
signal. A negative result is the anticipated finding and will be reported plainly as
such. The purpose of fixing the criterion now is that "no signal" remains a reportable
outcome rather than an invitation to keep searching.

### 6. The `unrate` coverage limitation, and the decision not to shorten the window

`unrate` has a permanent hole at 2025-10, which propagates through the 12-month window of
`unrate_trend_12m`. Measured against this panel, that feature is null at decision
month-ends **2025-12-31 through 2026-08-31** — and by the same arithmetic will remain null
until **2026-11-30**, the first decision point whose 12-month window clears the hole.

**The labelled modelling sample ends 2025-08-31, so the gap costs the study exactly zero
observations.** It costs live deployability: from 2025-12 to 2026-11 the model cannot
produce a signal at all, because one of its four inputs does not exist.

**Decision: the 12-month window is not shortened.** Cutting `unrate_trend_12m` to a 3- or
6-month window would restore live coverage sooner and is rejected. The window length was
chosen on design grounds before the gap was considered, and changing it now would be a
post-hoc specification change motivated purely by an inconvenience in recent data — the
precise practice this pre-registration exists to prevent. That the change would be
convenient is what makes it disqualifying.

The consequence is accepted: **no live signal for those months.** A model that abstains
when an input is genuinely missing is behaving correctly. Filling the gap, interpolating
it, or shrinking the window to step around it would all produce a signal in months where
no unemployment trend is knowable, which is a fabricated number, not a prediction.

If a shorter window is genuinely wanted later, it is a new feature under a dated
amendment, pre-registered with its own criterion, and reported **alongside** the 12-month
specification rather than replacing it.

### Amendments

Any change to sections 1-6 is recorded here with a date, the change, and the reason,
written before the change is made.

---

#### Amendment 1 — 2026-09-06

Made before any feature or label has been computed. Sections 1-6 above are unedited apart
from a superseding banner on section 2; nothing has been deleted.

**1.1 — Reported: which Shiller column `sp500_index` was built from.**

`sp500_index` is Shiller's **Real Total Return Price**, column 9 of the `Data` sheet in
`ie_data.xls`. Confirmed by exact match on all 1868 rows, not by reading the header alone.
It is **real** — CPI-deflated, quoted in the purchasing power of the source file's final
month — and it is a **total return** index, with dividends reinvested. The three
candidates are easy to confuse because two of them are indexed to the same base and are
identical in the first row:

| Workbook column | Series | Value at 1871-01 | Value at 2026-08 |
| --- | --- | --- | --- |
| col 1 | S&P Comp. P, nominal price | 4.44 | 7,711.32 |
| col 7 | Real Price, no dividends | 118.94 | 7,711.13 |
| **col 9** | **Real Total Return Price** | **118.94** | **5,260,117.47** |

Columns 7 and 9 agree at 1871-01 and differ by a factor of **682** by 2026-08. That factor
is the reinvested dividends. A study that believed it held column 7 while holding column 9
would be wrong about its own dependent variable by two and a half orders of magnitude.

**1.2 — Label changed to the 12-month excess return over 3-month bills.**

*Adopted.* At decision month-end M the label is **positive if the 12-month compounded
total return on the equity index exceeds the 12-month compounded return from rolling
3-month bills over the same window**, and negative otherwise. Ties count as negative.

Rationale, as directed: the decision this model informs is **stocks versus bills**. The
sign of the equity return alone does not answer that question — a year in which equities
return 3% while bills pay 5% is a positive label under the original specification and a
loss under the only decision the model is for.

Conventions fixed now, so they cannot be chosen to taste later:

- The bill leg compounds `dtb3` observed at each month-end within the window, treating the
  quoted annualised rate as an effective annual rate: a monthly factor of
  (1 + dtb3/100) raised to the power 1/12, multiplied across the 12 months. `DTB3` is a
  secondary-market discount-basis quote, so this is an approximation of a true rolling
  bill return; it is fixed here as the definition rather than tuned later.
- The bill leg deliberately uses `dtb3` values from months M+1 through M+12, which are in
  the future relative to the decision point. This is correct: it is part of the *label*,
  not a feature. It must not be "corrected" as a leak.

**Blocking prerequisite: the two legs are currently in different units.** `sp500_index` is
a **real** index (item 1.1); `dtb3` is a **nominal** yield. Subtracting a nominal bill
return from a real equity return subtracts the inflation of the window from one leg only.
Over the modelling sample, 12-month inflation averaged 3.9%, exceeded 5% in 174 of 763
months, and peaked at **14.8% in the year to 1979-03**. A mismatched excess return would
therefore mislabel the late 1970s and early 1980s wholesale — precisely the stretch where
the stocks-versus-bills choice mattered most, and precisely where a real-versus-nominal
error is invisible because the answer looks plausible either way.

The sign of an excess return is invariant to deflation applied consistently to both legs:
dividing both gross returns by the same inflation factor divides their difference by a
positive number and cannot change its sign. So a both-real label and a both-nominal label
are identical, and either is acceptable. Both require **CPI**, which the panel does not
carry: it is available as column 4 of the same Shiller workbook already downloaded.

Accordingly: **`cpi` must be added to the panel as a data-layer change, with its own
provenance and lag entry, before the label is computed.** Its publication lag is the same
mid-following-month CPI release already documented for `sp500_index` and `cape`. This
amendment adopts the label; the data layer must be extended before the label can be
computed correctly. The feature layer does not begin until that is done.

**1.3 — Effective sample size, recorded as the study's primary limitation.**

Because the label horizon is 12 months and observations are monthly, independent
non-overlapping observations are the month count divided by 12, rounded **down** to
complete blocks. These exact counts supersede the "roughly 64" approximation given in
section 2, which rounded rather than truncated:

| Split | Month-ends | **Independent non-overlapping 12-month observations** |
| --- | --- | --- |
| Development, 1962-02-28 to 2009-12-31 | 575 | **47** |
| Holdout, 2010-01-31 to 2025-08-31 | 188 | **15** |
| Total | 763 | **63** |

**This is the primary limitation of the study, ranking above data quality, feature choice
and model class.** The holdout contains **fifteen** independent observations. Fifteen
observations can distinguish a large effect from nothing; they cannot rank models, resolve
a modest edge, or support a confident negative. Four features fitted against 47
independent development observations is already a regime in which overfitting is the
default outcome rather than a risk to be managed.

Every reported result carries this number. Any conclusion phrased as though the study had
763 observations is a misreading of the design, and the write-up states the effective
count wherever a sample size is quoted.

**1.4 — The 1962 start, and what dropping `yield_slope` would recover.**

The modelling sample starts **1962-02-28 solely because `dgs10` begins 1962-01**, which
`yield_slope` requires. It is a consequence of data availability, not a design choice: the
equity series reaches back to 1871 and CAPE to 1881.

What dropping `yield_slope` would actually recover, measured:

| Feature set | Sample start | Month-ends | Independent obs |
| --- | --- | --- | --- |
| A — all four features | 1962-02-28 | 763 | 63 |
| B — drop `yield_slope` | 1960-03-31 | 786 | **65** |
| C — drop `yield_slope` **and** `unrate_trend_12m` | 1891-01-31 | 1616 | **134** |

Dropping `yield_slope` alone buys **two** additional independent observations, 63 to 65.
It recovers essentially nothing, because the binding constraint moves straight to
`unrate_trend_12m`, whose usable history starts in 1960. The century of extra data is
locked behind the **unemployment** feature, not the yield curve. Sacrificing the yield
curve to reach it would be a bad trade made on a false premise.

Scenario C — dropping both macro features and running CAPE and momentum from 1891 — would
more than double the independent observations, from 63 to 134, and is the only variant
that materially addresses the limitation in 1.3. It is **not adopted**: it abandons the
macro half of the study's premise, and choosing it now, before any result exists, would
still be a design chosen for sample size rather than for the question.

A further point in favour of A: starting in 1962 means every 12-month `unrate` window in
the sample lies entirely within the ALFRED first-release era, which begins 1960-02. Under
scenario B the earliest windows would reach back into the pre-1960 revised rows and
quietly reintroduce the lookahead contamination the panel was built to eliminate.

**Decision: `yield_slope` is retained and the sample starts 1962-02-28. This is final and
will not be revisited after results are seen.** Scenario C is recorded here as a
considered and rejected alternative so that adopting it later would be visible as what it
would be.

---

**Prerequisite of Amendment 1.2, satisfied 2026-09-06.** `cpi` has been added to the
panel as a data-layer change: CPI-U all items NSA, month-end aligned, verified by value
against `CPIAUCNS` (1362 of 1362 months exact), with Shiller's estimates for unpublished
months nulled and a 15-day publication lag column. The blocker on computing the amended
excess-return label is cleared. Note for the feature layer: `cpi` is null at 2025-10, so a
12-month excess-return label whose window spans that month cannot be deflated and must be
NaN under Rule 3 — the same treatment as `unrate`.

---

#### Amendment 2 — 2026-09-07 — ~~WITHDRAWN~~

> **WITHDRAWN 2026-09-07, in full, before any model or result existed.**
>
> This amendment recorded an **error in the instruction that prompted it**, not a design
> decision. `unrate_trend_12m` was never intended to change; the feature-layer
> specification restated it incorrectly, and the amendment faithfully wrote that mistake
> into the log. **The pre-registration governs**, so section 1's definition — latest
> available rate minus its 12-month trailing mean — stands unaltered and section 1 has
> been restored to its original wording.
>
> `data/features.parquet` has been rebuilt on the pre-registered definition and the full
> audit re-run: **761** complete rows, unchanged, and 0 availability violations across
> 3,052 checks.
>
> The entry is kept rather than deleted, because a withdrawn amendment is part of the
> record: it shows what was proposed, that it was acted on, and that it was reversed
> before any result could depend on it. Its text is preserved verbatim below and is **no
> longer operative**. Item 2.2 concerned `momentum_12m` as well and remains true of that
> feature; it is re-recorded as Amendment 3.2 so it does not lapse with this withdrawal.

*Original text, no longer in force:*

Made when the feature layer was built, before any model exists.

**2.1 — `unrate_trend_12m` redefined.** From *latest rate minus its 12-month trailing
mean* to the **12-month change**: the latest available rate minus the rate twelve months
earlier. Directed as part of the feature-layer specification. Section 1's original wording
is preserved above with a pointer to this entry.

Both definitions read the same 12-month window and both are void when that window spans
the 2025-10 gap, so coverage is unchanged: the study sample still holds **761** complete
observations, the figure computed read-only before the feature layer was written. The
change alters what the feature measures — a difference between two endpoints rather than a
deviation from a mean — not which rows exist.

**2.2 — Window-span convention recorded.** `momentum_12m` and `unrate_trend_12m` are
two-endpoint formulas, but Rule 4 says a window *spanning* the gap is void, so both are
computed as NaN unless all 13 months of their span are present. This is stricter than the
arithmetic needs. It costs nothing here — no in-sample feature window reaches 2025-10, the
newest reference month any feature uses being 2025-07 — and it is recorded so the stricter
reading is a stated choice rather than an accident.

---

#### Amendment 3 — 2026-09-07

Written before any model exists and before any result has been produced.

**3.1 — The sample refresh rule, fixed now.**

August 2026 CPI is due within days. When BLS publishes it, a panel rebuild will restore
`cpi` and `sp500_index` at 2026-08, which restores the label at decision **2025-08-31**
and takes the study sample from **761** usable rows to **762**. The extra row falls in the
holdout, taking it from 188 month-ends to 189 and from 15 to 15 independent
non-overlapping observations — the block count does not change.

**The sample may be refreshed exactly once, under all of these conditions:**

1. **Once only.** This single refresh is authorised and no other. Later CPI releases,
   later Shiller releases, and any change to the sample end date of 2025-08-31 are **not**
   authorised and would each need their own amendment, written before the fact.
2. **Before the model is frozen.** The refresh must happen while the model class,
   hyperparameters and feature transforms are still open, so it cannot be a response to
   anything the model has done.
3. **Never after the holdout is read.** Once the holdout is read the sample is closed
   permanently. If the holdout is read before the refresh happens, **the refresh is
   forfeited** and the 761-row sample stands as final.
4. **Whole rebuild, not a hand-edit.** `build_market_panel.py` and `build_features.py` are
   re-run end to end. No row is patched in place.
5. **Existing rows must not move.** The refresh is expected to *add* a row without
   perturbing the 761 already there: Shiller's revisions are confined to the final month
   or two of each release, far after the 2025-08-31 sample end, and his rebasing constant
   cancels in both the momentum ratio and the label ratio. The refresh must verify this —
   if any existing row's features or label change materially, stop and report rather than
   proceeding.
6. **Dated here when performed**, with the resulting row count.

**Why this is fixed now rather than decided later.** A sample that grows after results are
known is a researcher degree of freedom. One extra observation is exactly the size of
thing that can move a marginal result across a threshold, and an analyst who has seen a
near-miss and then chooses to refresh is making a different decision from one who fixed
the rule in advance — even with identical arithmetic and entirely honest intent. Writing
the rule down before any result exists removes the choice, which is the only reliable way
to remove the bias.

**3.2 — Window-span convention for `momentum_12m`** (re-recorded from the withdrawn
Amendment 2.2, which stated it correctly for this feature).

`momentum_12m` is a two-endpoint formula: a 12-month return needs only the index at both
ends, because the total-return index already compounds what happens in between. Rule 4
nonetheless says a window *spanning* the 2025-10 gap is void, so it is computed as NaN
unless all 13 months of its span are present. This is stricter than the arithmetic
requires and is a deliberate reading of the rule, not an accident. It costs nothing in
this sample: no in-sample feature window reaches 2025-10, the newest reference month any
feature uses being 2025-07.

`unrate_trend_12m` needs no such convention under the restored definition — a trailing
mean genuinely reads all 12 months of its window, so requiring the full span is simply
what the formula needs.

**Amendments after Amendment 3: none as of 2026-09-07.**

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

**The rule now covers four columns, not one.** `cpi`, `sp500_index` and `cape` are null at
2025-10 for the same single cause — see **2025-10 is a single-cause gap**. Everything below
applies to each of them, and they fail together: a window spanning that month voids all
four at once.

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

## Feature layer

`scripts/build_features.py` reads `data/market_panel.parquet` and writes
`data/features.parquet`: the four pre-registered features and the label, on monthly
decision points from 1962-02-28 to 2025-08-31. Features and label only — no model, no
training, no evaluation.

```bash
python scripts/build_features.py
```

| Column | Definition |
| --- | --- |
| `date` | decision month-end |
| `cape_z` | CAPE expanding z-score, 120-month warm-up |
| `yield_slope` | `dgs10` − `dtb3` |
| `momentum_12m` | 12-month trailing return of `sp500_index` |
| `unrate_trend_12m` | latest available `unrate` minus its 12-month trailing mean |
| `label` | 1 if the 12-month forward excess return of equities over bills is positive |

**763 rows, 761 complete.** The two incomplete rows are 2024-10-31 and 2025-08-31, both
because `cpi` is null at their label endpoint — 2025-10 permanently, 2026-08 until BLS
publishes. No feature has a single null: the expanding warm-up is exhausted long before
1962, and no in-sample feature window reaches the 2025-10 gap.

### The label deflates one leg, not two

The label is an excess return, so both legs must end up in the same units — real. They do
not start there, and that asymmetry is the whole point:

- **`sp500_index` is already real.** It is Shiller's *Real Total Return Price*, column 9,
  CPI-deflated at source. Its 12-month ratio `sp500_index(M+12)/sp500_index(M)` is
  therefore a real gross return with nothing left to do.
- **`dtb3` is a nominal yield.** Compounding it gives a nominal gross return, which must
  be deflated by the window's own inflation, `cpi(M)/cpi(M+12)`, to be comparable.

So exactly one leg is deflated. **Deflating both would double-deflate the equity leg**,
dividing a series that is already in constant dollars by inflation a second time. That
error is close to invisible in inspection — the result still looks like a plausible excess
return — and it would bite hardest in the high-inflation stretch of the late 1970s and
early 1980s, where 12-month inflation reached 14.8%, which is precisely the period where
the stocks-versus-bills question was most alive.

The equity index's rebasing constant cancels in the ratio, so the label is stable across
pulls of the Shiller file. Bill compounding follows the pre-registered convention: `dtb3`
from M+1 through M+12, each quoted annualised rate treated as an effective annual rate.

**Hand-check at decision 1995-03-31**, label window 1995-04-30 to 1996-03-31:

| Quantity | Value |
| --- | --- |
| equity real gross, `sp500_index(1996-03)/sp500_index(1995-03)` | 1.306278 |
| bill gross, nominal, `dtb3` compounded over the 12 months | 1.052588 |
| bill gross, real, × `cpi(1995-03)/cpi(1996-03)` | 1.023519 |
| excess | **+0.282759** |
| label | **1** — matches the built value |

Had the equity leg also been deflated it would have read 1.269 rather than 1.306 here, and
the excess would have been understated by roughly 3.6 percentage points in a single year —
enough to flip the sign in any year where equities beat bills by less than inflation.

**How the four rules are enforced.** Windows are computed on the reference-month series
that carries the NaN, and the availability shift is applied to the *result* — never the
reverse, which would turn the 2025-10 gap into a stale repeat. Availability is read from
each series' own lag column; no lag is hardcoded anywhere in the file, so the measured
per-row `unrate` lags are honoured alongside the constant 15- and 1-day ones. Rows are
ordered by availability date with a running maximum of the reference month, so a late
release such as `unrate` 2025-09, published 51 days behind, can never make an older month
look like the newest available one.

**Verified on every build**, not just at three sampled points: across all 763 decision
points and all four features, **zero inputs postdate their decision** — 3,052 checks. The
expanding z-score is recomputed by hand at several dates and matches to 1e-10, and it
differs from a full-sample z by 0.51 sd on average, disagreeing on sign in 10.4% of
in-sample months. `unrate_trend_12m` is likewise recomputed by hand: at decision
2025-08-31 the reference month is 2025-07, the trailing window is 2024-08 to 2025-07, the
latest rate is 4.2 against a window mean of 4.141667, giving +0.058333 — matching the
built value. If the complete-row count ever departs from 761 the script stops and says so
rather than adjusting anything.

**Not reported, by design.** The build prints no label positive rate, no class balance,
and no feature-label association. The baseline is re-estimated inside each walk-forward
fold, and seeing the pooled rate now would contaminate every later judgement.

---

## Provenance

Every build writes a header block into the parquet file's schema metadata and a
human-readable twin at `data/market_panel_provenance.txt`, containing source URLs, the
pull timestamp, row count, per-series observed date range and null count, every lag
assumption above, and the point-in-time status of each column.
