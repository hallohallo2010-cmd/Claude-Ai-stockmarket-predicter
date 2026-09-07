# Results — market-timing study, development sample

Status at this commit: **the walk-forward has been run on the development sample only.
The holdout, calendar years 2010 through 2025, has not been read.**

Every number below is traceable to a committed file: `data/market_panel_provenance.txt`,
`data/features.parquet`, `data/walkforward_results.csv`, `logs/experiment_log.md`, or the
pre-registration in `README.md`. Nothing here is a new run.

---

## 1. The question, and the criterion fixed before any result existed

**The question.** Do valuation and macro signals predict the direction of the S&P 500 over
the following 12 months, well enough to inform a stocks-versus-bills decision?

The design was registered in `README.md` on 2026-09-06, against panel build
`data/market_panel.parquet` (1868 rows, 1871-01-31 to 2026-08-31), **before any feature had
been computed and before any label had been examined**. What follows is what that document
fixed, not a description written afterwards.

**Primary criterion (section 5).** On the holdout, the model's **Brier score must be at
least 10% lower in relative terms** than the baseline's, where the baseline uses each
step's training-window positive rate. Brier and not accuracy, stated at the time because
the label is imbalanced and a constant predictor scores well on accuracy, so accuracy
cannot distinguish a signal from the base rate.

**Robustness.** The improvement must survive dropping calendar year 2020.

**Uncertainty.** Reported as a moving-block bootstrap confidence interval, block length 12
months, 10,000 resamples. Treating overlapping monthly rows as independent was prohibited
in section 2, as was random k-fold cross-validation.

**Secondary, reporting only, never promoted to primary:** accuracy, AUC, log loss, and the
Sharpe ratio of any trading rule. Section 5: "They cannot be substituted for the Brier
criterion if the Brier criterion fails."

**Declared expected outcome (section 5).** "The prior is that this study finds no useful
predictive signal. A negative result is the anticipated finding and will be reported
plainly as such."

Three dated amendments follow the registration. Amendment 1 (2026-09-06) identified the
Shiller column in use and changed the label to a 12-month **excess** return over bills.
Amendment 2 (2026-09-07) was withdrawn in full before any model existed, its text kept in
the log rather than deleted. Amendment 3 (2026-09-07) fixed a one-time sample-refresh rule
in advance. All were written before the change they authorise.

---

## 2. The data

`scripts/build_market_panel.py` builds `data/market_panel.parquet`: **1868 rows,
1871-01-31 to 2026-08-31**, monthly, month-end aligned, pulled 2026-09-06 21:54:23 UTC. It
is data only — no column is derived from another column in the file.

| Series | Source |
| --- | --- |
| `sp500_index`, `cape`, `cpi` | Robert Shiller, `ie_data.xls`, via `shillerdata.com` |
| `dgs10`, `dtb3` | FRED `DGS10`, `DTB3` |
| `unrate` | **ALFRED vintages**, not FRED's revised series |
| CPI verification authority | FRED `CPIAUCNS` |

### Point-in-time treatment

Every series carries an explicit `*_lag_days` column. A value on row `date` is unknown
until `date + lag_days`, and a signal formed at month-end M may only use rows where
`date + lag_days <= M`. `sp500_index`, `cape` and `cpi` carry an assumed 15 days — they all
depend on the same mid-month CPI release. `dgs10` and `dtb3` carry 1 day.

**`unrate` is the one series where the lag is measured rather than assumed.** FRED's
`UNRATE` is the revised series: seasonal factors are re-estimated every January and restate
years of history at once. The panel does not use it. It reads **799 ALFRED vintages** and
assigns each reference month the value from the earliest vintage in which that month
appears — its first published value — recording that vintage date and measuring the lag
from it. Measured lag across the first-release rows: **minimum −2 days, median 5, maximum
51**. Seven early-1960s rows have a non-positive lag, because BLS then published within the
reference month itself.

The alternative of a blanket `.shift(1)` was checked against the measured lag column and
**disagrees in 154 of 1868 months**: 146 leaks, 7 needlessly conservative, 1 lost value. The
leaks are 145 pre-1960 rows plus one modern case — at 2025-10-31 a `shift(1)` hands you
September 2025's rate, which was not published until 2025-11-20.

`sp500_index` and `cape` are **not** point-in-time and are recorded as such: Shiller
publishes no vintage archive, so they are revised data as of the pull date. `cape`
additionally depends on interpolated recent earnings, so its 15-day lag is a lower bound.
This is a known weakness of the study, stated rather than worked around.

`cpi` is verified by value, not by header: **1362 of 1362 shared months match `CPIAUCNS`
exactly, maximum difference 0.000000**. The 504 months before 1913-01 predate that
authority and are Shiller's Warren-Pearson splice; they cannot be verified and are recorded
as unverifiable.

### What was removed, and why

**Three fabricated CPI values.** Shiller's workbook carries his own estimates for months
BLS has not published, and says so in a footnote. Those are filled numbers. The build nulls
any month at or after 1913-01 that `CPIAUCNS` does not carry: **3 estimates removed —
2025-10, 2026-08 and 2026-09**. (The 2026-09 row is not in the shipped panel at all: the
build ran with `--drop-partial-month` and that month was still in progress at the pull
date.)

**Their derived columns were nulled with them: `sp500_index` ×3 and `cape` ×3.** Both are
Shiller's *real* series, deflated by the CPI of that same month. Where the deflator was his
estimate, the real value is fabricated — a real series divided by a fabricated deflator is
fabricated. Nulling them was measured read-only before it was enabled: it costs the
pre-registered study sample **zero labelled observations**, because every feature window
looks backward and cannot reach those months from a decision point at or before 2025-08-31.

**The October 2025 shutdown gap.** One event removed October 2025 from every column that
depends on a statistical agency:

| Column | Why it is null at 2025-10 |
| --- | --- |
| `cpi` | BLS cancelled the October 2025 CPI release |
| `unrate` | no household survey was conducted |
| `sp500_index` | derived — Shiller's real series needs the missing CPI |
| `cape` | derived — same missing deflator |

`dgs10` and `dtb3` are complete there; Treasury markets traded throughout. These are **not
four independent gaps**, and the panel records that they must not be treated as such: any
imputation borrowing across them is circular, because what is missing from each is the same
thing. A feature layer seeing all four fail at once is looking at one absence counted four
times, not four corroborating signals.

The gap is never filled, interpolated, forward-filled, or dropped. Any window spanning it
is NaN, enforced with `min_periods == window`. It is also never closed by re-ordering
operations: windows are computed on the reference-month series that carries the NaN, and
the availability shift is applied to the result. Doing it the other way round turns the hole
into a stale repeat and leaves nothing to propagate.

---

## 3. Features and label

`scripts/build_features.py` writes `data/features.parquet`: **763 decision month-ends,
1962-02-28 to 2025-08-31, of which 761 are complete.**

Four features, fixed in section 1 of the pre-registration before any were computed. No
interactions, no polynomial terms, no alternative windows, no regime dummies.

| Column | Definition |
| --- | --- |
| `cape_z` | Expanding z-score of CAPE — value minus expanding mean, over expanding sd. Minimum warm-up 120 months. |
| `yield_slope` | `dgs10` − `dtb3`, in percentage points |
| `momentum_12m` | 12-month log change in the real total return index |
| `unrate_trend_12m` | Latest available `unrate` minus its 12-month trailing mean |
| `label` | 1 if the 12-month forward excess return of equities over bills is positive |

**The label** (Amendment 1.2) is the sign of the 12-month **excess** return: positive if the
compounded total return on the equity index exceeds the compounded return from rolling
3-month bills over the same window. Ties count as negative. The decision the model informs
is stocks versus bills, and the sign of the equity return alone does not answer it — a year
returning 3% on equities against 5% on bills is a positive label under the original
specification and a loss under the only decision the model is for.

Exactly one leg is deflated. `sp500_index` is already real, so its 12-month ratio is a real
gross return; `dtb3` is a nominal yield, so the compounded bill leg is divided by the
window's own inflation. Deflating both would double-deflate the equity leg. Hand-checked at
decision 1995-03-31: equity real gross 1.306278, bill gross nominal 1.052588, bill gross
real 1.023519, excess +0.282759, label 1 — matching the built value. Had the equity leg
also been deflated it would have read 1.269, understating the excess by roughly 3.6
percentage points in a single year.

The bill leg deliberately reads `dtb3` from M+1 through M+12, which is in the future
relative to the decision point. That is part of the label, not a leak.

**Two rows are incomplete**, both from a null `cpi` at the label endpoint: 2024-10-31
(permanent — October 2025 CPI will never exist) and 2025-08-31 (temporary, pending August
2026 CPI). Neither is caused by the de-estimation above.

**The availability rule is verified on every build**: across all 763 decision points and all
four features, zero inputs postdate their decision — 3,052 checks. The expanding CAPE
z-score matters here rather than being a formality: it differs from a full-sample z-score by
0.51 sd on average and disagrees on the *sign* of the signal in 10.4% of in-sample months.
CAPE's mean by era is 14.87 (1881-1929), 14.91 (1930-1979), 21.15 (1980-2009) and 28.85
(2010-2026), against a full-sample mean of 17.78 — so a full-sample z-score would tell every
pre-1980 row the market was expensive relative to a future it could not see.

---

## 4. The protocol

`scripts/train_walkforward.py`, restricted to the development window **1962-02-28 to
2009-12-31, 575 month-ends**. The date restriction is pushed into the parquet reader, so
holdout rows are never materialised; the script then asserts nothing past 2009-12-31
survived.

- **Expanding walk-forward.** Training runs from 1962-02-28 to a cut and never slides or
  resets. Test blocks are calendar years, **1980 through 2009 — 30 folds**. Training windows
  run from 204 month-ends (17 independent 12-month blocks) to 552.
- **12-month embargo.** The cut sits 12 months before the fold's first test point, so for
  test year *Y* training ends at *(Y−1)*-01-31 and the last training row's 12-month label
  window closes exactly where the first test row's opens. The two touch at an endpoint and
  do not overlap. Without it, the last year of training labels reaches into the test period
   — the same leak as random folds in different clothing. Labels are 12-month overlapping,
  so this is not optional. Asserted fold by fold, not assumed. No random k-fold anywhere.
- **Per-fold baseline.** A constant predictor whose probability is **that fold's own
  training-window positive rate**, recomputed at every step. Never the full-sample rate.
  The per-fold rates are in `data/walkforward_results.csv`.
- **Brier is primary.** Accuracy and log loss are carried as reporting-only columns.
- **Nothing crosses a fold boundary.** Standardisation is fit on each fold's training rows
  and applied to that fold's test rows. No mean, sd, quantile grid or positive rate is
  computed over anything wider than the fold's own training window.

**Two models, and nothing else.** The constant baseline, then unpenalised logistic
regression on the four features — no penalty, so no regularisation strength to choose. No
gradient boosting, no hyperparameter search, no feature selection. The point of running the
simplest thing first is to learn whether any signal exists before introducing a model with
the capacity to fit 47 independent observations by accident.

Test-block length and the 1980 start are **not** fixed by the pre-registration, which fixes
the expanding window and the embargo but not the fold granularity. They were written into
the script before its first run and are recorded in `logs/experiment_log.md`. They are
implementation choices, not amendments.

---

## 5. The result

Pooled over the 360 test month-ends from 1980-01-31 to 2009-12-31 — **30 independent
12-month blocks** — each row scored against its own fold's baseline:

| | Logistic regression | Baseline |
| --- | --- | --- |
| **Brier (primary)** | **0.231755** | **0.212031** |
| Accuracy (reporting only) | 0.688889 | 0.719444 |
| Log loss (reporting only) | 0.742115 | 0.616578 |

| | |
| --- | --- |
| Relative change in Brier | **−9.30%** |
| Moving-block bootstrap 95% CI, baseline minus model, block 12, 10,000 resamples | **[−0.072089, +0.027139]** |
| Folds with model Brier below baseline | **17 of 30** |

**The model does not beat the baseline.** Its Brier score is 9.30% *higher* — worse — than
that of a constant predictor set to each fold's own training positive rate. The
pre-registered criterion asks for a 10% relative *reduction*. It is not met, and it is not
close to met: the sign is wrong before the magnitude is considered.

The bootstrap interval spans zero, so the development sample does not even establish that
the model is reliably worse — only that it is not better.

Accuracy and log loss are reported because section 5 says they are reported. Both are also
worse than the baseline, so no question of promoting them arises. Had either been better,
the answer would still be no: section 5 fixed that in advance precisely so a failing primary
metric could not be rescued by a secondary one.

The model wins 17 folds of 30 and still loses pooled. A count of fold wins is not the
primary metric and is not being promoted to one; the reason the two point in opposite
directions is section 6.

The pre-registered criterion is a **holdout** criterion. It is not being evaluated here and
nothing above should be read as a pass or a failure of the study. What is reported is the
development result as it stands.

---

## 6. Where it failed

The pooled Brier shortfall is 0.019724 per row, or **7.100652** in summed squared error
across the 360 test rows. Three consecutive folds account for **11.616540** of it — more
than the entire shortfall, with the other 27 folds netting in the model's favour.

| Fold | Test span | Test positive rate | Brier, model | Brier, baseline | Relative |
| --- | --- | --- | --- | --- | --- |
| 2000 | 2000-01-31 .. 2000-12-31 | 0.0 | 0.841843 | 0.493791 | −70.49% |
| 2001 | 2001-01-31 .. 2001-12-31 | 0.0 | 0.919440 | 0.495542 | −85.54% |
| 2002 | 2002-01-31 .. 2002-12-31 | 0.5 | 0.480653 | 0.284558 | −68.91% |

In 2000 and 2001 **every** test month resolved negative — equities lost to bills over all
24 of those 12-month windows. Because every label is 0 in those folds, Brier is the mean
squared forecast, so the root-mean-square probability the model assigned to "up" is the
square root of its Brier score. The identity checks out on the baseline column, where
`sqrt(0.493791) = 0.702703` and `sqrt(0.495542) = 0.703947` reproduce those folds' baseline
rates exactly.

| Fold | RMS probability of "up", model | Baseline probability |
| --- | --- | --- |
| 2000 | 0.917520 | 0.702703 |
| 2001 | 0.958874 | 0.703947 |

So the failure is not that the model disagreed with the baseline. It agreed with it and bet
harder. Both called "up" every month of 2000 and 2001; both were wrong every month —
accuracy is 0.0 for model and baseline alike in each fold. The model lost on Brier because
it went into the dot-com peak at a root-mean-square 0.92 and then 0.96 confidence while the
baseline sat at 0.70, and confidence is exactly what Brier charges for.

This is not a general tendency to overconfidence. In 2007, another fold where every test
month resolved negative, the model's RMS probability was 0.614632 against a baseline of
0.689394 — less confident than the baseline, and it beat it.

Two things must be said plainly about this section. First, 2002 is not the third-worst fold:
ranked by absolute Brier shortfall the order is 2001 (0.423898), 2000 (0.348052), **1984**
(0.229665), then 2002 (0.196095). The 2000-2002 block is reported because it is contiguous
and shares a cause, not because it is the three worst folds.

Second: excluding those three folds, the pooled Brier over the remaining 324 rows is
0.174471 for the model against 0.188409 for the baseline, a 7.40% relative *improvement*.
**That number has no standing and is not an alternative result.** It is what you get by
deleting the folds where the model was worst, which is not a permitted operation — the
pre-registration authorises exactly one such exclusion, calendar year 2020, and that year is
not in the development sample. It is recorded here to locate the failure, not to soften it.
The result is −9.30%.

---

## 7. Limitations

**The effective sample size, which ranks above everything else here.** Labels are 12-month
overlapping, so consecutive monthly rows share 11 of their 12 label months. Independent
non-overlapping observations are the month count divided by 12, truncated to complete
blocks:

| Split | Month-ends | Independent 12-month observations |
| --- | --- | --- |
| Development, 1962-02-28 to 2009-12-31 | 575 | **47** |
| Holdout, 2010-01-31 to 2025-08-31 | 188 | **15** |
| Total | 763 | **63** |

The pooled test set above is 360 month-ends, which is **30** independent blocks, not 360
observations. Any reading of these results that treats 360 as the sample size is wrong by
roughly a factor of twelve in the standard errors.

**What 47 independent development observations permit.** Detecting a large effect, or
finding nothing. That is the whole list. This study found no large effect detectable in this
sample with these four features. That statement is what the design supports.

**What they forbid.**

- *Ranking models.* 47 blocks cannot resolve which of two model classes is better. Anything
  that looked like an improvement over the numbers above would be within the noise the
  bootstrap interval already shows, which spans zero.
- *Concluding that market timing does not work.* This is one feature set, four features, one
  model class, one label definition, one sample, one horizon. A null result at this sample
  size is weak evidence about this specification and close to no evidence about the general
  question. Section 5's declared prior was that the study would find no useful signal; a
  result matching a stated prior is not thereby strengthened, and it is not a demonstration
  that no signal exists.
- *Resolving a modest edge.* An effect too small for 47 blocks to see is not thereby absent.
  Nothing here distinguishes "no signal" from "a signal smaller than this design can detect."

**Other limitations, recorded rather than argued away.**

- `sp500_index` and `cape` are revised data with no vintage archive, so the two
  equity-derived inputs are not point-in-time. `cape` also depends on earnings Shiller
  interpolates and later revises.
- `yield_slope` is placed one month back by the uniform availability rule even though a
  trader at the month-end close can see that day's yields. One uniform rule was preferred to
  one month of extra signal on one feature.
- The 1962 start is a data constraint, not a choice: `dgs10` begins 1962-01. Dropping
  `yield_slope` would recover two independent observations, 63 to 65, because the binding
  constraint moves straight to `unrate_trend_12m`. Dropping both macro features would reach
  1891 and 134 blocks; Amendment 1.4 records that as considered and rejected, because it
  abandons the macro half of the study's premise.
- The fold schedule — calendar-year test blocks, first test year 1980 — is an implementation
  choice, not a pre-registered one. It was fixed before the first run and logged, but it was
  not registered in advance, and a different schedule would give somewhat different numbers.

---

## 8. Status

**The holdout — calendar years 2010 through 2025, 188 month-ends, 15 independent
observations — has not been read.** It was not loaded, not counted, and no number in this
document is a function of it. `scripts/train_walkforward.py` pushes the development date
range into the parquet reader so holdout rows are never materialised, then asserts that
nothing past 2009-12-31 survived.

Section 4 fixed the terms: the holdout is read once, after the model class,
hyperparameters and all feature transforms are frozen from development alone. One read, one
number. If the result fails the criterion, that is the result of the study — it will not be
re-tuned and re-read, and the holdout will not be reopened to diagnose the failure.

It is also recorded in advance that 2010-2025 is one long expansion plus a single sharp
crash at persistently high CAPE — roughly 15 independent blocks of a narrow regime. A single
read of it has limited power. That was accepted as the cost of having a genuine holdout at
all.

Amendment 3.1 permits exactly one sample refresh, once August 2026 CPI is published, taking
the study from 761 to 762 usable rows. It is authorised only while the model is still open
and is **forfeited if the holdout is read first**.

Every configuration run to date is recorded in `logs/experiment_log.md`, which the training
script appends to on every run and cannot be run without writing to.
