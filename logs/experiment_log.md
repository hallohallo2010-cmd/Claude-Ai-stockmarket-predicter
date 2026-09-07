# Experiment log

Every configuration tried, in the order it was tried, with the date, what
changed from the previous entry, and the result. Appended automatically by
`scripts/train_walkforward.py` on every run, so a configuration cannot be
run without being recorded, and "what changed" is computed against the
previous entry rather than remembered.

All entries below are **development sample only**. The holdout is unread.

---

## 2026-09-07T07:25:00Z - config `bb3316ac387a`

**What changed.** First entry. Nothing to compare against: this is the first configuration recorded for this study.

- **Sample.** development only, 1962-02-28 .. 2009-12-31. Holdout not read.
- **Scheme.** expanding-window walk-forward, calendar-year test blocks, 30 folds, test years 1980-2009, 12-month embargo.
- **Features.** cape_z, yield_slope, momentum_12m, unrate_trend_12m.
- **Models.** constant baseline at the fold's training positive rate; logistic regression, unpenalised (C=inf), lbfgs, max_iter=1000.
- **Transform.** standardisation fit on the training fold only.
- **Primary metric.** Brier.

| Quantity | Value |
| --- | --- |
| Pooled test rows | 360 (30 independent blocks) |
| Pooled Brier, logistic | 0.231755 |
| Pooled Brier, baseline | 0.212031 |
| Relative change in Brier | -0.093025 |
| Block-bootstrap 95% CI, baseline minus model | [-0.072089, 0.027139] |
| Folds with model Brier below baseline | 17 of 30 |
| Pooled accuracy, model / baseline (reporting only) | 0.688889 / 0.719444 |
| Pooled log loss, model / baseline (reporting only) | 0.742115 / 0.616578 |

Per-fold results: `data/walkforward_results.csv`.

<!-- config: {"embargo_months": 12, "features": ["cape_z", "yield_slope", "momentum_12m", "unrate_trend_12m"], "first_test_year": 1980, "last_test_year": 2009, "models": ["constant baseline at the fold's training positive rate", "logistic regression, unpenalised (C=inf), lbfgs, max_iter=1000"], "n_folds": 30, "primary_metric": "Brier", "sample": "development only, 1962-02-28 .. 2009-12-31", "scheme": "expanding-window walk-forward, calendar-year test blocks", "transform": "standardisation fit on the training fold only"} -->

---

## 2026-09-07T07:26:40Z - config `bb3316ac387a`

**What changed.** Nothing changed. Same configuration, re-run.

- **Sample.** development only, 1962-02-28 .. 2009-12-31. Holdout not read.
- **Scheme.** expanding-window walk-forward, calendar-year test blocks, 30 folds, test years 1980-2009, 12-month embargo.
- **Features.** cape_z, yield_slope, momentum_12m, unrate_trend_12m.
- **Models.** constant baseline at the fold's training positive rate; logistic regression, unpenalised (C=inf), lbfgs, max_iter=1000.
- **Transform.** standardisation fit on the training fold only.
- **Primary metric.** Brier.

| Quantity | Value |
| --- | --- |
| Pooled test rows | 360 (30 independent blocks) |
| Pooled Brier, logistic | 0.231755 |
| Pooled Brier, baseline | 0.212031 |
| Relative change in Brier | -0.093025 |
| Block-bootstrap 95% CI, baseline minus model | [-0.072089, 0.027139] |
| Folds with model Brier below baseline | 17 of 30 |
| Pooled accuracy, model / baseline (reporting only) | 0.688889 / 0.719444 |
| Pooled log loss, model / baseline (reporting only) | 0.742115 / 0.616578 |

Per-fold results: `data/walkforward_results.csv`.

<!-- config: {"embargo_months": 12, "features": ["cape_z", "yield_slope", "momentum_12m", "unrate_trend_12m"], "first_test_year": 1980, "last_test_year": 2009, "models": ["constant baseline at the fold's training positive rate", "logistic regression, unpenalised (C=inf), lbfgs, max_iter=1000"], "n_folds": 30, "primary_metric": "Brier", "sample": "development only, 1962-02-28 .. 2009-12-31", "scheme": "expanding-window walk-forward, calendar-year test blocks", "transform": "standardisation fit on the training fold only"} -->
