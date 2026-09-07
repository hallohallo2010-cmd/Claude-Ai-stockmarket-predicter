"""Walk-forward training on the development sample only.

Reads ``data/features.parquet``, restricted at read time to the pre-registered
development window 1962-02-28 .. 2009-12-31. The holdout (2010-01-31 onward) is
filtered out by the parquet reader: it is not loaded into a dataframe, not
counted, and no quantity computed here is a function of it.

Protocol, from README.md "PRE-REGISTRATION":

* Section 4 - expanding-window walk-forward. Each fold trains on everything from
  the sample start up to a cut, then tests on the following period. The window
  never slides or resets. No random k-fold (section 2).
* Section 4 - embargo. The training window ends at least 12 months before the
  fold's first test point, so the last training row's 12-month label window
  closes exactly where the first test row's opens and no training label reaches
  into the test period.
* Section 3 - baseline. A constant predictor whose probability is the positive
  rate of that fold's own training window. Never the full-sample rate.
* Section 5 - primary metric is the Brier score, model against that fold's
  baseline. Accuracy and log loss are reporting-only and cannot be promoted if
  Brier fails.

Models, in order, and nothing else:

1. The constant baseline.
2. Logistic regression on the four pre-registered features (section 1).

Every transform is fit inside the training fold and applied to the test fold.
No mean, no standard deviation, no quantile grid crosses a fold boundary.

Writes ``data/walkforward_results.csv`` and appends a dated entry to
``logs/experiment_log.md``.

    python scripts/train_walkforward.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = ROOT / "data" / "features.parquet"
RESULTS_PATH = ROOT / "data" / "walkforward_results.csv"
LOG_PATH = ROOT / "logs" / "experiment_log.md"

# --- Pre-registered constants (README sections 1 and 4). Not free parameters. ---
DEV_START = pd.Timestamp("1962-02-28")
DEV_END = pd.Timestamp("2009-12-31")
EMBARGO_MONTHS = 12
FEATURES = ["cape_z", "yield_slope", "momentum_12m", "unrate_trend_12m"]
LABEL = "label"

# --- Fold schedule. Not fixed by the pre-registration; chosen here and logged. ---
# Test blocks are calendar years, so each fold's test period is exactly one
# 12-month non-overlapping block and the development window divides evenly.
# The first test year is 1980, which leaves 1962-02 .. 1979-01 as the smallest
# training window: 204 month-ends, 17 independent 12-month blocks, for a model
# with five parameters.
FIRST_TEST_YEAR = 1980
LAST_TEST_YEAR = 2009

BOOTSTRAP_BLOCK_MONTHS = 12
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260907

EPS = 1e-15


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of a probability forecast. Primary metric."""
    return float(np.mean((p - y) ** 2))


def accuracy(p: np.ndarray, y: np.ndarray) -> float:
    """Reporting only. Cannot be promoted over Brier."""
    return float(np.mean((p >= 0.5).astype(float) == y))


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Reporting only. Cannot be promoted over Brier."""
    q = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def load_development_sample() -> pd.DataFrame:
    """Read the development window only.

    The date restriction is pushed into the parquet reader, so holdout rows are
    never materialised here. The assertions below fail loudly rather than let a
    holdout row through unnoticed.
    """
    df = pd.read_parquet(
        FEATURES_PATH,
        filters=[("date", ">=", DEV_START), ("date", "<=", DEV_END)],
    )
    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].max() > DEV_END or df["date"].min() < DEV_START:
        raise SystemExit("holdout or pre-sample row reached the dataframe; stopping")

    missing = [c for c in FEATURES + [LABEL, "date"] if c not in df.columns]
    if missing:
        raise SystemExit(f"features.parquet is missing columns: {missing}")

    incomplete = int(df[FEATURES + [LABEL]].isna().any(axis=1).sum())
    if incomplete:
        raise SystemExit(
            f"{incomplete} development rows carry a null feature or label; "
            "the feature layer says there should be none. Stopping rather than "
            "dropping rows silently."
        )

    df["period"] = df["date"].dt.to_period("M")
    return df


def build_folds(df: pd.DataFrame) -> list[dict]:
    """Expanding training window, one calendar year of test points per fold."""
    folds = []
    for year in range(FIRST_TEST_YEAR, LAST_TEST_YEAR + 1):
        test_mask = df["date"].dt.year == year
        if not test_mask.any():
            raise SystemExit(f"no development rows in test year {year}")

        first_test_period = df.loc[test_mask, "period"].min()
        cut_period = first_test_period - EMBARGO_MONTHS
        train_mask = df["period"] <= cut_period

        train = df.loc[train_mask]
        test = df.loc[test_mask]
        if train.empty:
            raise SystemExit(f"empty training window for test year {year}")

        # The embargo, asserted rather than assumed. The last training row's
        # label window closes at last_train_period + 12; it must not open the
        # test period late, i.e. must not exceed the first test point.
        gap = (first_test_period - train["period"].max()).n
        if gap < EMBARGO_MONTHS:
            raise SystemExit(
                f"embargo violated in test year {year}: {gap} months between "
                f"training end and first test point"
            )
        if train["period"].max() + EMBARGO_MONTHS > first_test_period:
            raise SystemExit(f"training label window reaches into test year {year}")

        folds.append({"year": year, "train": train, "test": test})
    return folds


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _unpenalised_kwargs() -> tuple[tuple[str, object], ...]:
    """Pick the spelling of "no penalty" this sklearn accepts.

    ``C=np.inf`` is the current one and ``penalty=None`` the older; they fit the
    same maximum-likelihood model to identical coefficients. sklearn validates
    parameters at fit time, not construction, so this probes with a fit.
    """
    probe_x = np.array([[-1.0], [-0.5], [0.5], [1.0]])
    probe_y = np.array([0.0, 1.0, 0.0, 1.0])  # not separable: no convergence warning
    for kwargs in ({"C": np.inf}, {"penalty": None}):
        try:
            LogisticRegression(solver="lbfgs", max_iter=1000, **kwargs).fit(probe_x, probe_y)
        except (TypeError, ValueError):
            continue
        return tuple(kwargs.items())
    raise SystemExit("this sklearn accepts no unpenalised logistic regression")


def unpenalised_logistic() -> LogisticRegression:
    """Plain maximum-likelihood logistic regression.

    No penalty, so there is no regularisation strength to choose and nothing to
    tune.
    """
    return LogisticRegression(solver="lbfgs", max_iter=1000, **dict(_unpenalised_kwargs()))


def fit_predict_fold(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Fit both models on the training fold, predict the test fold.

    Standardisation is fit on the training rows only. Nothing computed from the
    test rows is used to transform anything.
    """
    x_train = train[FEATURES].to_numpy(dtype=float)
    x_test = test[FEATURES].to_numpy(dtype=float)
    y_train = train[LABEL].to_numpy(dtype=float)
    y_test = test[LABEL].to_numpy(dtype=float)

    # Model 1: constant predictor at this fold's own training positive rate.
    baseline_rate = float(y_train.mean())
    p_baseline = np.full(len(y_test), baseline_rate)

    # Model 2: logistic regression on the four features. Standardisation fit
    # inside the training fold. No penalty, no tuning, no feature selection.
    mu = x_train.mean(axis=0)
    sd = x_train.std(axis=0, ddof=0)
    if not np.all(sd > 0):
        raise SystemExit("a feature is constant within a training fold")

    z_train = (x_train - mu) / sd
    z_test = (x_test - mu) / sd

    if len(np.unique(y_train)) < 2:
        raise SystemExit("training fold carries a single class; stopping")

    model = unpenalised_logistic()
    model.fit(z_train, y_train)
    p_model = model.predict_proba(z_test)[:, 1]

    return {
        "y_test": y_test,
        "p_model": p_model,
        "p_baseline": p_baseline,
        "baseline_rate": baseline_rate,
    }


# --------------------------------------------------------------------------
# uncertainty
# --------------------------------------------------------------------------

def moving_block_ci(diff: np.ndarray) -> tuple[float, float]:
    """Moving-block bootstrap CI for the mean of a per-row series.

    Section 2 of the pre-registration prohibits treating overlapping monthly
    rows as independent; block length is 12 months, the label horizon.
    """
    n = len(diff)
    if n < BOOTSTRAP_BLOCK_MONTHS:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_starts = n - BOOTSTRAP_BLOCK_MONTHS + 1
    n_blocks = int(np.ceil(n / BOOTSTRAP_BLOCK_MONTHS))
    offsets = np.arange(BOOTSTRAP_BLOCK_MONTHS)
    means = np.empty(BOOTSTRAP_RESAMPLES)
    for i in range(BOOTSTRAP_RESAMPLES):
        starts = rng.integers(0, n_starts, size=n_blocks)
        idx = (starts[:, None] + offsets[None, :]).ravel()[:n]
        means[i] = diff[idx].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def fold_row(name, train, test, res) -> dict:
    y = res["y_test"]
    pm, pb = res["p_model"], res["p_baseline"]
    b_model, b_base = brier(pm, y), brier(pb, y)
    return {
        "fold": name,
        "train_start": train["date"].min().date().isoformat(),
        "train_end": train["date"].max().date().isoformat(),
        "test_start": test["date"].min().date().isoformat(),
        "test_end": test["date"].max().date().isoformat(),
        "n_train": len(train),
        "n_test": len(test),
        "baseline_rate": round(res["baseline_rate"], 6),
        "test_positive_rate": round(float(y.mean()), 6),
        "brier_model": round(b_model, 6),
        "brier_baseline": round(b_base, 6),
        "brier_rel_improvement": round((b_base - b_model) / b_base, 6) if b_base > 0 else "",
        "accuracy_model": round(accuracy(pm, y), 6),
        "accuracy_baseline": round(accuracy(pb, y), 6),
        "logloss_model": round(log_loss(pm, y), 6),
        "logloss_baseline": round(log_loss(pb, y), 6),
    }


def main() -> int:
    df = load_development_sample()
    folds = build_folds(df)

    rows, y_all, pm_all, pb_all, rate_all = [], [], [], [], []
    for fold in folds:
        res = fit_predict_fold(fold["train"], fold["test"])
        rows.append(fold_row(str(fold["year"]), fold["train"], fold["test"], res))
        y_all.append(res["y_test"])
        pm_all.append(res["p_model"])
        pb_all.append(res["p_baseline"])
        rate_all.append(np.full(len(res["y_test"]), res["baseline_rate"]))

    y = np.concatenate(y_all)
    pm = np.concatenate(pm_all)
    pb = np.concatenate(pb_all)
    rates = np.concatenate(rate_all)

    pooled_b_model, pooled_b_base = brier(pm, y), brier(pb, y)
    pooled = {
        "fold": "POOLED",
        "train_start": folds[0]["train"]["date"].min().date().isoformat(),
        "train_end": folds[-1]["train"]["date"].max().date().isoformat(),
        "test_start": folds[0]["test"]["date"].min().date().isoformat(),
        "test_end": folds[-1]["test"]["date"].max().date().isoformat(),
        "n_train": "",  # expands per fold; no single number is true of the pool
        "n_test": len(y),
        "baseline_rate": round(float(rates.mean()), 6),  # mean of the per-fold rates
        "test_positive_rate": round(float(y.mean()), 6),
        "brier_model": round(pooled_b_model, 6),
        "brier_baseline": round(pooled_b_base, 6),
        "brier_rel_improvement": round((pooled_b_base - pooled_b_model) / pooled_b_base, 6),
        "accuracy_model": round(accuracy(pm, y), 6),
        "accuracy_baseline": round(accuracy(pb, y), 6),
        "logloss_model": round(log_loss(pm, y), 6),
        "logloss_baseline": round(log_loss(pb, y), 6),
    }
    rows.append(pooled)

    out = pd.DataFrame(rows)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_PATH.with_suffix(".csv.tmp")
    out.to_csv(tmp, index=False)
    tmp.replace(RESULTS_PATH)

    # Uncertainty on the pooled Brier difference, baseline minus model.
    # Positive means the model scored lower Brier than the baseline.
    se_diff = (pb - y) ** 2 - (pm - y) ** 2
    ci_lo, ci_hi = moving_block_ci(se_diff)

    config = {
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sample": "development only, 1962-02-28 .. 2009-12-31",
        "scheme": "expanding-window walk-forward, calendar-year test blocks",
        "embargo_months": EMBARGO_MONTHS,
        "first_test_year": FIRST_TEST_YEAR,
        "last_test_year": LAST_TEST_YEAR,
        "features": FEATURES,
        "models": [
            "constant baseline at the fold's training positive rate",
            "logistic regression, unpenalised (C=inf), lbfgs, max_iter=1000",
        ],
        "transform": "standardisation fit on the training fold only",
        "primary_metric": "Brier",
        "n_folds": len(folds),
    }

    print_report(df, folds, out, pooled, ci_lo, ci_hi, config)
    append_log(config, out, pooled, ci_lo, ci_hi)
    return 0


def print_report(df, folds, out, pooled, ci_lo, ci_hi, config) -> None:
    n_dev = len(df)
    n_test = int(pooled["n_test"])
    print("Walk-forward training - development sample only")
    print("=" * 78)
    print(f"development rows loaded : {n_dev}  ({n_dev // 12} independent 12-month blocks)")
    print(f"holdout                 : not read, not loaded, not counted")
    print(f"folds                   : {len(folds)}  (test years {FIRST_TEST_YEAR}-{LAST_TEST_YEAR})")
    print(f"embargo                 : {EMBARGO_MONTHS} months, training end to first test point")
    print(f"pooled test rows        : {n_test}  ({n_test // 12} independent 12-month blocks)")
    print()

    cols = ["fold", "train_start", "train_end", "test_start", "test_end", "n_train",
            "n_test", "baseline_rate", "brier_model", "brier_baseline",
            "brier_rel_improvement"]
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(out[cols].to_string(index=False))
    print()

    wins = int((out.iloc[:-1]["brier_model"] < out.iloc[:-1]["brier_baseline"]).sum())
    print(f"folds where the model's Brier is below its baseline's : {wins} of {len(folds)}")
    print()
    print("POOLED")
    print(f"  Brier, logistic regression : {pooled['brier_model']}")
    print(f"  Brier, baseline            : {pooled['brier_baseline']}")
    print(f"  relative change            : {pooled['brier_rel_improvement']}")
    print(f"  moving-block bootstrap 95% CI on the Brier difference")
    print(f"  (baseline minus model, block 12 months, {BOOTSTRAP_RESAMPLES} resamples)")
    print(f"                             : [{ci_lo:.6f}, {ci_hi:.6f}]")
    print()
    print("REPORTING ONLY - cannot be promoted over Brier")
    print(f"  accuracy  model {pooled['accuracy_model']}   baseline {pooled['accuracy_baseline']}")
    print(f"  log loss  model {pooled['logloss_model']}   baseline {pooled['logloss_baseline']}")
    print()
    print(f"wrote {RESULTS_PATH.relative_to(ROOT)}")
    print(f"logged to {LOG_PATH.relative_to(ROOT)}")


CONFIG_MARK = "<!-- config: "


def previous_config() -> dict | None:
    """The configuration of the last entry in the log, or None if there is none.

    Each entry embeds its own configuration as a comment, so "what changed" is
    computed against the record rather than remembered.
    """
    if not LOG_PATH.exists():
        return None
    last = None
    for line in LOG_PATH.read_text().splitlines():
        if line.startswith(CONFIG_MARK):
            last = line[len(CONFIG_MARK):].rsplit("-->", 1)[0].strip()
    if last is None:
        return None
    try:
        return json.loads(last)
    except json.JSONDecodeError:
        return None


def describe_change(config: dict, prev: dict | None) -> str:
    """What changed since the previous entry, key by key."""
    comparable = {k: v for k, v in config.items() if k != "run_utc"}
    if prev is None:
        return (
            "First entry. Nothing to compare against: this is the first "
            "configuration recorded for this study."
        )
    changed = []
    for key in sorted(set(comparable) | set(prev)):
        before, after = prev.get(key, "(absent)"), comparable.get(key, "(absent)")
        if before != after:
            changed.append(f"`{key}`: {before!r} -> {after!r}")
    if not changed:
        return "Nothing changed. Same configuration, re-run."
    return "Changed from the previous entry - " + "; ".join(changed) + "."


def append_log(config, out, pooled, ci_lo, ci_hi) -> None:
    comparable = {k: v for k, v in config.items() if k != "run_utc"}
    blob = json.dumps(comparable, sort_keys=True)
    config_sha = hashlib.sha256(blob.encode()).hexdigest()[:12]
    change = describe_change(config, previous_config())

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not LOG_PATH.exists():
        LOG_PATH.write_text(
            "# Experiment log\n\n"
            "Every configuration tried, in the order it was tried, with the date, what\n"
            "changed from the previous entry, and the result. Appended automatically by\n"
            "`scripts/train_walkforward.py` on every run, so a configuration cannot be\n"
            "run without being recorded, and \"what changed\" is computed against the\n"
            "previous entry rather than remembered.\n\n"
            "All entries below are **development sample only**. The holdout is unread.\n"
        )

    wins = int((out.iloc[:-1]["brier_model"] < out.iloc[:-1]["brier_baseline"]).sum())
    n_folds = len(out) - 1
    entry = [
        "",
        "---",
        "",
        f"## {config['run_utc']} - config `{config_sha}`",
        "",
        f"**What changed.** {change}",
        "",
        f"- **Sample.** {config['sample']}. Holdout not read.",
        f"- **Scheme.** {config['scheme']}, {config['n_folds']} folds, "
        f"test years {config['first_test_year']}-{config['last_test_year']}, "
        f"{config['embargo_months']}-month embargo.",
        f"- **Features.** {', '.join(config['features'])}.",
        f"- **Models.** {'; '.join(config['models'])}.",
        f"- **Transform.** {config['transform']}.",
        f"- **Primary metric.** {config['primary_metric']}.",
        "",
        "| Quantity | Value |",
        "| --- | --- |",
        f"| Pooled test rows | {pooled['n_test']} ({int(pooled['n_test']) // 12} independent blocks) |",
        f"| Pooled Brier, logistic | {pooled['brier_model']} |",
        f"| Pooled Brier, baseline | {pooled['brier_baseline']} |",
        f"| Relative change in Brier | {pooled['brier_rel_improvement']} |",
        f"| Block-bootstrap 95% CI, baseline minus model | [{ci_lo:.6f}, {ci_hi:.6f}] |",
        f"| Folds with model Brier below baseline | {wins} of {n_folds} |",
        f"| Pooled accuracy, model / baseline (reporting only) | "
        f"{pooled['accuracy_model']} / {pooled['accuracy_baseline']} |",
        f"| Pooled log loss, model / baseline (reporting only) | "
        f"{pooled['logloss_model']} / {pooled['logloss_baseline']} |",
        "",
        f"Per-fold results: `data/walkforward_results.csv`.",
        "",
        f"{CONFIG_MARK}{blob} -->",
        "",
    ]
    with LOG_PATH.open("a") as fh:
        fh.write("\n".join(entry))


if __name__ == "__main__":
    sys.exit(main())
