#!/usr/bin/env python3
"""Assemble the monthly market-timing panel from free public sources.

Data only. No features, no model, no predictions. Nothing in the output is derived
from anything else in the output.

Sources
    Shiller ie_data.xls  -> sp500_index (real total return index), cape
    FRED    DGS10        -> dgs10
    FRED    DTB3         -> dtb3
    ALFRED  UNRATE       -> unrate, as first published (point-in-time)

The point-in-time handling and every publication lag assumption are documented in
README.md, which was written before this file. Read it before using the output.

Usage
    python scripts/build_market_panel.py [--out PATH] [--no-cache] [--offline]
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import random
import datetime as dt
import io
import json
import os
import re
import sys
import textwrap
import threading
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "data" / "market_panel.parquet"
DEFAULT_PROVENANCE = REPO_ROOT / "data" / "market_panel_provenance.txt"
DEFAULT_CACHE = REPO_ROOT / "data" / ".cache"

SHILLER_LANDING = "https://shillerdata.com/"
SHILLER_FALLBACK = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
ALFRED_CSV = "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
ALFRED_DOWNLOAD_PAGE = "https://alfred.stlouisfed.org/series/downloaddata"

# Left unset by default: some egress proxies reject requests that override the
# User-Agent, and the sources here are happy with the library default. Set
# MARKET_PANEL_USER_AGENT to identify yourself if your environment allows it.
USER_AGENT = os.environ.get("MARKET_PANEL_USER_AGENT")
TIMEOUT = 90
ALFRED_WORKERS = 4
MAX_ATTEMPTS = 5      # outbound goes through a proxy that occasionally drops connections
BACKOFF_BASE = 1.5

# Publication lag assumptions. Fixed in README.md before this module was written.
# A value on row `date` must be treated as unknown until date + lag_days.
LAG_DAYS = {
    "sp500_index": 15,  # CPI for month M lands mid-M+1; the real series needs it
    "cape": 15,         # same CPI lag, and a LOWER BOUND: earnings tail is interpolated
    "dgs10": 1,         # H.15 publishes the month-end close next business day
    "dtb3": 1,          # same
    # unrate has no assumed lag: it is measured from the true ALFRED vintage date
}

LAG_BASIS = {
    "sp500_index": "assumed: CPI for month M published ~15th of M+1",
    "cape": "assumed LOWER BOUND: CPI lag; earnings tail interpolated and later revised",
    "dgs10": "assumed: month-end close published next business day (3 days if Fri month-end)",
    "dtb3": "assumed: month-end close published next business day (3 days if Fri month-end)",
    "unrate": "MEASURED per row from the ALFRED vintage in which the month first appeared",
}

VALUE_COLUMNS = ["sp500_index", "cape", "dgs10", "dtb3", "unrate"]


class Fetcher:
    """HTTP with an on-disk cache, so re-runs do not hammer the sources."""

    def __init__(self, cache_dir: Path, use_cache: bool = True, offline: bool = False):
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.offline = offline
        self._local = threading.local()
        self._lock = threading.Lock()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def session(self) -> requests.Session:
        """One session per thread; a dropped proxy connection poisons only its own pool."""
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            if USER_AGENT:
                s.headers["User-Agent"] = USER_AGENT
            self._local.session = s
        return s

    def _reset_session(self) -> None:
        s = getattr(self._local, "session", None)
        if s is not None:
            s.close()
        self._local.session = None

    def get(self, url: str, params: dict | None = None, cache_key: str | None = None) -> bytes:
        path = self.cache_dir / cache_key if cache_key else None
        if path is not None and self.use_cache and path.exists():
            return path.read_bytes()
        if self.offline:
            raise RuntimeError(f"--offline set and no cached copy for {cache_key or url}")

        last: Exception | None = None
        for attempt in range(MAX_ATTEMPTS):
            try:
                resp = self.session.get(url, params=params, timeout=TIMEOUT)
                resp.raise_for_status()
                body = resp.content
                break
            except Exception as exc:  # noqa: BLE001 - transient network/proxy faults
                last = exc
                if attempt == MAX_ATTEMPTS - 1:
                    raise
                self._reset_session()
                delay = BACKOFF_BASE ** attempt + random.uniform(0, 0.5)
                with self._lock:
                    log(f"    retry {attempt + 1}/{MAX_ATTEMPTS - 1} in {delay:.1f}s ({exc.__class__.__name__})")
                time.sleep(delay)
        else:  # pragma: no cover - loop always breaks or raises
            raise last  # type: ignore[misc]

        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        return body


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def display_path(path: Path) -> str:
    """Repo-relative when it is inside the repo, absolute otherwise."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def month_end(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


# --------------------------------------------------------------------------------------
# Shiller: sp500_index and cape
# --------------------------------------------------------------------------------------

def resolve_shiller_url(fetcher: Fetcher) -> str:
    """Find the live ie_data.xls link. Yale's copy is stale; shillerdata.com is current."""
    try:
        html = fetcher.get(SHILLER_LANDING, cache_key="shiller_landing.html").decode(
            "utf-8", errors="replace"
        )
        hits = re.findall(r'href="([^"]*ie_data\.xls[^"]*)"', html, flags=re.I)
        if hits:
            url = hits[0]
            if url.startswith("//"):
                url = "https:" + url
            elif url.startswith("/"):
                url = SHILLER_LANDING.rstrip("/") + url
            return url
    except Exception as exc:  # noqa: BLE001 - fall back, but say why
        log(f"  ! could not scrape shillerdata.com ({exc}); falling back to Yale")
    return SHILLER_FALLBACK


def _flatten_headers(raw: pd.DataFrame, header_rows: int = 8) -> dict[int, str]:
    """Shiller's header is smeared over ~8 rows. Join each column's pieces."""
    out = {}
    for col in raw.columns:
        parts = [
            str(v).strip()
            for v in raw.iloc[:header_rows, col].tolist()
            if pd.notna(v) and str(v).strip()
        ]
        out[col] = re.sub(r"\s+", " ", " ".join(parts)).lower()
    return out


def _locate(headers: dict[int, str], want: str, exclude: tuple[str, ...], fallback: int) -> int:
    matches = [
        c for c, h in headers.items()
        if want in h and not any(x in h for x in exclude)
    ]
    if len(matches) == 1:
        return matches[0]
    log(
        f"  ! header match for {want!r} was ambiguous ({matches}); "
        f"falling back to fixed column {fallback}"
    )
    return fallback


def fetch_shiller(fetcher: Fetcher) -> tuple[pd.DataFrame, dict]:
    url = resolve_shiller_url(fetcher)
    log(f"  Shiller: {url}")
    blob = fetcher.get(url, cache_key="shiller_ie_data.xls")
    raw = pd.read_excel(io.BytesIO(blob), sheet_name="Data", header=None)

    headers = _flatten_headers(raw)
    # "Real Total Return Price" vs the CAPE variant, which says "cyclically adjusted".
    col_tr = _locate(headers, "total return price", ("cyclically",), fallback=9)
    col_cape = _locate(headers, "p/e10 or cape", ("tr p/e10",), fallback=12)

    body = raw.iloc[8:].copy()
    frac = pd.to_numeric(body[0], errors="coerce")
    body = body[frac.notna()].copy()
    frac = frac[frac.notna()]

    years = frac.astype(int)
    months = ((frac - years) * 100).round().astype(int)
    ok = months.between(1, 12)
    if not ok.all():
        raise ValueError(f"Shiller date column produced impossible months: {months[~ok].head()}")

    df = pd.DataFrame({
        "date": [month_end(y, m) for y, m in zip(years, months)],
        "sp500_index": pd.to_numeric(body[col_tr], errors="coerce").to_numpy(),
        "cape": pd.to_numeric(body[col_cape], errors="coerce").to_numpy(),
    })
    df = df.drop_duplicates(subset="date", keep="last").sort_values("date").reset_index(drop=True)

    meta = {
        "url": url,
        "sheet": "Data",
        "column_sp500_index": f"col {col_tr}: {headers[col_tr]}",
        "column_cape": f"col {col_cape}: {headers[col_cape]}",
        "rows_parsed": int(len(df)),
    }
    return df, meta


# --------------------------------------------------------------------------------------
# FRED daily series: dgs10, dtb3
# --------------------------------------------------------------------------------------

def fetch_fred_daily(fetcher: Fetcher, series_id: str, out_name: str) -> tuple[pd.DataFrame, dict]:
    log(f"  FRED {series_id}")
    blob = fetcher.get(FRED_CSV, params={"id": series_id}, cache_key=f"fred_{series_id}.csv")
    raw = pd.read_csv(io.BytesIO(blob))
    date_col = raw.columns[0]
    raw[date_col] = pd.to_datetime(raw[date_col])
    values = pd.to_numeric(raw[raw.columns[1]], errors="coerce")

    daily = pd.Series(values.to_numpy(), index=raw[date_col]).dropna()
    # Month-end alignment: last *available* observation inside the month. This is
    # selection, not filling. A month with no observation at all stays null.
    monthly = daily.resample("ME").last()

    df = pd.DataFrame({"date": monthly.index, out_name: monthly.to_numpy()})
    meta = {
        "url": f"{FRED_CSV}?id={series_id}",
        "series_id": series_id,
        "native_frequency": "daily",
        "daily_observations": int(len(daily)),
        "daily_first": str(daily.index.min().date()),
        "daily_last": str(daily.index.max().date()),
        "month_end_rule": "last available observation within the calendar month",
    }
    return df, meta


# --------------------------------------------------------------------------------------
# ALFRED: point-in-time unrate
# --------------------------------------------------------------------------------------

def alfred_vintage_dates(fetcher: Fetcher, series_id: str) -> list[str]:
    """The real vintage dates, scraped from ALFRED's download page."""
    blob = fetcher.get(
        ALFRED_DOWNLOAD_PAGE,
        params={"seid": series_id},
        cache_key=f"alfred_{series_id}_vintages.html",
    ).decode("utf-8", errors="replace")

    # Restrict to the vintage-date <select>; the page has other date-valued options
    # (the observation-start picker in particular).
    m = re.search(
        r"<select[^>]*selected_vintage_dates[^>]*>(.*?)</select>", blob, re.S | re.I
    )
    if m is None:
        raise RuntimeError(
            "could not find the selected_vintage_dates <select> on the ALFRED download "
            "page; its markup may have changed"
        )
    dates = sorted(set(re.findall(r'value="(\d{4}-\d{2}-\d{2})"', m.group(1))))
    if len(dates) < 100:
        raise RuntimeError(f"implausibly few vintage dates scraped: {len(dates)}")
    return dates


def _parse_alfred_csv(blob: bytes) -> pd.Series:
    raw = pd.read_csv(io.BytesIO(blob))
    idx = pd.to_datetime(raw[raw.columns[0]])
    vals = pd.to_numeric(raw[raw.columns[1]], errors="coerce")
    return pd.Series(vals.to_numpy(), index=idx).dropna()


def fetch_unrate_point_in_time(fetcher: Fetcher) -> tuple[pd.DataFrame, dict]:
    """Each month gets the value from the earliest vintage in which it appears."""
    series_id = "UNRATE"
    vintages = alfred_vintage_dates(fetcher, series_id)
    log(f"  ALFRED {series_id}: {len(vintages)} vintages, {vintages[0]} -> {vintages[-1]}")

    def grab(v: str) -> tuple[str, pd.Series]:
        blob = fetcher.get(
            ALFRED_CSV,
            params={"id": series_id, "vintage_date": v},
            cache_key=f"alfred/{series_id}_{v}.csv",
        )
        return v, _parse_alfred_csv(blob)

    frames: dict[str, pd.Series] = {}
    with futures.ThreadPoolExecutor(ALFRED_WORKERS) as pool:
        for i, (v, s) in enumerate(pool.map(grab, vintages), start=1):
            frames[v] = s
            if i % 100 == 0 or i == len(vintages):
                log(f"    fetched {i}/{len(vintages)} vintages")

    first_value: dict[pd.Timestamp, float] = {}
    first_vintage: dict[pd.Timestamp, str] = {}
    for v in vintages:  # chronological
        for obs_date, val in frames[v].items():
            if obs_date not in first_value:
                first_value[obs_date] = float(val)
                first_vintage[obs_date] = v

    earliest_vintage = vintages[0]
    # Everything already published before ALFRED's archive opens is revised data. The
    # single exception is that vintage's own newest month, which IS its first release.
    backfilled_through = frames[earliest_vintage].index.max()

    rows = []
    for obs_date in sorted(first_value):
        v = first_vintage[obs_date]
        genuine = not (v == earliest_vintage and obs_date < backfilled_through)
        release = pd.Timestamp(v)
        end = obs_date + pd.offsets.MonthEnd(0)
        rows.append({
            "date": end,
            "unrate": first_value[obs_date],
            "unrate_release_date": release,
            "unrate_lag_days": int((release - end).days),
            "unrate_is_first_release": bool(genuine),
        })
    df = pd.DataFrame(rows)

    genuine_rows = df[df["unrate_is_first_release"]]
    meta = {
        "url": f"{ALFRED_CSV}?id={series_id}&vintage_date=<each vintage>",
        "vintage_list_url": f"{ALFRED_DOWNLOAD_PAGE}?seid={series_id}",
        "series_id": series_id,
        "point_in_time": "YES - first published value per reference month",
        "vintages_used": len(vintages),
        "vintage_first": vintages[0],
        "vintage_last": vintages[-1],
        "first_release_from": str(genuine_rows["date"].min().date()),
        "revised_row_count": int((~df["unrate_is_first_release"]).sum()),
        "revised_rows_span": (
            f"{df.loc[~df['unrate_is_first_release'], 'date'].min().date()} -> "
            f"{df.loc[~df['unrate_is_first_release'], 'date'].max().date()}"
            if (~df["unrate_is_first_release"]).any() else "none"
        ),
        "measured_lag_days_min": int(genuine_rows["unrate_lag_days"].min()),
        "measured_lag_days_median": int(genuine_rows["unrate_lag_days"].median()),
        "measured_lag_days_max": int(genuine_rows["unrate_lag_days"].max()),
        "measured_lag_days_nonpositive": int((genuine_rows["unrate_lag_days"] <= 0).sum()),
    }
    return df, meta


def fetch_unrate_revised_fallback(fetcher: Fetcher) -> tuple[pd.DataFrame, dict]:
    """Only if ALFRED is unreachable. Loudly flagged as contaminated."""
    log("  ! ALFRED unreachable - falling back to FRED's REVISED UNRATE")
    blob = fetcher.get(FRED_CSV, params={"id": "UNRATE"}, cache_key="fred_UNRATE.csv")
    raw = pd.read_csv(io.BytesIO(blob))
    idx = pd.to_datetime(raw[raw.columns[0]])
    vals = pd.to_numeric(raw[raw.columns[1]], errors="coerce")
    s = pd.Series(vals.to_numpy(), index=idx).dropna()

    ends = [d + pd.offsets.MonthEnd(0) for d in s.index]
    # BLS releases month M in early M+1; without vintages we can only assume.
    df = pd.DataFrame({
        "date": ends,
        "unrate": s.to_numpy(),
        "unrate_release_date": pd.NaT,
        "unrate_lag_days": 15,
        "unrate_is_first_release": False,
    })
    meta = {
        "url": f"{FRED_CSV}?id=UNRATE",
        "series_id": "UNRATE",
        "point_in_time": "NO - REVISED SERIES, LOOKAHEAD-CONTAMINATED",
        "warning": (
            "ALFRED was unreachable. Every unrate value here is the CURRENT revised "
            "estimate, not what was published at the time. Seasonal factors are "
            "re-estimated annually and restate years of history. unrate_lag_days is an "
            "assumed 15 days, not measured. DO NOT treat this column as point-in-time."
        ),
        "assumed_lag_days": 15,
    }
    return df, meta


# --------------------------------------------------------------------------------------
# Panel assembly
# --------------------------------------------------------------------------------------

def build_panel(parts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    starts = [p["date"].min() for p in parts.values()]
    ends = [p["date"].max() for p in parts.values()]
    index = pd.date_range(min(starts), max(ends), freq="ME")
    panel = pd.DataFrame({"date": index})

    for df in parts.values():
        panel = panel.merge(df, on="date", how="left")

    for col, lag in LAG_DAYS.items():
        panel[f"{col}_lag_days"] = lag

    ordered = (
        ["date"]
        + VALUE_COLUMNS
        + [f"{c}_lag_days" for c in VALUE_COLUMNS]
        + ["unrate_release_date", "unrate_is_first_release"]
    )
    panel = panel[ordered]

    panel["unrate_is_first_release"] = panel["unrate_is_first_release"].astype("boolean")
    for col in VALUE_COLUMNS:
        panel[f"{col}_lag_days"] = panel[f"{col}_lag_days"].astype("Int64")
    return panel


def series_ranges(panel: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for col in VALUE_COLUMNS:
        present = panel.loc[panel[col].notna(), "date"]
        if len(present):
            span = panel[(panel["date"] >= present.min()) & (panel["date"] <= present.max())]
            gaps = [str(d.date()) for d in span.loc[span[col].isna(), "date"]]
        else:
            gaps = []
        out[col] = {
            "start": str(present.min().date()) if len(present) else None,
            "end": str(present.max().date()) if len(present) else None,
            "non_null": int(len(present)),
            "null": int(panel[col].isna().sum()),
            "interior_gaps": gaps,
            "lag_days": ("measured per row" if col == "unrate" else LAG_DAYS[col]),
            "lag_basis": LAG_BASIS[col],
        }
    return out


def render_header(panel: pd.DataFrame, sources: dict, ranges: dict, pulled: str, partial: dict) -> str:
    unrate_pit = sources["unrate"].get("point_in_time", "unknown")
    lines = [
        "=" * 86,
        "MARKET TIMING PANEL - PROVENANCE AND POINT-IN-TIME HEADER",
        "=" * 86,
        "",
        f"Built by      : scripts/build_market_panel.py",
        f"Pull date     : {pulled}",
        f"Rows          : {len(panel)}",
        f"Date range    : {panel['date'].min().date()} -> {panel['date'].max().date()}",
        f"Frequency     : monthly, month-end aligned",
        "",
        "Data only. No features, no model, no predictions. No column is derived from",
        "any other column in this file.",
        "",
        "-" * 86,
        "SOURCES",
        "-" * 86,
    ]
    for name, meta in sources.items():
        lines.append(f"\n[{name}]")
        for k, v in meta.items():
            lines.append(f"  {k:<24}: {v}")

    lines += [
        "",
        "-" * 86,
        "PER-SERIES RANGE, NULLS, AND PUBLICATION LAG",
        "-" * 86,
        "",
        "A value on row `date` must be treated as unknown until date + lag_days.",
        "A signal formed at month-end M may only use rows where date + lag_days <= M.",
        "",
    ]
    for col, r in ranges.items():
        gaps = r["interior_gaps"]
        shown = ", ".join(gaps[:8]) + (" ..." if len(gaps) > 8 else "") if gaps else "none"
        lines += [
            f"[{col}]",
            f"  observed range          : {r['start']} -> {r['end']}",
            f"  non-null / null         : {r['non_null']} / {r['null']}",
            f"  gaps inside that range  : {len(gaps)}  {shown}",
            f"  lag_days                : {r['lag_days']}",
            f"  lag basis               : {r['lag_basis']}",
            "",
        ]

    lines += [
        "-" * 86,
        "POINT-IN-TIME STATUS",
        "-" * 86,
        "",
        f"  unrate       : {unrate_pit}",
    ]
    if sources["unrate"].get("warning"):
        lines += ["", textwrap.indent(textwrap.fill(sources["unrate"]["warning"], 78), "    ")]
    else:
        n_rev = sources["unrate"].get("revised_row_count", 0)
        lines += [
            f"                 First-release values from {sources['unrate']['first_release_from']}",
            f"                 onward, taken from the earliest ALFRED vintage in which each",
            f"                 month appears. Release dates are observed, so unrate_lag_days",
            f"                 is measured, not assumed.",
            f"                 {n_rev} earlier rows predate ALFRED's archive and are REVISED",
            f"                 data: they are lookahead-contaminated. Filter on",
            f"                 unrate_is_first_release to exclude them.",
        ]
    lines += [
        "",
        "  dgs10, dtb3  : Effectively point-in-time. Market rates, not revised. The",
        "                 month-end level is set at the close of the last business day",
        "                 of the month and published the next business day.",
        "",
        "  sp500_index  : NOT point-in-time. REVISED DATA. Shiller publishes no vintage",
        "  cape           archive, so these are as-of the pull date above.",
        "                 sp500_index is Shiller's Real Total Return Price, quoted in the",
        "                 purchasing power of the source file's final month. Every new",
        "                 release rescales the whole column by one constant (+2.8% between",
        "                 the 2023-09 and 2024-09 files), so LEVELS ARE NOT COMPARABLE",
        "                 ACROSS PULLS - month-over-month ratios are, since the constant",
        "                 cancels. Apart from that constant, only the final partial month",
        "                 of a release is materially revised.",
        "                 cape additionally depends on S&P earnings, which Shiller",
        "                 interpolates for recent months and revises later. Its 15-day lag",
        "                 is a lower bound; recent-month cape is provisional.",
        "",
        "-" * 86,
        "KNOWN GAPS AND LAG ANOMALIES",
        "-" * 86,
        "",
        "  unrate 2025-10 is absent from the source itself. The US government shutdown",
        "  meant no household survey was conducted for October 2025, so BLS never",
        "  published a rate for that month. FRED's own series is null there too. It is",
        "  left null here and is NOT filled.",
        "",
        "  unrate 2025-09 carries a 51-day lag: the same shutdown pushed its release to",
        "  2025-11-20. That is the measured release date, not an estimate.",
        "",
    ]
    nonpos = sources["unrate"].get("measured_lag_days_nonpositive")
    if nonpos:
        lines += [
            f"  {nonpos} early-1960s rows have unrate_lag_days <= 0. This is not an error. In",
            "  that era BLS published the Monthly Report on the Labor Force within the",
            "  reference month itself, so the number was genuinely knowable before month",
            "  end. A non-positive lag simply means no embargo applies to that row.",
            "",
        ]
    if partial.get("dropped"):
        lines += [
            "-" * 86,
            "TRAILING MONTH EXCLUDED",
            "-" * 86,
            "",
            f"  This build ran with --drop-partial-month. The row for {partial['month_end']},",
            f"  the calendar month still in progress at the pull date ({partial['today']}), was",
            "  present in the sources but is NOT in this file. What it would have held:",
            "",
        ]
        for col, last_obs in partial["last_obs"].items():
            lines.append(f"    {col:<12} {last_obs}")
        lines += [
            "",
            "  Rerun without --drop-partial-month to keep it as a month-to-date row.",
            "",
        ]
    if partial["is_partial"]:
        lines += [
            "-" * 86,
            "PARTIAL TRAILING MONTH - READ THIS",
            "-" * 86,
            "",
            f"  The last row, {partial['month_end']}, is the CURRENT calendar month, which had",
            f"  not ended at the pull date ({partial['today']}). Its values are month-to-date,",
            "  not month-end:",
            "",
        ]
        for col, last_obs in partial["last_obs"].items():
            lines.append(f"    {col:<12} last underlying observation: {last_obs}")
        lines += [
            "",
            "  The row is retained rather than silently dropped, because dropping data is",
            "  also a choice. Do not treat it as a completed month-end observation. Rerun",
            "  with --drop-partial-month to exclude it.",
            "",
        ]
    lines += [
        "-" * 86,
        "NULL POLICY",
        "-" * 86,
        "",
        "Nulls are never filled, interpolated, or forward-filled. A null means the source",
        "had no value for that month. Series begin as early as each source allows, so",
        "early rows are null for the series that did not exist yet.",
        "=" * 86,
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--no-cache", action="store_true", help="ignore cached downloads")
    ap.add_argument("--offline", action="store_true", help="use only cached downloads")
    ap.add_argument(
        "--drop-partial-month", action="store_true",
        help="drop the trailing row when it is the current, unfinished calendar month",
    )
    args = ap.parse_args()

    fetcher = Fetcher(args.cache_dir, use_cache=not args.no_cache, offline=args.offline)
    pulled = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    log("Fetching sources...")
    shiller, shiller_meta = fetch_shiller(fetcher)
    dgs10, dgs10_meta = fetch_fred_daily(fetcher, "DGS10", "dgs10")
    dtb3, dtb3_meta = fetch_fred_daily(fetcher, "DTB3", "dtb3")
    try:
        unrate, unrate_meta = fetch_unrate_point_in_time(fetcher)
    except Exception as exc:  # noqa: BLE001 - degrade loudly, never silently
        log(f"  ! ALFRED point-in-time path failed: {exc}")
        unrate, unrate_meta = fetch_unrate_revised_fallback(fetcher)

    panel = build_panel({
        "shiller": shiller, "dgs10": dgs10, "dtb3": dtb3, "unrate": unrate,
    })

    # The trailing row is the current calendar month if the month has not ended yet.
    # Its values are month-to-date, not month-end. Flag it loudly; drop only on request.
    today = dt.date.today()
    last_end = panel["date"].max()
    partial = {
        "is_partial": (last_end.year, last_end.month) == (today.year, today.month),
        "month_end": str(last_end.date()),
        "today": str(today),
        "last_obs": {
            "sp500_index": f"Shiller month {last_end.strftime('%Y.%m')} (source file is month-to-date)",
            "cape": f"Shiller month {last_end.strftime('%Y.%m')} (source file is month-to-date)",
            "dgs10": dgs10_meta["daily_last"],
            "dtb3": dtb3_meta["daily_last"],
            "unrate": "n/a - not yet released for this month",
        },
    }
    if partial["is_partial"] and args.drop_partial_month:
        panel = panel[panel["date"] < last_end].reset_index(drop=True)
        partial["is_partial"] = False
        partial["dropped"] = True
        log(f"  dropped partial trailing month {last_end.date()}")

    sources = {
        "sp500_index + cape": shiller_meta,
        "dgs10": dgs10_meta,
        "dtb3": dtb3_meta,
        "unrate": unrate_meta,
    }
    ranges = series_ranges(panel)
    header = render_header(panel, sources, ranges, pulled, partial)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(panel, preserve_index=False)
    table = table.replace_schema_metadata({
        **(table.schema.metadata or {}),
        b"market_panel_header": header.encode(),
        b"market_panel_provenance_json": json.dumps({
            "built_by": "scripts/build_market_panel.py",
            "pull_date": pulled,
            "rows": len(panel),
            "date_range": [str(panel["date"].min().date()), str(panel["date"].max().date())],
            "frequency": "monthly, month-end aligned",
            "partial_trailing_month": partial,
            "sources": sources,
            "series": ranges,
            "null_policy": "never filled or interpolated",
        }, indent=2, default=str).encode(),
    })
    pq.write_table(table, args.out, compression="snappy")
    args.provenance.write_text(header + "\n")

    # ---------------------------------------------------------------- required output
    print()
    print(f"Wrote {display_path(args.out)}")
    print(f"Wrote {display_path(args.provenance)}")
    print()
    print(f"Rows       : {len(panel)}")
    print(f"Date range : {panel['date'].min().date()} -> {panel['date'].max().date()}")
    print()
    print("Per-series observed start (as early as each source allows):")
    for col, r in ranges.items():
        print(f"  {col:<12} {r['start']} -> {r['end']}")
    print()
    print("Null count per column (nulls are NOT filled):")
    width = max(len(c) for c in panel.columns)
    for col in panel.columns:
        print(f"  {col:<{width}}  {int(panel[col].isna().sum())}")
    print()
    pit = sources["unrate"].get("point_in_time", "unknown")
    print(f"unrate point-in-time: {pit}")
    for col, r in ranges.items():
        if r["interior_gaps"]:
            print(f"gap inside {col} span (not filled): {', '.join(r['interior_gaps'])}")
    if partial["is_partial"]:
        print(
            f"WARNING: last row {partial['month_end']} is the current, unfinished month; "
            f"its values are month-to-date. Use --drop-partial-month to exclude it."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
