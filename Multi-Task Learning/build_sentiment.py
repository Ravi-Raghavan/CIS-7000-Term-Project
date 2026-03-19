"""
build_sentiment.py
──────────────────
Full sentiment pipeline for SPA-MSJF:
  Step 1 — Scrape GDELT month-by-month (2020-2026) for AAPL/GOOG/META/NVDA/TSLA
  Step 2 — Score article titles with FinBERT (GPU-accelerated)
  Step 3 — Aggregate to daily features, align to trading calendar, fill gaps

Outputs  →  Scraped_data/sentiment_daily/<TICKER>_sentiment.csv
Columns  →  date | sent_score | sent_count_log | sent_pos_ratio

  sent_score      : mean FinBERT compound score  (-1 = very negative, +1 = very positive)
  sent_count_log  : log(1 + num_articles)         — normalised article volume
  sent_pos_ratio  : fraction of positive articles  (0–1)

Dates with no coverage → forward-filled up to 3 days, then neutral (0, 0, 0.333).
Pre-2020 dates (training only) → neutral, model learns to ignore absent signal.

Usage:
    py -3.12 build_sentiment.py              # full pipeline
    py -3.12 build_sentiment.py --skip-scrape   # re-score existing raw data
    py -3.12 build_sentiment.py --skip-finbert  # re-aggregate already scored data
"""

import argparse
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

# Stocks to scrape — TSLA added (was missing from original scrape).
# Single keyword strings only — GDELT API rejects multi-keyword lists
# (raises "Parentheses may only be used around OR'd statements").
STOCKS = {
    "AAPL": "Apple",
    "GOOG": "Google",
    "META": "Meta",
    "NVDA": "NVIDIA",
    "TSLA": "Tesla",
}

SCRAPE_START  = "2020-01-01"   # covers all of val+test and last 3yr of train
SCRAPE_END    = "2026-02-28"

RAW_DIR  = Path("../Scraped_data/gdeltdoc_historical")   # monthly raw CSVs cache
OUT_DIR  = Path("../Scraped_data/sentiment_daily")        # final daily CSVs

FINBERT_MODEL       = "ProsusAI/finbert"
FINBERT_BATCH_SIZE  = 64                  # increase if GPU has headroom
GDELT_SLEEP_S       = 1.5                 # seconds between API calls (rate limit)
GDELT_MAX_PER_MONTH = 250                 # GDELT max records per query


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():   return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


def month_ranges(start: str, end: str):
    """Yield (start_str, end_str) for each calendar month between start and end."""
    cur     = datetime.strptime(start, "%Y-%m-%d").replace(day=1)
    end_dt  = datetime.strptime(end,   "%Y-%m-%d")
    while cur <= end_dt:
        next_m = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield (
            cur.strftime("%Y-%m-%d"),
            min(next_m - timedelta(days=1), end_dt).strftime("%Y-%m-%d"),
        )
        cur = next_m


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — GDELT historical scrape
# ─────────────────────────────────────────────────────────────────────────────

def scrape_ticker(ticker: str, keywords: list, raw_dir: Path) -> pd.DataFrame:
    """
    Query GDELT month by month and cache each month as its own CSV.
    Already-cached months are skipped so the script is safely resumable.
    """
    from gdeltdoc import GdeltDoc, Filters

    gd       = GdeltDoc()
    tick_dir = raw_dir / ticker
    tick_dir.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for start, end in month_ranges(SCRAPE_START, SCRAPE_END):
        cache = tick_dir / f"{start[:7]}.csv"

        if cache.exists():
            try:
                df = pd.read_csv(cache)
                if not df.empty:
                    all_dfs.append(df)
            except Exception:
                pass   # empty or corrupt cache file — skip silently
            continue

        try:
            # keyword must be a plain string — GDELT rejects list/tuple inputs
            kw = keywords if isinstance(keywords, str) else keywords[0]
            f  = Filters(
                keyword=kw,
                start_date=start,
                end_date=end,
                num_records=GDELT_MAX_PER_MONTH,
                language="English",
            )
            df = gd.article_search(f)
            df.to_csv(cache, index=False)
            if not df.empty:
                all_dfs.append(df)
            print(f"    {start[:7]} : {len(df):>4} articles")
        except Exception as exc:
            print(f"    {start[:7]} : FAILED — {exc}")
            pd.DataFrame().to_csv(cache, index=False)   # mark as attempted

        time.sleep(GDELT_SLEEP_S)

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def scrape_all(raw_dir: Path) -> dict[str, pd.DataFrame]:
    results = {}
    for ticker, keywords in STOCKS.items():
        print(f"\n  [{ticker}] scraping {SCRAPE_START} → {SCRAPE_END}  …")
        df = scrape_ticker(ticker, keywords, raw_dir)
        results[ticker] = df
        print(f"  [{ticker}] total: {len(df)} articles")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — FinBERT scoring
# ─────────────────────────────────────────────────────────────────────────────

def load_finbert_pipeline(device: str):
    """Load FinBERT text-classification pipeline once and return it."""
    from transformers import pipeline as hf_pipeline
    print(f"  Loading FinBERT on {device} …")
    return hf_pipeline(
        "text-classification",
        model=FINBERT_MODEL,
        device=0 if device == "cuda" else (-1 if device == "cpu" else 0),
        top_k=None,           # return all 3 label scores
        truncation=True,
        max_length=512,
    )


def finbert_compound(pipe, texts: list[str]) -> np.ndarray:
    """
    Run FinBERT on texts in batches.
    Compound = P(positive) – P(negative)  →  [-1, +1]
    """
    compounds = []
    for i in range(0, len(texts), FINBERT_BATCH_SIZE):
        batch   = texts[i : i + FINBERT_BATCH_SIZE]
        results = pipe(batch)
        for preds in results:
            scores   = {p["label"].lower(): p["score"] for p in preds}
            compound = scores.get("positive", 0.0) - scores.get("negative", 0.0)
            compounds.append(float(compound))
        if i % (FINBERT_BATCH_SIZE * 5) == 0 and i > 0:
            print(f"      … {i}/{len(texts)} scored")
    return np.array(compounds, dtype=np.float32)


def load_raw_for_ticker(ticker: str, raw_dir: Path) -> pd.DataFrame:
    """Concatenate all monthly CSVs for a ticker."""
    tick_dir = raw_dir / ticker
    if not tick_dir.exists():
        return pd.DataFrame()
    dfs = []
    for f in sorted(tick_dir.glob("*.csv")):
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def score_all(raw_dir: Path, device: str) -> dict[str, pd.DataFrame]:
    pipe    = load_finbert_pipeline(device)
    scored  = {}

    for ticker in STOCKS:
        df = load_raw_for_ticker(ticker, raw_dir)
        if df.empty or "title" not in df.columns:
            print(f"  [{ticker}] no data to score")
            scored[ticker] = pd.DataFrame()
            continue

        # ── Clean before scoring ──────────────────────────────────────────────
        # Drop rows with missing title
        df = df.dropna(subset=["title"])
        df = df[df["title"].str.strip() != ""]

        # Deduplicate by URL (same article republished)
        if "url" in df.columns:
            df = df.drop_duplicates(subset=["url"])

        # English only (GDELT filter sometimes leaks)
        if "language" in df.columns:
            df = df[df["language"].str.lower() == "english"]

        print(f"\n  [{ticker}] scoring {len(df)} articles …")
        df = df.copy()
        df["finbert_compound"] = finbert_compound(pipe, df["title"].tolist())
        df["finbert_positive"] = (df["finbert_compound"] > 0.05).astype(int)

        scored[ticker] = df
        print(f"  [{ticker}] done  |  compound: "
              f"{df['finbert_compound'].min():.3f} → {df['finbert_compound'].max():.3f}"
              f"  |  pos: {df['finbert_positive'].mean()*100:.1f}%")

    return scored


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Daily aggregation + gap filling
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse article-level rows to one row per trading day.
    Returns: date | sent_score | sent_count_log | sent_pos_ratio
    """
    if df.empty:
        return pd.DataFrame(columns=["date", "sent_score", "sent_count_log", "sent_pos_ratio"])

    # Parse date from seendate (GDELT format: 20240115T120000Z)
    if "seendate" in df.columns:
        df["_dt"] = pd.to_datetime(df["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    elif "date" in df.columns:
        df["_dt"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        return pd.DataFrame(columns=["date", "sent_score", "sent_count_log", "sent_pos_ratio"])

    df = df.dropna(subset=["_dt", "finbert_compound"])
    df["date"] = df["_dt"].dt.normalize().dt.strftime("%Y-%m-%d")

    daily = (
        df.groupby("date")
        .agg(
            sent_score     = ("finbert_compound", "mean"),
            article_count  = ("finbert_compound", "count"),
            positive_count = ("finbert_positive", "sum"),
        )
        .reset_index()
    )

    daily["sent_count_log"] = np.log1p(daily["article_count"])
    daily["sent_pos_ratio"] = daily["positive_count"] / daily["article_count"]

    return daily[["date", "sent_score", "sent_count_log", "sent_pos_ratio"]]


def align_to_trading_calendar(
    daily: pd.DataFrame,
    trading_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Reindex daily sentiment to every trading date:
      1. Forward-fill up to 3 days (covers weekends / public holidays)
      2. Remaining NaN → neutral: sent_score=0, count=0, pos_ratio=1/3
    Pre-2020 dates (no scraping coverage) will all be neutral.
    """
    all_date_strs = trading_dates.strftime("%Y-%m-%d")
    full = pd.DataFrame({"date": all_date_strs})
    merged = full.merge(daily, on="date", how="left")

    feat_cols = ["sent_score", "sent_count_log", "sent_pos_ratio"]
    merged[feat_cols] = merged[feat_cols].ffill(limit=3)

    merged["sent_score"]     = merged["sent_score"].fillna(0.0)
    merged["sent_count_log"] = merged["sent_count_log"].fillna(0.0)
    merged["sent_pos_ratio"] = merged["sent_pos_ratio"].fillna(1 / 3)

    return merged[["date"] + feat_cols]


def load_trading_dates(price_dir: Path) -> pd.DatetimeIndex:
    """Load AAPL price CSV to get the common trading calendar as DatetimeIndex."""
    try:
        df = pd.read_csv(price_dir / "Historical_Data_AAPL.csv")
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None)
        return pd.DatetimeIndex(df["Date"].sort_values().dt.normalize())
    except Exception:
        print("  WARNING: could not load price data — using business-day calendar")
        return pd.bdate_range(start="2012-01-01", end="2026-03-01")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build daily sentiment features for SPA-MSJF")
    parser.add_argument("--skip-scrape",   action="store_true",
                        help="skip GDELT scraping, use cached raw monthly CSVs")
    parser.add_argument("--skip-finbert",  action="store_true",
                        help="skip FinBERT scoring, reload pre-scored CSVs")
    args = parser.parse_args()

    device = get_device()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Sentiment Pipeline  |  device: {device}")
    print(f"  Scrape window: {SCRAPE_START} → {SCRAPE_END}")
    print(f"  Stocks: {list(STOCKS.keys())}")
    print(f"{'='*60}")

    # ── Step 1 ────────────────────────────────────────────────────────────────
    if not args.skip_scrape:
        print("\n── Step 1: GDELT Scrape ────────────────────────────────")
        scrape_all(RAW_DIR)
    else:
        print("\n── Step 1: Skipped (using cached raw data) ─────────────")

    # ── Step 2 ────────────────────────────────────────────────────────────────
    if not args.skip_finbert:
        print("\n── Step 2: FinBERT Scoring ─────────────────────────────")
        scored = score_all(RAW_DIR, device)
        for ticker, df in scored.items():
            if not df.empty:
                df.to_csv(RAW_DIR / f"{ticker}_scored.csv", index=False)
    else:
        print("\n── Step 2: Loading pre-scored data ─────────────────────")
        scored = {}
        for ticker in STOCKS:
            path = RAW_DIR / f"{ticker}_scored.csv"
            scored[ticker] = pd.read_csv(path) if path.exists() else pd.DataFrame()
            print(f"  {ticker}: {len(scored[ticker])} rows loaded")

    # ── Step 3 ────────────────────────────────────────────────────────────────
    print("\n── Step 3: Daily Aggregation ───────────────────────────")
    trading_dates = load_trading_dates(Path("../Finance Data"))
    print(f"  Trading calendar: {trading_dates.min().date()} → {trading_dates.max().date()} "
          f"({len(trading_dates)} days)")

    for ticker, df in scored.items():
        daily    = aggregate_daily(df)
        aligned  = align_to_trading_calendar(daily, trading_dates)
        out_path = OUT_DIR / f"{ticker}_sentiment.csv"
        aligned.to_csv(out_path, index=False)

        n_covered = (aligned["sent_count_log"] > 0).sum()
        pct = n_covered / len(aligned) * 100
        print(f"  {ticker}: {n_covered}/{len(aligned)} days have news ({pct:.1f}%) → {out_path.name}")

    print(f"\n  Done. CSVs in {OUT_DIR}/")
    print("  Next: run train.py — sentiment features loaded automatically\n")


if __name__ == "__main__":
    main()
