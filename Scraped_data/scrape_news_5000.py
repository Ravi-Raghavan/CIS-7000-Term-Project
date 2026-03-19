import os
import re
import time
import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from requests.exceptions import ConnectTimeout, ReadTimeout, Timeout, ConnectionError as ReqConnectionError
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from gdeltdoc import GdeltDoc, Filters
from gdeltdoc.errors import RateLimitError

try:
    import trafilatura
except Exception:
    trafilatura = None


STOCKS = {
    "AAPL": ["Apple", "Apple Inc"],
    "NVDA": ["NVIDIA", "Nvidia"],
    "GOOGL": ["Google", "Alphabet"],
    "META": ["Meta", "Facebook"],
    "ORCL": ["Oracle"],
    "NFLX": ["Netflix"],
    "MSFT": ["Microsoft"],
}


def sentiment_label(c):
    if c >= 0.05:
        return "Positive"
    if c <= -0.05:
        return "Negative"
    return "Neutral"


def safe_write_csv(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    try:
        os.replace(tmp, path)
        return True
    except PermissionError:
        print(f"File locked: {path} (close it). Data saved to: {tmp}")
        return False


def safe_keywords(keywords):
    if keywords is None:
        return []

    if isinstance(keywords, (str, int, float)):
        keywords = [keywords]
    else:
        try:
            iter(keywords)
        except TypeError:
            keywords = [keywords]

    out = []
    for k in keywords:
        if k is None:
            continue
        if isinstance(k, float) and pd.isna(k):
            continue
        k = str(k).replace("$", " ").replace("\n", " ").replace("\r", " ").strip()
        k = "".join(ch for ch in k if (ch.isalnum() or ch.isspace() or ch in ".'\"&/+-"))
        k = re.sub(r"\s+", " ", k).strip()
        if k:
            out.append(k)
    return out


def fetch_window(gd, keywords, start_dt, end_dt, language=None, max_records=250, min_window_minutes=30):
    kw = safe_keywords(keywords)
    args = dict(keyword=kw, start_date=start_dt, end_date=end_dt, num_records=max_records)
    if language:
        args["language"] = language

    df = None
    for i in range(10):
        try:
            df = gd.article_search(Filters(**args))
            break
        except (RateLimitError, ConnectTimeout, ReadTimeout, Timeout, ReqConnectionError):
            wait = min(20 * (2 ** i), 600)
            print(f"GDELT busy/timeout. Sleeping {wait}s...")
            time.sleep(wait)
        except ValueError:
            args["keyword"] = safe_keywords(args.get("keyword"))
            df = gd.article_search(Filters(**args))
            break

    if df is None or len(df) == 0:
        return pd.DataFrame()

    window_minutes = (end_dt - start_dt).total_seconds() / 60.0
    if len(df) >= max_records and window_minutes > min_window_minutes:
        mid = start_dt + (end_dt - start_dt) / 2
        left = fetch_window(gd, keywords, start_dt, mid, language, max_records, min_window_minutes)
        right = fetch_window(gd, keywords, mid, end_dt, language, max_records, min_window_minutes)
        return pd.concat([left, right], ignore_index=True)

    return df


def collect_until(ticker, names, out_path, target_rows, window_hours, max_days, sleep_s, language, save_every, days_back):
    gd = GdeltDoc()
    urls = set()
    now = datetime.now(timezone.utc)

    end_dt = now
    old = None

    if os.path.exists(out_path):
        old = pd.read_csv(out_path)
        if "url" in old.columns:
            urls.update(old["url"].dropna().astype(str).tolist())
        if days_back is None and "seendate" in old.columns:
            seen = pd.to_datetime(old["seendate"], errors="coerce", utc=True).dropna()
            if len(seen) > 0:
                end_dt = seen.min().to_pydatetime()

    start_limit = (now - timedelta(days=days_back)) if days_back is not None else (end_dt - timedelta(days=max_days))
    keywords = [ticker] + names

    chunks = []
    last_saved = len(urls)
    last_print = len(urls)

    while end_dt > start_limit and len(urls) < target_rows:
        start_dt = end_dt - timedelta(hours=window_hours)
        df = fetch_window(gd, keywords, start_dt, end_dt, language=language)

        if not df.empty and "url" in df.columns:
            df["url"] = df["url"].astype(str)
            df = df.drop_duplicates(subset=["url"])
            df = df[~df["url"].isin(urls)]
            if not df.empty:
                df["ticker"] = ticker
                urls.update(df["url"].tolist())
                chunks.append(df)

        if len(urls) - last_print >= 250:
            print(f"{ticker}: {len(urls)}/{target_rows}")
            last_print = len(urls)

        if len(urls) - last_saved >= save_every:
            new = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
            tmp = pd.concat([old, new], ignore_index=True) if old is not None else new
            if not tmp.empty and "url" in tmp.columns:
                tmp = tmp.drop_duplicates(subset=["url"])
            safe_write_csv(tmp, out_path)
            old = tmp
            chunks = []
            last_saved = len(urls)

        end_dt = start_dt
        if sleep_s:
            time.sleep(sleep_s)

    new = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    out = pd.concat([old, new], ignore_index=True) if old is not None else new
    if not out.empty and "url" in out.columns:
        out = out.drop_duplicates(subset=["url"])
    return out


def add_sentiment(df):
    if df.empty:
        return df

    analyzer = SentimentIntensityAnalyzer()
    if "title" not in df.columns:
        df["title"] = ""

    df["compound"] = df["title"].astype(str).apply(lambda t: analyzer.polarity_scores(t)["compound"])
    df["sentiment"] = df["compound"].apply(sentiment_label)

    if "seendate" in df.columns:
        seen = pd.to_datetime(df["seendate"], errors="coerce", utc=True)
        df["date"] = seen.dt.date
        df["_seen_dt_utc"] = seen

    return df


def filter_recent(df, days_back):
    if days_back is None or df.empty:
        return df
    if "_seen_dt_utc" not in df.columns:
        if "seendate" in df.columns:
            df["_seen_dt_utc"] = pd.to_datetime(df["seendate"], errors="coerce", utc=True)
        else:
            return df
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    return df[df["_seen_dt_utc"].notna() & (df["_seen_dt_utc"] >= cutoff)].copy()


def strip_html(html):
    html = re.sub(r"(?is)<(script|style|noscript).*?>.*?(</\1>)", " ", html)
    html = re.sub(r"(?is)<.*?>", " ", html)
    html = re.sub(r"\s+", " ", html).strip()
    return html


def fetch_article_text(url, session, timeout=20):
    if not isinstance(url, str) or not url.strip():
        return ""

    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}

    try:
        if trafilatura:
            downloaded = trafilatura.fetch_url(url)
            if downloaded:
                text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
                if text:
                    return re.sub(r"\s+", " ", text).strip()

        r = session.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return ""
        return strip_html(r.text)
    except Exception:
        return ""


def add_article_text(df, out_path, text_sleep, save_every, max_articles):
    if df.empty or "url" not in df.columns:
        return df

    if "article_text" not in df.columns:
        df["article_text"] = ""

    need = df["article_text"].isna() | (df["article_text"].astype(str).str.len() == 0)
    idxs = df.index[need].tolist()
    if max_articles is not None:
        idxs = idxs[:max_articles]

    session = requests.Session()
    done = 0

    for idx in idxs:
        df.at[idx, "article_text"] = fetch_article_text(df.at[idx, "url"], session=session)
        done += 1

        if done % 100 == 0:
            print(f"Text scraped: {done}/{len(idxs)}")

        if done % save_every == 0:
            safe_write_csv(df, out_path)

        if text_sleep:
            time.sleep(text_sleep)

    safe_write_csv(df, out_path)
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="gdeltdoc_out")
    p.add_argument("--target", type=int, default=5000)
    p.add_argument("--window-hours", type=int, default=12)
    p.add_argument("--max-days", type=int, default=3650)
    p.add_argument("--sleep", type=float, default=2.0)
    p.add_argument("--language", default="eng")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--days-back", type=int, default=None)

    p.add_argument("--scrape-text", action="store_true")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--text-sleep", type=float, default=0.5)
    p.add_argument("--text-save-every", type=int, default=50)
    p.add_argument("--text-max", type=int, default=None)

    args = p.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    for ticker, names in STOCKS.items():
        out_path = os.path.join(args.outdir, f"{ticker}_articles.csv")
        try:
            if args.text_only:
                if not os.path.exists(out_path):
                    print(f"{ticker}: no file yet -> {out_path}")
                    continue
                df = pd.read_csv(out_path)
            else:
                df = collect_until(
                    ticker=ticker,
                    names=names,
                    out_path=out_path,
                    target_rows=args.target,
                    window_hours=args.window_hours,
                    max_days=args.max_days,
                    sleep_s=args.sleep,
                    language=args.language,
                    save_every=args.save_every,
                    days_back=args.days_back,
                )

            df = add_sentiment(df)
            df = filter_recent(df, args.days_back)

            keep = [c for c in df.columns if c != "_seen_dt_utc"]
            df = df[keep]
            safe_write_csv(df, out_path)

            if args.scrape_text:
                df = add_article_text(
                    df=df,
                    out_path=out_path,
                    text_sleep=args.text_sleep,
                    save_every=args.text_save_every,
                    max_articles=args.text_max,
                )

            print(f"{ticker}: {len(df)} rows -> {out_path}")
        except Exception as e:
            print(f"{ticker}: failed ({type(e).__name__}) {e}")


if __name__ == "__main__":
    main()
