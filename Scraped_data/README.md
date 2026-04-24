# Stock News & Sentiment Pipeline (GDELT)

This module builds a scalable pipeline to collect, process, and analyze global news articles for major tech stocks using the GDELT dataset.

---

## Overview

For each stock ticker (e.g., AAPL, NVDA, MSFT), the pipeline:

- Queries the :contentReference[oaicite:0]{index=0}
- Expands queries using company aliases (e.g., Apple, Apple Inc)
- Handles rate limits and retries automatically
- Deduplicates articles by URL
- Saves results incrementally (resumable runs)
- Scrapes full article text

---

## Key Features

### News Collection
- Time-windowed search for scalable retrieval
- Automatic window splitting for large result sets
- Resume support from saved CSV files

### Article Text Extraction
Uses :contentReference[oaicite:2]{index=2} for:
- Clean text extraction
- HTML fallback parsing if needed

---

## Supported Stocks

- AAPL (Apple)
- NVDA (NVIDIA)
- GOOGL (Google / Alphabet)
- META (Facebook / Meta)
- MSFT (Microsoft)
- NFLX (Netflix)
- ORCL (Oracle)

---

## Output

Each ticker generates a CSV file:

### Columns include:
- title
- url
- seendate
- source metadata
- ticker
- optional article text