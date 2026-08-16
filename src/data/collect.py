"""Collector for the Pokemon TCG API (https://api.pokemontcg.io/v2).

Downloads every card page and caches the raw JSON under
data/raw/snapshots/<YYYY-MM-DD>/page_XXXX.json. Card metadata is immutable,
but the tcgplayer/cardmarket price blocks change daily, so re-running on a
new day creates a new snapshot directory and builds a price history over time.

Properties:
- resumable: a page already on disk for today's snapshot is never re-fetched;
- rate limited: sleeps between requests (config: api.rate_limit_rps);
- retries with exponential backoff on 429/5xx/network errors.

Usage:
    python -m src.data.collect            # pull today's snapshot
    python -m src.data.collect --max-pages 5   # smoke test
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import time
from pathlib import Path

import requests

from src.config import load_config, resolve_path

log = logging.getLogger("collect")


def _session(api_key: str | None) -> requests.Session:
    sess = requests.Session()
    if api_key:
        sess.headers["X-Api-Key"] = api_key
    sess.headers["User-Agent"] = "pokemon-tcg-price-model (educational project)"
    return sess


def _get_with_retries(
    sess: requests.Session,
    url: str,
    params: dict,
    max_retries: int,
    backoff: float,
) -> dict | None:
    """Fetch with retries; return None (instead of raising) when exhausted.

    api.pokemontcg.io intermittently answers ~half of all requests with an
    instant 500/502 regardless of load, so transient failures are the normal
    case: retry generously with a capped backoff, and let the caller decide
    what a permanently failed page means.
    """
    for attempt in range(1, max_retries + 1):
        wait = min(backoff * 1.5 ** (attempt - 1), 30.0)
        try:
            resp = sess.get(url, params=params, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504):
                log.warning("HTTP %s on %s, retry %d/%d in %.0fs",
                            resp.status_code, url, attempt, max_retries, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("Request error %s, retry %d/%d in %.0fs",
                        exc, attempt, max_retries, wait)
            time.sleep(wait)
    log.error("Giving up on %s params=%s after %d retries",
              url, params, max_retries)
    return None


def collect_sets(cfg: dict, sess: requests.Session, raw_dir: Path) -> None:
    """Cache the /sets endpoint once (set metadata rarely changes)."""
    out = raw_dir / "sets.json"
    if out.exists():
        log.info("sets.json already cached, skipping")
        return
    data = _get_with_retries(
        sess, f"{cfg['api']['base_url']}/sets", {"pageSize": 250},
        cfg["api"]["max_retries"], cfg["api"]["retry_backoff_seconds"],
    )
    if data is None:
        log.warning("Could not fetch /sets this run; will retry next run")
        return
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    log.info("Saved %d sets to %s", len(data.get("data", [])), out)


def collect_cards(
    cfg: dict,
    snapshot_date: str | None = None,
    max_pages: int | None = None,
) -> Path:
    """Download all card pages for one snapshot date. Returns snapshot dir."""
    snapshot_date = snapshot_date or dt.date.today().isoformat()
    raw_dir = resolve_path(cfg, "raw_dir")
    snap_dir = raw_dir / "snapshots" / snapshot_date
    snap_dir.mkdir(parents=True, exist_ok=True)

    sess = _session(os.environ.get("POKEMONTCG_API_KEY"))
    collect_sets(cfg, sess, raw_dir)

    base_url = f"{cfg['api']['base_url']}/cards"
    page_size = cfg["api"]["page_size"]
    min_interval = 1.0 / cfg["api"]["rate_limit_rps"]

    page = 1
    skipped: list[int] = []
    consecutive_failures = 0
    while True:
        if max_pages is not None and page > max_pages:
            log.info("Reached max_pages=%d, stopping", max_pages)
            break
        out = snap_dir / f"page_{page:04d}.json"
        if out.exists():
            # Resumability: never re-download what we already have.
            page += 1
            continue
        t0 = time.monotonic()
        data = _get_with_retries(
            sess, base_url, {"page": page, "pageSize": page_size},
            cfg["api"]["max_retries"], cfg["api"]["retry_backoff_seconds"],
        )
        if data is None:
            # One dead page must not kill the whole snapshot: skip it and
            # keep going. Its file is never written, so the next run of
            # `make collect` retries exactly the missing pages.
            skipped.append(page)
            consecutive_failures += 1
            if consecutive_failures >= 5:
                log.error("%d consecutive pages failed - the API looks down; "
                          "stopping this run (re-run later to resume)",
                          consecutive_failures)
                break
            page += 1
            continue
        consecutive_failures = 0
        cards = data.get("data", [])
        if not cards:
            log.info("Empty page %d, collection complete", page)
            break
        out.write_text(json.dumps(data), encoding="utf-8")
        total = data.get("totalCount", "?")
        log.info("Saved page %d (%d cards, totalCount=%s)", page, len(cards), total)
        elapsed = time.monotonic() - t0
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        page += 1
    if skipped:
        log.warning("Snapshot incomplete: pages %s failed. Re-run "
                    "`python -m src.data.collect --snapshot-date %s` to "
                    "fill the gaps.", skipped, snapshot_date)
    return snap_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-date", default=None,
                        help="Override snapshot date (YYYY-MM-DD), default today")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Limit pages fetched (smoke testing)")
    args = parser.parse_args()
    cfg = load_config()
    snap_dir = collect_cards(cfg, args.snapshot_date, args.max_pages)
    log.info("Snapshot stored in %s", snap_dir)


if __name__ == "__main__":
    main()
