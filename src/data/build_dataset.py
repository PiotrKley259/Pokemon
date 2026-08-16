"""Flatten raw JSON snapshots into data/processed/cards.parquet.

One row per (card_id, variant, snapshot_date), where variant is a tcgplayer
price block (normal / holofoil / reverseHolofoil / ...). Cardmarket averages
are attached at the card level where available.

Usage:
    python -m src.data.build_dataset
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path

log = logging.getLogger("build_dataset")


def _card_static_fields(card: dict) -> dict:
    """Metadata that does not depend on variant or snapshot date."""
    set_info = card.get("set", {}) or {}
    attacks = card.get("attacks") or []
    pokedex = card.get("nationalPokedexNumbers") or []
    subtypes = card.get("subtypes") or []
    types = card.get("types") or []
    return {
        "card_id": card["id"],
        "name": card.get("name"),
        "supertype": card.get("supertype"),
        "subtype": subtypes[0] if subtypes else None,
        "rarity": card.get("rarity"),
        "artist": card.get("artist"),
        "hp": pd.to_numeric(card.get("hp"), errors="coerce"),
        "energy_type": types[0] if types else None,
        "retreat_cost": card.get("convertedRetreatCost"),
        "attack_count": len(attacks),
        "pokedex_number": pokedex[0] if pokedex else None,
        "card_number": pd.to_numeric(card.get("number"), errors="coerce"),
        "set_id": set_info.get("id"),
        "set_name": set_info.get("name"),
        "series": set_info.get("series"),
        "set_size": set_info.get("total"),
        "release_date": set_info.get("releaseDate"),
        "image_url": (card.get("images") or {}).get("small"),
    }


def _price_rows(card: dict, snapshot_date: str, variants: list[str]) -> list[dict]:
    """One row per variant found in the tcgplayer price block."""
    static = _card_static_fields(card)
    tcg_prices = ((card.get("tcgplayer") or {}).get("prices")) or {}
    cm = ((card.get("cardmarket") or {}).get("prices")) or {}
    rows = []
    for variant, block in tcg_prices.items():
        if variant not in variants or not isinstance(block, dict):
            continue
        rows.append({
            **static,
            "variant": variant,
            "snapshot_date": snapshot_date,
            "price_low": block.get("low"),
            "price_mid": block.get("mid"),
            "price_high": block.get("high"),
            "price_market": block.get("market"),
            "cm_avg": cm.get("averageSellPrice"),
            "cm_trend": cm.get("trendPrice"),
            "cm_avg30": cm.get("avg30"),
        })
    return rows


def build_dataset(cfg: dict) -> pd.DataFrame:
    raw_dir = resolve_path(cfg, "raw_dir")
    snapshots_dir = raw_dir / "snapshots"
    variants = cfg["dataset"]["variants"]
    if not snapshots_dir.exists():
        raise FileNotFoundError(
            f"{snapshots_dir} not found - run `python -m src.data.collect` first"
        )

    rows: list[dict] = []
    for snap_dir in sorted(snapshots_dir.iterdir()):
        if not snap_dir.is_dir():
            continue
        snapshot_date = snap_dir.name
        pages = sorted(snap_dir.glob("page_*.json"))
        n_before = len(rows)
        for page in pages:
            payload = json.loads(page.read_text(encoding="utf-8"))
            for card in payload.get("data", []):
                rows.extend(_price_rows(card, snapshot_date, variants))
        log.info("Snapshot %s: %d pages, %d priced rows",
                 snapshot_date, len(pages), len(rows) - n_before)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No priced rows found in any snapshot")

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["release_date"] = pd.to_datetime(
        df["release_date"], format="%Y/%m/%d", errors="coerce"
    )
    df = df.sort_values(["card_id", "variant", "snapshot_date"]).reset_index(drop=True)

    out = resolve_path(cfg, "dataset")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    log.info("Wrote %d rows (%d cards, %d snapshots) to %s",
             len(df), df["card_id"].nunique(),
             df["snapshot_date"].nunique(), out)
    return df


def main() -> None:
    build_dataset(load_config())


if __name__ == "__main__":
    main()
