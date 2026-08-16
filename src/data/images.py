"""Download and preprocess card art for the cold-start visual model.

Downloads every unique card image URL from the processed dataset into
data/images/raw/<card_id>.jpg (resumable: existing files are skipped), then
produces two 336px crops per card:

- data/images/full/<card_id>.jpg : the whole card, resized;
- data/images/art/<card_id>.jpg  : the illustration window only, cropping out
  the frame, name bar and attack text box so the encoder sees mostly art.

Usage:
    python -m src.data.images            # download + crop everything
    python -m src.data.images --limit 50 # smoke test
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from PIL import Image

from src.config import load_config, resolve_path

log = logging.getLogger("images")


def download_images(cfg: dict, limit: int | None = None) -> Path:
    df = pd.read_parquet(resolve_path(cfg, "dataset"),
                         columns=["card_id", "image_url"])
    cards = (df.dropna(subset=["image_url"])
             .drop_duplicates("card_id")
             .reset_index(drop=True))
    if limit:
        cards = cards.head(limit)

    raw_dir = resolve_path(cfg, "images_dir") / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    sess = requests.Session()
    n_new = 0
    for row in cards.itertuples():
        out = raw_dir / f"{row.card_id}.jpg"
        if out.exists():
            continue
        try:
            resp = sess.get(row.image_url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("Failed %s: %s", row.card_id, exc)
            continue
        out.write_bytes(resp.content)
        n_new += 1
        time.sleep(0.1)  # be polite to the image CDN
        if n_new % 200 == 0:
            log.info("Downloaded %d new images", n_new)
    log.info("Download done: %d new, %d total on disk",
             n_new, len(list(raw_dir.glob("*.jpg"))))
    return raw_dir


def crop_card(img: Image.Image, box_frac: tuple[float, float, float, float] | None,
              size: int) -> Image.Image:
    """Optionally crop by fractional box, then resize to (size, size)."""
    img = img.convert("RGB")
    if box_frac is not None:
        w, h = img.size
        l, t, r, b = box_frac
        img = img.crop((int(l * w), int(t * h), int(r * w), int(b * h)))
    return img.resize((size, size), Image.BICUBIC)


def build_crops(cfg: dict, limit: int | None = None) -> None:
    images_dir = resolve_path(cfg, "images_dir")
    raw_dir = images_dir / "raw"
    size = cfg["coldstart"]["image_size"]
    art_box = tuple(cfg["coldstart"]["art_crop"])

    files = sorted(raw_dir.glob("*.jpg"))
    if limit:
        files = files[:limit]
    for kind, box in [("full", None), ("art", art_box)]:
        out_dir = images_dir / kind
        out_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for f in files:
            out = out_dir / f.name
            if out.exists():
                continue
            try:
                with Image.open(f) as img:
                    crop_card(img, box, size).save(out, quality=92)
                n += 1
            except OSError as exc:
                log.warning("Bad image %s: %s", f.name, exc)
        log.info("Crop '%s': %d new files in %s", kind, n, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    if not args.skip_download:
        download_images(cfg, args.limit)
    build_crops(cfg, args.limit)


if __name__ == "__main__":
    main()
