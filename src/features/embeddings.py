"""Frozen image embeddings + leakage controls for the cold-start model.

Computing embeddings (needs torch; imported lazily so the rest of the project
runs without it):

    python -m src.features.embeddings --crop art
    python -m src.features.embeddings --crop full --encoder dinov2

Embeddings are cached to data/embeddings/<encoder>_<crop>.npy with the card
ids in <encoder>_<crop>_ids.json - existing cards are never re-encoded.

Leakage controls (used per split by the trainer, never fitted globally):

- `fit_pca` reduces to cfg pca_dims and reports variance retained;
- `rarity_probe_accuracy`: how well a linear probe reads rarity straight off
  the embeddings. High accuracy = the embedding is mostly a rarity detector,
  and any "visual" lift may just be rarity leaking through pixels.
- `EmbeddingResidualiser`: regresses every embedding dimension on
  (rarity, variant, artist, era) and keeps the residuals - the "style"
  signal left after removing what the tabular features already know.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import load_config, resolve_path

log = logging.getLogger("embeddings")

RESIDUALISE_COLS = ["rarity", "variant", "artist", "era"]


# --------------------------------------------------------------------------
# Encoding (torch, lazy)
# --------------------------------------------------------------------------
def _load_encoder(cfg: dict, encoder: str):
    import torch

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    if encoder == "clip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["coldstart"]["clip_model"],
            pretrained=cfg["coldstart"]["clip_pretrained"],
        )
        model = model.eval().to(device)

        @torch.no_grad()
        def embed(batch):
            return model.encode_image(batch.to(device)).float().cpu().numpy()

        return embed, preprocess, device
    if encoder == "dinov2":
        import torch
        from torchvision import transforms

        model = torch.hub.load("facebookresearch/dinov2",
                               cfg["coldstart"]["dinov2_model"])
        model = model.eval().to(device)
        preprocess = transforms.Compose([
            transforms.Resize((336, 336)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

        @torch.no_grad()
        def embed(batch):
            return model(batch.to(device)).float().cpu().numpy()

        return embed, preprocess, device
    raise ValueError(f"Unknown encoder {encoder!r}")


def compute_embeddings(cfg: dict, crop: str, encoder: str,
                       limit: int | None = None) -> Path:
    """Encode every cropped image not yet in the cache; append to the cache."""
    import torch
    from PIL import Image

    emb_dir = resolve_path(cfg, "embeddings_dir")
    emb_dir.mkdir(parents=True, exist_ok=True)
    npy_path = emb_dir / f"{encoder}_{crop}.npy"
    ids_path = emb_dir / f"{encoder}_{crop}_ids.json"

    done_ids: list[str] = (json.loads(ids_path.read_text())
                           if ids_path.exists() else [])
    done = set(done_ids)
    img_dir = resolve_path(cfg, "images_dir") / crop
    files = [f for f in sorted(img_dir.glob("*.jpg")) if f.stem not in done]
    if limit:
        files = files[:limit]
    if not files:
        log.info("Embedding cache up to date (%d cards)", len(done_ids))
        return npy_path

    embed, preprocess, device = _load_encoder(cfg, encoder)
    log.info("Encoding %d images with %s on %s", len(files), encoder, device)

    batch_size = cfg["coldstart"]["batch_size"]
    chunks, new_ids = [], []
    for i in range(0, len(files), batch_size):
        batch_files = files[i:i + batch_size]
        tensors = []
        for f in batch_files:
            with Image.open(f) as img:
                tensors.append(preprocess(img.convert("RGB")))
        chunks.append(embed(torch.stack(tensors)))
        new_ids.extend(f.stem for f in batch_files)
        if (i // batch_size) % 20 == 0:
            log.info("  %d / %d", i + len(batch_files), len(files))

    new_emb = np.concatenate(chunks)
    if npy_path.exists():
        new_emb = np.concatenate([np.load(npy_path), new_emb])
    np.save(npy_path, new_emb.astype(np.float32))
    ids_path.write_text(json.dumps(done_ids + new_ids))
    log.info("Cache now holds %d embeddings (dim %d) at %s",
             len(new_emb), new_emb.shape[1], npy_path)
    return npy_path


def load_embeddings(cfg: dict, crop: str, encoder: str
                    ) -> tuple[np.ndarray, list[str]]:
    emb_dir = resolve_path(cfg, "embeddings_dir")
    npy_path = emb_dir / f"{encoder}_{crop}.npy"
    ids_path = emb_dir / f"{encoder}_{crop}_ids.json"
    if not npy_path.exists():
        raise FileNotFoundError(
            f"{npy_path} missing - run `python -m src.features.embeddings "
            f"--crop {crop} --encoder {encoder}` first")
    return np.load(npy_path), json.loads(ids_path.read_text())


# --------------------------------------------------------------------------
# Leakage controls (fitted per split by the caller)
# --------------------------------------------------------------------------
def fit_pca(emb_train: np.ndarray, n_dims: int, seed: int
            ) -> tuple[PCA, float]:
    """Fit PCA on TRAINING embeddings only; return (pca, variance_retained)."""
    pca = PCA(n_components=min(n_dims, emb_train.shape[1], len(emb_train)),
              random_state=seed)
    pca.fit(emb_train)
    return pca, float(pca.explained_variance_ratio_.sum())


def _meta_frame(meta: pd.DataFrame) -> pd.DataFrame:
    d = pd.DataFrame(index=meta.index)
    d["rarity"] = meta["rarity"].fillna("unknown")
    d["variant"] = meta["variant"].fillna("unknown")
    d["artist"] = meta["artist"].fillna("unknown")
    d["era"] = meta["series"].fillna("unknown")
    return d


def rarity_probe_accuracy(emb: np.ndarray, rarity: pd.Series, seed: int,
                          cv: int = 3) -> float:
    """CV accuracy of a linear probe predicting rarity from embeddings alone.
    Reported in the README: high accuracy means the 'visual' signal is
    substantially a rarity detector."""
    y = rarity.fillna("unknown").to_numpy()
    keep = pd.Series(y).map(pd.Series(y).value_counts()) >= cv
    probe = Pipeline([
        ("impute", SimpleImputer()),
        ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
    ])
    scores = cross_val_score(probe, emb[keep.to_numpy()], y[keep.to_numpy()],
                             cv=cv, scoring="accuracy")
    return float(scores.mean())


class EmbeddingResidualiser:
    """Removes the (rarity, variant, artist, era)-predictable part of every
    embedding dimension, leaving residual 'style' components."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, emb: np.ndarray, meta: pd.DataFrame) -> "EmbeddingResidualiser":
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", min_frequency=3)
        X = self.ohe_.fit_transform(_meta_frame(meta))
        self.reg_ = Ridge(alpha=self.alpha)
        self.reg_.fit(X, emb)
        return self

    def transform(self, emb: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        X = self.ohe_.transform(_meta_frame(meta))
        return emb - self.reg_.predict(X)

    def fit_transform(self, emb: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
        return self.fit(emb, meta).transform(emb, meta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop", choices=["art", "full"], default="art")
    parser.add_argument("--encoder", choices=["clip", "dinov2"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config()
    encoder = args.encoder or cfg["coldstart"]["encoder"]
    compute_embeddings(cfg, args.crop, encoder, args.limit)


if __name__ == "__main__":
    main()
