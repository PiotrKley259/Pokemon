import numpy as np
import pandas as pd
import pytest

from src.features.price_decomposition import PriceDecomposer, species_key


def make_frame():
    """3 species: dex 6 in 12 printings at +1.0 above its baseline, dex 999
    in 2 printings at +1.0, plus filler species pinning the baseline at 0."""
    rows = []
    for i in range(12):
        rows.append(dict(card_id=f"c6-{i}", name="Charizard", pokedex_number=6,
                         rarity="Rare", variant="holofoil", series="SWSH",
                         log_price=1.0))
    for i in range(2):
        rows.append(dict(card_id=f"c999-{i}", name="Obscuremon",
                         pokedex_number=999, rarity="Rare",
                         variant="holofoil", series="SWSH", log_price=1.0))
    for i in range(40):
        rows.append(dict(card_id=f"f-{i}", name=f"Filler{i}",
                         pokedex_number=100 + i, rarity="Rare",
                         variant="holofoil", series="SWSH", log_price=0.0))
    return pd.DataFrame(rows)


def test_species_key_falls_back_to_name():
    df = pd.DataFrame({"pokedex_number": [6, None], "name": ["X", "Prof Oak"]})
    keys = species_key(df)
    assert keys.iloc[0] == "6"
    assert keys.iloc[1] == "Prof Oak"


def test_shrinkage_scales_with_printing_count():
    df = make_frame()
    dec = PriceDecomposer(shrinkage_k=5.0).fit(df)
    comp = dec.transform(df)
    eq_many = comp.loc[df["name"] == "Charizard", "character_equity"].iloc[0]
    eq_few = comp.loc[df["name"] == "Obscuremon", "character_equity"].iloc[0]
    # Same raw premium, but 12 printings should retain much more of it
    # than 2 printings: n/(n+k) = 12/17 vs 2/7.
    assert eq_many > eq_few > 0
    assert eq_many == pytest.approx(1.0 * 12 / 17, rel=1e-6)
    assert eq_few == pytest.approx(1.0 * 2 / 7, rel=1e-6)


def test_decomposition_is_additive():
    df = make_frame()
    comp = PriceDecomposer(shrinkage_k=5.0).fit_transform(df)
    recon = (comp["rarity_set_baseline"] + comp["character_equity"]
             + comp["art_premium"])
    assert np.allclose(recon, df["log_price"])


def test_unseen_species_gets_zero_equity():
    df = make_frame()
    dec = PriceDecomposer().fit(df)
    new = pd.DataFrame([dict(card_id="new-1", name="Brandnewmon",
                             pokedex_number=1500, rarity="Rare",
                             variant="holofoil", series="SWSH")])
    comp = dec.transform(new)
    assert comp["character_equity"].iloc[0] == 0.0
    # Baseline still resolves through the (rarity, variant, era) table.
    assert np.isfinite(comp["rarity_set_baseline"].iloc[0])


def test_unseen_rarity_falls_back_gracefully():
    df = make_frame()
    dec = PriceDecomposer().fit(df)
    new = pd.DataFrame([dict(card_id="new-2", name="X", pokedex_number=6,
                             rarity="Never Seen Rare", variant="holofoil",
                             series="Future Era")])
    comp = dec.transform(new)
    assert np.isfinite(comp["rarity_set_baseline"].iloc[0])
