import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_cfg():
    return {
        "features": {"history": {"return_windows": [7, 30, 90],
                                 "vol_window": 30}},
        "targets": {"task_b_horizons": [30, 90],
                    "horizon_tolerance_days": 7},
    }


@pytest.fixture
def toy_panel():
    """Two cards, one variant, weekly snapshots for ~6 months.

    card 'a-1' doubles smoothly; card 'a-2' is flat at 10.
    """
    dates = pd.date_range("2025-01-01", periods=26, freq="7D")
    rows = []
    for i, d in enumerate(dates):
        rows.append({"card_id": "a-1", "variant": "holofoil",
                     "snapshot_date": d, "price_market": 10.0 * 2 ** (i / 25),
                     "set_id": "a", "set_name": "Alpha", "series": "S",
                     "rarity": "Rare Holo", "artist": "x", "supertype": "Pokémon",
                     "subtype": "Basic", "energy_type": "Fire", "hp": "120",
                     "retreat_cost": 2, "attack_count": 2, "pokedex_number": 6,
                     "card_number": 4, "set_size": 100,
                     "release_date": pd.Timestamp("2024-06-01")})
        rows.append({"card_id": "a-2", "variant": "holofoil",
                     "snapshot_date": d, "price_market": 10.0,
                     "set_id": "a", "set_name": "Alpha", "series": "S",
                     "rarity": "Common", "artist": "y", "supertype": "Pokémon",
                     "subtype": "Basic", "energy_type": "Water", "hp": "60",
                     "retreat_cost": 1, "attack_count": 1, "pokedex_number": 7,
                     "card_number": 10, "set_size": 100,
                     "release_date": pd.Timestamp("2024-06-01")})
    return pd.DataFrame(rows)
