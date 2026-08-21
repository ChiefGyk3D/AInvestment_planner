from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ainvest.config import from_dict  # noqa: E402
from ainvest.data.synthetic import SyntheticProvider  # noqa: E402
from ainvest.stats import align_prices, estimate_moments  # noqa: E402

END = date(2026, 1, 1)
START = END - timedelta(days=730)


def base_config(**overrides) -> dict:
    """A small but complete config: two equities, two coins, three venues."""
    raw = {
        "budget": 300.0,
        "risk_profile": "balanced",
        "venues": {
            "public": {"fee_bps": 0, "min_notional": 1, "size_decimals": 5},
            "coinbase_advanced": {
                "fee_bps": 60, "spread_bps": 5, "min_notional": 2,
                "size_decimals": 8, "supports_withdrawal": True, "withdrawal_fee": 1.5,
            },
            "pricey": {"fee_bps": 0, "fixed_fee": 1.0, "min_notional": 10},
        },
        "assets": [
            {"symbol": "VTI", "asset_class": "equity", "venues": ["public"], "max_weight": 0.6},
            {"symbol": "GLD", "asset_class": "equity", "venues": ["public"], "max_weight": 0.4},
            {"symbol": "BTC", "asset_class": "crypto", "data_symbol": "BTC-USD",
             "venues": ["coinbase_advanced"], "max_weight": 0.3, "self_custody": True},
            {"symbol": "ETH", "asset_class": "crypto", "data_symbol": "ETH-USD",
             "venues": ["coinbase_advanced"], "max_weight": 0.3, "self_custody": True},
        ],
        "constraints": {"single_max": 0.5, "crypto_max": 0.25, "include_threshold": 0.01},
        "optimizer": {"objective": "risk_parity", "blend_equal_weight": 0.0, "restarts": 3},
        "data": {"provider": "synthetic", "min_observations": 60},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    return raw


@pytest.fixture
def cfg():
    return from_dict(base_config())


@pytest.fixture
def prices(cfg):
    frame = SyntheticProvider(seed=7).fetch(
        [a.feed_symbol for a in cfg.assets], START, END
    )
    frame = frame.rename(columns={a.feed_symbol: a.symbol for a in cfg.assets})
    return align_prices(frame, 60)


@pytest.fixture
def moments(prices):
    return estimate_moments(prices, mean_shrinkage=0.5)


@pytest.fixture
def latest(prices):
    return {c: float(prices[c].iloc[-1]) for c in prices.columns}
