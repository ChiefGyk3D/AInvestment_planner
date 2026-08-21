from __future__ import annotations

from pathlib import Path

import pytest

from ainvest.config import RISK_PROFILES, ConfigError, from_dict, load_config
from ainvest.models import Venue
from conftest import base_config


def test_shipped_example_config_loads():
    example = Path(__file__).resolve().parents[1] / "config" / "portfolio.example.yaml"
    cfg = load_config(example)
    assert cfg.assets and cfg.venues
    assert any(a.asset_class == "crypto" for a in cfg.assets)


def test_missing_file_points_at_init():
    with pytest.raises(ConfigError, match="ainvest init"):
        load_config("does-not-exist.yaml")


@pytest.mark.parametrize("profile", sorted(RISK_PROFILES))
def test_risk_profiles_set_the_crypto_cap(profile):
    cfg = from_dict({**base_config(), "risk_profile": profile, "constraints": {}})
    assert cfg.constraints.crypto_max == RISK_PROFILES[profile]["crypto_max"]


def test_explicit_constraints_override_the_profile():
    cfg = from_dict({**base_config(), "risk_profile": "conservative",
                     "constraints": {"crypto_max": 0.42}})
    assert cfg.constraints.crypto_max == 0.42


def test_unknown_risk_profile_is_rejected():
    with pytest.raises(ConfigError, match="unknown risk_profile"):
        from_dict({**base_config(), "risk_profile": "yolo"})


def test_typo_in_a_venue_name_is_caught_at_load():
    raw = base_config()
    raw["assets"][0]["venues"] = ["pubIic"]  # capital I, not an l
    with pytest.raises(ConfigError, match="undefined venue"):
        from_dict(raw)


def test_unknown_keys_are_reported_rather_than_ignored():
    raw = base_config()
    raw["optimizer"] = {"objectve": "min_vol"}
    with pytest.raises(ConfigError, match="unknown key"):
        from_dict(raw)

    raw = base_config()
    raw["assets"][0]["max_wieght"] = 0.5
    with pytest.raises(ConfigError, match="unknown key"):
        from_dict(raw)


def test_duplicate_symbols_are_rejected():
    raw = base_config()
    raw["assets"].append(dict(raw["assets"][0]))
    with pytest.raises(ConfigError, match="duplicate asset"):
        from_dict(raw)


def test_holdings_must_reference_known_assets():
    raw = base_config()
    raw["holdings"] = [{"symbol": "DOGE", "quantity": 1}]
    with pytest.raises(ConfigError, match="not in the universe"):
        from_dict(raw)


def test_asset_class_must_be_known():
    raw = base_config()
    raw["assets"][0]["asset_class"] = "beanie babies"
    with pytest.raises(ConfigError, match="expected one of"):
        from_dict(raw)


def test_contradictory_bucket_bounds_are_caught():
    with pytest.raises(ConfigError, match="cannot all be satisfied"):
        from_dict({**base_config(), "constraints": {
            "crypto_min": 0.7, "crypto_max": 0.8, "equity_min": 0.7,
        }})
    with pytest.raises(ConfigError, match="no way to be fully invested"):
        from_dict({**base_config(), "constraints": {"crypto_max": 0.2, "equity_max": 0.3}})


def test_a_minimum_above_its_own_maximum_is_caught_first():
    with pytest.raises(ConfigError, match="crypto bounds"):
        from_dict({**base_config(), "constraints": {"crypto_min": 0.7, "crypto_max": 0.25}})


def test_reserving_the_whole_budget_is_an_error():
    with pytest.raises(ConfigError, match="nothing to invest"):
        from_dict({**base_config(), "budget": 100, "cash_reserve": 100})


def test_cash_reserve_is_held_back():
    cfg = from_dict({**base_config(), "budget": 500, "cash_reserve": 200})
    assert cfg.investable == 300


class TestVenueCosts:
    def test_percentage_and_fixed_components(self):
        venue = Venue("v", fee_bps=50, spread_bps=10, fixed_fee=0.25)
        assert venue.cost(100) == pytest.approx(0.25 + 100 * 0.006)

    def test_minimum_fee_floor(self):
        assert Venue("v", fee_bps=10, min_fee=0.99).cost(1.0) == pytest.approx(0.99)

    def test_zero_notional_is_free(self):
        assert Venue("v", fee_bps=100, fixed_fee=5).cost(0) == 0.0

    def test_whole_share_venues_truncate(self):
        assert Venue("v", fractional=False).quantize(3.9) == 3.0

    def test_fractional_sizes_round_down_never_up(self):
        """Rounding a quantity up would spend more than the budget allowed."""
        venue = Venue("v", fractional=True, size_decimals=4)
        assert venue.quantize(3.987654) == 3.9876
        assert venue.quantize(3.9) == 3.9
        assert venue.quantize(0.00001) == 0.0
        assert venue.quantize(-1.0) == 0.0


def test_cheapest_venue_is_chosen(cfg):
    btc = cfg.asset("BTC")
    assert cfg.cheapest_venue(btc, 50).name == "coinbase_advanced"


def test_cheapest_venue_accounts_for_fixed_fees():
    raw = base_config()
    raw["assets"][0]["venues"] = ["public", "pricey"]
    cfg = from_dict(raw)
    vti = cfg.asset("VTI")
    # `pricey` charges a flat dollar; `public` is free, so it wins at any size.
    assert cfg.cheapest_venue(vti, 500).name == "public"


def test_cheapest_venue_prefers_one_that_accepts_the_order_size():
    """A venue you cannot use is not cheap, however low its fees."""
    raw = base_config()
    raw["venues"]["cheap_but_big"] = {"fee_bps": 1, "min_notional": 100}
    raw["venues"]["dear_but_small"] = {"fee_bps": 200, "min_notional": 1}
    raw["assets"][0]["venues"] = ["cheap_but_big", "dear_but_small"]
    cfg = from_dict(raw)
    vti = cfg.asset("VTI")
    # A $5 order cannot go to the $100-minimum venue, cheap as it is.
    assert cfg.cheapest_venue(vti, 5).name == "dear_but_small"
    # A $500 order can, and should.
    assert cfg.cheapest_venue(vti, 500).name == "cheap_but_big"
