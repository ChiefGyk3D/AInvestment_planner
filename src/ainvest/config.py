"""Load and validate the YAML config that drives everything else."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import Asset, Holding, Venue

#: Sensible starting points so a new user does not face a blank config.
#: ``crypto_max`` is the cap on total crypto weight; ``target_vol`` is the
#: annualised volatility the ``target_vol`` objective aims at.
RISK_PROFILES: dict[str, dict[str, float]] = {
    "conservative": {"crypto_max": 0.10, "target_vol": 0.10, "single_max": 0.25},
    "balanced": {"crypto_max": 0.25, "target_vol": 0.15, "single_max": 0.35},
    "aggressive": {"crypto_max": 0.50, "target_vol": 0.25, "single_max": 0.45},
    "degen": {"crypto_max": 0.85, "target_vol": 0.60, "single_max": 0.60},
}


class ConfigError(ValueError):
    """Raised when the config is missing something or self-contradictory."""


@dataclass
class Constraints:
    """Every bound the optimiser has to respect."""

    single_max: float = 0.35
    single_min: float = 0.0
    crypto_min: float = 0.0
    crypto_max: float = 0.25
    equity_min: float = 0.0
    equity_max: float = 1.0
    #: floor applied to any asset the optimiser keeps, to avoid dust positions
    include_threshold: float = 0.01
    max_positions: int = 0  # 0 == unlimited

    def validate(self) -> None:
        pairs = [
            ("single", self.single_min, self.single_max),
            ("crypto", self.crypto_min, self.crypto_max),
            ("equity", self.equity_min, self.equity_max),
        ]
        for label, lo, hi in pairs:
            if not 0.0 <= lo <= hi <= 1.0:
                raise ConfigError(
                    f"{label} bounds must satisfy 0 <= min <= max <= 1, got {lo}..{hi}"
                )
        if self.crypto_min + self.equity_min > 1.0 + 1e-9:
            raise ConfigError(
                "crypto_min + equity_min exceeds 100% of the portfolio; "
                "the constraints cannot all be satisfied at once"
            )
        if self.crypto_max + self.equity_max < 1.0 - 1e-9:
            raise ConfigError(
                "crypto_max + equity_max is below 100% of the portfolio; "
                "there is no way to be fully invested"
            )


@dataclass
class DataSettings:
    provider: str = "yahoo"
    lookback_days: int = 730
    cache_dir: str = ".cache/prices"
    cache_ttl_hours: float = 12.0
    #: minimum overlapping observations required before an asset is usable
    min_observations: int = 60
    synthetic_seed: int = 7


@dataclass
class OptimizerSettings:
    objective: str = "risk_parity"
    risk_free_rate: float = 0.04
    #: shrink estimated means toward their cross-sectional average; 1.0 means
    #: "ignore the return estimates entirely and let risk drive the weights"
    mean_shrinkage: float = 0.5
    covariance_shrinkage: str = "ledoit_wolf"  # or "none"
    #: blend the optimised weights toward equal weight; a cheap, effective
    #: guard against over-fitting a two-year sample
    blend_equal_weight: float = 0.20
    target_vol: float = 0.15
    max_iter: int = 1500  # a ceiling; the solver stops early once converged
    tolerance: float = 1e-10
    restarts: int = 6


@dataclass
class CustodySettings:
    """Rules for when to sweep coins off an exchange into self-custody."""

    enabled: bool = True
    wallet: str = "Trezor"
    #: don't bother sweeping a position smaller than this
    sweep_threshold: float = 100.0
    #: complain when the network fee costs more than this share of the position
    max_fee_fraction: float = 0.02


@dataclass
class Config:
    budget: float = 300.0
    cash_reserve: float = 0.0
    base_currency: str = "USD"
    risk_profile: str = "balanced"
    assets: list[Asset] = field(default_factory=list)
    venues: dict[str, Venue] = field(default_factory=dict)
    holdings: list[Holding] = field(default_factory=list)
    constraints: Constraints = field(default_factory=Constraints)
    data: DataSettings = field(default_factory=DataSettings)
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    custody: CustodySettings = field(default_factory=CustodySettings)
    rebalance_band: float = 0.05
    min_trade: float = 5.0
    allow_sells: bool = False

    @property
    def investable(self) -> float:
        return max(0.0, self.budget - self.cash_reserve)

    def asset(self, symbol: str) -> Asset:
        for a in self.assets:
            if a.symbol == symbol:
                return a
        raise ConfigError(f"unknown asset {symbol!r}")

    def venues_for(self, asset: Asset) -> list[Venue]:
        names = asset.venues or tuple(self.venues)
        missing = [n for n in names if n not in self.venues]
        if missing:
            raise ConfigError(
                f"asset {asset.symbol!r} references undefined venue(s): "
                f"{', '.join(missing)}"
            )
        return [self.venues[n] for n in names]

    def cheapest_venue(self, asset: Asset, notional: float) -> Venue:
        """Pick the venue with the lowest cost for a trade of this size.

        Venues whose minimum order size is above ``notional`` are considered
        last, so a tiny order still gets a usable answer rather than none.
        """
        options = self.venues_for(asset)
        if not options:
            raise ConfigError(f"asset {asset.symbol!r} has no venues configured")
        eligible = [v for v in options if notional >= v.min_notional]
        pool = eligible or options
        return min(pool, key=lambda v: (v.cost(max(notional, v.min_notional)), v.name))


def _as_float(value: Any, field_name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{field_name} must be a number, got {value!r}") from exc


def _venue_from_dict(name: str, raw: dict[str, Any]) -> Venue:
    known = {f for f in Venue.__dataclass_fields__ if f != "name"}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"venue {name!r} has unknown key(s): {', '.join(sorted(unknown))}"
        )
    kwargs: dict[str, Any] = {"name": name}
    for key, value in raw.items():
        if key in {"fractional", "supports_withdrawal"}:
            kwargs[key] = bool(value)
        elif key == "size_decimals":
            kwargs[key] = int(value)
        else:
            kwargs[key] = _as_float(value, f"venues.{name}.{key}")
    return Venue(**kwargs)


def _asset_from_dict(raw: dict[str, Any] | str) -> Asset:
    if isinstance(raw, str):
        return Asset(symbol=raw)
    if "symbol" not in raw:
        raise ConfigError(f"asset entry is missing 'symbol': {raw!r}")
    known = set(Asset.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"asset {raw['symbol']!r} has unknown key(s): {', '.join(sorted(unknown))}"
        )
    kwargs = dict(raw)
    kwargs["venues"] = tuple(kwargs.get("venues") or ())
    asset_class = kwargs.get("asset_class", "equity")
    if asset_class not in {"equity", "crypto", "cash"}:
        raise ConfigError(
            f"asset {raw['symbol']!r} has asset_class {asset_class!r}; "
            "expected one of equity, crypto, cash"
        )
    for key in ("min_weight", "max_weight"):
        if key in kwargs:
            kwargs[key] = _as_float(kwargs[key], f"{raw['symbol']}.{key}")
    if kwargs.get("min_weight", 0.0) > kwargs.get("max_weight", 1.0):
        raise ConfigError(
            f"asset {raw['symbol']!r} has min_weight above max_weight"
        )
    return Asset(**kwargs)


def _fill(target: Any, raw: dict[str, Any] | None, section: str) -> Any:
    """Copy known keys from ``raw`` onto a dataclass instance, type-checked."""
    if not raw:
        return target
    fields = target.__dataclass_fields__
    unknown = set(raw) - set(fields)
    if unknown:
        raise ConfigError(
            f"section {section!r} has unknown key(s): {', '.join(sorted(unknown))}"
        )
    for key, value in raw.items():
        current = getattr(target, key)
        if isinstance(current, bool):
            setattr(target, key, bool(value))
        elif isinstance(current, int) and not isinstance(current, bool):
            setattr(target, key, int(value))
        elif isinstance(current, float):
            setattr(target, key, _as_float(value, f"{section}.{key}"))
        else:
            setattr(target, key, value)
    return target


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise ConfigError(
            f"config file not found: {path}. Run `ainvest init` to write a starter config."
        )
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return from_dict(raw)


def from_dict(raw: dict[str, Any]) -> Config:
    raw = copy.deepcopy(raw)
    cfg = Config()

    for key in ("budget", "cash_reserve", "rebalance_band", "min_trade"):
        if key in raw:
            setattr(cfg, key, _as_float(raw[key], key))
    if "base_currency" in raw:
        cfg.base_currency = str(raw["base_currency"])
    if "allow_sells" in raw:
        cfg.allow_sells = bool(raw["allow_sells"])

    profile = str(raw.get("risk_profile", cfg.risk_profile))
    if profile not in RISK_PROFILES:
        raise ConfigError(
            f"unknown risk_profile {profile!r}; expected one of "
            f"{', '.join(sorted(RISK_PROFILES))}"
        )
    cfg.risk_profile = profile

    # Profile defaults first, then any explicit constraints override them.
    preset = RISK_PROFILES[profile]
    cfg.constraints.crypto_max = preset["crypto_max"]
    cfg.constraints.single_max = preset["single_max"]
    cfg.optimizer.target_vol = preset["target_vol"]

    _fill(cfg.constraints, raw.get("constraints"), "constraints")
    _fill(cfg.data, raw.get("data"), "data")
    _fill(cfg.optimizer, raw.get("optimizer"), "optimizer")
    _fill(cfg.custody, raw.get("custody"), "custody")
    cfg.constraints.validate()

    venues_raw = raw.get("venues") or {}
    if not isinstance(venues_raw, dict):
        raise ConfigError("'venues' must be a mapping of name -> settings")
    cfg.venues = {name: _venue_from_dict(name, v or {}) for name, v in venues_raw.items()}

    assets_raw = raw.get("assets") or []
    if not isinstance(assets_raw, list):
        raise ConfigError("'assets' must be a list")
    cfg.assets = [_asset_from_dict(a) for a in assets_raw]
    if not cfg.assets:
        raise ConfigError("config defines no assets; nothing to optimise")

    seen: set[str] = set()
    for a in cfg.assets:
        if a.symbol in seen:
            raise ConfigError(f"duplicate asset symbol {a.symbol!r}")
        seen.add(a.symbol)
        cfg.venues_for(a)  # raises early on a typo'd venue name

    holdings_raw = raw.get("holdings") or []
    cfg.holdings = [
        Holding(
            symbol=str(h["symbol"]),
            quantity=_as_float(h.get("quantity", 0.0), "holdings.quantity"),
            venue=str(h.get("venue", "")),
            cost_basis=_as_float(h.get("cost_basis", 0.0), "holdings.cost_basis"),
        )
        for h in holdings_raw
    ]
    unknown_holdings = {h.symbol for h in cfg.holdings} - seen
    if unknown_holdings:
        raise ConfigError(
            "holdings reference assets not in the universe: "
            f"{', '.join(sorted(unknown_holdings))}"
        )

    if cfg.investable <= 0:
        raise ConfigError(
            f"nothing to invest: budget {cfg.budget} minus cash_reserve "
            f"{cfg.cash_reserve} is {cfg.investable}"
        )
    return cfg
