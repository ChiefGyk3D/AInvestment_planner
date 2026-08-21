"""Price sources and the caching layer in front of them."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd

from ..config import Config
from .base import PriceProvider, ProviderError
from .cache import PriceCache
from .csv_source import CsvProvider
from .synthetic import SyntheticProvider
from .yahoo import YahooProvider

__all__ = [
    "PriceProvider", "ProviderError", "PriceCache", "CsvProvider",
    "SyntheticProvider", "YahooProvider", "build_provider", "load_prices",
]

PROVIDERS = ("yahoo", "csv", "synthetic")


def build_provider(cfg: Config) -> PriceProvider:
    name = cfg.data.provider
    if name == "yahoo":
        return YahooProvider()
    if name == "csv":
        return CsvProvider(cfg.data.cache_dir)
    if name == "synthetic":
        return SyntheticProvider(seed=cfg.data.synthetic_seed)
    raise ValueError(
        f"unknown data provider {name!r}; expected one of {', '.join(PROVIDERS)}"
    )


def load_prices(
    cfg: Config,
    symbols: Sequence[str] | None = None,
    *,
    refresh: bool = False,
    offline: bool = False,
) -> pd.DataFrame:
    """Get prices for the universe, preferring the cache unless told otherwise.

    Falls back to stale cached data if the provider cannot be reached, because
    a slightly old allocation beats no allocation. The caller is told via the
    frame's ``stale`` attribute.
    """
    feed_symbols = list(symbols) if symbols else [a.feed_symbol for a in cfg.assets]
    end = date.today()
    start = end - timedelta(days=int(cfg.data.lookback_days))
    cache = PriceCache(cfg.data.cache_dir, cfg.data.cache_ttl_hours)

    if cfg.data.provider == "csv":
        frame = CsvProvider(cfg.data.cache_dir).fetch(feed_symbols, start, end)
        frame.attrs["source"] = "csv"
        frame.attrs["stale"] = False
        return frame

    missing = [s for s in feed_symbols if refresh or not cache.is_fresh(s)]
    if missing and not offline:
        try:
            fresh = build_provider(cfg).fetch(missing, start, end)
            cache.store_frame(fresh)
        except Exception as exc:  # noqa: BLE001 - any failure falls back to cache
            # A slightly stale allocation beats no allocation, so a failed
            # fetch degrades to cached prices instead of aborting the run.
            cached = cache.load_many(feed_symbols, start, end)
            if cached.empty:
                raise ProviderError(
                    f"could not fetch prices and the cache is empty: {exc}"
                ) from exc
            reason = str(exc).splitlines()[0][:80] if str(exc) else type(exc).__name__
            cached.attrs["source"] = f"cache (fetch failed: {reason})"
            cached.attrs["stale"] = True
            return cached

    frame = cache.load_many(feed_symbols, start, end)
    if frame.empty:
        raise ProviderError(
            f"no price data available for {', '.join(feed_symbols)}. "
            "Try `ainvest fetch --refresh`, or switch data.provider to 'synthetic' "
            "to see the tool run end to end with fake prices."
        )
    frame.attrs["source"] = cfg.data.provider if not offline else "cache (offline)"
    frame.attrs["stale"] = bool(offline and missing)
    return frame
