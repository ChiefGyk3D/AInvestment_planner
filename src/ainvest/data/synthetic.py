"""A deterministic fake market.

Exists so the whole pipeline can be exercised -- in tests, in CI, on a plane,
or on a machine whose network blocks the real providers -- without pretending
the numbers mean anything. Prices follow a one-factor model: a shared market
move plus idiosyncratic noise, which produces the positive-but-imperfect
correlations real portfolios have.

The shock series are standardised to zero mean and unit variance before use,
so each asset's *realised* drift and volatility come out close to the profile
it was given. Real markets do not oblige like this -- an unstandardised draw
can easily hand you a two-year bear market on a symbol whose profile says +9%
a year. Pinning the moments is what makes this data useful as a fixture: tests
can assert on it, and the demo shows the machinery rather than one unlucky
random seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd

#: Rough annualised (drift, vol, market beta) profiles keyed by symbol prefix.
#: Crypto gets the high vol and high beta you would expect.
_PROFILES: dict[str, tuple[float, float, float]] = {
    "BTC": (0.35, 0.65, 1.4),
    "ETH": (0.30, 0.80, 1.6),
    "SOL": (0.35, 1.10, 1.9),
    "LTC": (0.05, 0.75, 1.2),
    "ADA": (0.05, 0.95, 1.5),
    "DOGE": (0.00, 1.20, 1.7),
    "VTI": (0.09, 0.16, 1.0),
    "VOO": (0.09, 0.16, 1.0),
    "SPY": (0.09, 0.16, 1.0),
    "QQQ": (0.12, 0.22, 1.2),
    "SCHD": (0.07, 0.14, 0.8),
    "VXUS": (0.06, 0.17, 0.9),
    "GLD": (0.04, 0.14, 0.1),
    "BND": (0.03, 0.06, 0.1),
    "TLT": (0.02, 0.14, -0.2),
}
_DEFAULT_PROFILE = (0.07, 0.20, 1.0)


def _standardise(x: np.ndarray) -> np.ndarray:
    """Force a shock series to exactly zero mean and unit variance."""
    centred = x - x.mean()
    sd = centred.std()
    return centred / sd if sd > 0 else centred


def profile_for(symbol: str) -> tuple[float, float, float]:
    key = symbol.upper().split("-")[0].split(".")[0]
    return _PROFILES.get(key, _DEFAULT_PROFILE)


class SyntheticProvider:
    """Generates reproducible pseudo-prices. Not real market data."""

    name = "synthetic"

    def __init__(self, seed: int = 7, include_weekends: bool = True) -> None:
        self.seed = seed
        self.include_weekends = include_weekends

    def _calendar(self, start: date, end: date, symbol: str) -> pd.DatetimeIndex:
        days = pd.date_range(start=start, end=end, freq="D")
        is_crypto = "-USD" in symbol.upper() or profile_for(symbol)[1] > 0.5
        if is_crypto and self.include_weekends:
            return days
        return days[days.dayofweek < 5]

    def fetch(self, symbols: Sequence[str], start: date, end: date) -> pd.DataFrame:
        if end <= start:
            raise ValueError("end date must be after start date")
        total_days = (end - start).days + 1
        rng = np.random.default_rng(self.seed)
        # One shared market factor drives the common component of every series.
        market = _standardise(rng.normal(0.0, 1.0, size=total_days))
        all_days = [start + timedelta(days=i) for i in range(total_days)]
        day_pos = {d: i for i, d in enumerate(all_days)}

        columns: dict[str, pd.Series] = {}
        for offset, symbol in enumerate(symbols):
            drift, vol, beta = profile_for(symbol)
            sub = np.random.default_rng(self.seed * 1000 + offset)
            idio = _standardise(sub.normal(0.0, 1.0, size=total_days))
            # Split total variance between the market factor and the residual.
            market_share = min(0.9, beta**2 * 0.16 / max(vol**2, 1e-9))
            daily_vol = vol / np.sqrt(365.0)
            shocks = daily_vol * (
                np.sqrt(market_share) * market + np.sqrt(1.0 - market_share) * idio
            )
            log_path = np.cumsum(shocks + (drift / 365.0) - 0.5 * daily_vol**2)
            start_price = 100.0 * (1.0 + 0.37 * ((offset * 7919) % 13))
            prices = start_price * np.exp(log_path)

            index = self._calendar(start, end, symbol)
            values = [prices[day_pos[d.date()]] for d in index]
            columns[symbol] = pd.Series(values, index=index, name=symbol)

        return pd.DataFrame(columns).sort_index()
