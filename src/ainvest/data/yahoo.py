"""Yahoo Finance prices via yfinance.

Covers both equities (``VTI``) and crypto pairs (``BTC-USD``) from one source,
which keeps the calendars consistent. yfinance is an optional dependency:
install it with ``pip install 'ainvest[live]'``.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import pandas as pd

from .base import ProviderError


class YahooProvider:
    name = "yahoo"

    def __init__(self, auto_adjust: bool = True) -> None:
        self.auto_adjust = auto_adjust

    def fetch(self, symbols: Sequence[str], start: date, end: date) -> pd.DataFrame:
        try:
            import yfinance
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ProviderError(
                "the 'yahoo' provider needs yfinance. Install it with "
                "`pip install 'ainvest[live]'`, or set data.provider to "
                "'csv' or 'synthetic'."
            ) from exc

        symbols = list(symbols)
        try:
            raw = yfinance.download(
                tickers=symbols,
                start=start.isoformat(),
                # yfinance treats `end` as exclusive
                end=(end + timedelta(days=1)).isoformat(),
                auto_adjust=self.auto_adjust,
                progress=False,
                group_by="column",
                threads=False,
            )
        except Exception as exc:  # pragma: no cover - network dependent
            raise ProviderError(f"Yahoo Finance request failed: {exc}") from exc

        if raw is None or len(raw) == 0:
            raise ProviderError(
                f"Yahoo Finance returned no rows for {', '.join(symbols)}. "
                "Check the ticker spellings (crypto needs the -USD suffix)."
            )

        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(close, pd.Series):
            close = close.to_frame(symbols[0])
        if len(symbols) == 1 and list(close.columns) == ["Close"]:
            close.columns = symbols
        close.index = pd.to_datetime(close.index).tz_localize(None)
        return close.sort_index()
