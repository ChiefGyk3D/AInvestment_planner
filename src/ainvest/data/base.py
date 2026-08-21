"""The contract every price source implements."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Protocol

import pandas as pd


class PriceProvider(Protocol):
    """Returns a date-indexed frame of adjusted closes, one column per symbol.

    Implementations should return whatever they could fetch rather than
    raising when a single symbol fails; the caller reports missing columns.
    """

    name: str

    def fetch(
        self, symbols: Sequence[str], start: date, end: date
    ) -> pd.DataFrame: ...


class ProviderError(RuntimeError):
    """A price source could not be reached or returned nothing usable."""
