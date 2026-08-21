"""Read prices from CSV files you supply yourself.

The escape hatch for anything the bundled providers do not cover: export from
your broker, drop the files in a directory, and point ``data.cache_dir`` at
it. Each file is ``<SYMBOL>.csv`` with ``date`` and ``close`` columns -- the
same format the cache writes, so a warmed cache doubles as a CSV source.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from .base import ProviderError
from .cache import PriceCache


class CsvProvider:
    name = "csv"

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def fetch(self, symbols: Sequence[str], start: date, end: date) -> pd.DataFrame:
        if not self.directory.exists():
            raise ProviderError(f"CSV directory does not exist: {self.directory}")
        # ttl_hours=0 keeps freshness out of it; on-disk files are the truth here.
        store = PriceCache(self.directory, ttl_hours=0.0)
        frame = store.load_many(symbols, start, end)
        if frame.empty:
            available = sorted(p.stem for p in self.directory.glob("*.csv"))
            raise ProviderError(
                f"no CSV data found for {', '.join(symbols)} in {self.directory}. "
                f"Available: {', '.join(available) if available else '(none)'}"
            )
        return frame
