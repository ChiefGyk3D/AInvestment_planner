"""A plain-CSV price cache.

Deliberately boring: one CSV per symbol, human-readable, greppable, and
trivially inspected or hand-edited when a provider returns something odd.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import date, datetime
from pathlib import Path

import pandas as pd


def _safe_name(symbol: str) -> str:
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in symbol)


class PriceCache:
    def __init__(self, directory: str | Path, ttl_hours: float = 12.0) -> None:
        self.directory = Path(directory)
        self.ttl_seconds = max(0.0, ttl_hours * 3600.0)

    def path_for(self, symbol: str) -> Path:
        return self.directory / f"{_safe_name(symbol)}.csv"

    def is_fresh(self, symbol: str) -> bool:
        path = self.path_for(symbol)
        if not path.exists():
            return False
        if self.ttl_seconds <= 0:
            return False
        return (time.time() - path.stat().st_mtime) < self.ttl_seconds

    def read(self, symbol: str) -> pd.Series | None:
        path = self.path_for(symbol)
        if not path.exists():
            return None
        try:
            frame = pd.read_csv(path, parse_dates=["date"])
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            return None
        if "date" not in frame or "close" not in frame or frame.empty:
            return None
        series = frame.set_index("date")["close"].astype(float)
        series.name = symbol
        return series.sort_index()

    def write(self, symbol: str, series: pd.Series) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        frame = series.rename("close").to_frame()
        frame.index.name = "date"
        frame.to_csv(self.path_for(symbol))

    def load_many(
        self, symbols: Sequence[str], start: date | None = None, end: date | None = None
    ) -> pd.DataFrame:
        columns: dict[str, pd.Series] = {}
        for symbol in symbols:
            series = self.read(symbol)
            if series is None or series.empty:
                continue
            if start is not None:
                series = series[series.index >= pd.Timestamp(start)]
            if end is not None:
                series = series[series.index <= pd.Timestamp(end)]
            if not series.empty:
                columns[symbol] = series
        if not columns:
            return pd.DataFrame()
        return pd.DataFrame(columns).sort_index()

    def store_frame(self, frame: pd.DataFrame) -> None:
        """Merge a freshly fetched frame into the cache, newest value winning."""
        for symbol in frame.columns:
            fresh = frame[symbol].dropna()
            if fresh.empty:
                continue
            existing = self.read(symbol)
            if existing is not None and not existing.empty:
                merged = pd.concat([existing, fresh])
                merged = merged[~merged.index.duplicated(keep="last")].sort_index()
            else:
                merged = fresh.sort_index()
            self.write(symbol, merged)

    def describe(self) -> list[tuple[str, int, str]]:
        """(symbol, rows, last date) for everything currently cached."""
        if not self.directory.exists():
            return []
        rows = []
        for path in sorted(self.directory.glob("*.csv")):
            series = self.read(path.stem)
            if series is None or series.empty:
                rows.append((path.stem, 0, "-"))
            else:
                last = series.index[-1]
                stamp = last.strftime("%Y-%m-%d") if isinstance(last, datetime) else str(last)
                rows.append((path.stem, len(series), stamp))
        return rows
