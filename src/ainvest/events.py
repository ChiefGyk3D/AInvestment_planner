"""News and event context for the monthly check.

The stance this module takes, deliberately and in writing: headlines and
calendar entries are CONTEXT, never triggers. No order in any plan changes
because of anything fetched here. Trading on news is one of the most reliably
self-harming behaviours in retail investing -- by the time a headline reaches
you it is in the price, and what it mostly does is supply a reason to abandon
the plan at the worst moment. What news that genuinely matters does to a
portfolio shows up in prices, which is why the one *quantitative* item here
(the stress gauge) is computed from prices alone and works offline.

Headlines come from per-symbol RSS feeds via the standard library, best-effort
with a short timeout: a dead network degrades to "no headlines", never to a
failed check. The calendar is user-maintained in the config -- FOMC dates,
CPI prints, an expected hard fork -- so known volatility dates are
unsurprising rather than actionable.
"""

from __future__ import annotations

import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import Config
from .stats import periods_per_year

#: Yahoo Finance's per-symbol headline feed. Works for equities and, with the
#: -USD suffix the data layer already uses, for crypto pairs.
FEED_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

STRESS_WINDOW = 20          # trading days of realised volatility
STRESS_PERCENTILE = 0.90    # flag when current vol is above this share of history


@dataclass
class Headline:
    symbol: str
    title: str
    published: str = ""
    link: str = ""


@dataclass
class Stress:
    """Where today's realised volatility sits in this portfolio's own history."""

    current_vol: float
    median_vol: float
    percentile: float
    elevated: bool

    @property
    def ratio(self) -> float:
        return self.current_vol / self.median_vol if self.median_vol > 0 else 1.0


@dataclass
class EventsReport:
    upcoming: list = field(default_factory=list)  # (CalendarEvent, days_away)
    headlines: list[Headline] = field(default_factory=list)
    stress: Stress | None = None
    notes: list[str] = field(default_factory=list)


def upcoming_events(cfg: Config, today: date | None = None) -> list:
    """Calendar entries inside the lookahead window, soonest first."""
    today = today or date.today()
    horizon = today + timedelta(days=cfg.news.lookahead_days)
    out = []
    for event in cfg.calendar:
        when = date.fromisoformat(event.date)
        if today <= when <= horizon:
            out.append((event, (when - today).days))
    return sorted(out, key=lambda pair: pair[1])


def parse_rss(xml_text: str, symbol: str, limit: int) -> list[Headline]:
    """Extract headlines from an RSS 2.0 document. Separated for testing."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        items.append(
            Headline(
                symbol=symbol,
                title=title,
                published=(item.findtext("pubDate") or "").strip(),
                link=(item.findtext("link") or "").strip(),
            )
        )
        if len(items) >= limit:
            break
    return items


def fetch_headlines(cfg: Config, symbols: list[str]) -> tuple[list[Headline], list[str]]:
    """Best-effort headline fetch. Failure is a note, never an error."""
    headlines: list[Headline] = []
    notes: list[str] = []
    per_symbol = max(1, cfg.news.max_headlines // max(1, len(symbols)))
    failed = 0
    for symbol in symbols:
        url = FEED_URL.format(symbol=symbol)
        try:
            with urllib.request.urlopen(url, timeout=cfg.news.timeout_seconds) as response:
                text = response.read().decode("utf-8", errors="replace")
            headlines.extend(parse_rss(text, symbol, per_symbol))
        except (urllib.error.URLError, OSError, ValueError):
            failed += 1
    if failed == len(symbols) and symbols:
        notes.append(
            "headlines unavailable (network unreachable or feed down); the check "
            "is complete without them -- nothing in the plan depends on news"
        )
    return headlines[: cfg.news.max_headlines], notes


def measure_stress(prices: pd.DataFrame, index: pd.Series) -> Stress | None:
    """Realised portfolio volatility versus its own history.

    This is the honest, quantitative stand-in for "is something happening":
    if news is moving markets, it moves realised volatility, with no editor
    deciding what counts. Current vol is the trailing ``STRESS_WINDOW``
    annualised standard deviation of the portfolio index; the percentile ranks
    it against every same-length window in the sample.
    """
    rets = index.pct_change().dropna()
    if len(rets) < STRESS_WINDOW * 3:
        return None
    ppy = periods_per_year(index.index)
    rolling = rets.rolling(STRESS_WINDOW).std().dropna() * np.sqrt(ppy)
    if rolling.empty:
        return None
    current = float(rolling.iloc[-1])
    history = rolling.to_numpy()
    percentile = float(np.mean(history <= current))
    return Stress(
        current_vol=current,
        median_vol=float(np.median(history)),
        percentile=percentile,
        elevated=percentile >= STRESS_PERCENTILE,
    )


def review(
    cfg: Config,
    prices: pd.DataFrame,
    index: pd.Series,
    *,
    offline: bool = False,
    today: date | None = None,
) -> EventsReport:
    report = EventsReport()
    report.upcoming = upcoming_events(cfg, today)
    report.stress = measure_stress(prices, index)

    if report.stress is not None and report.stress.elevated:
        report.notes.append(
            f"realised volatility ({report.stress.current_vol:.0%} annualised) is in "
            f"the top {1 - STRESS_PERCENTILE:.0%} of this portfolio's history -- "
            f"{report.stress.ratio:.1f}x its median. Markets are stressed. The plan "
            "does not change: contributions at these prices buy more, and selling "
            "into stress is how drawdowns become losses."
        )
    if report.upcoming:
        report.notes.append(
            "scheduled events are listed so volatility around them is unsurprising, "
            "not so it can be traded; the contribution schedule proceeds regardless"
        )

    if cfg.news.headlines and not offline:
        feed_symbols = [a.feed_symbol for a in cfg.assets]
        headlines, notes = fetch_headlines(cfg, feed_symbols)
        report.headlines = headlines
        report.notes.extend(notes)
        if headlines:
            report.notes.append(
                "headlines are context only: nothing in any plan changes because "
                "of a story, and acting on one mid-plan is how plans die"
            )
    elif cfg.news.headlines and offline:
        report.notes.append("offline: headline fetch skipped")
    return report
