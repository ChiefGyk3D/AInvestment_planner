"""News and events are context, never triggers -- and everything must degrade
gracefully when there is no network, because there often isn't."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ainvest.config import from_dict
from ainvest.events import (
    STRESS_WINDOW,
    measure_stress,
    parse_rss,
    review,
    upcoming_events,
)
from ainvest.signals import portfolio_index
from conftest import base_config

TODAY = date(2026, 8, 21)


def cfg_with_events(entries, **overrides):
    raw = base_config(**overrides)
    raw["news"] = {**raw.get("news", {}), "events": entries}
    return from_dict(raw)


def stamp(days_from_today):
    return (TODAY + timedelta(days=days_from_today)).isoformat()


class TestCalendar:
    def test_upcoming_events_are_windowed_and_sorted(self):
        cfg = cfg_with_events([
            {"date": stamp(15), "title": "CPI print"},
            {"date": stamp(3), "title": "FOMC", "note": "expect volatility"},
            {"date": stamp(40), "title": "too far out"},
            {"date": stamp(-2), "title": "already happened"},
        ])
        upcoming = upcoming_events(cfg, TODAY)
        assert [e.title for e, _ in upcoming] == ["FOMC", "CPI print"]
        assert [d for _, d in upcoming] == [3, 15]

    def test_today_counts_as_upcoming(self):
        cfg = cfg_with_events([{"date": stamp(0), "title": "now"}])
        assert upcoming_events(cfg, TODAY)[0][1] == 0

    def test_bad_event_date_is_rejected_at_load(self):
        with pytest.raises(Exception, match="ISO date"):
            cfg_with_events([{"date": "next Tuesday", "title": "vibes"}])

    def test_event_needs_a_title(self):
        with pytest.raises(Exception, match="date.*title|title"):
            cfg_with_events([{"date": stamp(1)}])


RSS_FIXTURE = """<?xml version="1.0"?>
<rss version="2.0"><channel>
  <item><title>VTI does a thing</title><pubDate>Fri, 21 Aug 2026</pubDate>
    <link>https://example.com/1</link></item>
  <item><title>Second story</title></item>
  <item><description>no title here</description></item>
</channel></rss>"""


class TestRss:
    def test_parses_titles_dates_and_links(self):
        headlines = parse_rss(RSS_FIXTURE, "VTI", limit=10)
        assert [h.title for h in headlines] == ["VTI does a thing", "Second story"]
        assert headlines[0].published.startswith("Fri")
        assert headlines[0].link == "https://example.com/1"
        assert all(h.symbol == "VTI" for h in headlines)

    def test_limit_is_respected(self):
        assert len(parse_rss(RSS_FIXTURE, "VTI", limit=1)) == 1

    def test_garbage_xml_is_no_headlines_not_a_crash(self):
        assert parse_rss("this is not xml <", "VTI", limit=5) == []


def index_with_calm_then_storm(calm=300, storm=40):
    rng = np.random.default_rng(0)
    rets = np.concatenate([
        rng.normal(0.0003, 0.004, calm),
        rng.normal(-0.002, 0.03, storm),   # vol regime ~8x the calm one
    ])
    idx = pd.date_range("2024-01-01", periods=len(rets) + 1, freq="D")
    return pd.Series(np.cumprod(np.concatenate([[1.0], 1 + rets])), index=idx)


class TestStress:
    def test_a_volatility_spike_is_flagged_as_elevated(self):
        index = index_with_calm_then_storm()
        stress = measure_stress(pd.DataFrame(), index)
        assert stress is not None
        assert stress.elevated
        assert stress.percentile >= 0.9
        assert stress.ratio > 2.0

    def test_calm_markets_are_not_elevated(self):
        index = index_with_calm_then_storm(calm=340, storm=0)
        stress = measure_stress(pd.DataFrame(), index)
        assert stress is not None
        assert not stress.elevated

    def test_too_little_history_returns_none(self):
        short = index_with_calm_then_storm(calm=STRESS_WINDOW, storm=0)
        assert measure_stress(pd.DataFrame(), short) is None


class TestReview:
    def test_offline_review_never_touches_the_network(self, prices):
        cfg = cfg_with_events([{"date": stamp(2), "title": "FOMC"}])
        index = portfolio_index(prices, {"VTI": 1.0})
        report = review(cfg, prices, index, offline=True, today=TODAY)
        assert report.headlines == []
        assert any("offline" in n for n in report.notes)
        assert [e.title for e, _ in report.upcoming] == ["FOMC"]

    def test_headlines_off_means_no_fetch_and_no_note(self, prices):
        raw = base_config()
        raw["news"] = {"headlines": False}
        cfg = from_dict(raw)
        index = portfolio_index(prices, {"VTI": 1.0})
        report = review(cfg, prices, index, offline=False, today=TODAY)
        assert report.headlines == []

    def test_stress_note_tells_you_not_to_sell_into_it(self, prices):
        """The one place a 'signal' fires, the advice is the opposite of
        panic: this is the module's whole editorial stance in one string."""
        cfg = cfg_with_events([])
        index = index_with_calm_then_storm()
        report = review(cfg, prices, index, offline=True, today=TODAY)
        assert report.stress is not None and report.stress.elevated
        assert any("selling into stress" in n for n in report.notes)
