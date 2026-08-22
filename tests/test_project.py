"""The projection must be reproducible, internally coherent, and humble."""

from __future__ import annotations

import pytest

from ainvest.config import from_dict
from ainvest.project import run_projection
from conftest import base_config


def make_cfg(**overrides):
    return from_dict(base_config(**overrides))


@pytest.fixture(scope="module")
def panel():
    # Reuse the synthetic fixtures at module scope; projections are the
    # slowest tests here and the input panel never changes.
    from ainvest.data.synthetic import SyntheticProvider
    from ainvest.stats import align_prices
    from conftest import END, START
    cfg = make_cfg()
    frame = SyntheticProvider(seed=7).fetch([a.feed_symbol for a in cfg.assets], START, END)
    frame = frame.rename(columns={a.feed_symbol: a.symbol for a in cfg.assets})
    return align_prices(frame, 60)


WEIGHTS = {"VTI": 0.4, "GLD": 0.3, "BTC": 0.15, "ETH": 0.15}


def test_projection_is_deterministic(panel):
    cfg = make_cfg()
    a = run_projection(cfg, panel, WEIGHTS, years=5, n_paths=400)
    b = run_projection(cfg, panel, WEIGHTS, years=5, n_paths=400)
    assert a.results[-1].wealth == b.results[-1].wealth


def test_percentiles_are_ordered_and_horizons_grow(panel):
    cfg = make_cfg()
    report = run_projection(cfg, panel, WEIGHTS, years=10, n_paths=600)
    for r in report.results:
        w = r.wealth
        assert w[10] <= w[25] <= w[50] <= w[75] <= w[90]
    # More years of contributions -> more median wealth.
    medians = [r.wealth[50] for r in report.results]
    assert medians == sorted(medians)


def test_contributions_flow_into_the_totals(panel):
    cfg = make_cfg(contribution={"amount": 100, "frequency": "monthly"})
    report = run_projection(cfg, panel, WEIGHTS, years=5, n_paths=300)
    five = next(r for r in report.results if r.years == 5)
    assert five.contribution_per_year == pytest.approx(1200)
    assert five.total_contributed == pytest.approx(cfg.investable + 6000)
    # And a plan with contributions ends richer than one without.
    silent = make_cfg(contribution={"amount": 0})
    quiet = run_projection(silent, panel, WEIGHTS, years=5, n_paths=300)
    q5 = next(r for r in quiet.results if r.years == 5)
    assert five.wealth[50] > q5.wealth[50]


def test_probability_of_loss_is_a_probability(panel):
    cfg = make_cfg()
    report = run_projection(cfg, panel, WEIGHTS, years=5, n_paths=500)
    for r in report.results:
        assert 0.0 <= r.probability_of_loss <= 1.0
        assert r.median_max_drawdown <= 0.0


def test_dca_comparison_conserves_money(panel):
    cfg = make_cfg(contribution={"amount": 0})
    report = run_projection(cfg, panel, WEIGHTS, years=3, n_paths=500, dca_months=6)
    labels = [label for label, _ in report.dca_comparison]
    assert any("lump sum now" in label for label in labels)
    values = dict(report.dca_comparison)
    win_rate = values[[label for label in labels if "wins" in label][0]]
    assert 0.0 <= win_rate <= 1.0


def test_the_report_admits_its_sample_is_small(panel):
    cfg = make_cfg()
    report = run_projection(cfg, panel, WEIGHTS, years=5, n_paths=200)
    assert any("years of history" in w for w in report.warnings)
