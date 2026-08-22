"""Timing signals, and the machinery to test whether they earn their keep.

"Know when to pull out" sounds like safety but is usually a way to sell low,
pay tax and fees for the privilege, and miss the rebound. Rather than refuse
the feature or pretend a signal is wisdom, this module does both jobs
honestly: it computes simple, transparent rules (trend and drawdown), and it
backtests the exit rule WITH taxes and trading costs charged on every exit --
because that, not the raw signal, is what you would actually experience.

The rules here are alerts. Nothing in this tool sells anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config
from .stats import periods_per_year


@dataclass
class SignalState:
    """Where the portfolio stands against its configured rules, today."""

    trend_days: int = 0
    above_trend: bool = True
    trend_gap: float = 0.0  # portfolio index vs its moving average, fractional
    drawdown: float = 0.0   # current fall from the running peak (<= 0)
    drawdown_alert: float = 0.0
    in_drawdown_alert: bool = False
    deploy_reserve: bool = False
    notes: list[str] = field(default_factory=list)


def portfolio_index(prices: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """A fixed-weight index of the portfolio, normalised to start at 1.

    Fixed target weights, not drifting holdings: the question the signals ask
    is "how is the strategy doing", which should not change shape because one
    position happened to grow.
    """
    cols = [c for c in prices.columns if weights.get(c, 0.0) > 0]
    if not cols:
        raise ValueError("no overlap between weights and price columns")
    w = np.array([weights[c] for c in cols], dtype=float)
    w = w / w.sum()
    rets = prices[cols].pct_change().fillna(0.0).to_numpy()
    growth = np.cumprod(1.0 + rets @ w)
    return pd.Series(growth, index=prices.index, name="portfolio")


def evaluate(
    cfg: Config, prices: pd.DataFrame, weights: dict[str, float]
) -> SignalState:
    """Compute today's signal state for the given target weights."""
    state = SignalState(
        trend_days=cfg.signals.trend_days,
        drawdown_alert=cfg.signals.drawdown_alert,
    )
    index = portfolio_index(prices, weights)

    state.drawdown = float(index.iloc[-1] / index.max() - 1.0)
    state.in_drawdown_alert = state.drawdown <= -cfg.signals.drawdown_alert
    if state.in_drawdown_alert:
        state.notes.append(
            f"portfolio is {state.drawdown:.1%} below its peak over this lookback. "
            "For a steady investor this is historically when contributions buy the "
            "most -- the alert exists to inform, not to trigger selling."
        )

    reserve = cfg.contribution.reserve_fraction
    if reserve > 0:
        state.deploy_reserve = state.drawdown <= -cfg.contribution.deploy_on_drawdown
        if state.deploy_reserve:
            state.notes.append(
                f"drawdown has passed {cfg.contribution.deploy_on_drawdown:.0%}: the "
                f"configured reserve ({reserve:.0%} of contributions) is flagged for "
                "deployment"
            )

    if cfg.signals.trend_days > 0 and len(index) > cfg.signals.trend_days:
        sma = index.rolling(cfg.signals.trend_days).mean()
        current, ma = float(index.iloc[-1]), float(sma.iloc[-1])
        state.above_trend = current >= ma
        state.trend_gap = current / ma - 1.0
        if not state.above_trend:
            state.notes.append(
                f"portfolio index is {abs(state.trend_gap):.1%} below its "
                f"{cfg.signals.trend_days}-day average. Before treating that as a "
                "sell signal, run `ainvest backtest --exit-rule trend` and look at "
                "what exits would have cost after tax."
            )
    elif cfg.signals.trend_days > 0:
        state.notes.append(
            f"not enough history for a {cfg.signals.trend_days}-day trend average"
        )
    return state


@dataclass
class OverlayResult:
    """A strategy equity curve with an exit rule applied, costs included."""

    label: str
    equity: pd.Series
    exits: int = 0
    days_in_market: float = 1.0
    taxes_paid: float = 0.0
    fees_paid: float = 0.0
    final_liquidation_tax: float = 0.0

    @property
    def total_return(self) -> float:
        return float(self.equity.iloc[-1] / self.equity.iloc[0] - 1.0)

    @property
    def after_liquidation_return(self) -> float:
        """Return if you cashed out at the end -- the honest apples-to-apples
        number, since the always-in portfolio holds unrealised gains the
        exit-rule portfolio has already paid tax on."""
        final = float(self.equity.iloc[-1]) - self.final_liquidation_tax
        return final / float(self.equity.iloc[0]) - 1.0


def trend_overlay(
    cfg: Config,
    index: pd.Series,
    *,
    trend_days: int | None = None,
    cost_bps: float = 10.0,
) -> OverlayResult:
    """Apply "exit when below the moving average" to a portfolio index.

    Mechanics per day t: the position for day t is decided from data through
    day t-1 (no look-ahead). An exit sells everything: it pays ``cost_bps``
    and realises all accumulated gain at the configured short/long-term rate
    depending on how long the position was held; a re-entry pays ``cost_bps``
    again and resets the basis. The result carries the tax owed on final
    liquidation separately so it can be compared fairly with buy-and-hold.
    """
    trend_days = trend_days or cfg.signals.trend_days
    if trend_days <= 0 or len(index) <= trend_days + 2:
        raise ValueError("not enough history for the requested trend window")

    sma = index.rolling(trend_days).mean()
    # Position for day t comes from yesterday's close vs yesterday's average.
    # While the average does not exist yet (the first trend_days rows) there
    # is no signal, and "no signal" means stay invested -- comparing NaN would
    # read as False and silently park the strategy in cash for the whole
    # warm-up period.
    above = (index >= sma) | sma.isna()
    signal = above.shift(1).fillna(True).to_numpy()
    rets = index.pct_change().fillna(0.0).to_numpy()

    equity = np.empty(len(index))
    equity[0] = 1.0
    basis = 1.0
    entry_day = 0
    invested = True
    exits = 0
    taxes = fees = 0.0
    days_in = 0
    cost = cost_bps / 10_000.0
    ppy = periods_per_year(index.index)

    for t in range(1, len(index)):
        want = bool(signal[t])
        value = equity[t - 1]
        if invested and not want:
            value *= 1.0 - cost
            fees += equity[t - 1] * cost
            gain = value - basis
            if gain > 0:
                held_years = (t - entry_day) / ppy
                rate = (
                    cfg.tax.long_term_rate if held_years > 1.0
                    else cfg.tax.short_term_rate
                )
                taxes += gain * rate
                value -= gain * rate
            invested = False
        elif not invested and want:
            value *= 1.0 - cost
            fees += equity[t - 1] * cost
            basis = value
            entry_day = t
            invested = True
            exits += 1  # count round trips on re-entry
        if invested:
            value *= 1.0 + rets[t]
            days_in += 1
        equity[t] = value

    final_tax = 0.0
    if invested:
        gain = equity[-1] - basis
        if gain > 0:
            held_years = (len(index) - 1 - entry_day) / ppy
            rate = (
                cfg.tax.long_term_rate if held_years > 1.0
                else cfg.tax.short_term_rate
            )
            final_tax = gain * rate

    return OverlayResult(
        label=f"trend exit ({trend_days}d)",
        equity=pd.Series(equity, index=index.index),
        exits=exits,
        days_in_market=days_in / max(len(index) - 1, 1),
        taxes_paid=taxes,
        fees_paid=fees,
        final_liquidation_tax=final_tax,
    )


def buy_and_hold(cfg: Config, index: pd.Series) -> OverlayResult:
    """The benchmark: stay in, pay tax only on final liquidation."""
    equity = index / index.iloc[0]
    gain = float(equity.iloc[-1]) - 1.0
    ppy = periods_per_year(index.index)
    held_years = (len(index) - 1) / ppy
    rate = cfg.tax.long_term_rate if held_years > 1.0 else cfg.tax.short_term_rate
    return OverlayResult(
        label="stay invested",
        equity=equity,
        exits=0,
        days_in_market=1.0,
        final_liquidation_tax=max(0.0, gain) * rate,
    )
