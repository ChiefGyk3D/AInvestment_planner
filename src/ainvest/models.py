"""Core value types shared across the package."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

AssetClass = Literal["equity", "crypto", "cash"]


@dataclass(frozen=True)
class Venue:
    """A place you can trade, and what it costs you to do so.

    Fees are modelled as ``fixed + notional * (fee_bps + spread_bps) / 10_000``
    with a floor of ``min_fee``. ``spread_bps`` is where you put the implicit
    cost of a retail "no commission" venue that instead marks up the price.
    """

    name: str
    fee_bps: float = 0.0
    spread_bps: float = 0.0
    fixed_fee: float = 0.0
    min_fee: float = 0.0
    min_notional: float = 1.0
    fractional: bool = True
    #: decimal places an order size is rounded to (shares or coins)
    size_decimals: int = 6
    withdrawal_fee: float = 0.0
    supports_withdrawal: bool = False

    def cost(self, notional: float) -> float:
        """Round-trip-free, one-way cost of trading ``notional`` dollars."""
        if notional <= 0:
            return 0.0
        variable = notional * (self.fee_bps + self.spread_bps) / 10_000.0
        return max(self.fixed_fee + variable, self.min_fee)

    def quantize(self, size: float) -> float:
        """Round an order size DOWN to what the venue will actually accept.

        Down, not to nearest: order sizes here are derived from a dollar
        budget, and rounding a quantity up quietly spends more than was
        allocated to it. Half a cent per order does not matter on its own, but
        "the plan costs more than the budget" is exactly the kind of quiet
        wrongness this tool exists to avoid.
        """
        if size <= 0:
            return 0.0
        if not self.fractional:
            return float(math.floor(size))
        scale = 10.0**self.size_decimals
        # The nudge absorbs float representation error so a value that is
        # mathematically on a boundary does not drop a whole increment.
        return math.floor(size * scale + 1e-9) / scale


@dataclass(frozen=True)
class Asset:
    """One investable line item."""

    symbol: str
    asset_class: AssetClass = "equity"
    name: str = ""
    #: venue names this asset can be bought on, cheapest-first is not required
    venues: tuple[str, ...] = ()
    #: hard bounds on this asset's portfolio weight, as fractions
    min_weight: float = 0.0
    max_weight: float = 1.0
    #: whether the asset can be moved to self-custody (e.g. a Trezor)
    self_custody: bool = False
    #: data-provider symbol override, when it differs from ``symbol``
    data_symbol: str = ""

    @property
    def feed_symbol(self) -> str:
        return self.data_symbol or self.symbol

    @property
    def label(self) -> str:
        return self.name or self.symbol


@dataclass(frozen=True)
class Holding:
    """Something you already own."""

    symbol: str
    quantity: float
    venue: str = ""
    cost_basis: float = 0.0


@dataclass(frozen=True)
class Trade:
    """A single order the plan is asking you to place by hand."""

    symbol: str
    side: Literal["buy", "sell"]
    venue: str
    notional: float
    quantity: float
    price: float
    fee: float

    @property
    def signed_notional(self) -> float:
        return self.notional if self.side == "buy" else -self.notional


@dataclass
class Allocation:
    """Target weights plus the risk numbers that justify them."""

    symbols: list[str]
    weights: list[float]
    expected_return: float = 0.0
    expected_vol: float = 0.0
    sharpe: float = 0.0
    risk_contributions: list[float] = field(default_factory=list)
    diversification: float = 0.0
    objective: str = ""

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.symbols, self.weights, strict=True))

    def nonzero(self, eps: float = 1e-6) -> list[tuple[str, float]]:
        return [(s, w) for s, w in zip(self.symbols, self.weights, strict=True) if w > eps]


@dataclass
class Plan:
    """An allocation turned into orders you can actually place."""

    allocation: Allocation
    trades: list[Trade] = field(default_factory=list)
    budget: float = 0.0
    deployed: float = 0.0
    total_fees: float = 0.0
    cash_left: float = 0.0
    dropped: list[tuple[str, str]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def fee_drag_bps(self) -> float:
        if self.deployed <= 0:
            return 0.0
        return 10_000.0 * self.total_fees / self.deployed

    def by_venue(self) -> dict[str, list[Trade]]:
        out: dict[str, list[Trade]] = {}
        for t in self.trades:
            out.setdefault(t.venue, []).append(t)
        return out


def index_of(assets: Iterable[Asset]) -> dict[str, Asset]:
    return {a.symbol: a for a in assets}
