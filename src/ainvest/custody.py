"""Advice on moving crypto off an exchange into self-custody.

An exchange balance is a claim on a company; a hardware wallet holds the coin
itself. Which is right depends on size: sweeping $30 of BTC across a $2
network fee costs nearly 7% of the position, so the honest answer for a small
account is often "leave it, and move it once it is worth moving".
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import Config
from .models import Plan


@dataclass
class CustodyAction:
    symbol: str
    value: float
    venue: str
    recommendation: str
    fee: float = 0.0
    fee_fraction: float = 0.0
    reason: str = ""


@dataclass
class CustodyReport:
    wallet: str = "Trezor"
    actions: list[CustodyAction] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def sweep_total(self) -> float:
        return sum(a.value for a in self.actions if a.recommendation == "sweep")


def review(cfg: Config, plan: Plan, extra_values: dict[str, float] | None = None) -> CustodyReport:
    """Decide, per crypto position, whether it is worth withdrawing."""
    report = CustodyReport(wallet=cfg.custody.wallet)
    if not cfg.custody.enabled:
        report.notes.append("custody review disabled in config")
        return report

    values: dict[str, tuple[float, str]] = {}
    for trade in plan.trades:
        if trade.side != "buy":
            continue
        current = values.get(trade.symbol, (0.0, trade.venue))
        values[trade.symbol] = (current[0] + trade.notional, trade.venue)
    for symbol, value in (extra_values or {}).items():
        current = values.get(symbol, (0.0, ""))
        values[symbol] = (current[0] + value, current[1])

    on_exchange = 0.0
    for symbol, (value, venue_name) in sorted(values.items(), key=lambda kv: -kv[1][0]):
        asset = cfg.asset(symbol)
        if asset.asset_class != "crypto" or value <= 0:
            continue
        venue = cfg.venues.get(venue_name)

        if not asset.self_custody:
            on_exchange += value
            report.actions.append(
                CustodyAction(
                    symbol=symbol, value=value, venue=venue_name,
                    recommendation="leave",
                    reason=f"not marked self_custody in config; stays on {venue_name or 'the exchange'}",
                )
            )
            continue
        if venue is not None and not venue.supports_withdrawal:
            on_exchange += value
            report.actions.append(
                CustodyAction(
                    symbol=symbol, value=value, venue=venue_name,
                    recommendation="leave",
                    reason=f"{venue_name} is configured as not supporting withdrawals "
                           "(some venues sell exposure, not the coin)",
                )
            )
            continue

        fee = venue.withdrawal_fee if venue is not None else 0.0
        fraction = fee / value if value > 0 else 1.0
        if value < cfg.custody.sweep_threshold:
            report.actions.append(
                CustodyAction(
                    symbol=symbol, value=value, venue=venue_name, recommendation="wait",
                    fee=fee, fee_fraction=fraction,
                    reason=f"below the ${cfg.custody.sweep_threshold:,.0f} sweep threshold; "
                           "let it grow, then withdraw once in a single larger transfer",
                )
            )
            on_exchange += value
        elif fraction > cfg.custody.max_fee_fraction:
            report.actions.append(
                CustodyAction(
                    symbol=symbol, value=value, venue=venue_name, recommendation="wait",
                    fee=fee, fee_fraction=fraction,
                    reason=f"the ${fee:,.2f} network fee is {fraction:.1%} of the position, "
                           f"above the {cfg.custody.max_fee_fraction:.1%} limit",
                )
            )
            on_exchange += value
        else:
            report.actions.append(
                CustodyAction(
                    symbol=symbol, value=value, venue=venue_name, recommendation="sweep",
                    fee=fee, fee_fraction=fraction,
                    reason=f"withdraw to {cfg.custody.wallet}; the ${fee:,.2f} fee is "
                           f"{fraction:.1%} of the position",
                )
            )

    if report.sweep_total > 0:
        report.notes.append(
            f"send a small test transaction first, confirm it arrives in {report.wallet}, "
            "then send the rest"
        )
        report.notes.append(
            "verify the receiving address on the hardware wallet screen itself, not "
            "on the computer -- that check is the entire point of the device"
        )
    if on_exchange > 0:
        report.notes.append(
            f"${on_exchange:,.2f} of crypto stays on an exchange for now. That is a "
            "deliberate trade-off: you keep counterparty risk in exchange for not "
            "burning a large share of a small position on network fees."
        )
    return report
