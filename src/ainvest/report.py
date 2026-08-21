"""Rendering. Plain text for the terminal, Markdown for saving, JSON for code."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from . import DISCLAIMER
from .backtest import BacktestReport
from .config import Config
from .custody import CustodyReport
from .models import Allocation, Plan
from .plan import realised_weights, tracking_error
from .rebalance import RebalanceReport
from .stats import Moments


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]], indent: str = "  ") -> str:
    """A minimal fixed-width table. Numeric-looking columns right-align."""
    body = [[str(c) for c in row] for row in rows]
    if not body:
        return f"{indent}(nothing to show)"
    widths = [
        max(len(str(headers[i])), max((len(r[i]) for r in body), default=0))
        for i in range(len(headers))
    ]
    right = [
        all(_numeric(r[i]) for r in body) for i in range(len(headers))
    ]

    def fmt(cells: Sequence[str]) -> str:
        out = []
        for i, cell in enumerate(cells):
            out.append(cell.rjust(widths[i]) if right[i] else cell.ljust(widths[i]))
        return indent + "  ".join(out).rstrip()

    lines = [fmt([str(h) for h in headers]), indent + "  ".join("-" * w for w in widths)]
    lines.extend(fmt(r) for r in body)
    return "\n".join(lines)


#: Characters that decorate a number without stopping it being one.
_NUMBER_DECORATION = str.maketrans("", "", "+-$,.%")


def _numeric(text: str) -> bool:
    """Whether a rendered cell reads as a number, so the column right-aligns."""
    stripped = text.removesuffix("bps").removesuffix("x").strip()
    return bool(stripped) and stripped.translate(_NUMBER_DECORATION).strip().isdigit()


def _heading(text: str) -> str:
    return f"\n{text}\n{'=' * len(text)}"


def render_allocation(allocation: Allocation, moments: Moments | None = None) -> str:
    rc = allocation.risk_contributions
    headers = ["asset", "weight"]
    if moments is not None:
        headers += ["exp. return", "volatility", "risk share"]
    rows = []
    for i, (symbol, weight) in enumerate(zip(allocation.symbols, allocation.weights, strict=True)):
        if weight <= 1e-6:
            continue
        row = [symbol, f"{weight:.1%}"]
        if moments is not None:
            idx = moments.symbols.index(symbol)
            row += [
                f"{moments.mu[idx]:.1%}",
                f"{moments.vol[idx]:.1%}",
                f"{rc[i]:.1%}" if i < len(rc) else "-",
            ]
        rows.append(row)

    out = [_heading(f"Target allocation ({allocation.objective})"), table(headers, rows)]
    out.append(
        f"\n  expected return {allocation.expected_return:.1%} | "
        f"volatility {allocation.expected_vol:.1%} | "
        f"Sharpe {allocation.sharpe:.2f} | "
        f"effective positions {allocation.diversification:.1f}"
    )
    return "\n".join(out)


def render_plan(plan: Plan, cfg: Config | None = None) -> str:
    out = [_heading("Orders to place")]
    if not plan.trades:
        out.append("  (no orders)")
    for venue, trades in sorted(plan.by_venue().items()):
        out.append(f"\n  On {venue}:")
        rows = [
            [
                t.side.upper(), t.symbol, f"${t.notional:,.2f}",
                f"{t.quantity:,.6f}".rstrip("0").rstrip("."),
                f"${t.price:,.2f}", f"${t.fee:,.2f}",
            ]
            for t in trades
        ]
        out.append(table(["side", "asset", "amount", "units", "price", "fee"], rows, indent="    "))

    out.append(
        f"\n  budget ${plan.budget:,.2f} | invested ${plan.deployed:,.2f} | "
        f"fees ${plan.total_fees:,.2f} ({plan.fee_drag_bps:,.0f} bps) | "
        f"cash left ${plan.cash_left:,.2f}"
    )
    drift = tracking_error(plan)
    if drift > 0.005:
        out.append(
            f"  rounding and minimums moved you {drift:.1%} away from the target "
            "weights in total"
        )
    if plan.dropped:
        out.append("\n  Left out:")
        for symbol, reason in plan.dropped:
            out.append(f"    - {symbol}: {reason}")
    if plan.warnings:
        out.append("\n  Worth knowing:")
        for warning in plan.warnings:
            out.append(f"    ! {warning}")
    return "\n".join(out)


def render_rebalance(report: RebalanceReport) -> str:
    out = [_heading("Current vs target")]
    rows = [
        [
            d.symbol, f"${d.current_value:,.2f}", f"{d.current_weight:.1%}",
            f"{d.target_weight:.1%}", f"{d.drift:+.1%}",
        ]
        for d in sorted(report.drifts, key=lambda x: -abs(x.drift))
    ]
    out.append(table(["asset", "value", "now", "target", "drift"], rows))
    out.append(
        f"\n  portfolio ${report.total_value:,.2f} | new cash ${report.new_cash:,.2f} | "
        f"largest drift {report.max_drift:.1%}"
    )
    if report.plan is not None:
        out.append(render_plan(report.plan))
    if report.skipped:
        out.append("\n  Skipped:")
        for symbol, reason in report.skipped:
            out.append(f"    - {symbol}: {reason}")
    if report.notes:
        out.append("\n  Notes:")
        for note in report.notes:
            out.append(f"    * {note}")
    return "\n".join(out)


def render_custody(report: CustodyReport) -> str:
    out = [_heading(f"Custody ({report.wallet})")]
    if not report.actions:
        out.append("  no crypto positions to review")
    else:
        rows = [
            [
                a.recommendation.upper(), a.symbol, f"${a.value:,.2f}",
                a.venue or "-", a.reason,
            ]
            for a in report.actions
        ]
        out.append(table(["action", "asset", "value", "venue", "why"], rows))
    for note in report.notes:
        out.append(f"\n  * {note}")
    return "\n".join(out)


def render_backtest(report: BacktestReport) -> str:
    out = [_heading("Walk-forward backtest")]
    rows = [
        [
            r.label, f"{r.total_return:+.1%}", f"{r.cagr:+.1%}", f"{r.volatility:.1%}",
            f"{r.sharpe:.2f}", f"{r.max_drawdown:.1%}",
            f"{r.turnover:.1f}x" if r.rebalances else "-",
        ]
        for r in report.all_results
    ]
    out.append(
        table(["strategy", "total", "CAGR", "vol", "Sharpe", "max DD", "turnover"], rows)
    )
    for warning in report.warnings:
        out.append(f"\n  ! {warning}")
    return "\n".join(out)


def render_reality_check(cfg: Config, plan: Plan) -> str:
    """The part a portfolio optimiser cannot tell you, so it usually doesn't."""
    budget = plan.budget
    lines = [_heading("Reality check")]
    annual = plan.allocation.expected_return * max(plan.deployed, 0.0)
    vol_dollars = plan.allocation.expected_vol * max(plan.deployed, 0.0)
    lines.append(
        f"  On ${plan.deployed:,.2f} invested, the modelled expected return is about "
        f"${annual:,.2f} a year, and a typical year swings roughly ±${vol_dollars:,.2f}."
    )
    recovery = _months_to_recover(plan)
    if recovery is None:
        lines.append(
            f"  You paid ${plan.total_fees:,.2f} in fees to open this "
            f"({plan.fee_drag_bps / 100:.2f}%). The modelled expected return is "
            "negative over this history, so there is no payback period to quote -- "
            "which is itself worth sitting with before you place the orders."
        )
    else:
        lines.append(
            f"  You paid ${plan.total_fees:,.2f} in fees to open this. That is "
            f"{plan.fee_drag_bps / 100:.2f}% -- earned back only after roughly "
            f"{recovery} of the modelled return."
        )
    if budget < 1000:
        lines.append(
            "  At this size the decisions that matter are how much you add each "
            "month and whether you leave it alone. Optimisation is a rounding "
            "error next to contributions at a few hundred dollars."
        )
    lines.append(
        "  Everything above rests on the past two years resembling the next one. "
        "It may not. Size this at what you can genuinely afford to lose."
    )
    return "\n".join(lines)


def _months_to_recover(plan: Plan) -> str | None:
    """How long the modelled return takes to earn the fees back, or None."""
    annual = plan.allocation.expected_return * max(plan.deployed, 1e-9)
    if annual <= 0:
        return None
    if plan.total_fees <= 0:
        return "no time at all -- this plan costs nothing to open"
    months = 12.0 * plan.total_fees / annual
    if months < 1:
        return f"{months * 30:.0f} days"
    return f"{months:.1f} months" if months < 24 else f"{months / 12:.1f} years"


def render_footer() -> str:
    return f"\n{'-' * 72}\n{DISCLAIMER}"


def to_json(payload: dict[str, Any]) -> str:
    def default(obj: Any) -> Any:
        if is_dataclass(obj) and not isinstance(obj, type):
            return asdict(obj)
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)

    return json.dumps(payload, indent=2, default=default)


def plan_payload(cfg: Config, allocation: Allocation, plan: Plan) -> dict[str, Any]:
    return {
        "disclaimer": DISCLAIMER,
        "budget": plan.budget,
        "objective": allocation.objective,
        "risk_profile": cfg.risk_profile,
        "allocation": {
            "weights": allocation.as_dict(),
            "expected_return": allocation.expected_return,
            "expected_vol": allocation.expected_vol,
            "sharpe": allocation.sharpe,
            "effective_positions": allocation.diversification,
        },
        "orders": [
            {
                "symbol": t.symbol, "side": t.side, "venue": t.venue,
                "notional": round(t.notional, 2), "quantity": t.quantity,
                "price": t.price, "fee": round(t.fee, 4),
            }
            for t in plan.trades
        ],
        "realised_weights": realised_weights(plan),
        "totals": {
            "deployed": round(plan.deployed, 2),
            "fees": round(plan.total_fees, 4),
            "fee_drag_bps": round(plan.fee_drag_bps, 1),
            "cash_left": round(plan.cash_left, 2),
        },
        "dropped": [{"symbol": s, "reason": r} for s, r in plan.dropped],
        "warnings": plan.warnings,
    }


def render_markdown(cfg: Config, allocation: Allocation, plan: Plan) -> str:
    lines = [
        "# Investment plan",
        "",
        f"- Budget: **${plan.budget:,.2f}**",
        f"- Risk profile: **{cfg.risk_profile}**",
        f"- Objective: **{allocation.objective}**",
        f"- Modelled return / volatility: **{allocation.expected_return:.1%}** / "
        f"**{allocation.expected_vol:.1%}**",
        "",
        "## Target weights",
        "",
        "| Asset | Weight |",
        "| --- | ---: |",
    ]
    lines += [f"| {s} | {w:.1%} |" for s, w in allocation.nonzero()]
    lines += ["", "## Orders", "", "| Venue | Side | Asset | Amount | Units | Fee |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for t in plan.trades:
        units = f"{t.quantity:,.6f}".rstrip("0").rstrip(".")
        lines.append(
            f"| {t.venue} | {t.side} | {t.symbol} | ${t.notional:,.2f} | {units} | ${t.fee:,.2f} |"
        )
    lines += [
        "",
        f"Invested **${plan.deployed:,.2f}**, fees **${plan.total_fees:,.2f}** "
        f"({plan.fee_drag_bps:,.0f} bps), cash left **${plan.cash_left:,.2f}**.",
        "",
        "---",
        "",
        f"_{DISCLAIMER}_",
    ]
    return "\n".join(lines)
