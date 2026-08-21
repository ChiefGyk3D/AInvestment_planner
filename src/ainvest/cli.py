"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import DISCLAIMER, __version__
from .allocator import optimize_weights
from .backtest import run_backtest
from .config import RISK_PROFILES, Config, ConfigError, load_config
from .custody import review as custody_review
from .data import PriceCache, load_prices
from .data.base import ProviderError
from .optimize.core import InfeasibleError
from .optimize.objectives import DESCRIPTIONS, OBJECTIVES
from .plan import build_plan
from .rebalance import holdings_value, plan_rebalance
from .report import (
    plan_payload,
    render_allocation,
    render_backtest,
    render_custody,
    render_footer,
    render_markdown,
    render_plan,
    render_reality_check,
    render_rebalance,
    table,
    to_json,
)
from .stats import align_prices, estimate_moments
from .templates import EXAMPLE_CONFIG

DEFAULT_CONFIG = "portfolio.yaml"


def _feed_map(cfg: Config) -> dict[str, str]:
    """feed symbol -> portfolio symbol, so columns come back named as you named them."""
    return {a.feed_symbol: a.symbol for a in cfg.assets}


def _prepare(cfg: Config, args) -> tuple[pd.DataFrame, list[str]]:
    """Load prices, rename to portfolio symbols, align calendars."""
    frame = load_prices(
        cfg, refresh=getattr(args, "refresh", False), offline=getattr(args, "offline", False)
    )
    frame = frame.rename(columns=_feed_map(cfg))
    known = [a.symbol for a in cfg.assets]
    frame = frame[[c for c in frame.columns if c in known]]
    missing = [s for s in known if s not in frame.columns]
    aligned = align_prices(frame, cfg.data.min_observations)
    missing += [c for c in aligned.attrs.get("dropped", []) if c not in missing]
    return aligned, missing


def _latest_prices(prices: pd.DataFrame) -> dict[str, float]:
    return {c: float(prices[c].iloc[-1]) for c in prices.columns}


def _report_missing(missing: list[str]) -> None:
    if missing:
        print(
            f"  note: no usable price history for {', '.join(sorted(missing))} -- "
            "excluded from this run",
            file=sys.stderr,
        )


def cmd_init(args) -> int:
    path = Path(args.output)
    if path.exists() and not args.force:
        print(f"{path} already exists; pass --force to overwrite", file=sys.stderr)
        return 1
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(EXAMPLE_CONFIG)
    print(f"Wrote {path}")
    print("\nNext:")
    print(f"  1. Edit {path} -- especially the venue fees, which are estimates.")
    print("  2. ainvest optimize          # see the plan")
    print("  3. ainvest backtest          # sanity-check the method")
    print(f"\n{DISCLAIMER}")
    return 0


def cmd_fetch(args) -> int:
    cfg = load_config(args.config)
    cache = PriceCache(cfg.data.cache_dir, cfg.data.cache_ttl_hours)
    if args.list:
        rows = cache.describe()
        print(f"Cache: {Path(cfg.data.cache_dir).resolve()}")
        print(table(["symbol", "rows", "last date"], rows))
        return 0
    frame = load_prices(cfg, refresh=True)
    print(f"Fetched {frame.shape[1]} symbols, {frame.shape[0]} rows "
          f"via {frame.attrs.get('source', cfg.data.provider)}")
    print(table(["symbol", "rows", "last date"], cache.describe()))
    return 0


def _build(cfg: Config, args):
    """Load prices, estimate moments and optimise, in one step."""
    prices, missing = _prepare(cfg, args)
    moments = estimate_moments(
        prices,
        covariance_shrinkage=cfg.optimizer.covariance_shrinkage,
        mean_shrinkage=cfg.optimizer.mean_shrinkage,
    )
    allocation = optimize_weights(cfg, moments)
    return allocation, _latest_prices(prices), prices, missing, moments


def cmd_optimize(args) -> int:
    cfg = load_config(args.config)
    if args.objective:
        cfg.optimizer.objective = args.objective
    if args.budget is not None:
        cfg.budget = args.budget
    if args.profile:
        cfg.risk_profile = args.profile
        preset = RISK_PROFILES[args.profile]
        cfg.constraints.crypto_max = preset["crypto_max"]
        cfg.constraints.single_max = preset["single_max"]
        cfg.optimizer.target_vol = preset["target_vol"]
        cfg.constraints.validate()

    allocation, prices, panel, missing, moments = _build(cfg, args)
    plan = build_plan(cfg, allocation, prices)

    if args.json:
        payload = plan_payload(cfg, allocation, plan)
        payload["excluded"] = missing
        print(to_json(payload))
        return 0
    if args.markdown:
        print(render_markdown(cfg, allocation, plan))
        return 0

    _report_missing(missing)
    print(f"AInvestment Planner {__version__} | {panel.shape[0]} observations "
          f"through {panel.index[-1].date()} | source: {panel.attrs.get('source', cfg.data.provider)}")
    print(render_allocation(allocation, moments))
    print(render_plan(plan, cfg))
    print(render_custody(custody_review(cfg, plan)))
    print(render_reality_check(cfg, plan))
    print(render_footer())
    return 0


def cmd_rebalance(args) -> int:
    cfg = load_config(args.config)
    if args.allow_sells:
        cfg.allow_sells = True
    allocation, prices, _panel, missing, _moments = _build(cfg, args)
    new_cash = args.cash if args.cash is not None else cfg.investable
    report = plan_rebalance(cfg, allocation, prices, new_cash=new_cash)

    if args.json:
        payload = {
            "disclaimer": DISCLAIMER,
            "total_value": report.total_value,
            "new_cash": report.new_cash,
            "max_drift": report.max_drift,
            "drifts": [
                {
                    "symbol": d.symbol, "value": round(d.current_value, 2),
                    "current_weight": d.current_weight, "target_weight": d.target_weight,
                    "drift": d.drift,
                }
                for d in report.drifts
            ],
            "orders": [
                {"symbol": t.symbol, "side": t.side, "venue": t.venue,
                 "notional": round(t.notional, 2), "quantity": t.quantity, "fee": round(t.fee, 4)}
                for t in (report.plan.trades if report.plan else [])
            ],
            "skipped": [{"symbol": s, "reason": r} for s, r in report.skipped],
            "notes": report.notes,
        }
        print(to_json(payload))
        return 0

    _report_missing(missing)
    held, _ = holdings_value(cfg.holdings, prices)
    print(f"Holdings worth ${held:,.2f}; adding ${new_cash:,.2f}")
    print(render_allocation(allocation))
    print(render_rebalance(report))
    if report.plan is not None:
        print(render_custody(custody_review(cfg, report.plan)))
    print(render_footer())
    return 0


def cmd_backtest(args) -> int:
    cfg = load_config(args.config)
    if args.objective:
        cfg.optimizer.objective = args.objective
    prices, missing = _prepare(cfg, args)
    report = run_backtest(
        cfg, prices,
        lookback_days=args.lookback,
        rebalance_days=args.rebalance_days,
    )
    if args.json:
        print(to_json({
            "disclaimer": DISCLAIMER,
            "results": [
                {
                    "label": r.label, "total_return": r.total_return, "cagr": r.cagr,
                    "volatility": r.volatility, "sharpe": r.sharpe,
                    "max_drawdown": r.max_drawdown, "turnover": r.turnover,
                    "rebalances": r.rebalances,
                }
                for r in report.all_results
            ],
            "warnings": report.warnings,
        }))
        return 0
    _report_missing(missing)
    print(render_backtest(report))
    print(render_footer())
    return 0


def cmd_explain(args) -> int:
    cfg = load_config(args.config)
    prices, missing = _prepare(cfg, args)
    moments = estimate_moments(
        prices,
        covariance_shrinkage=cfg.optimizer.covariance_shrinkage,
        mean_shrinkage=cfg.optimizer.mean_shrinkage,
    )
    _report_missing(missing)

    print(f"\nUniverse ({moments.observations} observations, "
          f"{moments.periods:.0f} periods/year, covariance shrinkage "
          f"{moments.shrinkage:.0%})")
    rows = [
        [s, cfg.asset(s).asset_class, f"{moments.mu[i]:.1%}", f"{moments.vol[i]:.1%}",
         f"{(moments.mu[i] - cfg.optimizer.risk_free_rate) / moments.vol[i]:.2f}"
         if moments.vol[i] > 0 else "-"]
        for i, s in enumerate(moments.symbols)
    ]
    print(table(["asset", "class", "exp. return", "volatility", "Sharpe"], rows))

    print("\nCorrelations")
    corr = moments.correlation()
    header = ["", *moments.symbols]
    rows = [
        [s, *[f"{corr[i][j]:+.2f}" for j in range(len(moments.symbols))]]
        for i, s in enumerate(moments.symbols)
    ]
    print(table(header, rows))
    print("\n  Correlation is what diversification is made of: two assets at +0.95 "
          "\n  are one position wearing two tickers.")

    print("\nObjectives")
    for name in OBJECTIVES:
        marker = "*" if name == cfg.optimizer.objective else " "
        print(f"  {marker} {name:<14} {DESCRIPTIONS[name]}")
    print(f"\n  (* = currently selected in {args.config})")
    print(render_footer())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ainvest",
        description="Fee-aware allocation planning for small stock and crypto accounts. "
                    "Produces plans, never orders. Not financial advice.",
    )
    parser.add_argument("--version", action="version", version=f"ainvest {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(sub, *, data: bool = True) -> None:
        sub.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="path to the YAML config")
        if data:
            sub.add_argument("--refresh", action="store_true", help="ignore the cache and refetch")
            sub.add_argument("--offline", action="store_true", help="use cached prices only")

    p_init = subparsers.add_parser("init", help="write a starter config")
    p_init.add_argument("-o", "--output", default=DEFAULT_CONFIG)
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_fetch = subparsers.add_parser("fetch", help="refresh or inspect the price cache")
    common(p_fetch, data=False)
    p_fetch.add_argument("--list", action="store_true", help="show what is cached and exit")
    p_fetch.set_defaults(func=cmd_fetch, refresh=True, offline=False)

    p_opt = subparsers.add_parser("optimize", help="build an allocation and a buy list")
    common(p_opt)
    p_opt.add_argument("--budget", type=float, help="override the configured budget")
    p_opt.add_argument("--objective", choices=OBJECTIVES)
    p_opt.add_argument(
        "--profile", choices=sorted(RISK_PROFILES),
        help="apply a risk profile, overriding single_max, crypto_max and "
             "target_vol from the config",
    )
    p_opt.add_argument("--json", action="store_true")
    p_opt.add_argument("--markdown", action="store_true")
    p_opt.set_defaults(func=cmd_optimize)

    p_reb = subparsers.add_parser("rebalance", help="plan trades from your current holdings")
    common(p_reb)
    p_reb.add_argument("--cash", type=float, help="new money to add (default: budget)")
    p_reb.add_argument("--allow-sells", action="store_true",
                       help="permit sales; may realise taxable gains")
    p_reb.add_argument("--json", action="store_true")
    p_reb.set_defaults(func=cmd_rebalance)

    p_bt = subparsers.add_parser("backtest", help="walk the objective forward through history")
    common(p_bt)
    p_bt.add_argument("--objective", choices=OBJECTIVES)
    p_bt.add_argument("--lookback", type=int, default=252, help="rows used to fit each refit")
    p_bt.add_argument("--rebalance-days", type=int, default=63, help="rows between refits")
    p_bt.add_argument("--json", action="store_true")
    p_bt.set_defaults(func=cmd_backtest)

    p_exp = subparsers.add_parser("explain", help="show the inputs behind the numbers")
    common(p_exp)
    p_exp.set_defaults(func=cmd_explain)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except InfeasibleError as exc:
        print(f"constraints cannot be satisfied: {exc}", file=sys.stderr)
        return 3
    except ProviderError as exc:
        print(f"data error: {exc}", file=sys.stderr)
        return 4
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
