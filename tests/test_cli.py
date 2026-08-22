"""End-to-end runs through the actual entry point, on synthetic prices."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ainvest.cli import main
from ainvest.templates import EXAMPLE_CONFIG
from conftest import base_config


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = base_config()
    raw["data"] = {**raw["data"], "cache_dir": str(tmp_path / "cache")}
    path = tmp_path / "portfolio.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def run(args, config=None) -> int:
    return main([*args, "-c", str(config)] if config else args)


def test_init_writes_a_config_that_loads(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    written = tmp_path / "portfolio.yaml"
    assert written.exists()
    assert yaml.safe_load(written.read_text())["budget"] > 0
    assert "NOT FINANCIAL ADVICE" in capsys.readouterr().out


def test_init_refuses_to_clobber_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "portfolio.yaml").write_text("budget: 1\n")
    assert main(["init"]) == 1
    assert (tmp_path / "portfolio.yaml").read_text() == "budget: 1\n"
    assert main(["init", "--force"]) == 0
    assert len((tmp_path / "portfolio.yaml").read_text()) > 100


def test_the_shipped_template_matches_the_checked_in_example():
    example = Path(__file__).resolve().parents[1] / "config" / "portfolio.example.yaml"
    assert example.read_text() == EXAMPLE_CONFIG, (
        "config/portfolio.example.yaml has drifted from templates.EXAMPLE_CONFIG"
    )


def test_optimize_prints_a_plan_and_the_disclaimer(workspace, capsys):
    assert run(["optimize"], workspace) == 0
    out = capsys.readouterr().out
    assert "Target allocation" in out
    assert "Orders to place" in out
    assert "Reality check" in out
    assert "NOT FINANCIAL ADVICE" in out


def test_optimize_json_is_valid_and_self_consistent(workspace, capsys):
    assert run(["optimize", "--json"], workspace) == 0
    payload = json.loads(capsys.readouterr().out)
    assert sum(payload["allocation"]["weights"].values()) == pytest.approx(1.0, abs=1e-6)
    spent = sum(o["notional"] for o in payload["orders"]) + payload["totals"]["fees"]
    assert spent <= payload["budget"] + 0.01
    assert payload["disclaimer"].startswith("NOT FINANCIAL ADVICE")


def test_budget_override_flows_through(workspace, capsys):
    assert run(["optimize", "--budget", "1000", "--json"], workspace) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["budget"] == pytest.approx(1000.0)
    assert payload["totals"]["deployed"] > 900


@pytest.fixture
def wide_workspace(tmp_path, monkeypatch):
    """A universe broad enough that even the tightest profile is feasible."""
    monkeypatch.chdir(tmp_path)
    raw = base_config()
    raw["data"] = {**raw["data"], "cache_dir": str(tmp_path / "cache")}
    for symbol in ("VOO", "QQQ", "BND", "TLT"):
        raw["assets"].append(
            {"symbol": symbol, "asset_class": "equity", "venues": ["public"]}
        )
    path = tmp_path / "wide.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


@pytest.mark.parametrize(
    "profile, limit", [("conservative", 0.10), ("balanced", 0.25), ("aggressive", 0.50)]
)
def test_profile_override_changes_the_crypto_cap(wide_workspace, capsys, profile, limit):
    assert run(["optimize", "--profile", profile, "--json"], wide_workspace) == 0
    payload = json.loads(capsys.readouterr().out)
    crypto = sum(
        w for s, w in payload["allocation"]["weights"].items() if s in {"BTC", "ETH"}
    )
    assert crypto <= limit + 1e-6


def test_a_profile_too_tight_for_the_universe_explains_itself(workspace, capsys):
    """Four assets cannot fill a portfolio when each is capped at 25%."""
    assert run(["optimize", "--profile", "conservative"], workspace) == 3
    err = capsys.readouterr().err
    assert "4 available assets" in err
    assert "single_max" in err
    assert "add more assets" in err


def test_markdown_output_is_a_document(workspace, capsys):
    assert run(["optimize", "--markdown"], workspace) == 0
    out = capsys.readouterr().out
    assert out.startswith("# Investment plan")
    assert "| Venue | Side |" in out


def test_explain_shows_correlations_and_objectives(workspace, capsys):
    assert run(["explain"], workspace) == 0
    out = capsys.readouterr().out
    assert "Correlations" in out
    assert "risk_parity" in out


def test_backtest_runs(workspace, capsys):
    assert run(["backtest", "--lookback", "150", "--rebalance-days", "45"], workspace) == 0
    out = capsys.readouterr().out
    assert "Walk-forward backtest" in out
    assert "equal weight" in out


def test_rebalance_with_holdings(workspace, capsys):
    raw = yaml.safe_load(workspace.read_text())
    raw["holdings"] = [{"symbol": "VTI", "quantity": 0.5, "venue": "public"}]
    workspace.write_text(yaml.safe_dump(raw))
    assert run(["rebalance", "--cash", "250", "--json"], workspace) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total_value"] > 0
    assert all(o["side"] == "buy" for o in payload["orders"])


def test_check_runs_the_whole_monthly_review(workspace, capsys):
    raw = yaml.safe_load(workspace.read_text())
    raw["holdings"] = [{
        "symbol": "VTI", "quantity": 1.0, "venue": "public",
        "cost_basis": 500.0, "acquired": "2025-01-15",
    }]
    workspace.write_text(yaml.safe_dump(raw))
    assert run(["check"], workspace) == 0
    out = capsys.readouterr().out
    assert "Portfolio check" in out
    assert "contribution" in out
    assert "Tax review" in out
    assert "Signals" in out
    assert "NOT FINANCIAL ADVICE" in out


def test_check_shows_events_and_never_fetches_offline(workspace, capsys):
    from datetime import date, timedelta
    raw = yaml.safe_load(workspace.read_text())
    raw["news"] = {"events": [{
        "date": (date.today() + timedelta(days=4)).isoformat(),
        "title": "FOMC rate decision",
    }]}
    workspace.write_text(yaml.safe_dump(raw))
    assert run(["fetch"], workspace) == 0  # warm the cache so --offline has data
    capsys.readouterr()
    assert run(["check", "--offline"], workspace) == 0
    out = capsys.readouterr().out
    assert "News & events (context, never triggers)" in out
    assert "FOMC rate decision" in out
    assert "offline: headline fetch skipped" in out


def test_rebalance_withholds_sells_without_approval(workspace, capsys):
    raw = yaml.safe_load(workspace.read_text())
    raw["holdings"] = [{"symbol": "VTI", "quantity": 5.0, "venue": "public"}]
    raw["allow_sells"] = True
    workspace.write_text(yaml.safe_dump(raw))
    assert run(["rebalance", "--cash", "0", "--json"], workspace) == 0
    payload = json.loads(capsys.readouterr().out)
    assert not any(o["side"] == "sell" for o in payload["orders"])
    assert any("withheld" in item["reason"] for item in payload["skipped"])
    capsys.readouterr()
    assert run(["rebalance", "--cash", "0", "--approve-sells", "--json"], workspace) == 0
    approved = json.loads(capsys.readouterr().out)
    assert any(o["side"] == "sell" for o in approved["orders"])


def test_project_reports_percentile_bands(workspace, capsys):
    assert run(["project", "--years", "3", "--paths", "300"], workspace) == 0
    out = capsys.readouterr().out
    assert "Monte Carlo" in out
    assert "median" in out
    assert "p10" in out


def test_project_json_is_valid(workspace, capsys):
    assert run(
        ["project", "--years", "3", "--paths", "200", "--dca-months", "3", "--json"],
        workspace,
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["results"]
    for r in payload["results"]:
        bands = r["wealth_percentiles"]
        assert bands["10"] <= bands["50"] <= bands["90"]
    assert payload["dca_comparison"]


def test_backtest_exit_rule_charges_taxes(workspace, capsys):
    raw = yaml.safe_load(workspace.read_text())
    raw["signals"] = {"trend_days": 100}
    workspace.write_text(yaml.safe_dump(raw))
    assert run(["backtest", "--exit-rule", "trend"], workspace) == 0
    out = capsys.readouterr().out
    assert "stay invested" in out
    assert "after cash-out" in out
    assert "trend exit" in out


def test_fetch_populates_and_lists_the_cache(workspace, capsys, tmp_path):
    assert run(["fetch"], workspace) == 0
    assert (tmp_path / "cache").exists()
    capsys.readouterr()
    assert run(["fetch", "--list"], workspace) == 0
    listing = capsys.readouterr().out
    assert "BTC-USD" in listing


def test_offline_mode_uses_the_cache(workspace, capsys):
    assert run(["fetch"], workspace) == 0
    capsys.readouterr()
    assert run(["optimize", "--offline", "--json"], workspace) == 0
    assert json.loads(capsys.readouterr().out)["orders"]


class TestErrorHandling:
    def test_missing_config_exits_cleanly(self, tmp_path, capsys):
        assert main(["optimize", "-c", str(tmp_path / "nope.yaml")]) == 2
        assert "ainvest init" in capsys.readouterr().err

    def test_bad_config_exits_cleanly(self, tmp_path, capsys):
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.safe_dump({**base_config(), "risk_profile": "wat"}))
        assert main(["optimize", "-c", str(path)]) == 2
        assert "unknown risk_profile" in capsys.readouterr().err

    def test_impossible_constraints_exit_cleanly(self, tmp_path, capsys, monkeypatch):
        monkeypatch.chdir(tmp_path)
        raw = base_config()
        raw["data"] = {**raw["data"], "cache_dir": str(tmp_path / "cache")}
        # Every asset capped at 10%, so four assets cannot add up to 100%.
        for asset in raw["assets"]:
            asset["max_weight"] = 0.10
        path = tmp_path / "tight.yaml"
        path.write_text(yaml.safe_dump(raw))
        assert main(["optimize", "-c", str(path)]) == 3
        assert "cannot be satisfied" in capsys.readouterr().err

    def test_an_unknown_subcommand_is_rejected(self):
        with pytest.raises(SystemExit):
            main(["moon"])
