"""compare.py 测试。"""
from __future__ import annotations

import pandas as pd

from dso.compare import StrategyComparison, StrategyRow, compare_stops
from dso.stops import ATRStop, ChandelierStop, TimeStop


def test_compare_returns_one_row_per_strategy(synth_mixed_df):
    stops = {"atr": ATRStop(), "chandelier": ChandelierStop()}
    cmp = compare_stops(synth_mixed_df, stops)
    assert len(cmp.rows) == 2
    assert {r.stop_name for r in cmp.rows} == {"atr", "chandelier"}


def test_compare_sorted_by_sharpe_desc(synth_mixed_df):
    stops = {f"s{i}": ATRStop(period=p, multiplier=2.5)
             for i, p in enumerate([5, 10, 20])}
    cmp = compare_stops(synth_mixed_df, stops)
    sharpes = [r.sharpe for r in cmp.rows]
    assert sharpes == sorted(sharpes, reverse=True)


def test_compare_best_strategy_set(synth_mixed_df):
    stops = {"atr": ATRStop(), "time": TimeStop(max_bars=10)}
    cmp = compare_stops(synth_mixed_df, stops)
    assert cmp.best_strategy in stops
    # 应该是 sharpe 最高的那个
    assert cmp.best_strategy == cmp.rows[0].stop_name


def test_compare_empty_stops_returns_empty(synth_mixed_df):
    cmp = compare_stops(synth_mixed_df, {})
    assert cmp.rows == []
    assert cmp.best_strategy is None


def test_compare_to_dict_serializable(synth_mixed_df):
    import json
    stops = {"atr": ATRStop(), "chandelier": ChandelierStop()}
    cmp = compare_stops(synth_mixed_df, stops)
    json.dumps(cmp.to_dict())


def test_compare_to_table_includes_strategy_names(synth_mixed_df):
    stops = {"atr": ATRStop(), "time": TimeStop()}
    cmp = compare_stops(synth_mixed_df, stops)
    table = cmp.to_table()
    assert "atr" in table
    assert "time" in table
    assert "Sharpe" in table


def test_compare_to_table_has_separator(synth_mixed_df):
    stops = {"atr": ATRStop()}
    cmp = compare_stops(synth_mixed_df, stops)
    lines = cmp.to_table().split("\n")
    assert lines[1].startswith("-")


def test_compare_dsr_computed_when_multiple_strategies(synth_mixed_df):
    """超过 1 个策略才有意义算 deflated Sharpe。"""
    stops = {"atr": ATRStop(), "chandelier": ChandelierStop(),
             "time": TimeStop()}
    cmp = compare_stops(synth_mixed_df, stops)
    # 不强制大于 0，但至少不应该是 NaN
    assert 0.0 <= cmp.deflated_sharpe_pct <= 1.0


def test_compare_strategy_row_to_dict():
    r = StrategyRow(stop_name="x", total_return=0.1, sharpe=1.5,
                    sortino=2.0, calmar=1.0, max_dd=-0.05,
                    n_trades=20, hit_rate=0.6)
    d = r.to_dict()
    assert d["sharpe"] == 1.5
    assert d["n_trades"] == 20
