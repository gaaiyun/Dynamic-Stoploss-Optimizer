"""attribution.py 测试。"""
from __future__ import annotations

import pandas as pd

from dso.attribution import AttributionReport, attribute_pnl
from dso.backtest import run_backtest
from dso.stops import ATRStop


def test_attribute_empty_backtest_safe(synth_mixed_df):
    """没交易时也应该不崩，返回 0。"""
    n = len(synth_mixed_df)
    result = run_backtest(synth_mixed_df, ATRStop(),
                          entry_signal=[False] * n)
    rep = attribute_pnl(result, synth_mixed_df)
    assert rep.n_trades == 0
    assert rep.actual_total_pnl == 0
    assert rep.perfect_total_pnl == 0


def test_attribute_with_trades_records_n_trades(synth_mixed_df):
    result = run_backtest(synth_mixed_df, ATRStop())
    rep = attribute_pnl(result, synth_mixed_df)
    assert rep.n_trades == result.n_trades


def test_perfect_pnl_at_least_actual(synth_mixed_df):
    """完美出场点 P/L 不会比实际更差（理论上 ≥）。"""
    result = run_backtest(synth_mixed_df, ATRStop())
    rep = attribute_pnl(result, synth_mixed_df)
    if rep.n_trades > 0:
        # perfect_total >= actual_total（perfect 是最优 exit，actual 是实际 exit）
        # 但加上佣金成本可能让 actual 略低
        assert rep.perfect_total_pnl >= rep.actual_total_pnl - 0.1


def test_stop_pnl_loss_is_negative_or_zero(synth_mixed_df):
    """实际 - perfect ≤ 0（停损让出了一部分潜在利润）。"""
    result = run_backtest(synth_mixed_df, ATRStop())
    rep = attribute_pnl(result, synth_mixed_df)
    if rep.n_trades > 0:
        assert rep.stop_pnl_loss <= 0.01    # 允许浮点误差


def test_attribution_contribution_pcts_sum_to_one_or_zero(synth_mixed_df):
    result = run_backtest(synth_mixed_df, ATRStop())
    rep = attribute_pnl(result, synth_mixed_df)
    if rep.entry_contribution_pct > 0 or rep.stop_contribution_pct > 0:
        # 加起来应该是 1
        total = rep.entry_contribution_pct + rep.stop_contribution_pct
        assert abs(total - 1.0) < 1e-9


def test_attribute_to_dict_serializable(synth_mixed_df):
    import json
    result = run_backtest(synth_mixed_df, ATRStop())
    rep = attribute_pnl(result, synth_mixed_df)
    json.dumps(rep.to_dict())


def test_attribute_capture_ratio_reasonable(synth_mixed_df):
    """capture ratio ∈ [-large, 1] 区间。"""
    result = run_backtest(synth_mixed_df, ATRStop())
    rep = attribute_pnl(result, synth_mixed_df)
    if rep.n_trades > 0:
        # 平均捕获率不应该是 NaN
        import math
        assert math.isfinite(rep.avg_pct_of_perfect_captured)
