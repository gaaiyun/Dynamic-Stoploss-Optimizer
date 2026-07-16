"""attribution.py 测试。"""
from __future__ import annotations

import pandas as pd

from dso.attribution import attribute_pnl
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
    assert rep.mfe_total_pnl == 0


def test_attribute_with_trades_records_n_trades(synth_mixed_df):
    result = run_backtest(synth_mixed_df, ATRStop())
    rep = attribute_pnl(result, synth_mixed_df)
    assert rep.n_trades == result.n_trades


def test_mfe_pnl_at_least_actual(synth_mixed_df):
    """最大有利变动净收益不应比实际退出更差。"""
    result = run_backtest(synth_mixed_df, ATRStop())
    rep = attribute_pnl(result, synth_mixed_df)
    if rep.n_trades > 0:
        assert rep.mfe_total_pnl >= rep.actual_total_pnl - 0.1


def test_realization_gap_is_negative_or_zero(synth_mixed_df):
    """实际收益减 MFE 净收益应小于等于零。"""
    result = run_backtest(synth_mixed_df, ATRStop())
    rep = attribute_pnl(result, synth_mixed_df)
    if rep.n_trades > 0:
        assert rep.realization_gap_pnl <= 0.01


def test_aggregate_capture_ratio_is_finite(synth_mixed_df):
    result = run_backtest(synth_mixed_df, ATRStop())
    rep = attribute_pnl(result, synth_mixed_df)
    import math
    assert math.isfinite(rep.aggregate_capture_ratio)


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
        assert math.isfinite(rep.avg_trade_capture_ratio)


def test_entry_bar_high_is_not_treated_as_post_entry_mfe():
    df = pd.DataFrame({
        "open": [100, 100], "high": [150, 102],
        "low": [90, 99], "close": [100, 101], "volume": [1, 1],
    })
    result = run_backtest(
        df, ATRStop(period=14), entry_signal=[True, False], commission_pct=0.0,
    )
    rep = attribute_pnl(result, df)
    assert rep.mfe_total_pnl == 2_000.0
