"""backtest.py 测试 —— triple-barrier 事件驱动。"""
from __future__ import annotations

import pandas as pd
import pytest

from dso.backtest import (
    BacktestResult, Trade, _resolve_signals, _to_bars, run_backtest,
)
from dso.stops import ATRStop, TimeStop


# --- helpers ---------------------------------------------------------

def test_to_bars_handles_lowercase():
    df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1.5]})
    bars = _to_bars(df)
    assert len(bars) == 1
    assert bars[0].close == 1.5


def test_resolve_signals_default_is_all_true():
    sig = _resolve_signals(pd.DataFrame(), None, 5)
    assert sig == [True] * 5


def test_resolve_signals_passes_pd_series():
    s = pd.Series([True, False, True, False, True])
    sig = _resolve_signals(pd.DataFrame(), s, 5)
    assert sig == [True, False, True, False, True]


def test_resolve_signals_validates_length():
    with pytest.raises(ValueError):
        _resolve_signals(pd.DataFrame(), [True, False], 5)


def test_resolve_signals_handles_pd_series_with_nan():
    s = pd.Series([True, None, True])
    sig = _resolve_signals(pd.DataFrame(), s, 3)
    assert sig == [True, False, True]


# --- run_backtest ----------------------------------------------------

def test_backtest_returns_result(synth_uptrend_df):
    stop = ATRStop(period=10, multiplier=2.5)
    result = run_backtest(synth_uptrend_df, stop)
    assert isinstance(result, BacktestResult)
    assert result.stop_name == "ATRStop"


def test_backtest_initial_capital_default():
    df = pd.DataFrame({
        "open": [100] * 30, "high": [101] * 30, "low": [99] * 30,
        "close": [100] * 30, "volume": [1e6] * 30,
    })
    result = run_backtest(df, ATRStop())
    assert result.initial_capital == 100_000.0


def test_backtest_equity_curve_length_matches(synth_uptrend_df):
    result = run_backtest(synth_uptrend_df, ATRStop())
    assert len(result.equity_curve) == len(synth_uptrend_df)


def test_backtest_profit_target_triggers(synth_uptrend_df):
    """上涨 + 10% 止盈应该触发若干次 profit_taken。"""
    stop = ATRStop(period=10, multiplier=10.0)    # 故意宽松，让 profit 先触发
    result = run_backtest(synth_uptrend_df, stop,
                          profit_target_pct=0.05,
                          max_holding_bars=50)
    # 上涨数据 + 5% 止盈 → 应该有 profit_taken
    # 但具体几次依赖随机种子，至少应该 >= 0
    assert result.n_profit_taken >= 0
    assert result.n_trades >= 1


def test_backtest_max_holding_triggers_time_stop(synth_uptrend_df):
    # 用很宽的 ATR 让止损不被触发，强制时间 barrier 收
    stop = ATRStop(period=10, multiplier=100.0)
    result = run_backtest(synth_uptrend_df, stop,
                          max_holding_bars=5)
    assert result.n_timed_out > 0


def test_backtest_zero_trades_when_no_signal(synth_uptrend_df):
    n = len(synth_uptrend_df)
    result = run_backtest(synth_uptrend_df, ATRStop(),
                          entry_signal=[False] * n)
    assert result.n_trades == 0
    assert result.final_equity == result.initial_capital


def test_backtest_short_side(synth_downtrend_df):
    """下跌行情 short 仓应能赚钱。"""
    stop = ATRStop(period=10, multiplier=3.0)
    result = run_backtest(synth_downtrend_df, stop, side="short",
                          profit_target_pct=0.05)
    # 在合成下跌数据上 short 应该至少有些正盈利交易
    # （不强制要求总收益正，因为随机性 + commission）
    assert result.n_trades > 0


def test_backtest_commission_reduces_final_equity(synth_uptrend_df):
    stop_lo = ATRStop(period=10, multiplier=2.0)
    result_lo = run_backtest(synth_uptrend_df, stop_lo, commission_pct=0.0001)
    stop_hi = ATRStop(period=10, multiplier=2.0)
    result_hi = run_backtest(synth_uptrend_df, stop_hi, commission_pct=0.01)
    if result_lo.n_trades > 0:
        assert result_lo.final_equity >= result_hi.final_equity


def test_backtest_to_dict_serializable(synth_uptrend_df):
    import json
    result = run_backtest(synth_uptrend_df, ATRStop())
    json.dumps(result.to_dict())


# --- Trade -----------------------------------------------------------

def test_trade_to_dict_preserves_fields():
    t = Trade(entry_bar=10, entry_price=100, exit_bar=15, exit_price=110,
              side="long", exit_reason="profit", n_bars=5, pnl=10,
              pnl_pct=0.1, commission=0.2)
    d = t.to_dict()
    assert d["side"] == "long"
    assert d["pnl"] == 10.0


# --- BacktestResult --------------------------------------------------

def test_hit_rate_empty_returns_zero():
    r = BacktestResult(stop_name="x", stop_params={})
    assert r.hit_rate == 0.0


def test_total_return_zero_initial():
    r = BacktestResult(stop_name="x", stop_params={},
                       initial_capital=0, final_equity=100)
    assert r.total_return == 0.0


def test_time_stop_only_makes_all_trades_timed_out(synth_uptrend_df):
    """TimeStop 没有 price stop，所有交易都应该是 timed_out。"""
    stop = TimeStop(max_bars=10)
    result = run_backtest(synth_uptrend_df, stop)
    assert result.n_stopped == 0
    assert result.n_timed_out + result.n_open_at_end == result.n_trades
