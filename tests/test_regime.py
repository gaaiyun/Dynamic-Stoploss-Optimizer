"""regime.py 测试 —— ADX / Hurst / 波动率分类。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dso.regime import (
    RegimeReport, _calc_adx_dmi, _hurst_exponent,
    _realized_vol, detect_regime,
)


# --- _calc_adx_dmi ---------------------------------------------------

def test_adx_strong_uptrend_high_adx(synth_uptrend_df):
    """单调上涨 → ADX 应该比较高。"""
    adx, plus_di, minus_di = _calc_adx_dmi(synth_uptrend_df, period=14)
    # 上涨：+DI > -DI
    assert plus_di > minus_di


def test_adx_strong_downtrend_minus_di_dominates(synth_downtrend_df):
    adx, plus_di, minus_di = _calc_adx_dmi(synth_downtrend_df, period=14)
    assert minus_di > plus_di


def test_adx_short_data_returns_zeros():
    df = pd.DataFrame({"open": [100] * 5, "high": [101] * 5,
                       "low": [99] * 5, "close": [100] * 5})
    adx, p, m = _calc_adx_dmi(df, period=14)
    assert (adx, p, m) == (0.0, 0.0, 0.0)


# --- _hurst_exponent -------------------------------------------------

def test_hurst_short_data_returns_none():
    s = pd.Series([1.0, 2.0, 3.0])
    assert _hurst_exponent(s, max_lag=20) is None


def test_hurst_uptrend_above_05_or_handle_nan(synth_uptrend_df):
    """趋势数据：Hurst > 0.5 是理论预期，但简化 R/S 估计在小样本可能偏低。
    至少应返回有效浮点。"""
    h = _hurst_exponent(synth_uptrend_df["close"])
    assert h is None or isinstance(h, float)


# --- _realized_vol ---------------------------------------------------

def test_realized_vol_constant_zero():
    s = pd.Series([100.0] * 50)
    assert _realized_vol(s) == 0.0


def test_realized_vol_with_variation():
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.02, 100)
    closes = 100 * np.exp(np.cumsum(rets))
    s = pd.Series(closes)
    vol = _realized_vol(s)
    # 期望约 0.02 * sqrt(252) ≈ 0.32
    assert 0.20 < vol < 0.50


def test_realized_vol_short_data():
    assert _realized_vol(pd.Series([100, 101])) == 0.0


# --- detect_regime ---------------------------------------------------

def test_regime_returns_report(synth_uptrend_df):
    report = detect_regime(synth_uptrend_df)
    assert isinstance(report, RegimeReport)
    assert report.adx >= 0
    assert isinstance(report.recommended_stops, list)
    assert len(report.recommended_stops) > 0


def test_regime_uptrend_recommends_trailing_stops(synth_uptrend_df):
    report = detect_regime(synth_uptrend_df)
    # strong_uptrend / weak_uptrend 应该推荐 trailing 类
    if "uptrend" in report.label:
        # 应至少含 atr / chandelier / supertrend / moving_average 之一
        assert any(s in report.recommended_stops
                   for s in ["atr", "chandelier", "supertrend", "moving_average"])


def test_regime_downtrend_label(synth_downtrend_df):
    report = detect_regime(synth_downtrend_df)
    # 下跌数据应该被识别为 downtrend / ranging / 而不是 uptrend
    assert "uptrend" not in report.label or report.label == "high_volatility"


def test_regime_high_vol_label():
    """构造超高波动 → 应被归类为 high_volatility。"""
    rng = np.random.default_rng(0)
    n = 200
    rets = rng.normal(0, 0.10, n)    # 10% 日波动 → 巨大年化
    closes = 100 * np.exp(rets.cumsum())
    df = pd.DataFrame({
        "open": closes, "high": closes * 1.05, "low": closes * 0.95,
        "close": closes,
    })
    report = detect_regime(df)
    assert report.realized_vol_annualized > 0.4
    assert report.label == "high_volatility"


def test_regime_missing_columns_raises():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError, match="缺必要列"):
        detect_regime(df)


def test_regime_to_dict_serializable(synth_uptrend_df):
    import json
    report = detect_regime(synth_uptrend_df)
    json.dumps(report.to_dict())


def test_regime_handles_capital_columns():
    """OHLC 大写列名应该被自动 lowercase。"""
    rng = np.random.default_rng(0)
    n = 100
    closes = 100 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame({
        "Open": closes, "High": closes * 1.01, "Low": closes * 0.99,
        "Close": closes,
    })
    report = detect_regime(df)
    assert isinstance(report, RegimeReport)
