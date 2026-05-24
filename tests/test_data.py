"""data.py 测试 —— CSV / yfinance / 合成。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dso.data import (
    REQUIRED_COLS, load_csv, load_yfinance, synthetic_ohlcv,
)


# --- synthetic_ohlcv ----------------------------------------------

def test_synthetic_default_shape():
    df = synthetic_ohlcv()
    assert len(df) == 500
    assert set(REQUIRED_COLS).issubset(df.columns)


def test_synthetic_deterministic_with_seed():
    a = synthetic_ohlcv(n=100, seed=42)
    b = synthetic_ohlcv(n=100, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_synthetic_different_seeds_differ():
    a = synthetic_ohlcv(seed=1)
    b = synthetic_ohlcv(seed=2)
    assert not a["close"].equals(b["close"])


def test_synthetic_ohlc_invariants():
    """对每一行：high >= max(open, close)，low <= min(open, close)。"""
    df = synthetic_ohlcv(n=200, seed=0)
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()


def test_synthetic_positive_prices():
    df = synthetic_ohlcv(n=300, daily_vol=0.05, seed=0)
    assert (df["close"] > 0).all()


def test_synthetic_uptrend_has_positive_drift():
    df = synthetic_ohlcv(n=200, regime="uptrend", seed=0)
    # 上涨：最后收盘 > 起始收盘
    assert df["close"].iloc[-1] > df["close"].iloc[0]


def test_synthetic_downtrend_has_negative_drift():
    df = synthetic_ohlcv(n=200, regime="downtrend", seed=0)
    assert df["close"].iloc[-1] < df["close"].iloc[0]


def test_synthetic_ranging_has_low_vol():
    df = synthetic_ohlcv(n=200, regime="ranging", seed=0)
    # ranging：daily std 应该比 mixed / uptrend 低
    ranging_std = df["close"].pct_change().std()
    mixed_std = synthetic_ohlcv(n=200, regime="mixed",
                                 seed=0)["close"].pct_change().std()
    assert ranging_std < mixed_std


def test_synthetic_invalid_regime_raises():
    with pytest.raises(ValueError, match="未知 regime"):
        synthetic_ohlcv(regime="bogus")


# --- load_csv ----------------------------------------------------

def test_load_csv_roundtrip(tmp_path):
    src = synthetic_ohlcv(n=100, seed=0)
    p = tmp_path / "data.csv"
    src.to_csv(p)
    loaded = load_csv(str(p))
    assert len(loaded) == 100
    assert set(REQUIRED_COLS).issubset(loaded.columns)
    np.testing.assert_allclose(loaded["close"].values,
                                src["close"].values, rtol=1e-9)


def test_load_csv_missing_critical_columns_raises(tmp_path):
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5),
                       "price": [100] * 5})
    p = tmp_path / "bad.csv"
    df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="缺必要列"):
        load_csv(str(p))


def test_load_csv_missing_volume_filled_with_zero(tmp_path):
    df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "open": [100] * 5, "high": [101] * 5,
        "low": [99] * 5, "close": [100] * 5,
    })
    p = tmp_path / "no_vol.csv"
    df.to_csv(p, index=False)
    loaded = load_csv(str(p))
    assert "volume" in loaded.columns
    assert (loaded["volume"] == 0.0).all()


# --- load_yfinance -----------------------------------------------

def test_load_yfinance_missing_module_raises(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **kw):
        if name == "yfinance":
            raise ImportError("simulated")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(ImportError, match="yfinance"):
        load_yfinance("AAPL", start="2024-01-01")
