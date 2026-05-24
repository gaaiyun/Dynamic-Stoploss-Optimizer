"""Shared fixtures for dso tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# 让 tests/ 能 import dso
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@pytest.fixture
def synth_uptrend_df() -> pd.DataFrame:
    """200-bar 单调上涨合成 OHLCV，方便测试 long trailing 行为。"""
    import numpy as np
    n = 200
    rng = np.random.default_rng(0)
    log_rets = rng.normal(0.001, 0.01, n)
    closes = 100.0 * np.exp(log_rets.cumsum())
    intra = rng.uniform(-0.002, 0.002, (n, 3))
    opens = closes * (1 + intra[:, 0])
    highs = np.maximum(opens, closes) * (1 + abs(intra[:, 1]))
    lows = np.minimum(opens, closes) * (1 - abs(intra[:, 2]))
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": [1e6] * n,
    })


@pytest.fixture
def synth_downtrend_df() -> pd.DataFrame:
    import numpy as np
    n = 200
    rng = np.random.default_rng(1)
    log_rets = rng.normal(-0.001, 0.01, n)
    closes = 100.0 * np.exp(log_rets.cumsum())
    intra = rng.uniform(-0.002, 0.002, (n, 3))
    opens = closes * (1 + intra[:, 0])
    highs = np.maximum(opens, closes) * (1 + abs(intra[:, 1]))
    lows = np.minimum(opens, closes) * (1 - abs(intra[:, 2]))
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": [1e6] * n,
    })


@pytest.fixture
def synth_mixed_df() -> pd.DataFrame:
    """混合 regime（先涨后跌）—— 用于 walk-forward / compare 测试。"""
    import numpy as np
    rng = np.random.default_rng(2)
    seg1 = rng.normal(0.003, 0.01, 100)   # 涨
    seg2 = rng.normal(0.0, 0.005, 100)    # 横
    seg3 = rng.normal(-0.002, 0.012, 100) # 跌
    log_rets = np.concatenate([seg1, seg2, seg3])
    closes = 100.0 * np.exp(log_rets.cumsum())
    intra = rng.uniform(-0.002, 0.002, (300, 3))
    opens = closes * (1 + intra[:, 0])
    highs = np.maximum(opens, closes) * (1 + abs(intra[:, 1]))
    lows = np.minimum(opens, closes) * (1 - abs(intra[:, 2]))
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": [1e6] * 300,
    })
