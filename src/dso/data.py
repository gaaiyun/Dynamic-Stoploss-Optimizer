"""OHLCV 数据加载：CSV / yfinance / 合成。

不强制依赖 yfinance —— 用户用本地 CSV 或合成数据也能完整 demo。
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


REQUIRED_COLS = ["open", "high", "low", "close", "volume"]


def load_csv(path: str) -> pd.DataFrame:
    """从 CSV 读 OHLCV。"""
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    # 找第一列是日期就当 index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        # volume 是可选的
        critical = [c for c in missing if c != "volume"]
        if critical:
            raise ValueError(f"CSV 缺必要列：{critical}")
        df["volume"] = 0.0

    return df[REQUIRED_COLS].astype(float)


def load_yfinance(symbol: str, start: str, end: Optional[str] = None,
                  auto_adjust: bool = True) -> pd.DataFrame:
    """从 yfinance 抓 OHLCV。"""
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance 未装。pip install yfinance 后再用。"
        ) from e
    df = yf.download(symbol, start=start, end=end,
                     auto_adjust=auto_adjust, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"yfinance 没数据：{symbol} {start} → {end}")
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower()
                  for c in df.columns]
    for col in REQUIRED_COLS:
        if col not in df.columns:
            df[col] = 0.0
    return df[REQUIRED_COLS].astype(float)


def synthetic_ohlcv(
    n: int = 500,
    start_price: float = 100.0,
    daily_drift: float = 0.0005,
    daily_vol: float = 0.015,
    seed: Optional[int] = 42,
    regime: str = "mixed",     # "uptrend" / "downtrend" / "ranging" / "mixed"
) -> pd.DataFrame:
    """生成合成 OHLCV。可指定 regime 强制构造特定市场状态。"""
    rng = np.random.default_rng(seed)

    if regime == "uptrend":
        drift = max(daily_drift, 0.002)
        vol = daily_vol * 0.7
    elif regime == "downtrend":
        drift = -abs(daily_drift) - 0.002
        vol = daily_vol * 0.7
    elif regime == "ranging":
        drift = 0.0
        vol = daily_vol * 0.5
    elif regime == "mixed":
        # 上涨 → 横盘 → 下跌 三段
        third = n // 3
        seg1 = rng.normal(0.003, daily_vol * 0.7, third)
        seg2 = rng.normal(0.0, daily_vol * 0.4, third)
        seg3 = rng.normal(-0.002, daily_vol * 0.9, n - 2 * third)
        log_rets = np.concatenate([seg1, seg2, seg3])
    else:
        raise ValueError(f"未知 regime {regime}")

    if regime != "mixed":
        log_rets = rng.normal(drift, vol, n)

    closes = start_price * np.exp(log_rets.cumsum())
    intraday = rng.uniform(-0.003, 0.003, (n, 3))
    opens = closes * (1 + intraday[:, 0])
    highs = np.maximum(opens, closes) * (1 + np.abs(intraday[:, 1]))
    lows = np.minimum(opens, closes) * (1 - np.abs(intraday[:, 2]))
    vols = rng.uniform(1e6, 5e6, n)

    dates = pd.date_range(end=datetime.today().date(), periods=n, freq="B")
    df = pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": vols,
    }, index=dates)
    df.index.name = "date"
    return df
