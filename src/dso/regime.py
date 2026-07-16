"""市场状态识别（ADX + 波动率 + Hurst）。

不同 regime 可优先比较不同停损候选。映射是解释性启发式，不是经样本外
验证的“最优策略”结论：

| Regime | 推荐停损 | 理由 |
|---|---|---|
| 强趋势上行 (ADX > 25, +DI > -DI) | Chandelier / SuperTrend / PSAR | trailing 让趋势走 |
| 强趋势下行 (ADX > 25, +DI < -DI) | 同上（short 仓） | 同上 |
| 横盘震荡 (ADX < 20) | Donchian / MA | 窄止损快速止损 |
| 高波动 / 低 Hurst (< 0.5) | Volatility / ATR | 适配 vol 变化 |

参考：
- Wilder (1978) Chapter 13 — ADX / +DI / -DI 定义
- Hurst (1951) — Hurst 指数（趋势 vs 反转判断），现代 R/S 估计法
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd


RegimeLabel = Literal[
    "strong_uptrend", "strong_downtrend",
    "weak_uptrend", "weak_downtrend",
    "ranging", "high_volatility",
]


@dataclass
class RegimeReport:
    label: RegimeLabel
    adx: float
    plus_di: float
    minus_di: float
    hurst_exponent: Optional[float]
    realized_vol_annualized: float
    recommended_stops: List[str]
    reasoning: List[str]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "adx": float(self.adx),
            "plus_di": float(self.plus_di),
            "minus_di": float(self.minus_di),
            "hurst_exponent": (float(self.hurst_exponent)
                               if self.hurst_exponent is not None else None),
            "realized_vol_annualized": float(self.realized_vol_annualized),
            "recommended_stops": list(self.recommended_stops),
            "mapping_basis": "heuristic-candidates-not-validated",
            "reasoning": list(self.reasoning),
        }


def _calc_adx_dmi(df: pd.DataFrame, period: int = 14) -> tuple:
    """ADX + +DI + -DI（Wilder 平滑）。返回 (adx, +di, -di)。"""
    if len(df) < period * 2:
        return 0.0, 0.0, 0.0
    highs = df["high"].astype(float).values
    lows = df["low"].astype(float).values
    closes = df["close"].astype(float).values

    tr = np.zeros(len(df))
    plus_dm = np.zeros(len(df))
    minus_dm = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0

    # Wilder 平滑
    def wilder_smooth(arr: np.ndarray) -> np.ndarray:
        out = np.zeros(len(arr))
        out[period] = arr[1: period + 1].sum()
        for i in range(period + 1, len(arr)):
            out[i] = out[i - 1] - (out[i - 1] / period) + arr[i]
        return out

    smoothed_tr = wilder_smooth(tr)
    smoothed_plus_dm = wilder_smooth(plus_dm)
    smoothed_minus_dm = wilder_smooth(minus_dm)

    # 防止除零
    safe_tr = np.where(smoothed_tr > 0, smoothed_tr, 1e-12)
    plus_di = 100 * smoothed_plus_dm / safe_tr
    minus_di = 100 * smoothed_minus_dm / safe_tr

    dx_denom = plus_di + minus_di
    safe_dx_denom = np.where(dx_denom > 1e-12, dx_denom, 1)
    dx = 100 * np.abs(plus_di - minus_di) / safe_dx_denom

    # ADX = DX 的 Wilder 平滑
    adx = np.zeros(len(df))
    adx[2 * period] = dx[period + 1: 2 * period + 1].mean()
    for i in range(2 * period + 1, len(df)):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    return float(adx[-1]), float(plus_di[-1]), float(minus_di[-1])


def _hurst_exponent(prices: pd.Series, max_lag: int = 20) -> Optional[float]:
    """简化 Hurst 指数（R/S 分析的对数斜率近似）。

    H ≈ 0.5 → 随机游走
    H > 0.5 → 趋势性
    H < 0.5 → 反转性 / 均值回归
    """
    closes = prices.dropna().astype(float).values
    if len(closes) < max_lag * 4:
        return None
    log_prices = np.log(closes)
    lags = list(range(2, max_lag + 1))
    rs_values = []
    for lag in lags:
        # variance of lag differences
        diffs = log_prices[lag:] - log_prices[:-lag]
        if len(diffs) < 2:
            continue
        std = diffs.std()
        if std <= 1e-12:
            continue
        rs_values.append((math.log(lag), math.log(std)))
    if len(rs_values) < 4:
        return None
    xs = np.array([x for x, _ in rs_values])
    ys = np.array([y for _, y in rs_values])
    slope = np.polyfit(xs, ys, 1)[0]
    # H = slope of log(R/S) ~ log(lag)；近似下 H = slope of log(std) ~ log(lag)
    return float(slope)


def _realized_vol(prices: pd.Series, window: int = 20,
                  periods_per_year: int = 252) -> float:
    closes = prices.dropna().astype(float)
    if len(closes) < window + 1:
        return 0.0
    log_rets = np.log(closes / closes.shift(1)).dropna()
    if len(log_rets) < window:
        return 0.0
    return float(log_rets.tail(window).std() * math.sqrt(periods_per_year))


def detect_regime(df: pd.DataFrame, adx_period: int = 14,
                  hurst_max_lag: int = 20,
                  vol_window: int = 20) -> RegimeReport:
    """从 OHLC DataFrame 推断市场状态并给出待比较的 stop 候选列表。

    Parameters
    ----------
    df : 必须含 open/high/low/close 列
    """
    required = {"high", "low", "close"}
    cols_lower = {c.lower() for c in df.columns}
    if not required.issubset(cols_lower):
        raise ValueError(f"缺必要列：{required - cols_lower}")
    df = df.rename(columns={c: c.lower() for c in df.columns})

    adx, plus_di, minus_di = _calc_adx_dmi(df, period=adx_period)
    hurst = _hurst_exponent(df["close"], max_lag=hurst_max_lag)
    vol = _realized_vol(df["close"], window=vol_window)

    reasoning: List[str] = []
    # ADX 主导规则
    if adx >= 25:
        if plus_di > minus_di:
            label: RegimeLabel = "strong_uptrend"
            reasoning.append(f"ADX={adx:.1f} ≥ 25 + +DI={plus_di:.1f} > -DI={minus_di:.1f}")
            recommended = ["chandelier", "supertrend", "parabolic_sar", "atr"]
        else:
            label = "strong_downtrend"
            reasoning.append(f"ADX={adx:.1f} ≥ 25 + -DI={minus_di:.1f} > +DI={plus_di:.1f}")
            recommended = ["chandelier", "supertrend", "parabolic_sar", "atr"]
    elif adx >= 20:
        if plus_di > minus_di:
            label = "weak_uptrend"
            reasoning.append(f"ADX={adx:.1f} ∈ [20, 25) + +DI 略大")
            recommended = ["moving_average", "atr", "supertrend"]
        else:
            label = "weak_downtrend"
            reasoning.append(f"ADX={adx:.1f} ∈ [20, 25) + -DI 略大")
            recommended = ["moving_average", "atr", "supertrend"]
    else:
        label = "ranging"
        reasoning.append(f"ADX={adx:.1f} < 20，无趋势")
        recommended = ["donchian", "max_drawdown", "moving_average"]

    # 高波动覆盖
    if vol > 0.40:    # 40% 年化是高波动
        reasoning.append(f"年化波动率 {vol:.1%} > 40%，归类高波动")
        label = "high_volatility"
        recommended = ["volatility", "atr", "max_drawdown"]

    # Hurst 加注
    if hurst is not None:
        if hurst > 0.55:
            reasoning.append(f"Hurst≈{hurst:.2f} > 0.55，趋势性较强")
        elif hurst < 0.45:
            reasoning.append(f"Hurst≈{hurst:.2f} < 0.45，反转 / 均值回归倾向")

    return RegimeReport(
        label=label, adx=adx, plus_di=plus_di, minus_di=minus_di,
        hurst_exponent=hurst, realized_vol_annualized=vol,
        recommended_stops=recommended, reasoning=reasoning,
    )
