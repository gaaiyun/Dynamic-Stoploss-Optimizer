"""移动平均线停损。

止损 = MA(period) ± offset。常用 SMA 或 EMA。
"""
from __future__ import annotations

from typing import Optional

from .base import BaseStop, Bar


def _sma(values: list[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _ema(values: list[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    # 初始 EMA = 前 period 的 SMA
    ema_val = sum(values[:period]) / period
    for v in values[period:]:
        ema_val = v * k + ema_val * (1 - k)
    return ema_val


class MovingAverageStop(BaseStop):
    """MA 作为 trailing stop。"""

    DEFAULT_PARAMS = {
        "period": 20,
        "ma_type": "ema",      # "sma" / "ema"
        "offset_pct": 0.0,     # 止损偏离 MA 多少百分比（向不利方向）
    }

    def _update_stop(self, bar: Bar) -> Optional[float]:
        period = self.params["period"]
        ma_type = self.params["ma_type"].lower()
        offset_pct = self.params["offset_pct"]

        closes = [b.close for b in self._history]
        if ma_type == "sma":
            ma = _sma(closes, period)
        elif ma_type == "ema":
            ma = _ema(closes, period)
        else:
            raise ValueError(f"未知 ma_type {ma_type}（用 sma 或 ema）")

        if ma is None:
            return None

        # offset：long 止损在 MA 下方 offset_pct%；short 在上方
        if self.state.side == "long":
            return ma * (1 - offset_pct)
        return ma * (1 + offset_pct)
