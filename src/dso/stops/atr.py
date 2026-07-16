"""ATR 停损（Wilder 1978）。

止损价 = 当前价 ∓ multiplier × ATR(period)。long 仓向上 trail，short 向下。

参考：Welles Wilder Jr. (1978), *New Concepts in Technical Trading Systems*.
"""
from __future__ import annotations

from typing import Optional

from .base import BaseStop, Bar


def _wilder_atr(history: list[Bar], period: int) -> Optional[float]:
    """Wilder 平滑 ATR。需要 period+1 根 bar。"""
    n = len(history)
    if n < period + 1:
        return None
    trs = []
    for i in range(1, n):
        high, low = history[i].high, history[i].low
        prev_close = history[i - 1].close
        trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
    # 初始 ATR：前 period 个 TR 的简单平均
    atr = sum(trs[:period]) / period
    # 后续 Wilder 平滑：ATR_n = (ATR_{n-1} × (period-1) + TR_n) / period
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
    return atr


class ATRStop(BaseStop):
    """ATR × multiplier 距离的 trailing stop（Wilder 1978）。"""

    DEFAULT_PARAMS = {
        "period": 14,
        "multiplier": 2.5,
    }

    def _update_stop(self, bar: Bar) -> Optional[float]:
        period = self.params["period"]
        multiplier = self.params["multiplier"]
        atr = _wilder_atr(self._history, period)
        if atr is None:
            return None
        if self.state.side == "long":
            # long 用 bar.close（不用 high，避免日内 spike 拉太远）
            return bar.close - multiplier * atr
        return bar.close + multiplier * atr
