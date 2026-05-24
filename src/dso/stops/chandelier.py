"""Chandelier Exit（Charles Le Beau 1992）。

把止损"挂"在 N 周期最高点下方 multiplier × ATR 处。long 仓只向上调整。

公式（long）：

    Stop = max(H_n) - multiplier × ATR(period)

其中 max(H_n) 是过去 N 根 K 线的最高高点。

参考：Le Beau & Lucas (1992), *Computer Analysis of the Futures Markets*.
"""
from __future__ import annotations

from typing import Optional

from .atr import _wilder_atr
from .base import BaseStop, Bar


class ChandelierStop(BaseStop):
    """Chandelier Exit（Le Beau 1992）。"""

    DEFAULT_PARAMS = {
        "period": 22,         # 经典 22 个 bar（约 1 月）
        "multiplier": 3.0,    # Le Beau 原书推荐 3
    }

    def _update_stop(self, bar: Bar) -> Optional[float]:
        period = self.params["period"]
        multiplier = self.params["multiplier"]
        if len(self._history) < period:
            return None
        atr = _wilder_atr(self._history, period)
        if atr is None:
            return None

        window = self._history[-period:]
        if self.state.side == "long":
            highest_high = max(b.high for b in window)
            return highest_high - multiplier * atr
        else:
            lowest_low = min(b.low for b in window)
            return lowest_low + multiplier * atr
