"""Donchian Channel 停损（Turtle Traders, Dennis & Eckhardt 1980s）。

Turtle 规则的核心退出：long 在 N 周期最低点离场，short 在 N 周期最高点。
经典 N = 10（短线退出）或 N = 20（与入场对称）。

参考：Curtis Faith (2007), *Way of the Turtle* —— 原汁原味的海龟规则。
"""
from __future__ import annotations

from typing import Optional

from .base import BaseStop, Bar


class DonchianStop(BaseStop):
    """Donchian Channel 退出（Turtle Traders）。"""

    DEFAULT_PARAMS = {
        "period": 10,    # 短线退出，长线 20
    }

    def _update_stop(self, bar: Bar) -> Optional[float]:
        period = self.params["period"]
        if len(self._history) < period:
            return None
        window = self._history[-period:]
        if self.state.side == "long":
            return min(b.low for b in window)
        return max(b.high for b in window)
