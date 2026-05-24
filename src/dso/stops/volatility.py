"""波动率停损。

止损 = 入场价 ∓ multiplier × 年化波动率 × 入场价（long 时减、short 时加）。
这是 v1 原版 ``VolatilityStoploss`` 的清洗版本：state-machine 化 + 修正若干
edge case。
"""
from __future__ import annotations

import math
from typing import Optional

from .base import BaseStop, Bar


def _annualized_vol(closes: list[float], window: int,
                    use_log_returns: bool = True,
                    periods_per_year: int = 252) -> Optional[float]:
    if len(closes) < window + 1:
        return None
    if use_log_returns:
        rets = [math.log(closes[i] / closes[i - 1])
                for i in range(1, len(closes))
                if closes[i - 1] > 0]
    else:
        rets = [closes[i] / closes[i - 1] - 1
                for i in range(1, len(closes))
                if closes[i - 1] > 0]
    if len(rets) < window:
        return None
    recent = rets[-window:]
    mean = sum(recent) / len(recent)
    var = sum((r - mean) ** 2 for r in recent) / (len(recent) - 1)
    return math.sqrt(var) * math.sqrt(periods_per_year)


class VolatilityStop(BaseStop):
    """波动率距离 trailing stop。"""

    DEFAULT_PARAMS = {
        "window": 20,
        "multiplier": 2.0,
        "use_log_returns": True,
        "periods_per_year": 252,
    }

    def _update_stop(self, bar: Bar) -> Optional[float]:
        window = self.params["window"]
        multiplier = self.params["multiplier"]
        use_log = self.params["use_log_returns"]
        ppy = self.params["periods_per_year"]

        closes = [b.close for b in self._history]
        vol = _annualized_vol(closes, window, use_log_returns=use_log,
                              periods_per_year=ppy)
        if vol is None:
            return None

        # vol 已经是年化小数（如 0.30 = 30%）。止损距离 = 入场价 × vol × multiplier
        # 这里用入场价做基准更稳定（与 v1 一致）；想随当前价变可改 bar.close
        distance = self.state.entry_price * vol * multiplier
        if self.state.side == "long":
            return bar.close - distance
        return bar.close + distance
