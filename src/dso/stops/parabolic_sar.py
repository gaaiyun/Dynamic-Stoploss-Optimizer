"""Parabolic SAR（Welles Wilder Jr. 1978）。

Wilder 的经典 SAR：long 仓时止损每根 bar 加速向价格靠拢。

    SAR_t = SAR_{t-1} + AF × (EP - SAR_{t-1})

其中：
- EP（Extreme Point）：开仓以来的最高点（long）或最低点（short）
- AF（Acceleration Factor）：每次 EP 更新时 +increment，最大 0.20

参考：Wilder (1978), *New Concepts in Technical Trading Systems*, Chapter 5.
"""
from __future__ import annotations

from typing import Optional

from .base import BaseStop, Bar


class ParabolicSARStop(BaseStop):
    """Parabolic SAR 停损（Wilder 1978）。"""

    DEFAULT_PARAMS = {
        "af_init": 0.02,         # 起始 AF
        "af_increment": 0.02,    # 每次 EP 更新时 +
        "af_max": 0.20,          # AF 上限
    }

    def _init_state(self) -> None:
        # 初始 SAR：long 用 entry_price 略下方；short 用上方
        # （Wilder 规则更复杂，但开仓后跟 entry 是标准简化）
        self.state.metadata["ep"] = self.state.entry_price
        self.state.metadata["af"] = self.params["af_init"]

    def _update_stop(self, bar: Bar) -> Optional[float]:
        ep = self.state.metadata["ep"]
        af = self.state.metadata["af"]
        af_inc = self.params["af_increment"]
        af_max = self.params["af_max"]

        prev_stop = self.state.current_stop
        if prev_stop is None:
            # 首次：SAR 起点 = entry_price ± 一个小距离（用 0.5% 启动）
            init_dist = self.state.entry_price * 0.005
            if self.state.side == "long":
                prev_stop = self.state.entry_price - init_dist
            else:
                prev_stop = self.state.entry_price + init_dist

        # 更新 EP（极值点）
        if self.state.side == "long":
            if bar.high > ep:
                ep = bar.high
                af = min(af + af_inc, af_max)
        else:
            if bar.low < ep:
                ep = bar.low
                af = min(af + af_inc, af_max)

        # 新 SAR
        new_sar = prev_stop + af * (ep - prev_stop)

        # Wilder 规则：SAR 不能进入今日 / 昨日的 high-low 区间（long）
        if self.state.side == "long" and len(self._history) >= 2:
            new_sar = min(new_sar, self._history[-1].low, self._history[-2].low)
        elif self.state.side == "short" and len(self._history) >= 2:
            new_sar = max(new_sar, self._history[-1].high, self._history[-2].high)

        self.state.metadata["ep"] = ep
        self.state.metadata["af"] = af
        return new_sar
