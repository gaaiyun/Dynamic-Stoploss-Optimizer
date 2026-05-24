"""SuperTrend 停损。

经典公式（基于 ATR 的趋势带）：

    HL2 = (high + low) / 2
    Upper = HL2 + multiplier × ATR
    Lower = HL2 - multiplier × ATR

    Final Upper / Lower 随 trail，long 用 Lower 作为止损。

参考：Olivier Seban — SuperTrend Indicator（社区起源），现广泛被
TradingView / freqtrade / vectorbt 等实现。
"""
from __future__ import annotations

from typing import Optional

from .atr import _wilder_atr
from .base import BaseStop, Bar


class SuperTrendStop(BaseStop):
    """SuperTrend 停损。"""

    DEFAULT_PARAMS = {
        "period": 10,         # 主流默认
        "multiplier": 3.0,
    }

    def _init_state(self) -> None:
        # 维护 final_lower / final_upper 用于 trail
        self.state.metadata["final_lower"] = None
        self.state.metadata["final_upper"] = None

    def _update_stop(self, bar: Bar) -> Optional[float]:
        period = self.params["period"]
        multiplier = self.params["multiplier"]
        atr = _wilder_atr(self._history, period)
        if atr is None:
            return None

        hl2 = (bar.high + bar.low) / 2.0
        basic_upper = hl2 + multiplier * atr
        basic_lower = hl2 - multiplier * atr

        # Final upper/lower：经典 SuperTrend trail 规则
        prev_lower = self.state.metadata.get("final_lower")
        prev_upper = self.state.metadata.get("final_upper")
        prev_close = self._history[-2].close if len(self._history) >= 2 else bar.close

        if prev_lower is None:
            final_lower = basic_lower
        else:
            final_lower = (basic_lower if basic_lower > prev_lower
                                          or prev_close < prev_lower
                           else prev_lower)
        if prev_upper is None:
            final_upper = basic_upper
        else:
            final_upper = (basic_upper if basic_upper < prev_upper
                                          or prev_close > prev_upper
                           else prev_upper)

        self.state.metadata["final_lower"] = final_lower
        self.state.metadata["final_upper"] = final_upper

        if self.state.side == "long":
            return final_lower
        return final_upper
