"""峰值回撤停损。

记录开仓以来的最高浮盈点（long）或最低浮盈点（short），跌穿 max_drawdown%
就触发。也叫 trailing stop（最朴素的 trailing 形式）。
"""
from __future__ import annotations

from typing import Optional

from .base import BaseStop, Bar


class MaxDrawdownStop(BaseStop):
    """从峰值回撤超过阈值就止损。"""

    DEFAULT_PARAMS = {
        "max_drawdown_pct": 0.05,      # 5%
        "use_entry_price": False,      # True = 相对于入场价的固定 stoploss
    }

    def _init_state(self) -> None:
        self.state.metadata["peak_price"] = self.state.entry_price

    def _update_stop(self, bar: Bar) -> Optional[float]:
        max_dd = self.params["max_drawdown_pct"]
        use_entry = self.params["use_entry_price"]
        peak = self.state.metadata["peak_price"]

        # 更新峰值
        if self.state.side == "long":
            peak = max(peak, bar.high)
        else:
            peak = min(peak, bar.low)
        self.state.metadata["peak_price"] = peak

        # 止损 = peak × (1 - max_dd) 或 entry × (1 - max_dd)
        base = self.state.entry_price if use_entry else peak
        if self.state.side == "long":
            return base * (1 - max_dd)
        return base * (1 + max_dd)
