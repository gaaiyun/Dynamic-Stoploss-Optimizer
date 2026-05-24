"""时间停损：持仓超过 N 根 bar 强制平仓。

学术上叫"vertical barrier"（Lopez de Prado 2018 triple-barrier method 的第三个 barrier）。
"""
from __future__ import annotations

from typing import Optional

from .base import BaseStop, Bar


class TimeStop(BaseStop):
    """N 根 bar 后强制平仓。"""

    DEFAULT_PARAMS = {
        "max_bars": 20,
    }

    NEEDS_WARMUP = False

    @property
    def exit_reason(self) -> str:
        return "time"

    def _update_stop(self, bar: Bar) -> Optional[float]:
        # Time stop 不维护"价格止损"，只在 _check_triggered 里看 n_bars_held
        return None

    def _check_triggered(self, bar: Bar) -> bool:
        return self.state.n_bars_held >= self.params["max_bars"]
