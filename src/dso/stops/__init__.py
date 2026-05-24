"""8 个内置停损策略 + 注册表。

策略名 → 类的映射，方便 CLI / compare / walk_forward 按字符串名引用。
"""
from __future__ import annotations

from .atr import ATRStop
from .base import BaseStop, Bar, StopState, warmup
from .chandelier import ChandelierStop
from .donchian import DonchianStop
from .max_drawdown import MaxDrawdownStop
from .moving_average import MovingAverageStop
from .parabolic_sar import ParabolicSARStop
from .supertrend import SuperTrendStop
from .time_stop import TimeStop
from .volatility import VolatilityStop


STOPS: dict[str, type[BaseStop]] = {
    "atr": ATRStop,
    "chandelier": ChandelierStop,
    "supertrend": SuperTrendStop,
    "parabolic_sar": ParabolicSARStop,
    "donchian": DonchianStop,
    "moving_average": MovingAverageStop,
    "volatility": VolatilityStop,
    "max_drawdown": MaxDrawdownStop,
    "time": TimeStop,
}


__all__ = [
    "STOPS", "BaseStop", "Bar", "StopState", "warmup",
    "ATRStop", "ChandelierStop", "SuperTrendStop", "ParabolicSARStop",
    "DonchianStop", "MovingAverageStop", "VolatilityStop",
    "MaxDrawdownStop", "TimeStop",
]
