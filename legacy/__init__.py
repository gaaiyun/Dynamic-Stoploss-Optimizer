"""
Dynamic Stoploss Optimizer - 动态止损策略优化器

派蒙的最强止损策略工具包！包含多种动态止损算法和回测功能。
"""

try:
    from .stoploss_strategies import (
        VolatilityStoploss,
        ATRStoploss,
        MAStoploss,
        MaxDrawdownStoploss,
        BaseStoploss
    )
    from .backtester import StoplossBacktester, BacktestResult
    from .optimizer import StoplossOptimizer
except ImportError:
    from stoploss_strategies import (
        VolatilityStoploss,
        ATRStoploss,
        MAStoploss,
        MaxDrawdownStoploss,
        BaseStoploss
    )
    from backtester import StoplossBacktester, BacktestResult
    from optimizer import StoplossOptimizer

__version__ = '1.0.0'
__author__ = '派蒙 ⭐'

__all__ = [
    'BaseStoploss',
    'VolatilityStoploss',
    'ATRStoploss',
    'MAStoploss',
    'MaxDrawdownStoploss',
    'StoplossBacktester',
    'BacktestResult',
    'StoplossOptimizer'
]
