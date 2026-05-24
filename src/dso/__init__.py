"""dso — Dynamic Stoploss Optimizer v2.

按学界 + 实务界已成熟的 8 种停损策略（Wilder ATR / Le Beau Chandelier /
SuperTrend / Wilder Parabolic SAR / Turtle Donchian / MA / Volatility /
Time-based），加 walk-forward 优化 + ADX 市场状态识别 + Lopez de Prado
triple-barrier 回测 + Deflated Sharpe Ratio 多重检验校正。

公共 API：

    from dso import (
        STOPS, run_backtest, compare_stops,
        walk_forward, detect_regime, load_ohlcv,
    )

详见 README.md / docs/RESEARCH_NOTES.md。
"""
from __future__ import annotations

__version__ = "2.0.0"

from dso.stops import STOPS, BaseStop, StopState  # noqa: E402
from dso.backtest import BacktestResult, run_backtest  # noqa: E402
from dso.compare import StrategyComparison, compare_stops  # noqa: E402
from dso.walk_forward import WalkForwardResult, walk_forward  # noqa: E402
from dso.regime import RegimeReport, detect_regime  # noqa: E402
from dso.attribution import attribute_pnl  # noqa: E402
from dso.metrics import (  # noqa: E402
    deflated_sharpe_ratio, max_drawdown, sharpe_ratio, sortino_ratio,
)
from dso.data import load_csv, load_yfinance, synthetic_ohlcv  # noqa: E402


__all__ = [
    "__version__",
    "STOPS", "BaseStop", "StopState",
    "BacktestResult", "run_backtest",
    "StrategyComparison", "compare_stops",
    "WalkForwardResult", "walk_forward",
    "RegimeReport", "detect_regime",
    "attribute_pnl",
    "deflated_sharpe_ratio", "max_drawdown",
    "sharpe_ratio", "sortino_ratio",
    "load_csv", "load_yfinance", "synthetic_ohlcv",
]
