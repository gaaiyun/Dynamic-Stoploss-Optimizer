"""把 N 个 stop 策略在同一份数据上跑一遍，按风险调整收益排序。

加入 Deflated Sharpe Ratio 校正多重检验 — 这是大部分回测库忽略的关键。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .metrics import (
    calmar_ratio, deflated_sharpe_ratio, max_drawdown,
    returns_from_equity, sharpe_ratio, sortino_ratio,
)
from .stops.base import BaseStop


@dataclass
class StrategyRow:
    stop_name: str
    total_return: float
    sharpe: float
    sortino: float
    calmar: float
    max_dd: float
    n_trades: int
    hit_rate: float

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class StrategyComparison:
    rows: List[StrategyRow] = field(default_factory=list)
    deflated_sharpe_pct: float = 0.0    # 最好策略的 DSR 概率
    best_strategy: Optional[str] = None
    n_strategies_tested: int = 0

    def to_dict(self) -> dict:
        return {
            "best_strategy": self.best_strategy,
            "deflated_sharpe_pct": float(self.deflated_sharpe_pct),
            "n_strategies_tested": int(self.n_strategies_tested),
            "rows": [r.to_dict() for r in self.rows],
        }

    def to_table(self) -> str:
        """渲染对比表（等宽字体）。"""
        headers = ["Strategy", "TotalRet", "Sharpe", "Sortino",
                   "Calmar", "MaxDD", "Trades", "HitRate"]
        rows = [headers]
        for r in self.rows:
            rows.append([
                r.stop_name,
                f"{r.total_return:+.2%}",
                f"{r.sharpe:+.2f}",
                f"{r.sortino:+.2f}",
                f"{r.calmar:+.2f}",
                f"{r.max_dd:+.2%}",
                str(r.n_trades),
                f"{r.hit_rate:.1%}",
            ])
        widths = [max(len(row[c]) for row in rows) for c in range(len(headers))]
        lines = []
        for i, row in enumerate(rows):
            line = "  ".join(c.ljust(widths[j]) for j, c in enumerate(row))
            lines.append(line)
            if i == 0:
                lines.append("-" * len(line))
        if self.best_strategy:
            lines.append("")
            lines.append(
                f"Best by Sharpe: {self.best_strategy}  "
                f"(Deflated Sharpe probability: {self.deflated_sharpe_pct:.1%})"
            )
        return "\n".join(lines)


def compare_stops(
    df: pd.DataFrame,
    stops: Dict[str, BaseStop],
    *,
    side: str = "long",
    entry_signal=None,
    profit_target_pct: Optional[float] = None,
    max_holding_bars: Optional[int] = None,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0,
    initial_capital: float = 100_000.0,
    position_fraction: float = 1.0,
    periods_per_year: int = 252,
) -> StrategyComparison:
    """对一组 stops 同数据同 entry signal 跑回测，按 Sharpe 排序。

    Parameters
    ----------
    stops : {名字: BaseStop 实例}
    """
    if not stops:
        return StrategyComparison()

    rows: List[StrategyRow] = []
    best_sharpe = -np.inf
    best_name = None
    best_returns: List[float] = []

    for name, stop in stops.items():
        result = run_backtest(
            df, stop, side=side, entry_signal=entry_signal,
            profit_target_pct=profit_target_pct,
            max_holding_bars=max_holding_bars,
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
            position_fraction=position_fraction,
        )
        rets = returns_from_equity(result.equity_curve)
        sr = sharpe_ratio(rets, periods_per_year=periods_per_year)
        so = sortino_ratio(rets, periods_per_year=periods_per_year)
        cl = calmar_ratio(rets, result.equity_curve,
                          periods_per_year=periods_per_year)
        mdd = max_drawdown(result.equity_curve)

        rows.append(StrategyRow(
            stop_name=name,
            total_return=result.total_return,
            sharpe=sr, sortino=so, calmar=cl, max_dd=mdd,
            n_trades=result.n_trades, hit_rate=result.hit_rate,
        ))
        if sr > best_sharpe:
            best_sharpe = sr
            best_name = name
            best_returns = rets

    rows.sort(key=lambda r: r.sharpe, reverse=True)

    # Deflated Sharpe Ratio：用最佳策略
    dsr_prob = 0.0
    if best_returns and len(stops) >= 2:
        try:
            arr = np.asarray(best_returns)
            skew = float(((arr - arr.mean()) ** 3).mean() /
                         max(arr.std() ** 3, 1e-12))
            kurt = float(((arr - arr.mean()) ** 4).mean() /
                         max(arr.std() ** 4, 1e-12))
            dsr_prob = deflated_sharpe_ratio(
                sharpe_observed=best_sharpe,
                n_trials=len(stops),
                n_obs=len(best_returns),
                skewness=skew,
                kurtosis=kurt,
                periods_per_year=periods_per_year,
            )
        except ImportError:
            dsr_prob = 0.0

    return StrategyComparison(
        rows=rows,
        best_strategy=best_name,
        deflated_sharpe_pct=dsr_prob,
        n_strategies_tested=len(stops),
    )
