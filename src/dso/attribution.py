"""退出实现诊断：实际 P/L 与最大有利变动（MFE）之间的差距。

这不是因果意义上的“入场贡献 vs 止损贡献”。它只回答：在同一入场和同一
持仓区间内，实际退出捕获了多少事后可见的最大有利变动。

    actual_pnl = mfe_net_pnl + realization_gap

MFE 使用入场后的价格区间，排除入场 K 线在收盘前已经发生的 high/low。
它是事后上界，不证明止损导致了差距，也不能单独评价入场质量。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .backtest import BacktestResult


@dataclass
class AttributionReport:
    n_trades: int
    actual_total_pnl: float
    mfe_total_pnl: float
    realization_gap_pnl: float
    aggregate_capture_ratio: float
    avg_trade_capture_ratio: float

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def attribute_pnl(result: BacktestResult, df: pd.DataFrame) -> AttributionReport:
    """计算同一持仓区间内的 MFE 净收益与实际实现差距。

    long 使用入场后到退出时的最高 high，short 使用最低 low。MFE 扣除与
    实际交易相同的佣金，但仍是事后最优价上界，不是可交易策略。
    """
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if not result.trades:
        return AttributionReport(
            n_trades=0, actual_total_pnl=0.0, mfe_total_pnl=0.0,
            realization_gap_pnl=0.0, aggregate_capture_ratio=0.0,
            avg_trade_capture_ratio=0.0,
        )

    mfe_pnls: List[float] = []
    actual_pnls: List[float] = []
    capture_ratios: List[float] = []

    for trade in result.trades:
        start = trade.entry_bar + 1
        end = max(trade.exit_bar + 1, start + 1)
        segment = df.iloc[start:end]
        if segment.empty:
            # 入场即在样本末尾平仓时，没有入场后的价格区间。
            mfe = trade.pnl
        elif trade.side == "long":
            best_exit = float(segment["high"].max())
            mfe = ((best_exit - trade.entry_price) * trade.quantity -
                   trade.commission)
        else:
            best_exit = float(segment["low"].min())
            mfe = ((trade.entry_price - best_exit) * trade.quantity -
                   trade.commission)

        mfe_pnls.append(mfe)
        actual_pnls.append(trade.pnl)
        if mfe > 1e-12:
            capture_ratios.append(trade.pnl / mfe)
        elif abs(trade.pnl) < 1e-12:
            capture_ratios.append(1.0)

    total_mfe = sum(mfe_pnls)
    total_actual = sum(actual_pnls)
    gap = total_actual - total_mfe
    aggregate_capture = (total_actual / total_mfe
                         if abs(total_mfe) > 1e-12 else 0.0)

    return AttributionReport(
        n_trades=len(result.trades),
        actual_total_pnl=float(total_actual),
        mfe_total_pnl=float(total_mfe),
        realization_gap_pnl=float(gap),
        aggregate_capture_ratio=float(aggregate_capture),
        avg_trade_capture_ratio=(
            float(sum(capture_ratios) / len(capture_ratios))
            if capture_ratios else 0.0
        ),
    )
