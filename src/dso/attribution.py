"""P/L 归因：区分入场 timing 贡献 vs 止损贡献。

一个常见误区：策略整体收益差就归咎"停损不好"，但实际可能入场点本来就烂。
这层把 P/L 分两块：

    P/L = (perfect_exit_pnl) + (stop_pnl_loss)

其中：
    perfect_exit_pnl = 假设每笔都按"持仓期间最高点"（long）平仓的 P/L
    stop_pnl_loss   = 实际 P/L - perfect_exit_pnl（负数，表示停损"失去的"利润）

如果 perfect_exit_pnl 本身很负 → 是入场问题；
如果 perfect_exit_pnl 强但 stop_pnl_loss 很负 → 是停损太紧 / 太晚。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from .backtest import BacktestResult, Trade


@dataclass
class AttributionReport:
    n_trades: int
    actual_total_pnl: float
    perfect_total_pnl: float           # 假设每笔都顶峰平仓
    stop_pnl_loss: float               # 停损相对完美的差
    entry_contribution_pct: float      # 入场贡献占比
    stop_contribution_pct: float       # 停损贡献占比
    avg_pct_of_perfect_captured: float  # 平均每笔捕获了完美收益的几成

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def attribute_pnl(result: BacktestResult, df: pd.DataFrame) -> AttributionReport:
    """从 backtest 结果 + 原始 OHLC 推 perfect-exit 反事实。

    "Perfect exit" 定义：long 仓在 [entry_bar, exit_bar] 区间内的最高 high；
    short 仓在该区间的最低 low。
    """
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if not result.trades:
        return AttributionReport(
            n_trades=0, actual_total_pnl=0.0, perfect_total_pnl=0.0,
            stop_pnl_loss=0.0, entry_contribution_pct=0.0,
            stop_contribution_pct=0.0, avg_pct_of_perfect_captured=0.0,
        )

    perfect_pnls: List[float] = []
    actual_pnls: List[float] = []
    capture_ratios: List[float] = []

    for t in result.trades:
        # 在 [entry_bar, exit_bar] 内找最优出场价
        segment = df.iloc[t.entry_bar:max(t.exit_bar + 1, t.entry_bar + 1)]
        if len(segment) == 0:
            continue
        if t.side == "long":
            best_exit = float(segment["high"].max())
            perfect = best_exit - t.entry_price
        else:
            best_exit = float(segment["low"].min())
            perfect = t.entry_price - best_exit

        perfect_pnls.append(perfect)
        actual_pnls.append(t.pnl)

        if abs(perfect) > 1e-12:
            capture_ratios.append(t.pnl / perfect)
        elif abs(t.pnl) < 1e-12:
            capture_ratios.append(1.0)

    total_perfect = sum(perfect_pnls)
    total_actual = sum(actual_pnls)
    loss = total_actual - total_perfect    # 通常 ≤ 0

    # 贡献占比
    abs_total = abs(total_perfect) + abs(loss)
    if abs_total < 1e-12:
        entry_pct = 0.0
        stop_pct = 0.0
    else:
        entry_pct = abs(total_perfect) / abs_total
        stop_pct = abs(loss) / abs_total

    return AttributionReport(
        n_trades=len(result.trades),
        actual_total_pnl=float(total_actual),
        perfect_total_pnl=float(total_perfect),
        stop_pnl_loss=float(loss),
        entry_contribution_pct=float(entry_pct),
        stop_contribution_pct=float(stop_pct),
        avg_pct_of_perfect_captured=(
            float(sum(capture_ratios) / len(capture_ratios))
            if capture_ratios else 0.0
        ),
    )
