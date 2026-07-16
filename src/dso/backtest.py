"""Triple-barrier 事件驱动回测（Lopez de Prado 2018）。

经典 Lopez de Prado 的三 barrier 方法：

    1. 止损 barrier（horizontal lower）—— 由 BaseStop 提供
    2. 止盈 barrier（horizontal upper）—— ``profit_target_pct`` 参数
    3. 时间 barrier（vertical）—— ``max_holding_bars`` 参数

任一 barrier 先到即平仓。引擎当前只维护一笔持仓；重叠持仓会显式拒绝，
避免一个 stop state 被多笔交易共享。

参考：Lopez de Prado (2018), *Advances in Financial Machine Learning*,
Chapter 3 "Labeling".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import pandas as pd

from .stops.base import BaseStop, Bar


@dataclass
class Trade:
    entry_bar: int
    entry_price: float
    exit_bar: int
    exit_price: float
    side: str
    exit_reason: str             # "stop" / "profit" / "time" / "end"
    n_bars: int
    pnl: float                   # 单笔净 PnL（已扣佣金与滑点）
    pnl_pct: float
    commission: float
    quantity: float = 1.0
    final_stop: Optional[float] = None    # 平仓时止损价

    def to_dict(self) -> dict:
        return {k: (float(v) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


@dataclass
class BacktestResult:
    stop_name: str
    stop_params: dict
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)    # 每根 bar 的盯市权益
    initial_capital: float = 100_000.0
    final_equity: float = 0.0
    n_signals: int = 0
    n_trades: int = 0
    n_stopped: int = 0           # 因止损平仓数
    n_profit_taken: int = 0      # 因止盈平仓
    n_timed_out: int = 0         # 因时间 barrier 平仓
    n_open_at_end: int = 0       # 回测结束时还未平仓的

    @property
    def total_return(self) -> float:
        if self.initial_capital <= 0:
            return 0.0
        return self.final_equity / self.initial_capital - 1

    @property
    def hit_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def avg_pnl_pct(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.pnl_pct for t in self.trades) / len(self.trades)

    def to_dict(self) -> dict:
        return {
            "stop_name": self.stop_name,
            "stop_params": self.stop_params,
            "initial_capital": float(self.initial_capital),
            "final_equity": float(self.final_equity),
            "total_return": float(self.total_return),
            "n_signals": int(self.n_signals),
            "n_trades": int(self.n_trades),
            "n_stopped": int(self.n_stopped),
            "n_profit_taken": int(self.n_profit_taken),
            "n_timed_out": int(self.n_timed_out),
            "n_open_at_end": int(self.n_open_at_end),
            "hit_rate": float(self.hit_rate),
            "avg_pnl_pct": float(self.avg_pnl_pct),
            "trades": [t.to_dict() for t in self.trades],
        }


def _to_bars(df: pd.DataFrame) -> List[Bar]:
    bars = []
    cols = {c.lower(): c for c in df.columns}
    for _, row in df.iterrows():
        bars.append(Bar(
            open=float(row[cols.get("open", "open")]),
            high=float(row[cols.get("high", "high")]),
            low=float(row[cols.get("low", "low")]),
            close=float(row[cols.get("close", "close")]),
            volume=float(row.get(cols.get("volume", "volume"), 0)),
        ))
    return bars


def _resolve_signals(df: pd.DataFrame,
                     entry_signal: Optional[Sequence[bool]],
                     n_bars: int) -> List[bool]:
    """规范化 entry signal 为长度 n_bars 的 bool 列表。

    默认（entry_signal=None）：**每个 bar 都允许入场**，前提是当前无持仓
    （即上一笔已平）。这是停损策略对比场景的合理默认 —— 想测停损本身效果，
    不应被人为的 entry timing 限制成"只入场一次"。

    自定义 entry：传 ``entry_signal=`` 为 list[bool] 或 pd.Series[bool]，
    长度必须与 df 一致。
    """
    if entry_signal is None:
        return [True] * n_bars
    if isinstance(entry_signal, pd.Series):
        return [False if pd.isna(x) else bool(x) for x in entry_signal.tolist()]
    sig = list(entry_signal)
    if len(sig) != n_bars:
        raise ValueError(
            f"entry_signal 长度 {len(sig)} 与数据 {n_bars} 不一致"
        )
    return [bool(x) for x in sig]


def run_backtest(
    df: pd.DataFrame,
    stop: BaseStop,
    *,
    side: str = "long",
    entry_signal: Optional[Sequence[bool]] = None,
    profit_target_pct: Optional[float] = None,    # e.g. 0.10 = 10% 止盈
    max_holding_bars: Optional[int] = None,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0,
    initial_capital: float = 100_000.0,
    position_fraction: float = 1.0,
    allow_overlap: bool = False,
) -> BacktestResult:
    """对一个 stop 跑 triple-barrier 回测。

    Parameters
    ----------
    df : OHLC DataFrame
    stop : BaseStop 实例（**不要在外部 reset，会被本函数 reset**）
    side : "long" / "short"
    entry_signal : bool 序列；True 的 bar 尝试开仓
    profit_target_pct : 止盈百分比；None 表示不设
    max_holding_bars : 持仓最多多少 bar 后强制平；None 不限
    commission_pct : 单边佣金
    slippage_pct : 单边滑点
    initial_capital : 起始资金
    position_fraction : 每次开仓使用当前已实现权益的比例，范围 (0, 1]
    allow_overlap : 保留参数；当前单持仓引擎不支持 True
    """
    if allow_overlap:
        raise NotImplementedError(
            "allow_overlap=True 需要独立 position/stop state，当前单持仓引擎不支持 overlap"
        )
    if df.empty:
        raise ValueError("OHLC 数据为空")
    if side not in ("long", "short"):
        raise ValueError(f"side 必须是 long/short，得到 {side}")
    if initial_capital <= 0:
        raise ValueError("initial_capital 必须大于 0")
    if not 0 < position_fraction <= 1:
        raise ValueError("position_fraction 必须在 (0, 1] 内")
    if commission_pct < 0 or slippage_pct < 0:
        raise ValueError("commission_pct 和 slippage_pct 不能为负")

    bars = _to_bars(df)
    n = len(bars)
    signals = _resolve_signals(df, entry_signal, n)

    result = BacktestResult(
        stop_name=stop.name,
        stop_params=dict(stop.params),
        initial_capital=initial_capital,
    )

    equity = initial_capital
    open_position = None

    def _marked_equity(mark_price: float) -> float:
        """按当前收盘价估计平仓后的权益，供逐 bar 风险指标使用。"""
        if open_position is None:
            return equity
        pos = open_position
        if pos["side"] == "long":
            adj_exit = mark_price * (1 - slippage_pct)
            pnl_per = (adj_exit - pos["entry_price"]) * pos["quantity"]
        else:
            adj_exit = mark_price * (1 + slippage_pct)
            pnl_per = (pos["entry_price"] - adj_exit) * pos["quantity"]
        estimated_commission = (
            (pos["entry_price"] + adj_exit) * pos["quantity"] * commission_pct
        )
        return equity + pnl_per - estimated_commission

    def _close(exit_bar: int, exit_price: float, reason: str) -> None:
        nonlocal equity, open_position
        pos = open_position
        # 滑点 + 佣金
        if pos["side"] == "long":
            adj_exit = exit_price * (1 - slippage_pct)
            pnl_per = (adj_exit - pos["entry_price"]) * pos["quantity"]
        else:
            adj_exit = exit_price * (1 + slippage_pct)
            pnl_per = (pos["entry_price"] - adj_exit) * pos["quantity"]
        commission = (
            (pos["entry_price"] + adj_exit) * pos["quantity"] * commission_pct
        )
        pnl = pnl_per - commission
        entry_notional = pos["entry_price"] * pos["quantity"]
        pnl_pct = pnl / entry_notional
        equity += pnl

        result.trades.append(Trade(
            entry_bar=pos["entry_bar"], entry_price=pos["entry_price"],
            exit_bar=exit_bar, exit_price=adj_exit,
            side=pos["side"], exit_reason=reason,
            n_bars=exit_bar - pos["entry_bar"], pnl=pnl, pnl_pct=pnl_pct,
            commission=commission, quantity=pos["quantity"],
            final_stop=stop.state.current_stop if stop.state else None,
        ))
        open_position = None
        if reason == "stop":
            result.n_stopped += 1
        elif reason == "profit":
            result.n_profit_taken += 1
        elif reason == "time":
            result.n_timed_out += 1

    for i, bar in enumerate(bars):
        # 先处理在本 bar 开始前已经存在的持仓。止损位只能使用上一收盘前
        # 已知的信息，不能先读当前 high/low 更新 stop 再回看同一 bar 是否触发。
        if open_position is not None:
            prior_stop = stop.state.current_stop if stop.state else None
            stop_hit = False
            stop_exit = None
            if stop.exit_reason == "stop" and prior_stop is not None:
                if side == "long" and bar.low <= prior_stop:
                    stop_hit = True
                    stop_exit = bar.open if bar.open <= prior_stop else prior_stop
                elif side == "short" and bar.high >= prior_stop:
                    stop_hit = True
                    stop_exit = bar.open if bar.open >= prior_stop else prior_stop

            # OHLC 无法识别同一 bar 内止损与止盈的先后，采用 stop-first 保守约定。
            if stop_hit:
                _close(i, float(stop_exit), reason="stop")
            elif profit_target_pct is not None:
                ep = open_position["entry_price"]
                if side == "long" and bar.high >= ep * (1 + profit_target_pct):
                    target = ep * (1 + profit_target_pct)
                    _close(i, max(bar.open, target), reason="profit")
                elif side == "short" and bar.low <= ep * (1 - profit_target_pct):
                    target = ep * (1 - profit_target_pct)
                    _close(i, min(bar.open, target), reason="profit")

            if open_position is not None:
                state = stop.update(bar)
                if stop.exit_reason == "time" and state.triggered:
                    _close(i, bar.close, reason="time")
                elif (max_holding_bars is not None and
                      state.n_bars_held >= max_holding_bars):
                    _close(i, bar.close, reason="time")

        # entry_signal 解释为本 bar 收盘信号；新仓从下一根 bar 起才可能触发。
        if open_position is None and signals[i] and i < n - 1:
            result.n_signals += 1
            entry_price = bar.close * (1 + slippage_pct if side == "long"
                                        else 1 - slippage_pct)
            stop.reset(side=side, entry_price=entry_price, entry_bar_index=i)
            stop.prime_at_entry(bars[:i], bar)
            quantity = equity * position_fraction / entry_price
            open_position = {
                "entry_bar": i, "entry_price": entry_price, "side": side,
                "quantity": quantity,
            }

        result.equity_curve.append(_marked_equity(bar.close))

    # 回测结束时如还有持仓，按最后收盘平
    if open_position is not None:
        _close(n - 1, bars[-1].close, reason="end")
        result.n_open_at_end += 1
        result.equity_curve[-1] = equity

    result.final_equity = equity
    result.n_trades = len(result.trades)
    return result
