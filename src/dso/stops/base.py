"""停损策略基类与共享数据结构。

停损是**有状态**的（要随价格变动 trail），因此每个策略实例都维护一个
``StopState``：初始化时 reset，每根 K 线调 ``update(bar)`` 推进，触发时
``triggered=True``。

这与 v1 的 stateless 设计不同 —— v1 每次重算所有历史，慢且容易出现
look-ahead bias。v2 的状态机一次只看一根 bar。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Bar:
    """一根 K 线最少需要的字段。"""
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "Bar":
        return cls(
            open=float(d.get("open", d.get("Open", 0))),
            high=float(d.get("high", d.get("High", 0))),
            low=float(d.get("low", d.get("Low", 0))),
            close=float(d.get("close", d.get("Close", 0))),
            volume=float(d.get("volume", d.get("Volume", 0))),
        )


@dataclass
class StopState:
    """停损策略的运行状态。

    Attributes
    ----------
    side : "long" / "short"
    entry_price : 入场价
    entry_bar_index : 入场时的 bar index
    current_stop : 当前止损价（None 表示尚未生成）
    triggered : 当前 bar 是否触发止损
    n_bars_held : 已经持仓多少根 bar
    metadata : 策略私有状态（如 SAR 的 EP / AF、Chandelier 的 highest high）
    """
    side: str = "long"
    entry_price: float = 0.0
    entry_bar_index: int = 0
    current_stop: Optional[float] = None
    triggered: bool = False
    n_bars_held: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "side": self.side,
            "entry_price": float(self.entry_price),
            "entry_bar_index": int(self.entry_bar_index),
            "current_stop": (float(self.current_stop)
                             if self.current_stop is not None else None),
            "triggered": bool(self.triggered),
            "n_bars_held": int(self.n_bars_held),
            "metadata": dict(self.metadata),
        }


class BaseStop(ABC):
    """所有停损策略的基类。"""

    # 默认参数（子类覆盖）
    DEFAULT_PARAMS: Dict[str, Any] = {}

    # 在 update() 看到足够数据前，是否允许提前触发
    NEEDS_WARMUP: bool = True

    def __init__(self, **params):
        merged = dict(self.DEFAULT_PARAMS)
        merged.update({k: v for k, v in params.items() if v is not None})
        self.params: Dict[str, Any] = merged
        self.state: Optional[StopState] = None
        self._history: list[Bar] = []

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def exit_reason(self) -> str:
        """触发时记录的 exit_reason。默认 "stop"；time-based stop 覆盖为 "time"。"""
        return "stop"

    def reset(self, side: str, entry_price: float,
              entry_bar_index: int = 0) -> StopState:
        """开新仓时调一次。返回初始 state。"""
        if side not in ("long", "short"):
            raise ValueError(f"side 必须是 long/short，得到 {side}")
        if entry_price <= 0:
            raise ValueError(f"entry_price 必须 > 0，得到 {entry_price}")

        self.state = StopState(
            side=side, entry_price=entry_price,
            entry_bar_index=entry_bar_index,
            metadata={},
        )
        self._history = []
        self._init_state()
        return self.state

    def _init_state(self) -> None:
        """子类覆盖：初始化策略私有 metadata（默认 no-op）。"""

    def update(self, bar: Bar) -> StopState:
        """喂一根 K 线，推进 state。返回更新后的 state。"""
        if self.state is None:
            raise RuntimeError("先调 reset(side, entry_price) 才能 update")
        self._history.append(bar)
        self.state.n_bars_held += 1

        new_stop = self._update_stop(bar)
        # trailing：long 只能向上调整（保护已有利润），short 只能向下
        if new_stop is not None:
            self.state.current_stop = self._tighten(new_stop)

        # 触发判断
        self.state.triggered = self._check_triggered(bar)
        return self.state

    def prime_at_entry(self, history: list[Bar], entry_bar: Bar) -> StopState:
        """用截至入场收盘可见的历史建立下一根 K 线使用的止损位。

        入场价按 ``entry_bar.close`` 形成，因此该 K 线的 high/low 不能再反过来
        触发这笔新仓。这里更新指标与止损状态，但不增加持仓 bar 数，也不做
        触发判断。
        """
        if self.state is None:
            raise RuntimeError("先调 reset(side, entry_price) 才能 prime")
        self._history = [*history, entry_bar]
        new_stop = self._update_stop(entry_bar)
        if new_stop is not None:
            self.state.current_stop = self._tighten(new_stop)
        self.state.triggered = False
        self.state.n_bars_held = 0
        return self.state

    @abstractmethod
    def _update_stop(self, bar: Bar) -> Optional[float]:
        """子类必须实现：根据当前 bar 算出新的止损价。

        返回 None = 不更新（保持原 stop）。返回的值不会自动 tighten —— 由
        基类 ``_tighten`` 统一处理"只向有利方向调整"。
        """

    def _tighten(self, new_stop: float) -> float:
        """trailing：long 取 max(旧, 新)，short 取 min(旧, 新)。"""
        if self.state.current_stop is None:
            return new_stop
        if self.state.side == "long":
            return max(self.state.current_stop, new_stop)
        return min(self.state.current_stop, new_stop)

    def _check_triggered(self, bar: Bar) -> bool:
        """默认触发逻辑：触及止损价。

        - long: bar.low <= stop → 触发
        - short: bar.high >= stop → 触发
        """
        if self.state.current_stop is None:
            return False
        if self.state.side == "long":
            return bar.low <= self.state.current_stop
        return bar.high >= self.state.current_stop


# --- 工具：bar 序列预热（不持仓时用历史填充指标缓存） -------------------

def warmup(stop: BaseStop, bars: list[Bar]) -> None:
    """先把若干根历史 bar 喂给 stop 让它的指标暖机，不算持仓。"""
    if stop.state is None:
        return
    for b in bars:
        stop._history.append(b)
