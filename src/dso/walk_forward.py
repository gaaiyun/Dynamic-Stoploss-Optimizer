"""Walk-forward 参数优化（Pardo 1992 / Lopez de Prado 2018）。

把数据按时间切若干段：

    [train_1 | test_1 | train_2 | test_2 | ...]    # rolling
    [train_1 | test_1 ]
    [    train_2     | test_2 ]                    # anchored
    [        train_3         | test_3 ]

每段在 train 上 grid search 选最优参数，到 test 上评估。最终结果是
out-of-sample 拼接的 equity 曲线 — 这才是诚实的"参数有效性"评估。

v1 的 optimizer 用整个历史做 grid search，是典型的"数据挖掘偏倚"。v2 强制 走 walk-forward。

参考：
- Pardo (1992), *Design, Testing, and Optimization of Trading Systems*
- Lopez de Prado (2018), ch. 12 "Backtesting through Cross-Validation"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .backtest import BacktestResult, run_backtest
from .metrics import returns_from_equity, sharpe_ratio
from .stops.base import BaseStop


WalkMode = Literal["anchored", "rolling"]


@dataclass
class Fold:
    fold_idx: int
    train_start: int
    train_end: int     # exclusive
    test_start: int
    test_end: int      # exclusive


@dataclass
class FoldResult:
    fold: Fold
    best_params: Dict[str, Any]
    train_score: float
    test_score: float
    test_total_return: float
    test_max_drawdown: float


@dataclass
class WalkForwardResult:
    stop_name: str
    folds: List[FoldResult] = field(default_factory=list)
    oos_total_return: float = 0.0
    mean_train_score: float = 0.0
    mean_test_score: float = 0.0

    @property
    def overfitting_gap(self) -> float:
        """train - test 越大越过拟合。"""
        return self.mean_train_score - self.mean_test_score

    def to_dict(self) -> dict:
        return {
            "stop_name": self.stop_name,
            "n_folds": len(self.folds),
            "oos_total_return": float(self.oos_total_return),
            "mean_train_score": float(self.mean_train_score),
            "mean_test_score": float(self.mean_test_score),
            "overfitting_gap": float(self.overfitting_gap),
            "folds": [
                {
                    "fold_idx": f.fold.fold_idx,
                    "train_range": [f.fold.train_start, f.fold.train_end],
                    "test_range": [f.fold.test_start, f.fold.test_end],
                    "best_params": f.best_params,
                    "train_score": float(f.train_score),
                    "test_score": float(f.test_score),
                    "test_total_return": float(f.test_total_return),
                    "test_max_drawdown": float(f.test_max_drawdown),
                }
                for f in self.folds
            ],
        }


def make_folds(n: int, n_folds: int = 5, train_ratio: float = 0.7,
               mode: WalkMode = "anchored") -> List[Fold]:
    """按时间切折。"""
    if n_folds < 2:
        raise ValueError("n_folds 至少 2")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio ∈ (0, 1)")

    folds = []
    if mode == "anchored":
        segment = n // (n_folds + 1)
        for k in range(n_folds):
            tr_end = segment * (k + 1)
            te_end = min(segment * (k + 2), n)
            folds.append(Fold(
                fold_idx=k, train_start=0, train_end=tr_end,
                test_start=tr_end, test_end=te_end,
            ))
    elif mode == "rolling":
        total_segs = n_folds + 1
        seg = n // total_segs
        train_len = max(int(seg * train_ratio / (1 - train_ratio)), seg)
        for k in range(n_folds):
            te_start = train_len + k * seg
            te_end = min(te_start + seg, n)
            tr_start = max(te_start - train_len, 0)
            if te_start >= n:
                break
            folds.append(Fold(
                fold_idx=k, train_start=tr_start, train_end=te_start,
                test_start=te_start, test_end=te_end,
            ))
    else:
        raise ValueError(f"未知 mode {mode}")
    return folds


def _evaluate_params(
    df: pd.DataFrame, stop_class: type[BaseStop],
    params: dict, entry_signal: Optional[Sequence[bool]],
    side: str, profit_target_pct: Optional[float],
    max_holding_bars: Optional[int],
    commission_pct: float, slippage_pct: float,
    initial_capital: float, periods_per_year: int,
) -> Tuple[float, BacktestResult]:
    stop = stop_class(**params)
    if entry_signal is not None:
        sig = list(entry_signal)[:len(df)]
        # 长度 < df 时补 False
        sig += [False] * (len(df) - len(sig))
        entry_segment = sig
    else:
        entry_segment = None
    result = run_backtest(
        df, stop, side=side, entry_signal=entry_segment,
        profit_target_pct=profit_target_pct,
        max_holding_bars=max_holding_bars,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        initial_capital=initial_capital,
    )
    rets = returns_from_equity(result.equity_curve)
    score = sharpe_ratio(rets, periods_per_year=periods_per_year)
    return score, result


def walk_forward(
    df: pd.DataFrame,
    stop_class: type[BaseStop],
    param_grid: Dict[str, List[Any]],
    *,
    n_folds: int = 5,
    train_ratio: float = 0.7,
    mode: WalkMode = "anchored",
    side: str = "long",
    entry_signal: Optional[Sequence[bool]] = None,
    profit_target_pct: Optional[float] = None,
    max_holding_bars: Optional[int] = None,
    commission_pct: float = 0.001,
    slippage_pct: float = 0.0,
    initial_capital: float = 100_000.0,
    periods_per_year: int = 252,
) -> WalkForwardResult:
    """对一个停损策略类做 walk-forward 优化。

    Parameters
    ----------
    stop_class : 必须可以 ``stop_class(**params)`` 实例化
    param_grid : 例如 ``{"period": [10, 14, 20], "multiplier": [2.0, 2.5, 3.0]}``
    """
    n = len(df)
    folds = make_folds(n, n_folds=n_folds, train_ratio=train_ratio, mode=mode)

    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]

    result = WalkForwardResult(stop_name=stop_class.__name__)
    train_scores, test_scores = [], []
    oos_returns = []

    for fold in folds:
        train_df = df.iloc[fold.train_start:fold.train_end].reset_index(drop=True)
        test_df = df.iloc[fold.test_start:fold.test_end].reset_index(drop=True)
        if len(train_df) < 30 or len(test_df) < 5:
            continue

        # ----- train: grid search -----
        best_params: Dict[str, Any] = {}
        best_score = -np.inf
        for combo in product(*value_lists):
            params = dict(zip(keys, combo))
            try:
                score, _ = _evaluate_params(
                    train_df, stop_class, params, entry_signal=None,
                    side=side, profit_target_pct=profit_target_pct,
                    max_holding_bars=max_holding_bars,
                    commission_pct=commission_pct,
                    slippage_pct=slippage_pct,
                    initial_capital=initial_capital,
                    periods_per_year=periods_per_year,
                )
            except Exception:
                score = -np.inf
            if score > best_score:
                best_score = score
                best_params = params

        # ----- test: 用最佳参数评估 -----
        try:
            test_score, test_bt = _evaluate_params(
                test_df, stop_class, best_params, entry_signal=None,
                side=side, profit_target_pct=profit_target_pct,
                max_holding_bars=max_holding_bars,
                commission_pct=commission_pct,
                slippage_pct=slippage_pct,
                initial_capital=initial_capital,
                periods_per_year=periods_per_year,
            )
        except Exception:
            test_score = -np.inf
            test_bt = None

        train_scores.append(best_score)
        test_scores.append(test_score)
        if test_bt is not None:
            oos_returns.append(test_bt.total_return)

        # max drawdown for test
        if test_bt is not None:
            from .metrics import max_drawdown
            mdd = max_drawdown(test_bt.equity_curve)
        else:
            mdd = 0.0

        result.folds.append(FoldResult(
            fold=fold,
            best_params=best_params,
            train_score=best_score,
            test_score=test_score,
            test_total_return=(test_bt.total_return if test_bt else 0.0),
            test_max_drawdown=mdd,
        ))

    if train_scores:
        result.mean_train_score = float(np.mean(train_scores))
    if test_scores:
        result.mean_test_score = float(np.mean(test_scores))
    if oos_returns:
        # OOS 总收益：按 fold 累乘
        product_ret = 1.0
        for r in oos_returns:
            product_ret *= (1 + r)
        result.oos_total_return = float(product_ret - 1)
    return result
