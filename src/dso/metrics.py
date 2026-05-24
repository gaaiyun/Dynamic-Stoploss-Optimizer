"""组合表现 + Deflated Sharpe Ratio。

Deflated Sharpe Ratio（Bailey & Lopez de Prado 2014）：当用同一份数据
比较 N 个策略时，最高的 Sharpe 因为多重检验偏倚，需要"打折"。这是
学界共识但工程项目里几乎没人接入。
"""
from __future__ import annotations

import math
from typing import List, Sequence

import numpy as np


def sharpe_ratio(returns: Sequence[float], periods_per_year: int = 252,
                 rf: float = 0.0) -> float:
    """单期年化 Sharpe。"""
    arr = np.asarray(returns, dtype=float)
    if len(arr) < 2:
        return 0.0
    excess = arr - rf / periods_per_year
    std = excess.std(ddof=1)
    if std < 1e-12:
        return 0.0
    return float(excess.mean() / std * math.sqrt(periods_per_year))


def sortino_ratio(returns: Sequence[float], periods_per_year: int = 252,
                  rf: float = 0.0) -> float:
    """只用下行波动算分母的 Sortino。"""
    arr = np.asarray(returns, dtype=float)
    if len(arr) < 2:
        return 0.0
    excess = arr - rf / periods_per_year
    downside = np.minimum(excess, 0)
    downside_std = math.sqrt((downside ** 2).mean())
    if downside_std < 1e-12:
        return 0.0
    return float(excess.mean() / downside_std * math.sqrt(periods_per_year))


def max_drawdown(equity_curve: Sequence[float]) -> float:
    """从 equity 序列算最大回撤（负数）。"""
    if not equity_curve:
        return 0.0
    arr = np.asarray(equity_curve, dtype=float)
    peak = np.maximum.accumulate(arr)
    # 避免 peak=0 时除零
    valid = peak > 0
    if not valid.any():
        return 0.0
    dd = (arr[valid] - peak[valid]) / peak[valid]
    return float(dd.min())


def calmar_ratio(returns: Sequence[float],
                 equity_curve: Sequence[float],
                 periods_per_year: int = 252) -> float:
    """Calmar = 年化收益 / |最大回撤|."""
    arr = np.asarray(returns, dtype=float)
    if len(arr) < 2:
        return 0.0
    annual_ret = float((1 + arr.mean()) ** periods_per_year - 1)
    mdd = max_drawdown(equity_curve)
    if abs(mdd) < 1e-12:
        return 0.0
    return float(annual_ret / abs(mdd))


# --- Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014) ---------------

def _emc() -> float:
    """Euler-Mascheroni 常数。"""
    return 0.57721566490153286061


def _approx_expected_max_z(n_trials: int) -> float:
    """E[max(Z_1...Z_N)] for i.i.d. standard normal — 用 Bailey-LdP 的近似。

    E[max] ≈ (1 - γ) × Φ⁻¹(1 - 1/N) + γ × Φ⁻¹(1 - 1/(N·e))

    其中 γ = Euler-Mascheroni 常数，Φ⁻¹ 是标准正态的 inverse CDF。
    """
    from scipy.stats import norm   # type: ignore
    if n_trials < 2:
        return 0.0
    gamma = _emc()
    e = math.e
    a = norm.ppf(1 - 1 / n_trials)
    b = norm.ppf(1 - 1 / (n_trials * e))
    return (1 - gamma) * a + gamma * b


def deflated_sharpe_ratio(
    sharpe_observed: float,
    n_trials: int,
    n_obs: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
) -> float:
    """Deflated Sharpe Ratio（Bailey & Lopez de Prado 2014）。

    对"最高的 Sharpe" 做多重检验校正，返回 [0, 1] 区间的概率：观测到的
    Sharpe 比随机 baseline 更好的统计置信度。

    Parameters
    ----------
    sharpe_observed : 观测到的 Sharpe（年化）
    n_trials : 测试过的策略数（grid search 的 grid size 也算）
    n_obs : 用了多少 sample 观测
    skewness : 收益分布偏度（默认 0 = 假设正态）
    kurtosis : 收益分布峰度（默认 3 = 正态）

    Returns
    -------
    概率值 ∈ [0, 1]。> 0.95 = 在 95% 置信下确实 outperform。

    Notes
    -----
    论文：Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio:
    Correcting for Selection Bias, Backtest Overfitting, and
    Non-Normality"，JPM.
    """
    try:
        from scipy.stats import norm    # type: ignore
    except ImportError:
        raise ImportError("deflated_sharpe_ratio 需要 scipy：pip install scipy")

    if n_trials < 1 or n_obs < 2:
        return 0.0

    # Step 1: Expected max Sharpe under null（i.i.d. zero-mean）
    e_max_z = _approx_expected_max_z(n_trials)
    # 把 z-score 转回 Sharpe scale
    sr_zero = e_max_z / math.sqrt(n_obs)

    # Step 2: 调整非正态性（Bailey-LdP eq. 13）
    # SE(SR) = sqrt((1 - skew*SR + (kurt-1)/4 * SR^2) / (n_obs - 1))
    sr2 = sharpe_observed ** 2
    se_sr_sq = (1 - skewness * sharpe_observed +
                (kurtosis - 1) / 4 * sr2) / max(n_obs - 1, 1)
    if se_sr_sq <= 0:
        return 0.0
    se_sr = math.sqrt(se_sr_sq)

    # Step 3: DSR = Φ((SR_observed - SR_zero) / SE(SR))
    z = (sharpe_observed - sr_zero) / se_sr
    return float(norm.cdf(z))


def returns_from_equity(equity_curve: Sequence[float]) -> List[float]:
    """从 equity 序列推回每期收益率。"""
    arr = np.asarray(equity_curve, dtype=float)
    if len(arr) < 2:
        return []
    rets = []
    for i in range(1, len(arr)):
        prev = arr[i - 1]
        if prev <= 0:
            rets.append(0.0)
        else:
            rets.append(float(arr[i] / prev - 1))
    return rets
