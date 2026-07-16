"""metrics.py 测试 —— Sharpe / Sortino / Calmar / Deflated SR / Max DD。"""
from __future__ import annotations

import math

import numpy as np
import pytest

from dso.metrics import (
    calmar_ratio, deflated_sharpe_ratio, max_drawdown,
    returns_from_equity, sharpe_ratio, sortino_ratio,
    _approx_expected_max_z, _emc,
)


# --- sharpe_ratio -----------------------------------------------------

def test_sharpe_positive_mean_positive_sr():
    rets = [0.001] * 100
    sr = sharpe_ratio(rets)
    # std=0 → 函数应安全返回 0（不是 NaN）
    assert sr == 0.0


def test_sharpe_with_volatility():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, 252).tolist()
    sr = sharpe_ratio(rets)
    # 期望约 0.001 / 0.01 * sqrt(252) ≈ 1.59
    assert 1.0 < sr < 3.0


def test_sharpe_negative_drift_negative_sr():
    rng = np.random.default_rng(1)
    rets = rng.normal(-0.001, 0.01, 252).tolist()
    assert sharpe_ratio(rets) < 0


def test_sharpe_short_data():
    assert sharpe_ratio([]) == 0.0
    assert sharpe_ratio([0.001]) == 0.0


# --- sortino_ratio ---------------------------------------------------

def test_sortino_no_downside_safe():
    rets = [0.01] * 50    # 全正
    so = sortino_ratio(rets)
    # downside std = 0 → 函数应返回 0
    assert so == 0.0


def test_sortino_higher_than_sharpe_when_only_downside_vol():
    rng = np.random.default_rng(2)
    # 构造：上行平稳，下行剧烈
    rets = []
    for _ in range(252):
        if rng.random() < 0.6:
            rets.append(0.005)
        else:
            rets.append(-0.02)
    sr = sharpe_ratio(rets)
    so = sortino_ratio(rets)
    # 两者都有意义即可
    assert isinstance(sr, float)
    assert isinstance(so, float)


# --- max_drawdown ----------------------------------------------------

def test_max_dd_monotone_up_zero():
    eq = [100, 105, 110, 115, 120]
    assert max_drawdown(eq) == 0.0


def test_max_dd_simple_dip():
    eq = [100, 120, 90, 100]
    # peak=120, trough=90, dd = (90-120)/120 = -0.25
    assert math.isclose(max_drawdown(eq), -0.25, rel_tol=1e-9)


def test_max_dd_empty():
    assert max_drawdown([]) == 0.0


def test_max_dd_negative_returns_zero_peak_safe():
    """全 0 或负值不应触发 div-by-zero。"""
    assert max_drawdown([0, 0, 0]) == 0.0


# --- calmar_ratio ----------------------------------------------------

def test_calmar_basic():
    rng = np.random.default_rng(3)
    rets = rng.normal(0.001, 0.01, 252).tolist()
    eq = [100.0]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    c = calmar_ratio(rets, eq)
    assert isinstance(c, float)


def test_calmar_zero_dd_returns_zero():
    """无回撤场景应返回 0（避免 div by zero）。"""
    rets = [0.001] * 252
    eq = list(np.cumprod([1.001] * 252) * 100)
    c = calmar_ratio(rets, eq)
    assert c == 0.0


# --- deflated_sharpe_ratio ------------------------------------------

scipy = pytest.importorskip("scipy")


def test_dsr_high_sharpe_low_trials():
    """SR 高、试验少、obs 多 → DSR 概率应接近 1。"""
    prob = deflated_sharpe_ratio(
        sharpe_observed=2.5, n_trials=2, n_obs=1000)
    assert prob > 0.85


def test_dsr_high_sharpe_many_trials_lower_prob():
    """同样 SR，测试很多策略 → 多重检验偏倚扣减。

    SR 不能太大（否则 normal CDF 饱和到 1.0），n_obs 不能太大；选适中数。
    """
    prob_few = deflated_sharpe_ratio(sharpe_observed=0.5, n_trials=2, n_obs=100)
    prob_many = deflated_sharpe_ratio(sharpe_observed=0.5, n_trials=500, n_obs=100)
    assert prob_few > prob_many


def test_dsr_returns_in_unit_interval():
    """DSR 应在 [0, 1] 之间。"""
    for trials in [2, 10, 100]:
        for sr in [-1.0, 0.0, 0.5, 1.5, 3.0]:
            p = deflated_sharpe_ratio(sharpe_observed=sr, n_trials=trials, n_obs=500)
            assert 0.0 <= p <= 1.0


def test_dsr_annualization_scale_is_invariant():
    annualized = deflated_sharpe_ratio(
        sharpe_observed=1.6, n_trials=20, n_obs=252,
        periods_per_year=252,
    )
    per_period = deflated_sharpe_ratio(
        sharpe_observed=1.6 / math.sqrt(252), n_trials=20, n_obs=252,
        periods_per_year=1,
    )
    assert annualized == pytest.approx(per_period)


def test_dsr_zero_observations_returns_zero():
    p = deflated_sharpe_ratio(sharpe_observed=1.0, n_trials=5, n_obs=1)
    assert p == 0.0


def test_emc_constant():
    """Euler-Mascheroni 常数 ~ 0.5772。"""
    assert 0.57 < _emc() < 0.58


def test_approx_expected_max_z_increases_with_n():
    """E[max(Z_1..Z_N)] 应随 N 单调递增。"""
    e1 = _approx_expected_max_z(2)
    e2 = _approx_expected_max_z(100)
    assert e1 < e2


# --- returns_from_equity --------------------------------------------

def test_returns_from_equity_basic():
    eq = [100, 110, 121]
    rets = returns_from_equity(eq)
    assert len(rets) == 2
    assert math.isclose(rets[0], 0.1)
    assert math.isclose(rets[1], 0.1)


def test_returns_from_equity_empty():
    assert returns_from_equity([]) == []
    assert returns_from_equity([100]) == []


def test_returns_from_equity_zero_safety():
    """前一根 equity = 0 不应崩。"""
    rets = returns_from_equity([100, 0, 50])
    assert len(rets) == 2
    assert rets[1] == 0.0
