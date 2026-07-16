"""walk_forward.py 测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dso.stops import ATRStop
from dso.walk_forward import (
    WalkForwardResult, make_folds, walk_forward,
)


# --- make_folds ----------------------------------------------------

def test_make_folds_anchored_count():
    folds = make_folds(300, n_folds=4, train_ratio=0.7, mode="anchored")
    assert len(folds) == 4
    # anchored：所有 train 都从 0 开始
    assert all(f.train_start == 0 for f in folds)
    # train_end 单调递增
    for i in range(1, 4):
        assert folds[i].train_end > folds[i - 1].train_end
    assert folds[0].train_end == 210
    assert folds[-1].test_end == 300


def test_make_folds_rolling_window_constant_train_length():
    folds = make_folds(500, n_folds=4, mode="rolling", train_ratio=0.7)
    train_lengths = [f.train_end - f.train_start for f in folds]
    # rolling 的 train 长度应该一致
    assert all(length == train_lengths[0] for length in train_lengths)


def test_make_folds_rejects_invalid():
    with pytest.raises(ValueError, match="n_folds"):
        make_folds(100, n_folds=1)
    with pytest.raises(ValueError, match="train_ratio"):
        make_folds(100, n_folds=3, train_ratio=1.5)
    with pytest.raises(ValueError, match="未知 mode"):
        make_folds(100, n_folds=3, mode="bogus")


def test_make_folds_test_segments_non_overlapping_anchored():
    folds = make_folds(300, n_folds=4, mode="anchored")
    for i in range(1, len(folds)):
        # 上一折 test_end <= 这一折 test_start
        assert folds[i - 1].test_end <= folds[i].test_start


# --- walk_forward end-to-end --------------------------------------

def test_walk_forward_returns_result(synth_mixed_df):
    grid = {"period": [10, 20], "multiplier": [2.0, 3.0]}
    result = walk_forward(synth_mixed_df, ATRStop, grid,
                           n_folds=3, mode="anchored")
    assert isinstance(result, WalkForwardResult)
    assert result.stop_name == "ATRStop"


def test_walk_forward_picks_best_params_each_fold(synth_mixed_df):
    grid = {"period": [10, 20], "multiplier": [2.0, 3.0]}
    result = walk_forward(synth_mixed_df, ATRStop, grid, n_folds=3)
    for f in result.folds:
        assert "period" in f.best_params
        assert "multiplier" in f.best_params
        assert f.best_params["period"] in [10, 20]


def test_walk_forward_train_test_scores_recorded(synth_mixed_df):
    grid = {"period": [10], "multiplier": [2.5]}
    result = walk_forward(synth_mixed_df, ATRStop, grid, n_folds=2)
    for f in result.folds:
        assert isinstance(f.train_score, float)
        assert isinstance(f.test_score, float)


def test_walk_forward_overfitting_gap_computed(synth_mixed_df):
    grid = {"period": [10, 20, 30], "multiplier": [2.0, 2.5, 3.0]}
    result = walk_forward(synth_mixed_df, ATRStop, grid, n_folds=3)
    # gap = mean_train - mean_test
    assert isinstance(result.overfitting_gap, float)
    assert np.isfinite(result.overfitting_gap)


def test_walk_forward_to_dict_serializable(synth_mixed_df):
    import json
    grid = {"period": [10], "multiplier": [2.5]}
    result = walk_forward(synth_mixed_df, ATRStop, grid, n_folds=2)
    json.dumps(result.to_dict())


def test_walk_forward_rolling_mode(synth_mixed_df):
    grid = {"period": [10], "multiplier": [2.5]}
    result = walk_forward(synth_mixed_df, ATRStop, grid,
                           n_folds=3, mode="rolling", train_ratio=0.6)
    assert isinstance(result, WalkForwardResult)


def test_walk_forward_skips_too_short_folds():
    """数据极少时应该没崩，只是 fold 数会少。"""
    rng = np.random.default_rng(0)
    n = 50  # 太短
    closes = 100 + rng.normal(0, 1, n).cumsum()
    df = pd.DataFrame({
        "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "volume": [1e6] * n,
    })
    grid = {"period": [10], "multiplier": [2.5]}
    result = walk_forward(df, ATRStop, grid, n_folds=5)
    # 不强制要求多少 fold，但至少不应该崩
    assert isinstance(result, WalkForwardResult)


def test_walk_forward_respects_entry_signal(synth_mixed_df):
    grid = {"period": [10], "multiplier": [2.5]}
    no_entries = [False] * len(synth_mixed_df)
    result = walk_forward(
        synth_mixed_df, ATRStop, grid, n_folds=3,
        entry_signal=no_entries,
    )
    assert result.folds
    assert all(f.test_total_return == 0.0 for f in result.folds)
    assert result.mean_train_score == 0.0
    assert result.mean_test_score == 0.0


def test_walk_forward_rejects_empty_parameter_values(synth_mixed_df):
    with pytest.raises(ValueError, match="param_grid"):
        walk_forward(
            synth_mixed_df, ATRStop,
            {"period": [], "multiplier": [2.5]}, n_folds=2,
        )
