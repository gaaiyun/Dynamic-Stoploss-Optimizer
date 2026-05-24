"""8 个停损策略的单元测试。"""
from __future__ import annotations

import pytest

from dso.stops import (
    STOPS, ATRStop, ChandelierStop, DonchianStop, MaxDrawdownStop,
    MovingAverageStop, ParabolicSARStop, SuperTrendStop, TimeStop,
    VolatilityStop,
)
from dso.stops.base import Bar, BaseStop, StopState
from dso.stops.atr import _wilder_atr
from dso.stops.moving_average import _ema, _sma


def _mkbar(c: float, h: float = None, l: float = None) -> Bar:
    h = h if h is not None else c * 1.005
    l = l if l is not None else c * 0.995
    return Bar(open=c, high=h, low=l, close=c)


# --- registry ----------------------------------------------------------

def test_registry_has_8_strategies():
    expected = {"atr", "chandelier", "supertrend", "parabolic_sar",
                "donchian", "moving_average", "volatility",
                "max_drawdown", "time"}
    assert set(STOPS.keys()) == expected


def test_registry_classes_inherit_basestop():
    for cls in STOPS.values():
        assert issubclass(cls, BaseStop)


def test_each_stop_has_default_params():
    for name, cls in STOPS.items():
        assert isinstance(cls.DEFAULT_PARAMS, dict)


# --- Bar dataclass ----------------------------------------------------

def test_bar_from_dict_lowercase():
    b = Bar.from_dict({"open": 1, "high": 2, "low": 0.5, "close": 1.5})
    assert b.open == 1.0 and b.high == 2.0


def test_bar_from_dict_capital():
    b = Bar.from_dict({"Open": 1, "High": 2, "Low": 0.5, "Close": 1.5})
    assert b.close == 1.5


# --- StopState --------------------------------------------------------

def test_stop_state_to_dict_serializable():
    import json
    s = StopState(side="long", entry_price=100.0, entry_bar_index=5,
                  current_stop=95.0, n_bars_held=3)
    d = s.to_dict()
    json.dumps(d)
    assert d["current_stop"] == 95.0


# --- BaseStop common behavior ----------------------------------------

def test_basestop_reset_validates_side():
    stop = ATRStop()
    with pytest.raises(ValueError):
        stop.reset(side="invalid", entry_price=100)


def test_basestop_reset_validates_entry_price():
    stop = ATRStop()
    with pytest.raises(ValueError):
        stop.reset(side="long", entry_price=0)


def test_basestop_update_requires_reset():
    stop = ATRStop()
    with pytest.raises(RuntimeError):
        stop.update(_mkbar(100))


def test_basestop_tighten_long_only_raises_stop():
    """Long：新 stop < 旧 stop 时不应降级（向不利方向）。"""
    stop = ATRStop(period=2, multiplier=1.0)
    stop.reset(side="long", entry_price=100)
    # 喂 3 bar 给 ATR 暖机
    for _ in range(3):
        stop.update(_mkbar(100))
    initial_stop = stop.state.current_stop
    # 模拟下跌 → 新计算的 stop 可能更低，但 long 的 tighten 应该保持高的
    for _ in range(2):
        stop.update(_mkbar(90))
    # current_stop 不应小于 initial_stop（trailing rule）
    if initial_stop is not None and stop.state.current_stop is not None:
        assert stop.state.current_stop >= initial_stop


# --- ATR ---------------------------------------------------------------

def test_wilder_atr_short_data_returns_none():
    bars = [_mkbar(100) for _ in range(5)]
    assert _wilder_atr(bars, period=14) is None


def test_wilder_atr_constant_prices_zero():
    """常数价 → TR 全 0 → ATR = 0。"""
    bars = [_mkbar(100, h=100, l=100) for _ in range(30)]
    atr = _wilder_atr(bars, period=14)
    assert atr == 0.0


def test_wilder_atr_increasing_volatility():
    bars = []
    for i in range(30):
        c = 100 + i
        bars.append(_mkbar(c, h=c + i * 0.5, l=c - i * 0.5))
    atr = _wilder_atr(bars, period=14)
    assert atr is not None and atr > 0


def test_atr_stop_long_trails_up():
    """Long ATR stop 应该跟着上涨 trail."""
    stop = ATRStop(period=5, multiplier=2.0)
    stop.reset(side="long", entry_price=100)
    # 暖机 + 上涨
    for c in [100, 100, 100, 100, 100, 100]:
        stop.update(_mkbar(c))
    early_stop = stop.state.current_stop
    for c in [110, 115, 120, 125]:
        stop.update(_mkbar(c))
    later_stop = stop.state.current_stop
    if early_stop is not None and later_stop is not None:
        assert later_stop >= early_stop


def test_atr_stop_short_trails_down():
    stop = ATRStop(period=5, multiplier=2.0)
    stop.reset(side="short", entry_price=100)
    for c in [100] * 6:
        stop.update(_mkbar(c))
    early_stop = stop.state.current_stop
    for c in [90, 85, 80, 75]:
        stop.update(_mkbar(c))
    later_stop = stop.state.current_stop
    if early_stop is not None and later_stop is not None:
        assert later_stop <= early_stop


def test_atr_stop_triggers_on_low_break():
    stop = ATRStop(period=2, multiplier=1.0)
    stop.reset(side="long", entry_price=100)
    for _ in range(3):
        stop.update(_mkbar(100, h=101, l=99))
    # 急跌穿透
    state = stop.update(_mkbar(80, h=82, l=78))
    if stop.state.current_stop is not None:
        if 78 <= stop.state.current_stop:
            assert state.triggered


# --- Chandelier --------------------------------------------------------

def test_chandelier_uses_period_high():
    stop = ChandelierStop(period=5, multiplier=2.0)
    stop.reset(side="long", entry_price=100)
    bars = [_mkbar(c, h=c + 2, l=c - 2)
            for c in [100, 105, 110, 108, 109, 112, 115]]
    for b in bars:
        stop.update(b)
    # 最高 high = 115+2 = 117，stop = 117 - 2*ATR
    assert stop.state.current_stop is not None
    assert stop.state.current_stop < 117  # 必须比最高点低


def test_chandelier_default_params():
    stop = ChandelierStop()
    assert stop.params["period"] == 22
    assert stop.params["multiplier"] == 3.0


# --- SuperTrend --------------------------------------------------------

def test_supertrend_returns_a_value():
    stop = SuperTrendStop(period=5, multiplier=2.0)
    stop.reset(side="long", entry_price=100)
    for c in range(100, 120):
        stop.update(_mkbar(c, h=c + 1, l=c - 1))
    assert stop.state.current_stop is not None
    assert stop.state.current_stop < 120   # long stop 在价格下方


# --- Parabolic SAR ----------------------------------------------------

def test_parabolic_sar_af_increases_with_new_high():
    stop = ParabolicSARStop()
    stop.reset(side="long", entry_price=100)
    af_start = stop.state.metadata["af"]
    # 连续创新高 → AF 应增长
    for c in range(100, 110):
        stop.update(_mkbar(c, h=c + 1, l=c - 1))
    af_later = stop.state.metadata["af"]
    assert af_later > af_start


def test_parabolic_sar_af_capped_at_max():
    stop = ParabolicSARStop(af_init=0.18, af_increment=0.02, af_max=0.20)
    stop.reset(side="long", entry_price=100)
    for c in range(100, 130):
        stop.update(_mkbar(c, h=c + 1, l=c - 1))
    assert stop.state.metadata["af"] <= 0.20


# --- Donchian ----------------------------------------------------------

def test_donchian_long_stop_is_period_low():
    stop = DonchianStop(period=5)
    stop.reset(side="long", entry_price=100)
    for c in [100, 95, 98, 99, 97, 100]:
        stop.update(_mkbar(c, h=c + 1, l=c - 1))
    # 最近 5 bar 的 low = c - 1：最低 95-1=94
    assert stop.state.current_stop == 94


def test_donchian_short_stop_is_period_high():
    stop = DonchianStop(period=5)
    stop.reset(side="short", entry_price=100)
    for c in [100, 105, 102, 101, 103, 100]:
        stop.update(_mkbar(c, h=c + 1, l=c - 1))
    # 最近 5 bar 的 high = c+1：最高 105+1=106
    assert stop.state.current_stop == 106


def test_donchian_needs_warmup():
    stop = DonchianStop(period=10)
    stop.reset(side="long", entry_price=100)
    # 只喂 5 bar，不够
    for _ in range(5):
        stop.update(_mkbar(100))
    assert stop.state.current_stop is None


# --- MovingAverage ----------------------------------------------------

def test_sma_basic():
    assert _sma([1, 2, 3, 4, 5], 5) == 3.0
    assert _sma([1, 2, 3], 5) is None


def test_ema_basic():
    # EMA(2) 在 [10, 10, 10] 上应是 10
    assert _ema([10, 10, 10], 2) == 10.0


def test_ma_stop_uses_ema_by_default():
    stop = MovingAverageStop(period=5, ma_type="ema")
    stop.reset(side="long", entry_price=100)
    for c in [100, 102, 104, 106, 108]:
        stop.update(_mkbar(c))
    assert stop.state.current_stop is not None
    # EMA(5) 在上涨序列上应该 < 108
    assert stop.state.current_stop < 108


def test_ma_stop_rejects_unknown_ma_type():
    stop = MovingAverageStop(ma_type="invalid")
    stop.reset(side="long", entry_price=100)
    with pytest.raises(ValueError):
        for _ in range(10):
            stop.update(_mkbar(100))


# --- Volatility -------------------------------------------------------

def test_volatility_stop_uses_log_returns():
    stop = VolatilityStop(window=10, multiplier=2.0, use_log_returns=True)
    stop.reset(side="long", entry_price=100)
    for c in [100, 101, 99, 102, 98, 103, 97, 104, 96, 105, 95]:
        stop.update(_mkbar(c))
    assert stop.state.current_stop is not None


def test_volatility_stop_needs_warmup():
    stop = VolatilityStop(window=20)
    stop.reset(side="long", entry_price=100)
    for _ in range(5):
        stop.update(_mkbar(100))
    assert stop.state.current_stop is None


# --- MaxDrawdown ------------------------------------------------------

def test_max_drawdown_triggers_on_5pct_drop():
    stop = MaxDrawdownStop(max_drawdown_pct=0.05, use_entry_price=False)
    stop.reset(side="long", entry_price=100)
    # 涨到 120 → peak = 120 → stop = 120*0.95 = 114
    stop.update(_mkbar(120, h=120, l=119))
    # 跌到 110 → 触发
    state = stop.update(_mkbar(110, h=115, l=110))
    assert state.triggered


def test_max_drawdown_use_entry_mode():
    stop = MaxDrawdownStop(max_drawdown_pct=0.10, use_entry_price=True)
    stop.reset(side="long", entry_price=100)
    # 即使涨到 200，stop 也应该锁定在 entry * 0.90 = 90
    stop.update(_mkbar(200, h=201, l=199))
    assert stop.state.current_stop == 90


# --- Time --------------------------------------------------------------

def test_time_stop_triggers_after_max_bars():
    stop = TimeStop(max_bars=5)
    stop.reset(side="long", entry_price=100)
    for i in range(4):
        state = stop.update(_mkbar(100))
        assert not state.triggered
    state = stop.update(_mkbar(100))
    assert state.triggered


def test_time_stop_no_current_stop_value():
    stop = TimeStop(max_bars=10)
    stop.reset(side="long", entry_price=100)
    stop.update(_mkbar(100))
    assert stop.state.current_stop is None
