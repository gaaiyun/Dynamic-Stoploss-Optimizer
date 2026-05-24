"""
止损策略实现模块

派蒙精心打造的四种动态止损策略！
- 基于波动率的动态止损
- 基于 ATR 的止损
- 基于移动均线的止损
- 基于最大回撤的止损
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class StoplossSignal:
    """止损信号数据类"""
    triggered: bool  # 是否触发止损
    stop_price: float  # 止损价格
    current_price: float  # 当前价格
    stop_type: str  # 止损类型
    metadata: Dict[str, Any] = None  # 额外信息
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseStoploss(ABC):
    """止损策略基类
    
    所有止损策略都需要继承这个类并实现 calculate_stop 方法
    """
    
    def __init__(self, name: str = "base"):
        self.name = name
        self.params = {}
    
    @abstractmethod
    def calculate_stop(
        self,
        data: pd.DataFrame,
        position_type: str = 'long',
        entry_price: float = None,
        current_bar: int = None
    ) -> StoplossSignal:
        """
        计算止损位
        
        Args:
            data: 包含 OHLC 数据的历史 DataFrame
            position_type: 持仓类型 ('long' 或 'short')
            entry_price: 入场价格
            current_bar: 当前 K 线索引
            
        Returns:
            StoplossSignal: 止损信号
        """
        pass
    
    def update_params(self, **kwargs):
        """更新策略参数"""
        self.params.update(kwargs)
    
    def get_params(self) -> Dict[str, Any]:
        """获取当前参数"""
        return self.params.copy()


class VolatilityStoploss(BaseStoploss):
    """基于波动率的动态止损策略
    
    使用历史波动率动态调整止损幅度：
    - 波动率高时放宽止损
    - 波动率低时收紧止损
    
    参数:
        window: 计算波动率的时间窗口 (默认 20)
        multiplier: 波动率乘数 (默认 2.0)
        use_log_returns: 是否使用对数收益率 (默认 True)
    """
    
    def __init__(
        self,
        window: int = 20,
        multiplier: float = 2.0,
        use_log_returns: bool = True
    ):
        super().__init__(name="volatility")
        self.window = window
        self.multiplier = multiplier
        self.use_log_returns = use_log_returns
        self.params = {
            'window': window,
            'multiplier': multiplier,
            'use_log_returns': use_log_returns
        }
    
    def calculate_volatility(self, prices: pd.Series) -> pd.Series:
        """计算历史波动率"""
        if self.use_log_returns:
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()
        
        volatility = returns.rolling(window=self.window).std() * np.sqrt(252)
        return volatility
    
    def calculate_stop(
        self,
        data: pd.DataFrame,
        position_type: str = 'long',
        entry_price: float = None,
        current_bar: int = None
    ) -> StoplossSignal:
        """计算基于波动率的止损位"""
        if current_bar is None or current_bar < self.window:
            return StoplossSignal(
                triggered=False,
                stop_price=None,
                current_price=data['close'].iloc[-1],
                stop_type='volatility',
                metadata={'reason': 'insufficient_data'}
            )
        
        current_price = data['close'].iloc[-1]
        volatility = self.calculate_volatility(data['close']).iloc[-1]
        
        if pd.isna(volatility) or entry_price is None:
            return StoplossSignal(
                triggered=False,
                stop_price=None,
                current_price=current_price,
                stop_type='volatility',
                metadata={'reason': 'invalid_volatility_or_entry'}
            )
        
        # 动态止损幅度 = 波动率 * 乘数
        stop_distance = volatility * self.multiplier * entry_price
        
        if position_type == 'long':
            stop_price = entry_price - stop_distance
            triggered = current_price <= stop_price
        else:  # short
            stop_price = entry_price + stop_distance
            triggered = current_price >= stop_price
        
        return StoplossSignal(
            triggered=triggered,
            stop_price=stop_price,
            current_price=current_price,
            stop_type='volatility',
            metadata={
                'volatility': volatility,
                'stop_distance': stop_distance,
                'stop_distance_pct': stop_distance / entry_price
            }
        )


class ATRStoploss(BaseStoploss):
    """基于 ATR (平均真实波幅) 的止损策略
    
    ATR 是衡量市场波动性的经典指标，由 Welles Wilder 提出。
    止损位 = 入场价 ± (ATR × 乘数)
    
    参数:
        period: ATR 计算周期 (默认 14)
        multiplier: ATR 乘数 (默认 2.5)
        use_current_atr: 是否使用当前 ATR 值 (默认 True)
    """
    
    def __init__(
        self,
        period: int = 14,
        multiplier: float = 2.5,
        use_current_atr: bool = True
    ):
        super().__init__(name="atr")
        self.period = period
        self.multiplier = multiplier
        self.use_current_atr = use_current_atr
        self.params = {
            'period': period,
            'multiplier': multiplier,
            'use_current_atr': use_current_atr
        }
    
    def calculate_atr(self, data: pd.DataFrame) -> pd.Series:
        """
        计算 ATR (Average True Range)
        
        True Range = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = MA(True Range, period)
        """
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 计算 True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算 ATR (使用 Wilder 的平滑方法)
        atr = true_range.ewm(span=self.period, adjust=False, min_periods=self.period).mean()
        return atr
    
    def calculate_stop(
        self,
        data: pd.DataFrame,
        position_type: str = 'long',
        entry_price: float = None,
        current_bar: int = None
    ) -> StoplossSignal:
        """计算基于 ATR 的止损位"""
        if current_bar is None or current_bar < self.period:
            return StoplossSignal(
                triggered=False,
                stop_price=None,
                current_price=data['close'].iloc[-1],
                stop_type='atr',
                metadata={'reason': 'insufficient_data'}
            )
        
        current_price = data['close'].iloc[-1]
        atr = self.calculate_atr(data).iloc[-1]
        
        if pd.isna(atr) or entry_price is None:
            return StoplossSignal(
                triggered=False,
                stop_price=None,
                current_price=current_price,
                stop_type='atr',
                metadata={'reason': 'invalid_atr_or_entry'}
            )
        
        # 止损距离 = ATR × 乘数
        stop_distance = atr * self.multiplier
        
        if position_type == 'long':
            stop_price = entry_price - stop_distance
            triggered = current_price <= stop_price
        else:  # short
            stop_price = entry_price + stop_distance
            triggered = current_price >= stop_price
        
        return StoplossSignal(
            triggered=triggered,
            stop_price=stop_price,
            current_price=current_price,
            stop_type='atr',
            metadata={
                'atr': atr,
                'stop_distance': stop_distance,
                'stop_distance_pct': stop_distance / entry_price if entry_price else None
            }
        )


class MAStoploss(BaseStoploss):
    """基于移动平均线的止损策略
    
    使用移动平均线作为动态止损位：
    - 多头：MA 作为支撑位，跌破止损
    - 空头：MA 作为阻力位，突破止损
    
    参数:
        period: 均线周期 (默认 50)
        ma_type: 均线类型 ('sma', 'ema', 'wma') (默认 'ema')
        offset: 均线偏移量 (百分比，默认 0)
    """
    
    def __init__(
        self,
        period: int = 50,
        ma_type: str = 'ema',
        offset: float = 0.0
    ):
        super().__init__(name="ma")
        self.period = period
        self.ma_type = ma_type.lower()
        self.offset = offset
        self.params = {
            'period': period,
            'ma_type': ma_type,
            'offset': offset
        }
    
    def calculate_ma(self, prices: pd.Series) -> pd.Series:
        """计算移动平均线"""
        if self.ma_type == 'sma':
            ma = prices.rolling(window=self.period).mean()
        elif self.ma_type == 'ema':
            ma = prices.ewm(span=self.period, adjust=False).mean()
        elif self.ma_type == 'wma':
            weights = np.arange(1, self.period + 1)
            ma = prices.rolling(window=self.period).apply(
                lambda x: np.dot(x, weights) / weights.sum(),
                raw=True
            )
        else:
            raise ValueError(f"Unknown ma_type: {self.ma_type}")
        
        return ma
    
    def calculate_stop(
        self,
        data: pd.DataFrame,
        position_type: str = 'long',
        entry_price: float = None,
        current_bar: int = None
    ) -> StoplossSignal:
        """计算基于均线的止损位"""
        if current_bar is None or current_bar < self.period:
            return StoplossSignal(
                triggered=False,
                stop_price=None,
                current_price=data['close'].iloc[-1],
                stop_type='ma',
                metadata={'reason': 'insufficient_data'}
            )
        
        current_price = data['close'].iloc[-1]
        ma_value = self.calculate_ma(data['close']).iloc[-1]
        
        if pd.isna(ma_value):
            return StoplossSignal(
                triggered=False,
                stop_price=None,
                current_price=current_price,
                stop_type='ma',
                metadata={'reason': 'invalid_ma'}
            )
        
        # 应用偏移
        if self.offset != 0:
            if position_type == 'long':
                ma_value = ma_value * (1 - self.offset)
            else:
                ma_value = ma_value * (1 + self.offset)
        
        if position_type == 'long':
            stop_price = ma_value
            triggered = current_price <= stop_price
        else:  # short
            stop_price = ma_value
            triggered = current_price >= stop_price
        
        return StoplossSignal(
            triggered=triggered,
            stop_price=stop_price,
            current_price=current_price,
            stop_type='ma',
            metadata={
                'ma_value': ma_value,
                'ma_type': self.ma_type,
                'period': self.period
            }
        )


class MaxDrawdownStoploss(BaseStoploss):
    """基于最大回撤的止损策略
    
    监控持仓期间的最大回撤（从最高价的跌幅）：
    - 当回撤超过阈值时触发止损
    - 可以保护浮动利润
    
    参数:
        max_drawdown: 最大允许回撤 (默认 0.05 即 5%)
        use_entry_price: 是否相对于入场价计算 (默认 False，相对于最高价)
    """
    
    def __init__(
        self,
        max_drawdown: float = 0.05,
        use_entry_price: bool = False
    ):
        super().__init__(name="max_drawdown")
        self.max_drawdown = max_drawdown
        self.use_entry_price = use_entry_price
        self.params = {
            'max_drawdown': max_drawdown,
            'use_entry_price': use_entry_price
        }
    
    def calculate_stop(
        self,
        data: pd.DataFrame,
        position_type: str = 'long',
        entry_price: float = None,
        current_bar: int = None
    ) -> StoplossSignal:
        """计算基于最大回撤的止损位"""
        if current_bar is None or entry_price is None:
            return StoplossSignal(
                triggered=False,
                stop_price=None,
                current_price=data['close'].iloc[-1],
                stop_type='max_drawdown',
                metadata={'reason': 'insufficient_data'}
            )
        
        # 获取持仓期间的价格序列
        position_data = data['close'].iloc[-(current_bar + 1):] if current_bar < len(data) else data['close']
        current_price = position_data.iloc[-1]
        
        if position_type == 'long':
            # 多头：计算从最高价的回撤
            if self.use_entry_price:
                peak_price = entry_price
            else:
                peak_price = position_data.max()
            
            if peak_price <= 0:
                return StoplossSignal(
                    triggered=False,
                    stop_price=None,
                    current_price=current_price,
                    stop_type='max_drawdown',
                    metadata={'reason': 'invalid_peak_price'}
                )
            
            drawdown = (peak_price - current_price) / peak_price
            stop_price = peak_price * (1 - self.max_drawdown)
            triggered = drawdown >= self.max_drawdown
            
        else:  # short
            # 空头：计算从最低价的回撤
            if self.use_entry_price:
                trough_price = entry_price
            else:
                trough_price = position_data.min()
            
            if trough_price <= 0:
                return StoplossSignal(
                    triggered=False,
                    stop_price=None,
                    current_price=current_price,
                    stop_type='max_drawdown',
                    metadata={'reason': 'invalid_trough_price'}
                )
            
            drawdown = (current_price - trough_price) / trough_price
            stop_price = trough_price * (1 + self.max_drawdown)
            triggered = drawdown >= self.max_drawdown
        
        return StoplossSignal(
            triggered=triggered,
            stop_price=stop_price,
            current_price=current_price,
            stop_type='max_drawdown',
            metadata={
                'drawdown': drawdown if 'drawdown' in dir() else None,
                'peak_price': peak_price if 'peak_price' in dir() else None,
                'max_allowed_drawdown': self.max_drawdown
            }
        )


# 工具函数
def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """计算夏普比率"""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """计算最大回撤"""
    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    return abs(drawdown.min())


def calculate_win_rate(trades: pd.DataFrame) -> float:
    """计算胜率"""
    if len(trades) == 0:
        return 0.0
    winning_trades = trades[trades['pnl'] > 0]
    return len(winning_trades) / len(trades)
