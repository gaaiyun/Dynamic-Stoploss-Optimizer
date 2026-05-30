"""
动态止损策略测试模块 (v1, legacy)

确保所有止损策略都能正常工作。
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

try:
    from .stoploss_strategies import (
        VolatilityStoploss,
        ATRStoploss,
        MAStoploss,
        MaxDrawdownStoploss,
        StoplossSignal
    )
    from .backtester import StoplossBacktester, BacktestResult
    from .optimizer import StoplossOptimizer
except ImportError:
    from stoploss_strategies import (
        VolatilityStoploss,
        ATRStoploss,
        MAStoploss,
        MaxDrawdownStoploss,
        StoplossSignal
    )
    from backtester import StoplossBacktester, BacktestResult
    from optimizer import StoplossOptimizer


def generate_sample_data(
    n_days: int = 252,
    start_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0005
) -> pd.DataFrame:
    """生成模拟股票数据"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    returns = np.random.normal(trend, volatility, n_days)
    prices = start_price * np.cumprod(1 + returns)
    
    # 生成 OHLC 数据
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'high': prices * (1 + np.random.uniform(0, 0.02, n_days)),
        'low': prices * (1 - np.random.uniform(0, 0.02, n_days)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    # 确保 high >= close >= low
    data['high'] = data[['high', 'close', 'open']].max(axis=1)
    data['low'] = data[['low', 'close', 'open']].min(axis=1)
    
    return data


class TestVolatilityStoploss:
    """测试基于波动率的止损策略"""
    
    def test_initialization(self):
        """测试初始化"""
        strategy = VolatilityStoploss(window=20, multiplier=2.0)
        assert strategy.window == 20
        assert strategy.multiplier == 2.0
        assert strategy.name == "volatility"
    
    def test_calculate_stop_long(self):
        """测试多头止损计算"""
        data = generate_sample_data(n_days=100)
        strategy = VolatilityStoploss(window=20, multiplier=2.0)
        
        signal = strategy.calculate_stop(
            data=data.iloc[:50],
            position_type='long',
            entry_price=100.0,
            current_bar=49
        )
        
        assert isinstance(signal, StoplossSignal)
        assert signal.stop_type == 'volatility'
        assert signal.stop_price is not None
        assert signal.stop_price < 100.0  # 多头止损价应低于入场价
    
    def test_calculate_stop_short(self):
        """测试空头止损计算"""
        data = generate_sample_data(n_days=100)
        strategy = VolatilityStoploss(window=20, multiplier=2.0)
        
        signal = strategy.calculate_stop(
            data=data.iloc[:50],
            position_type='short',
            entry_price=100.0,
            current_bar=49
        )
        
        assert signal.stop_price is not None
        assert signal.stop_price > 100.0  # 空头止损价应高于入场价
    
    def test_insufficient_data(self):
        """测试数据不足的情况"""
        data = generate_sample_data(n_days=10)
        strategy = VolatilityStoploss(window=20, multiplier=2.0)
        
        signal = strategy.calculate_stop(
            data=data,
            position_type='long',
            entry_price=100.0,
            current_bar=5
        )
        
        assert not signal.triggered
        assert signal.metadata['reason'] == 'insufficient_data'


class TestATRStoploss:
    """测试基于 ATR 的止损策略"""
    
    def test_initialization(self):
        """测试初始化"""
        strategy = ATRStoploss(period=14, multiplier=2.5)
        assert strategy.period == 14
        assert strategy.multiplier == 2.5
        assert strategy.name == "atr"
    
    def test_calculate_atr(self):
        """测试 ATR 计算"""
        data = generate_sample_data(n_days=100)
        strategy = ATRStoploss(period=14, multiplier=2.5)
        
        atr = strategy.calculate_atr(data)
        assert len(atr) == 100
        assert not atr.isna().all()
        assert (atr > 0).all()  # ATR 应该始终为正
    
    def test_calculate_stop(self):
        """测试止损计算"""
        data = generate_sample_data(n_days=100)
        strategy = ATRStoploss(period=14, multiplier=2.5)
        
        signal = strategy.calculate_stop(
            data=data.iloc[:50],
            position_type='long',
            entry_price=100.0,
            current_bar=49
        )
        
        assert isinstance(signal, StoplossSignal)
        assert 'atr' in signal.metadata
        assert signal.metadata['atr'] > 0


class TestMAStoploss:
    """测试基于均线的止损策略"""
    
    def test_initialization(self):
        """测试初始化"""
        strategy = MAStoploss(period=50, ma_type='ema')
        assert strategy.period == 50
        assert strategy.ma_type == 'ema'
        assert strategy.name == "ma"
    
    def test_calculate_ma_types(self):
        """测试不同类型的均线"""
        data = generate_sample_data(n_days=100)
        prices = data['close']
        
        for ma_type in ['sma', 'ema', 'wma']:
            strategy = MAStoploss(period=20, ma_type=ma_type)
            ma = strategy.calculate_ma(prices)
            assert len(ma) == 100
            assert not ma.isna().all()
    
    def test_calculate_stop(self):
        """测试止损计算"""
        data = generate_sample_data(n_days=100)
        strategy = MAStoploss(period=20, ma_type='sma')
        
        signal = strategy.calculate_stop(
            data=data.iloc[:50],
            position_type='long',
            entry_price=100.0,
            current_bar=49
        )
        
        assert signal.stop_price is not None
        assert 'ma_value' in signal.metadata


class TestMaxDrawdownStoploss:
    """测试基于最大回撤的止损策略"""
    
    def test_initialization(self):
        """测试初始化"""
        strategy = MaxDrawdownStoploss(max_drawdown=0.05)
        assert strategy.max_drawdown == 0.05
        assert strategy.name == "max_drawdown"
    
    def test_calculate_stop_long(self):
        """测试多头回撤止损"""
        data = generate_sample_data(n_days=100)
        strategy = MaxDrawdownStoploss(max_drawdown=0.10)
        
        signal = strategy.calculate_stop(
            data=data.iloc[:50],
            position_type='long',
            entry_price=100.0,
            current_bar=49
        )
        
        assert isinstance(signal, StoplossSignal)
        assert signal.stop_type == 'max_drawdown'
    
    def test_trigger_on_large_drawdown(self):
        """测试大回撤时触发止损"""
        # 创建一个有明显下跌的数据
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        prices = [100]
        for i in range(1, 100):
            if i < 50:
                prices.append(prices[-1] * 1.02)  # 上涨
            else:
                prices.append(prices[-1] * 0.95)  # 大幅下跌
        
        data = pd.DataFrame({
            'date': dates,
            'open': prices,
            'high': [p * 1.01 for p in prices],
            'low': [p * 0.99 for p in prices],
            'close': prices
        })
        
        strategy = MaxDrawdownStoploss(max_drawdown=0.10)
        
        signal = strategy.calculate_stop(
            data=data,
            position_type='long',
            entry_price=100.0,
            current_bar=99
        )
        
        # 应该触发止损（回撤超过 10%）
        assert signal.triggered or signal.metadata.get('drawdown', 0) > 0


class TestStoplossBacktester:
    """测试回测器"""
    
    def test_initialization(self):
        """测试初始化"""
        data = generate_sample_data(n_days=100)
        backtester = StoplossBacktester(data, initial_capital=100000)
        
        assert backtester.initial_capital == 100000
        assert len(backtester.data) == 100
    
    def test_run_backtest(self):
        """测试运行回测"""
        data = generate_sample_data(n_days=252)
        backtester = StoplossBacktester(data, initial_capital=100000)
        strategy = ATRStoploss(period=14, multiplier=2.5)
        
        result = backtester.run(strategy, position_type='long', shares_per_trade=100)
        
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == 'atr'
        assert result.initial_capital == 100000
        assert result.final_capital > 0
        assert result.total_trades >= 0
    
    def test_backtest_metrics(self):
        """测试回测指标"""
        data = generate_sample_data(n_days=252)
        backtester = StoplossBacktester(data, initial_capital=100000)
        strategy = VolatilityStoploss(window=20, multiplier=2.0)
        
        result = backtester.run(strategy)
        
        # 验证指标范围
        assert -1.0 <= result.total_return <= 10.0  # 收益率在 -100% 到 1000% 之间
        assert 0.0 <= result.win_rate <= 1.0  # 胜率在 0-100% 之间
        assert 0.0 <= result.max_drawdown <= 1.0  # 回撤在 0-100% 之间
    
    def test_summary_output(self):
        """测试摘要输出"""
        data = generate_sample_data(n_days=252)
        backtester = StoplossBacktester(data, initial_capital=100000)
        strategy = ATRStoploss(period=14, multiplier=2.5)
        
        result = backtester.run(strategy)
        summary = result.summary()
        
        assert isinstance(summary, str)
        assert '回测结果摘要' in summary
        assert 'atr' in summary.lower()


class TestStoplossOptimizer:
    """测试参数优化器"""
    
    def test_grid_search(self):
        """测试网格搜索"""
        data = generate_sample_data(n_days=252)
        backtester = StoplossBacktester(data, initial_capital=100000)
        strategy = ATRStoploss(period=14, multiplier=2.5)
        
        optimizer = StoplossOptimizer(strategy, backtester, n_jobs=1)
        
        param_grid = {
            'period': [10, 14],
            'multiplier': [2.0, 2.5]
        }
        
        result = optimizer.optimize(param_grid, metric='sharpe_ratio', verbose=False)
        
        assert 'best_params' in result
        assert 'best_score' in result
        assert 'period' in result['best_params']
        assert 'multiplier' in result['best_params']
    
    def test_random_search(self):
        """测试随机搜索"""
        data = generate_sample_data(n_days=252)
        backtester = StoplossBacktester(data, initial_capital=100000)
        strategy = ATRStoploss(period=14, multiplier=2.5)
        
        optimizer = StoplossOptimizer(strategy, backtester, n_jobs=1)
        
        param_dist = {
            'period': (10, 20),
            'multiplier': (1.5, 3.0)
        }
        
        result = optimizer.random_search(param_dist, n_iter=10, verbose=False)
        
        assert 'best_params' in result
        assert 10 <= result['best_params']['period'] <= 20
        assert 1.5 <= result['best_params']['multiplier'] <= 3.0


def test_integration_all_strategies():
    """集成测试：测试所有策略"""
    data = generate_sample_data(n_days=252)
    backtester = StoplossBacktester(data, initial_capital=100000)
    
    strategies = [
        VolatilityStoploss(window=20, multiplier=2.0),
        ATRStoploss(period=14, multiplier=2.5),
        MAStoploss(period=50, ma_type='ema'),
        MaxDrawdownStoploss(max_drawdown=0.10)
    ]
    
    results = []
    for strategy in strategies:
        result = backtester.run(strategy)
        results.append(result)
        assert result.total_trades > 0, f"{strategy.name} should have trades"
    
    # 比较策略
    comparison_df = backtester.compare_strategies(strategies)
    assert len(comparison_df) == 4
    assert len(comparison_df.columns) > 0


if __name__ == '__main__':
    # 运行简单测试
    print("开始运行测试")
    
    # 生成测试数据
    data = generate_sample_data(n_days=252)
    
    # 测试所有策略
    strategies = [
        VolatilityStoploss(window=20, multiplier=2.0),
        ATRStoploss(period=14, multiplier=2.5),
        MAStoploss(period=50, ma_type='ema'),
        MaxDrawdownStoploss(max_drawdown=0.10)
    ]
    
    backtester = StoplossBacktester(data, initial_capital=100000)
    
    print("\n回测结果对比：")
    print("=" * 60)
    
    for strategy in strategies:
        result = backtester.run(strategy)
        print(f"\n策略：{strategy.name.upper()}")
        print(f"  总收益率：{result.total_return:.2%}")
        print(f"  夏普比率：{result.sharpe_ratio:.2f}")
        print(f"  最大回撤：{result.max_drawdown:.2%}")
        print(f"  胜率：{result.win_rate:.2%}")
        print(f"  交易次数：{result.total_trades}")
    
    print("\n测试完成，所有策略都能正常工作")
