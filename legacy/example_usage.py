"""
动态止损策略优化器 - 使用示例

演示止损策略工具包的常见用法。
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 导入止损策略模块
try:
    from .stoploss_strategies import (
        VolatilityStoploss,
        ATRStoploss,
        MAStoploss,
        MaxDrawdownStoploss
    )
    from .backtester import StoplossBacktester
    from .optimizer import StoplossOptimizer
except ImportError:
    from stoploss_strategies import (
        VolatilityStoploss,
        ATRStoploss,
        MAStoploss,
        MaxDrawdownStoploss
    )
    from backtester import StoplossBacktester
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
    
    data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
        'high': prices * (1 + np.random.uniform(0, 0.02, n_days)),
        'low': prices * (1 - np.random.uniform(0, 0.02, n_days)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, n_days)
    })
    
    data['high'] = data[['high', 'close', 'open']].max(axis=1)
    data['low'] = data[['low', 'close', 'open']].min(axis=1)
    
    return data


def example_1_basic_stoploss():
    """示例 1：基本止损策略使用"""
    print("\n" + "="*60)
    print("示例 1：基本止损策略使用")
    print("="*60)
    
    # 准备数据
    data = generate_sample_data(n_days=252)
    
    # 创建止损策略
    vol_stop = VolatilityStoploss(window=20, multiplier=2.0)
    atr_stop = ATRStoploss(period=14, multiplier=2.5)
    ma_stop = MAStoploss(period=50, ma_type='ema')
    dd_stop = MaxDrawdownStoploss(max_drawdown=0.10)
    
    # 计算止损信号
    entry_price = 100.0
    current_bar = 100
    
    print("\n当前价格：${:.2f}".format(data['close'].iloc[current_bar]))
    print(f"入场价格：${entry_price}")
    
    strategies = [vol_stop, atr_stop, ma_stop, dd_stop]
    
    for strategy in strategies:
        signal = strategy.calculate_stop(
            data=data.iloc[:current_bar+1],
            position_type='long',
            entry_price=entry_price,
            current_bar=current_bar
        )
        
        print(f"\n{strategy.name.upper()} 止损策略:")
        print(f"  止损价格：${signal.stop_price:.2f}" if signal.stop_price else "  止损价格：N/A")
        print(f"  是否触发：{'是' if signal.triggered else '否'}")
        if signal.metadata:
            print(f"  详细信息：{signal.metadata}")


def example_2_backtesting():
    """示例 2：回测止损策略"""
    print("\n" + "="*60)
    print("示例 2：回测止损策略")
    print("="*60)
    
    # 准备数据
    data = generate_sample_data(n_days=252)
    
    # 创建回测器
    backtester = StoplossBacktester(
        data=data,
        initial_capital=100000,
        commission=0.001,
        slippage=0.001
    )
    
    # 创建策略
    strategy = ATRStoploss(period=14, multiplier=2.5)
    
    # 运行回测
    print("\n开始回测...")
    result = backtester.run(
        strategy=strategy,
        position_type='long',
        shares_per_trade=100,
        max_positions=1
    )
    
    # 打印结果
    print(result.summary())
    
    # 访问详细数据
    print("\n详细指标:")
    print(f"  最终资金：${result.final_capital:,.2f}")
    print(f"  盈利交易：{result.winning_trades} 笔")
    print(f"  亏损交易：{result.losing_trades} 笔")
    print(f"  平均持仓：{result.average_holding_period:.1f} 天")
    print(f"  最大连续盈利：{result.max_consecutive_wins} 次")
    print(f"  最大连续亏损：{result.max_consecutive_losses} 次")


def example_3_strategy_comparison():
    """示例 3：对比不同止损策略"""
    print("\n" + "="*60)
    print("示例 3：对比不同止损策略")
    print("="*60)
    
    # 准备数据
    data = generate_sample_data(n_days=252)
    backtester = StoplossBacktester(data, initial_capital=100000)
    
    # 创建多个策略
    strategies = [
        VolatilityStoploss(window=20, multiplier=2.0),
        VolatilityStoploss(window=20, multiplier=2.5),
        ATRStoploss(period=14, multiplier=2.0),
        ATRStoploss(period=14, multiplier=2.5),
        ATRStoploss(period=14, multiplier=3.0),
        MAStoploss(period=20, ma_type='sma'),
        MAStoploss(period=50, ma_type='ema'),
        MaxDrawdownStoploss(max_drawdown=0.05),
        MaxDrawdownStoploss(max_drawdown=0.10)
    ]
    
    # 运行回测并对比
    print("\n策略对比结果:\n")
    
    results = []
    for strategy in strategies:
        result = backtester.run(strategy, shares_per_trade=100)
        results.append({
            '策略': f"{strategy.name} ({strategy.get_params()})",
            '总收益': f"{result.total_return:.2%}",
            '夏普比率': f"{result.sharpe_ratio:.2f}",
            '最大回撤': f"{result.max_drawdown:.2%}",
            '胜率': f"{result.win_rate:.2%}",
            '交易次数': result.total_trades
        })
    
    # 转换为 DataFrame 并排序
    df = pd.DataFrame(results)
    df = df.sort_values('夏普比率', ascending=False, key=lambda x: x.str.replace('%', '').astype(float) if x.dtype == 'object' else x)
    
    print(df.to_string(index=False))


def example_4_parameter_optimization():
    """示例 4：参数优化"""
    print("\n" + "="*60)
    print("示例 4：参数优化")
    print("="*60)
    
    # 准备数据
    data = generate_sample_data(n_days=252)
    backtester = StoplossBacktester(data, initial_capital=100000)
    
    # 创建策略
    strategy = ATRStoploss(period=14, multiplier=2.5)
    
    # 创建优化器
    optimizer = StoplossOptimizer(strategy, backtester, n_jobs=1)
    
    # 网格搜索
    print("\n开始网格搜索优化...")
    param_grid = {
        'period': [10, 14, 20],
        'multiplier': [1.5, 2.0, 2.5, 3.0]
    }
    
    result = optimizer.optimize(
        param_grid=param_grid,
        metric='sharpe_ratio',
        maximize=True,
        verbose=True
    )
    
    print("\n最优参数:")
    print(f"  period: {result['best_params']['period']}")
    print(f"  multiplier: {result['best_params']['multiplier']}")
    print(f"  夏普比率：{result['best_score']:.2f}")
    
    # 获取优化历史
    summary_df = optimizer.get_optimization_summary()
    print("\n所有参数组合表现:")
    print(summary_df.head(10).to_string(index=False))


def example_5_custom_entry_signal():
    """示例 5：自定义入场信号"""
    print("\n" + "="*60)
    print("示例 5：自定义入场信号")
    print("="*60)
    
    # 准备数据
    data = generate_sample_data(n_days=252)
    
    # 创建简单的均线交叉入场信号
    data['sma_20'] = data['close'].rolling(20).mean()
    data['sma_50'] = data['close'].rolling(50).mean()
    
    # 金叉入场：20 日均线上穿 50 日均线
    entry_signal = (
        (data['sma_20'] > data['sma_50']) & 
        (data['sma_20'].shift(1) <= data['sma_50'].shift(1))
    )
    
    print(f"\n入场信号数量：{entry_signal.sum()} 次")
    
    # 回测
    backtester = StoplossBacktester(data, initial_capital=100000)
    strategy = ATRStoploss(period=14, multiplier=2.5)
    
    result = backtester.run(
        strategy=strategy,
        entry_signal=entry_signal,
        shares_per_trade=100
    )
    
    print(result.summary())


def example_6_real_data():
    """示例 6：使用真实数据（需要 yfinance）"""
    print("\n" + "="*60)
    print("示例 6：使用真实数据")
    print("="*60)
    
    try:
        import yfinance as yf
        
        # 下载数据
        print("\n下载 AAPL 数据...")
        ticker = yf.Ticker("AAPL")
        df = ticker.history(period="2y", interval="1d")
        
        if df.empty:
            print("无法获取数据，使用模拟数据演示")
            df = generate_sample_data(n_days=504)
        
        # 重命名列
        data = df.rename(columns=str.lower)[['open', 'high', 'low', 'close', 'volume']]
        data = data.reset_index()
        data = data.rename(columns={'date': 'date'})
        
        print(f"数据加载成功：{len(data)} 天")
        
        # 回测
        backtester = StoplossBacktester(data, initial_capital=100000)
        strategy = ATRStoploss(period=14, multiplier=2.5)
        
        result = backtester.run(strategy, shares_per_trade=10)
        print(result.summary())
        
    except ImportError:
        print("\n未安装 yfinance，使用模拟数据演示")
        print("安装方法：pip install yfinance")
        
        # 使用模拟数据
        data = generate_sample_data(n_days=504)
        backtester = StoplossBacktester(data, initial_capital=100000)
        strategy = ATRStoploss(period=14, multiplier=2.5)
        
        result = backtester.run(strategy, shares_per_trade=100)
        print(result.summary())


def example_7_visualization():
    """示例 7：可视化（需要 matplotlib）"""
    print("\n" + "="*60)
    print("示例 7：可视化权益曲线")
    print("="*60)
    
    try:
        import matplotlib.pyplot as plt
        
        # 准备数据
        data = generate_sample_data(n_days=252)
        backtester = StoplossBacktester(data, initial_capital=100000)
        
        # 运行多个策略
        strategies = [
            ATRStoploss(period=14, multiplier=2.0),
            ATRStoploss(period=14, multiplier=2.5),
            ATRStoploss(period=14, multiplier=3.0)
        ]
        
        results = []
        for strategy in strategies:
            result = backtester.run(strategy)
            results.append(result)
        
        # 绘制权益曲线
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for result in results:
            ax.plot(
                result.equity_curve.index,
                result.equity_curve.values,
                label=f"{result.strategy_name} (Sharpe: {result.sharpe_ratio:.2f})",
                linewidth=2
            )
        
        ax.axhline(y=100000, color='gray', linestyle='--', label='Initial Capital')
        ax.set_title('Stoploss Strategy Comparison', fontsize=14)
        ax.set_xlabel('Date')
        ax.set_ylabel('Capital ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('stoploss_comparison.png', dpi=300, bbox_inches='tight')
        print("图表已保存：stoploss_comparison.png")
        plt.show()
        
    except ImportError:
        print("\n未安装 matplotlib")
        print("安装方法：pip install matplotlib")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("动态止损策略优化器 - 使用示例")
    print("=" * 60)
    
    # 运行示例
    example_1_basic_stoploss()
    example_2_backtesting()
    example_3_strategy_comparison()
    example_4_parameter_optimization()
    example_5_custom_entry_signal()
    example_6_real_data()
    # example_7_visualization()  # 可选，需要 matplotlib
    
    print("\n" + "="*60)
    print("示例演示完成")
    print("="*60)
    print("\n提示:")
    print("  - 修改参数来测试不同的止损策略")
    print("  - 使用真实数据进行回测")
    print("  - 结合自己的交易策略使用")
    print("  - 注意过拟合风险，在不同市场环境下测试")
    print("\n更多信息请查看 SKILL.md 和源代码")
    print("\n")


if __name__ == '__main__':
    main()
