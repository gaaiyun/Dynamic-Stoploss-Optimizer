# Dynamic Stoploss Optimizer - 动态止损策略优化器

## 技能描述

本技能提供多种动态止损策略的实现和回测功能，帮助量化交易者优化止损策略，降低交易风险。

## 功能特性

### 核心止损策略

1. **基于波动率的动态止损 (Volatility-based Stoploss)**
   - 使用历史波动率动态调整止损幅度
   - 波动率高时放宽止损，波动率低时收紧止损
   - 适合趋势跟踪策略

2. **基于 ATR 的止损 (ATR-based Stoploss)**
   - 使用平均真实波幅 (Average True Range) 计算止损位
   - 自适应市场波动性
   - 广泛应用于趋势跟踪系统

3. **基于移动均线的止损 (Moving Average-based Stoploss)**
   - 使用 SMA/EMA 作为动态止损线
   - 支持多种均线周期
   - 适合趋势跟随策略

4. **基于最大回撤的止损 (Max Drawdown-based Stoploss)**
   - 监控持仓期间的最大回撤
   - 当回撤超过阈值时触发止损
   - 保护利润，限制损失

### 回测模块

- 支持历史数据回测
- 提供详细的性能指标
- 可视化止损触发点
- 对比不同止损策略效果

## 使用方法

### 基本使用

```python
from skills.dynamic-stoploss import (
    VolatilityStoploss,
    ATRStoploss,
    MAStoploss,
    MaxDrawdownStoploss,
    StoplossBacktester
)

# 创建止损策略
vol_stop = VolatilityStoploss(window=20, multiplier=2.0)
atr_stop = ATRStoploss(period=14, multiplier=2.5)
ma_stop = MAStoploss(period=50, ma_type='ema')
dd_stop = MaxDrawdownStoploss(max_dd=0.05)

# 回测
backtester = StoplossBacktester(data=df, initial_capital=100000)
results = backtester.run(strategy=vol_stop)
print(results.summary())
```

### 参数优化

```python
# 优化止损参数
optimizer = StoplossOptimizer(strategy=atr_stop)
best_params = optimizer.optimize(
    param_grid={
        'period': [10, 14, 20],
        'multiplier': [1.5, 2.0, 2.5, 3.0]
    },
    metric='sharpe_ratio'
)
```

## 输出指标

- 总收益率 (Total Return)
- 年化收益率 (Annualized Return)
- 夏普比率 (Sharpe Ratio)
- 最大回撤 (Max Drawdown)
- 胜率 (Win Rate)
- 盈亏比 (Profit/Loss Ratio)
- 平均持仓时间 (Average Holding Period)
- 止损触发次数 (Stoploss Trigger Count)

## 依赖

见 `requirements.txt`

## 注意事项

- 历史回测结果不代表未来表现
- 止损策略需结合具体交易策略调整
- 建议在不同市场环境下测试策略鲁棒性
- 注意过拟合风险

## 作者

派蒙 ⭐ - 提瓦特最强的量化助手！
