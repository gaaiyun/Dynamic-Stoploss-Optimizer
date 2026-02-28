# Dynamic Stoploss Optimizer - 动态止损策略优化器

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**派蒙的最强止损策略工具包！** 🧙‍♀️⭐

本工具包提供多种动态止损策略的实现和回测功能，帮助量化交易者优化止损策略，降低交易风险。

---

## 🌟 功能特性

### 核心止损策略

| 策略名称 | 描述 | 适用场景 |
|---------|------|---------|
| **VolatilityStoploss** | 基于波动率的动态止损 | 趋势跟踪策略 |
| **ATRStoploss** | 基于平均真实波幅 (ATR) 的止损 | 广泛适用于各种策略 |
| **MAStoploss** | 基于移动平均线的止损 | 趋势跟随策略 |
| **MaxDrawdownStoploss** | 基于最大回撤的止损 | 保护浮动利润 |

### 回测模块

- ✅ 完整的历史回测功能
- ✅ 详细的性能指标（夏普比率、最大回撤、胜率等）
- ✅ 交易记录追踪
- ✅ 权益曲线生成
- ✅ 多策略对比

### 参数优化

- ✅ 网格搜索优化
- ✅ 随机搜索优化
- ✅ 并行计算支持
- ✅ 优化结果可视化

---

## 📦 安装

### 依赖

```bash
pip install numpy pandas matplotlib scipy
```

### 可选依赖（用于真实数据）

```bash
pip install yfinance
```

---

## 🚀 快速开始

### 1. 基本使用

```python
from dynamic-stoploss import (
    ATRStoploss,
    StoplossBacktester
)
import pandas as pd

# 准备数据 (需要包含 open, high, low, close 列)
data = pd.read_csv('your_data.csv')

# 创建止损策略
strategy = ATRStoploss(period=14, multiplier=2.5)

# 创建回测器
backtester = StoplossBacktester(
    data=data,
    initial_capital=100000,
    commission=0.001,
    slippage=0.001
)

# 运行回测
result = backtester.run(strategy, position_type='long', shares_per_trade=100)

# 查看结果
print(result.summary())
```

### 2. 对比多个策略

```python
from dynamic-stoploss import (
    VolatilityStoploss,
    ATRStoploss,
    MAStoploss,
    MaxDrawdownStoploss
)

strategies = [
    VolatilityStoploss(window=20, multiplier=2.0),
    ATRStoploss(period=14, multiplier=2.5),
    MAStoploss(period=50, ma_type='ema'),
    MaxDrawdownStoploss(max_drawdown=0.10)
]

# 对比所有策略
comparison_df = backtester.compare_strategies(strategies)
print(comparison_df)
```

### 3. 参数优化

```python
from dynamic-stoploss import StoplossOptimizer

# 创建优化器
optimizer = StoplossOptimizer(strategy, backtester)

# 网格搜索
best_params = optimizer.optimize(
    param_grid={
        'period': [10, 14, 20],
        'multiplier': [1.5, 2.0, 2.5, 3.0]
    },
    metric='sharpe_ratio'
)

print(f"最优参数：{best_params['best_params']}")
```

---

## 📊 输出指标

回测结果包含以下关键指标：

| 指标 | 说明 |
|-----|------|
| **总收益率 (Total Return)** | 整个回测期间的总收益百分比 |
| **年化收益率 (Annualized Return)** | 年化后的收益率 |
| **夏普比率 (Sharpe Ratio)** | 风险调整后收益，越高越好 |
| **最大回撤 (Max Drawdown)** | 最大资金回撤幅度 |
| **胜率 (Win Rate)** | 盈利交易占比 |
| **盈亏比 (Profit Factor)** | 总盈利/总亏损 |
| **平均交易盈亏** | 平均每笔交易的盈亏金额 |
| **平均持仓周期** | 平均持仓天数 |
| **最大连续盈利/亏损** | 最大连续盈利/亏损次数 |

---

## 📚 示例

### 示例 1：ATR 止损策略

```python
from dynamic-stoploss import ATRStoploss, StoplossBacktester

# 创建 ATR 止损策略
strategy = ATRStoploss(period=14, multiplier=2.5)

# 回测
backtester = StoplossBacktester(data, initial_capital=100000)
result = backtester.run(strategy)

print(result.summary())
```

### 示例 2：自定义入场信号

```python
# 创建均线交叉入场信号
data['sma_20'] = data['close'].rolling(20).mean()
data['sma_50'] = data['close'].rolling(50).mean()

entry_signal = (data['sma_20'] > data['sma_50']) & \
               (data['sma_20'].shift(1) <= data['sma_50'].shift(1))

# 使用自定义信号回测
result = backtester.run(
    strategy,
    entry_signal=entry_signal,
    shares_per_trade=100
)
```

### 示例 3：参数优化

```python
optimizer = StoplossOptimizer(strategy, backtester)

# 随机搜索
result = optimizer.random_search(
    param_distributions={
        'period': (10, 30),
        'multiplier': (1.5, 3.5)
    },
    n_iter=50,
    metric='sharpe_ratio'
)

print(f"最优夏普比率：{result['best_score']:.2f}")
```

---

## 🧪 测试

运行测试套件：

```bash
cd skills/dynamic-stoploss
python test_stoploss.py
```

或使用 pytest：

```bash
pytest test_stoploss.py -v
```

---

## 📖 API 文档

### 止损策略类

#### `BaseStoploss` (抽象基类)

所有止损策略的基类。

**方法:**
- `calculate_stop(data, position_type, entry_price, current_bar)` - 计算止损信号
- `update_params(**kwargs)` - 更新策略参数
- `get_params()` - 获取当前参数

#### `VolatilityStoploss`

基于波动率的动态止损。

**参数:**
- `window` (int): 计算波动率的时间窗口，默认 20
- `multiplier` (float): 波动率乘数，默认 2.0
- `use_log_returns` (bool): 是否使用对数收益率，默认 True

#### `ATRStoploss`

基于 ATR 的止损。

**参数:**
- `period` (int): ATR 计算周期，默认 14
- `multiplier` (float): ATR 乘数，默认 2.5
- `use_current_atr` (bool): 是否使用当前 ATR 值，默认 True

#### `MAStoploss`

基于移动平均线的止损。

**参数:**
- `period` (int): 均线周期，默认 50
- `ma_type` (str): 均线类型 ('sma', 'ema', 'wma')，默认 'ema'
- `offset` (float): 均线偏移量（百分比），默认 0

#### `MaxDrawdownStoploss`

基于最大回撤的止损。

**参数:**
- `max_drawdown` (float): 最大允许回撤，默认 0.05
- `use_entry_price` (bool): 是否相对于入场价计算，默认 False

### 回测器类

#### `StoplossBacktester`

**参数:**
- `data` (pd.DataFrame): OHLC 数据
- `initial_capital` (float): 初始资金，默认 100000
- `commission` (float): 交易佣金比例，默认 0.001
- `slippage` (float): 滑点比例，默认 0.001

**方法:**
- `run(strategy, position_type, entry_signal, exit_target, shares_per_trade, max_positions)` - 运行回测
- `compare_strategies(strategies, **kwargs)` - 对比多个策略
- `plot_equity_curve(result, title, save_path)` - 绘制权益曲线

### 优化器类

#### `StoplossOptimizer`

**参数:**
- `strategy` (BaseStoploss): 止损策略实例
- `backtester` (StoplossBacktester): 回测器实例
- `n_jobs` (int): 并行工作进程数，默认 -1

**方法:**
- `optimize(param_grid, metric, maximize, n_folds, verbose)` - 网格搜索优化
- `random_search(param_distributions, n_iter, metric, maximize, verbose)` - 随机搜索
- `plot_optimization_results(param1, param2, metric, save_path)` - 绘制优化结果
- `get_optimization_summary()` - 获取优化摘要

---

## ⚠️ 注意事项

1. **历史回测不代表未来表现** - 回测结果仅供参考
2. **避免过拟合** - 在不同市场环境下测试策略
3. **参数敏感性** - 止损参数对结果影响很大，需谨慎优化
4. **交易成本** - 回测中已考虑佣金和滑点，但实际交易可能不同
5. **风险管理** - 止损只是风险管理的一部分，需结合仓位管理

---

## 📝 更新日志

### v1.0.0 (2026-02-28)
- ✨ 初始版本发布
- ✅ 实现 4 种动态止损策略
- ✅ 完整的回测功能
- ✅ 参数优化模块
- ✅ 测试和文档

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

## 📄 许可证

MIT License

---

## 👤 作者

**派蒙 ⭐** - 提瓦特最强的量化助手！

---

## 🙏 致谢

感谢以下项目提供的灵感：
- [pybroker](https://github.com/edwardlee91/pybroker) - Python 算法交易框架
- [ML for Trading Book](https://github.com/stefan-jansen/machine-learning-for-trading) - 机器学习交易书籍

---

**✨ 派蒙提示：止损是交易成功的关键！合理使用止损策略，保护你的本金~**
