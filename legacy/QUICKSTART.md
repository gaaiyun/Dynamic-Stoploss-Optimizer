# 快速开始 - 动态止损策略优化器

## 5 分钟上手指南

### 1️⃣ 安装依赖

```bash
pip install numpy pandas
```

### 2️⃣ 导入模块

```python
from dynamic-stoploss import (
    ATRStoploss,
    StoplossBacktester
)
```

### 3️⃣ 准备数据

```python
import pandas as pd

# 读取你的数据（必须包含 open, high, low, close 列）
data = pd.read_csv('your_stock_data.csv')

# 或者使用 yfinance 下载数据
import yfinance as yf
df = yf.download('AAPL', period='2y')
data = df[['Open', 'High', 'Low', 'Close']].rename(
    columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
)
```

### 4️⃣ 创建策略

```python
# ATR 止损策略
strategy = ATRStoploss(period=14, multiplier=2.5)

# 或者其他策略
from dynamic-stoploss import VolatilityStoploss, MAStoploss, MaxDrawdownStoploss

vol_strategy = VolatilityStoploss(window=20, multiplier=2.0)
ma_strategy = MAStoploss(period=50, ma_type='ema')
dd_strategy = MaxDrawdownStoploss(max_drawdown=0.10)
```

### 5️⃣ 运行回测

```python
# 创建回测器
backtester = StoplossBacktester(
    data=data,
    initial_capital=100000,  # 初始资金
    commission=0.001,        # 佣金 0.1%
    slippage=0.001           # 滑点 0.1%
)

# 运行回测
result = backtester.run(
    strategy,
    position_type='long',    # 多头
    shares_per_trade=100     # 每笔 100 股
)

# 查看结果
print(result.summary())
```

### 6️⃣ 优化参数

```python
from dynamic-stoploss import StoplossOptimizer

# 创建优化器
optimizer = StoplossOptimizer(strategy, backtester)

# 网格搜索最优参数
best = optimizer.optimize(
    param_grid={
        'period': [10, 14, 20],
        'multiplier': [1.5, 2.0, 2.5, 3.0]
    },
    metric='sharpe_ratio'  # 优化夏普比率
)

print(f"最优参数：{best['best_params']}")
print(f"最优夏普比率：{best['best_score']:.2f}")
```

## 常用策略参数参考

### ATR 止损
- **短线交易**: period=10, multiplier=2.0
- **中线交易**: period=14, multiplier=2.5
- **长线交易**: period=20, multiplier=3.0

### 波动率止损
- **激进**: window=10, multiplier=1.5
- **稳健**: window=20, multiplier=2.0
- **保守**: window=30, multiplier=2.5

### 均线止损
- **短线**: period=20, ma_type='ema'
- **中线**: period=50, ma_type='ema'
- **长线**: period=200, ma_type='sma'

### 最大回撤止损
- **紧止损**: max_drawdown=0.05 (5%)
- **中等**: max_drawdown=0.10 (10%)
- **宽松**: max_drawdown=0.15 (15%)

## 完整示例

```python
import pandas as pd
import yfinance as yf
from dynamic-stoploss import ATRStoploss, StoplossBacktester

# 1. 下载数据
print("下载数据...")
df = yf.download('AAPL', period='2y')
data = df[['Open', 'High', 'Low', 'Close']].rename(
    columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'}
)

# 2. 创建策略
strategy = ATRStoploss(period=14, multiplier=2.5)

# 3. 回测
backtester = StoplossBacktester(data, initial_capital=100000)
result = backtester.run(strategy, shares_per_trade=10)

# 4. 输出结果
print(result.summary())

# 5. 优化
from dynamic-stoploss import StoplossOptimizer
optimizer = StoplossOptimizer(strategy, backtester)
best = optimizer.optimize(
    param_grid={
        'period': [10, 14, 20],
        'multiplier': [2.0, 2.5, 3.0]
    },
    metric='sharpe_ratio'
)
print(f"\n最优参数：{best['best_params']}")
```

## 常见问题

### Q: 数据格式要求？
A: DataFrame 必须包含 `open`, `high`, `low`, `close` 四列，可选 `volume` 和`date` 列。

### Q: 如何结合自己的交易策略？
A: 在 `backtester.run()` 中传入 `entry_signal` 参数，指定你的入场信号。

### Q: 回测结果为负怎么办？
A: 尝试调整止损参数，或结合更好的入场策略。止损只是风险管理工具，不能单独盈利。

### Q: 如何保存回测结果？
A: `result.to_dict()` 可转换为字典，然后保存为 CSV 或 JSON。

## 下一步

- 📖 查看 `README.md` 了解完整 API
- 📝 运行 `example_usage.py` 查看更多示例
- 🧪 运行 `test_stoploss.py` 了解测试方法
- 🔧 修改参数，测试你自己的策略！

---

**✨ 派蒙提示：止损是保护本金的关键！但也要记住，好的止损策略需要配合好的入场策略哦~**
