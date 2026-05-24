# Dynamic Stoploss Optimizer (v2)

研究级动态停损实验室：**8 个学界 + 实务界已成熟的停损策略**，按 **市场状态 (regime)
推荐 + walk-forward 优化 + triple-barrier 回测 + Deflated Sharpe Ratio** 全套
框架，专门解决"哪个停损在我的数据上真好用"这个问题。

## 与现有库对比

大部分回测库（freqtrade / backtrader / vectorbt / lumibot / pybroker / QuantConnect）
都让你选一个固定的停损策略然后跑回测。本仓库不一样：

| 现有库的做法 | 我们的做法 |
|---|---|
| 选一个停损，比如 ATR×2.5 | 跑 **8 个停损** 在同一份数据上，按 Sharpe 排 |
| Grid search 在全样本上选最优 | **Walk-forward** 滚动训练 / 样本外评估，给出 overfitting gap |
| 最高 Sharpe = 胜出 | 加 **Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)** 多重检验校正 |
| 用户自己判断哪种行情用哪个停损 | **ADX + Hurst 自动识别 regime**，推荐对应停损 |
| P/L 差就归咎"停损不好" | **P/L 归因**：拆出入场 vs 停损各占多少 |

文献支撑见 [`docs/RESEARCH_NOTES.md`](./docs/RESEARCH_NOTES.md)。

## 8 个内置停损策略

| 策略 | 算法来源 | 适用场景 |
|---|---|---|
| `atr` | Wilder (1978) | 通用 trailing |
| `chandelier` | Le Beau (1992) | 趋势跟踪 |
| `supertrend` | Olivier Seban (社区) | 趋势 + 反转 |
| `parabolic_sar` | Wilder (1978) | 趋势加速 |
| `donchian` | Turtle Traders / Faith (2007) | 横盘 / 通道突破 |
| `moving_average` | 经典 | 趋势确认 |
| `volatility` | 滚动 stdev | 波动率自适应 |
| `max_drawdown` | 朴素 trailing | 浮盈保护 |
| `time` | Lopez de Prado (2018) vertical barrier | 防止"枯坐" |

## 安装

```bash
pip install pandas numpy scipy
# 可选：yfinance 真数据
pip install yfinance
```

## 快速开始

### CLI

```bash
# 列 8 个内置停损
python __main__.py list-stops

# 1) 检测市场状态 + 推荐停损
python __main__.py regime --symbol AAPL --start 2022-01-01

# 2) 8 策略在同一数据上对比 + Deflated Sharpe
python __main__.py compare --symbol AAPL --start 2022-01-01

# 3) 单策略回测（triple-barrier：止损 + 止盈 + 时间）
python __main__.py backtest --stop chandelier --symbol AAPL --start 2022-01-01 \
    --profit-target 0.10 --max-holding 30

# 4) Walk-forward 优化（避免 in-sample 过拟合）
python __main__.py walk-forward --stop atr --symbol AAPL --start 2020-01-01 \
    --n-folds 4 --mode anchored

# 5) P/L 归因（停损 vs 入场各占多少）
python __main__.py attribution --stop atr --symbol AAPL --start 2022-01-01

# 6) 离线 demo（不需要 yfinance）
python __main__.py compare --synthetic --regime mixed --n-bars 500
```

### Python 库

```python
import pandas as pd
from dso import (
    STOPS, compare_stops, detect_regime, walk_forward,
    run_backtest, attribute_pnl, synthetic_ohlcv,
)

# 加载数据
df = synthetic_ohlcv(n=500, regime="mixed")    # 或 pd.read_csv / load_yfinance

# 1. Regime 推荐
regime = detect_regime(df)
print(regime.label, regime.recommended_stops)
# 'weak_uptrend'  ['moving_average', 'atr', 'supertrend']

# 2. 8 策略对比
stops = {name: cls() for name, cls in STOPS.items()}
cmp = compare_stops(df, stops)
print(cmp.to_table())

# 3. 单策略回测（含 triple-barrier）
result = run_backtest(df, STOPS["chandelier"](),
                      profit_target_pct=0.10,
                      max_holding_bars=30)
print(f"trades={result.n_trades}, stopped={result.n_stopped}, "
      f"profit={result.n_profit_taken}, timed_out={result.n_timed_out}")

# 4. Walk-forward 优化
from dso.stops import ATRStop
wf = walk_forward(df, ATRStop,
                  param_grid={"period": [10, 14, 20],
                              "multiplier": [2.0, 2.5, 3.0]},
                  n_folds=4)
print(f"train Sharpe: {wf.mean_train_score:.2f}, "
      f"test Sharpe: {wf.mean_test_score:.2f}, "
      f"overfitting gap: {wf.overfitting_gap:.2f}")

# 5. P/L 归因
attr = attribute_pnl(result, df)
print(f"perfect exit P/L: {attr.perfect_total_pnl:.2f}")
print(f"actual P/L:       {attr.actual_total_pnl:.2f}")
print(f"stop cost:        {attr.stop_pnl_loss:.2f}")
print(f"entry vs stop:    {attr.entry_contribution_pct:.0%} / {attr.stop_contribution_pct:.0%}")
```

## 真实输出例子

```
$ python __main__.py compare --synthetic --regime mixed --n-bars 500

Strategy        TotalRet  Sharpe  Sortino  Calmar   MaxDD   Trades  HitRate
---------------------------------------------------------------------------
chandelier      +0.91%    +8.35   +93.90   +115.78  -0.00%  215     85.1%
moving_average  +0.55%    +7.25   +77.55   +48.87   -0.01%  228     81.6%
parabolic_sar   +0.03%    +0.76   +2.20    +0.92    -0.01%  83      28.9%
atr             +0.03%    +0.60   +1.57    +0.49    -0.03%  27      37.0%
supertrend      +0.02%    +0.51   +1.44    +0.44    -0.03%  22      36.4%
max_drawdown    +0.02%    +0.31   +1.12    +0.38    -0.03%  12      33.3%
donchian        +0.01%    +0.17   +0.73    +0.11    -0.04%  83      15.7%
time            +0.00%    +0.02   +0.03    +0.01    -0.05%  25      48.0%
volatility      +0.00%    +0.02   +0.02    +0.02    -0.04%  2       50.0%

Best by Sharpe: chandelier  (Deflated Sharpe probability: 0.0%)
```

`Deflated Sharpe probability: 0.0%` 提醒你：测试了 9 个策略后，最高 Sharpe 不能
直接当真 — 多重检验偏倚需要更多样本量或更稳健的 SR 差距才能给出"真的赢"
的统计置信度。

```
$ python __main__.py attribution --stop atr --synthetic --regime mixed --n-bars 300

{
  "n_trades": 25,
  "actual_total_pnl": -2.08,
  "perfect_total_pnl": 85.37,
  "stop_pnl_loss": -87.45,
  "entry_contribution_pct": 0.49,
  "stop_contribution_pct": 0.51
}
```

"完美 exit P/L 是 85，实际是 -2，停损损失 87" — 直接告诉你 ATR 停损在这段
数据上太紧 / 太晚了，让出了所有潜在利润。

## 核心架构

```
src/dso/
├── stops/                  # 8 个停损策略（state machine）
│   ├── base.py             # BaseStop + StopState + Bar
│   ├── atr.py / chandelier.py / supertrend.py / parabolic_sar.py
│   ├── donchian.py / moving_average.py / volatility.py
│   ├── max_drawdown.py / time_stop.py
│   └── __init__.py         # STOPS 注册表
├── backtest.py             # Triple-barrier 事件驱动回测
├── compare.py              # 多策略 + Deflated Sharpe Ratio
├── walk_forward.py         # Pardo / LdP 滚动 train/test
├── regime.py               # ADX + Hurst + 推荐 stop
├── attribution.py          # 入场 vs 停损 P/L 拆分
├── metrics.py              # Sharpe / Sortino / Calmar / DSR
└── data.py                 # CSV / yfinance / 合成
```

## 设计取舍

1. **State machine 而非 stateless 重算**：v1 每次 `calculate_stop` 都重算整个历史，
   慢且容易 look-ahead bias。v2 的 `update(bar)` 一次推进一根 bar，
   `_history` 暖机历史指标，但触发只看当下 + 状态。
2. **Trailing 默认走单方向**：long 的 stop 只能往上调，short 只能往下。这是
   trailing stop 的正确语义。
3. **Entry signal 默认全 True**：测停损时不希望被 entry timing 干扰。要套自己
   的 entry signal 传 `entry_signal=` 参数。
4. **Triple-barrier 默认只接 stop barrier**：profit_target / max_holding 可选。
   这与 Lopez de Prado 2018 的 labeling 方法对齐。
5. **Deflated Sharpe 自动启用**：每次 compare 都算 DSR，让用户警惕多重检验偏倚。
   这是大部分回测库忽略的关键。

## 测试

```bash
pytest tests/
```

**128 个测试**，2 秒内跑完，全部不联网：

| 文件 | 测试数 | 覆盖 |
|---|---|---|
| `test_stops.py` | 34 | 8 个停损策略的 trailing 行为 + 触发逻辑 |
| `test_backtest.py` | 18 | Triple-barrier 各 exit reason + commission + side |
| `test_metrics.py` | 21 | Sharpe / Sortino / Calmar + Deflated SR + Max DD |
| `test_regime.py` | 15 | ADX / Hurst / vol 分类 + 推荐 stop |
| `test_walk_forward.py` | 11 | Anchored / rolling fold + 最佳参数选择 |
| `test_compare.py` | 9 | 排序 + best strategy + DSR 集成 |
| `test_attribution.py` | 7 | P/L 拆分 + 边界 |
| `test_data.py` | 13 | CSV / yfinance / synthetic + OHLC invariants |

## 文献支撑

详见 [`docs/RESEARCH_NOTES.md`](./docs/RESEARCH_NOTES.md)。核心：

- Wilder (1978), *New Concepts in Technical Trading Systems* — ATR, Parabolic SAR, ADX
- Le Beau & Lucas (1992), *Computer Analysis of the Futures Markets* — Chandelier Exit
- Faith (2007), *Way of the Turtle* — Donchian Channel exit
- Lopez de Prado (2018), *Advances in Financial Machine Learning* — Triple-barrier method, walk-forward CV
- Bailey & Lopez de Prado (2014, JPM) — Deflated Sharpe Ratio

## v1（legacy）

v1 的 4 个停损 + 单 grid optimizer + 简单 backtest 保留在 [`legacy/`](./legacy/)
下，供需要参考的人查阅。v1 API 不再演进，v1 的 README 也有派蒙 / OpenClaw 等
AI 痕迹，仅作历史归档。

## 已知限制

- 回测假设按 stop_price 平仓（无 spike-through 模拟）；高波动数据上实际成交可能
  更差。
- Walk-forward 默认 grid 是手工挑的"合理范围"，超出该范围（如 ATR period=200）
  需要传自定义 `param_grid`。
- Deflated Sharpe Ratio 假设收益分布的偏度 / 峰度估计稳定；样本量 < 100 时
  数值不稳。

## 许可

MIT
