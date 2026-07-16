# Dynamic Stoploss Optimizer (v2)

动态停损研究实验室：在统一的事件回测口径下比较 **9 个规则型停损策略**，并提供
walk-forward、市场状态诊断、MFE 实现差距和 Deflated Sharpe Ratio。它用于形成
可复核的研究候选，不直接证明某个停损在未来或实盘中有效。

## 与现有库对比

大部分回测库（freqtrade / backtrader / vectorbt / lumibot / pybroker / QuantConnect）
都让你选一个固定的停损策略然后跑回测。本仓库不一样：

| 现有库的做法 | 我们的做法 |
|---|---|
| 选一个停损，比如 ATR×2.5 | 跑 **9 个停损** 在同一份数据上，按同一口径比较 |
| Grid search 在全样本上选最优 | **Walk-forward** 滚动训练 / 样本外评估，给出 overfitting gap |
| 最高 Sharpe = 胜出 | 加 **Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)** 多重检验校正 |
| 用户自己判断哪种行情用哪个停损 | **ADX + Hurst 识别 regime**，返回待验证的策略候选 |
| 只看最终 P/L | 对比实际退出与事后最大有利变动（MFE）的实现差距 |

文献支撑见 [`docs/RESEARCH_NOTES.md`](./docs/RESEARCH_NOTES.md)。

## 9 个内置停损策略

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
# 列 9 个内置停损
python __main__.py list-stops

# 1) 检测市场状态 + 给出启发式候选
python __main__.py regime --symbol AAPL --start 2022-01-01

# 2) 9 策略在同一数据上对比 + Deflated Sharpe
python __main__.py compare --symbol AAPL --start 2022-01-01

# 3) 单策略回测（triple-barrier：止损 + 止盈 + 时间）
python __main__.py backtest --stop chandelier --symbol AAPL --start 2022-01-01 \
    --profit-target 0.10 --max-holding 30

# 4) Walk-forward 优化（避免 in-sample 过拟合）
python __main__.py walk-forward --stop atr --symbol AAPL --start 2020-01-01 \
    --n-folds 4 --mode anchored

# 5) 实际退出相对 MFE 的实现差距
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

# 1. Regime 候选映射（仍需样本外验证）
regime = detect_regime(df)
print(regime.label, regime.recommended_stops)
# 'weak_uptrend'  ['moving_average', 'atr', 'supertrend']

# 2. 9 策略对比
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

# 5. MFE 实现差距诊断
attr = attribute_pnl(result, df)
print(f"MFE net P/L:      {attr.mfe_total_pnl:.2f}")
print(f"actual net P/L:   {attr.actual_total_pnl:.2f}")
print(f"realization gap:  {attr.realization_gap_pnl:.2f}")
print(f"capture ratio:    {attr.aggregate_capture_ratio:.1%}")
```

## 真实输出例子

```
$ python __main__.py compare --synthetic --regime mixed --n-bars 500

Strategy        TotalRet  Sharpe  Sortino  Calmar  MaxDD    Trades  HitRate
---------------------------------------------------------------------------
volatility      +3.33%    +0.19   +0.26    +0.09   -33.38%  2       50.0%
max_drawdown    +0.49%    +0.10   +0.14    +0.05   -34.94%  14      28.6%
time            -1.36%    +0.04   +0.06    +0.02   -34.34%  25      48.0%
supertrend      -1.89%    +0.03   +0.04    +0.01   -35.82%  24      33.3%
atr             -4.86%    -0.06   -0.08    -0.03   -37.03%  31      32.3%
donchian        -16.67%   -0.45   -0.60    -0.17   -44.25%  82      18.3%
parabolic_sar   -18.20%   -0.50   -0.67    -0.19   -43.87%  99      24.2%
chandelier      -33.42%   -1.13   -1.46    -0.32   -54.07%  218     35.8%
moving_average  -33.79%   -1.15   -1.50    -0.32   -54.73%  222     35.6%

Best by Sharpe: volatility  (Deflated Sharpe probability: 10.5%)
```

这个固定种子的合成样例只验证执行链，不是策略有效性证据。低 DSR 提醒我：测试
多个候选后，最高样本 Sharpe 不能直接当成稳定优势。

```
$ python __main__.py attribution --stop atr --synthetic --regime mixed --n-bars 300

{
  "n_trades": 25,
  "actual_total_pnl": -11528.76,
  "mfe_total_pnl": 74077.32,
  "realization_gap_pnl": -85606.08,
  "aggregate_capture_ratio": -0.1556,
  "avg_trade_capture_ratio": -2.1064
}
```

MFE 是在相同入场和持仓区间内事后可见的价格上界。负实现比例说明这组入场和退出
没有捕获该上界，但不能据此把差额因果归于止损，也不能单独判断入场质量。

## 核心架构

```
src/dso/
├── stops/                  # 9 个停损策略（state machine）
│   ├── base.py             # BaseStop + StopState + Bar
│   ├── atr.py / chandelier.py / supertrend.py / parabolic_sar.py
│   ├── donchian.py / moving_average.py / volatility.py
│   ├── max_drawdown.py / time_stop.py
│   └── __init__.py         # STOPS 注册表
├── backtest.py             # Triple-barrier 事件驱动回测
├── compare.py              # 多策略 + Deflated Sharpe Ratio
├── walk_forward.py         # Pardo / LdP 滚动 train/test
├── regime.py               # ADX + Hurst + 启发式候选 stop
├── attribution.py          # 实际退出相对 MFE 的实现差距
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
4. **事件回测借鉴 triple-barrier 结构**：profit target、stop 和 time barrier 可选。
   它不是 Lopez de Prado meta-labeling 流程的完整复现；OHLC 同 bar 冲突采用
   stop-first 保守规则。
5. **Deflated Sharpe 自动启用**：每次 compare 都算 DSR，让用户警惕多重检验偏倚。
   这是大部分回测库忽略的关键。

## 测试

```bash
pytest tests/
```

**139 个测试**，全部不联网：

| 文件 | 测试数 | 覆盖 |
|---|---|---|
| `test_stops.py` | 34 | 9 个停损策略的 trailing 行为 + 触发逻辑 |
| `test_backtest.py` | 25 | 时序、跳空、盯市权益、仓位、成本和各 exit reason |
| `test_metrics.py` | 22 | Sharpe / Sortino / Calmar + Deflated SR + Max DD |
| `test_regime.py` | 15 | ADX / Hurst / vol 分类 + 推荐 stop |
| `test_walk_forward.py` | 13 | Anchored / rolling fold、信号切片与参数选择 |
| `test_compare.py` | 9 | 排序 + best strategy + DSR 集成 |
| `test_attribution.py` | 8 | MFE 实现差距 + 边界 |
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
下，供需要参考的人查阅。v1 API 不再演进，仅作历史归档。

## 已知限制

- 只有 OHLC 时无法知道同一根 K 线内止损与止盈先后，当前采用 stop-first；跳空穿越
  止损时按开盘价成交，仍不模拟盘口深度和冲击成本。
- 引擎当前只支持单标的、单持仓；默认每次使用当前权益的 100%，可用
  `position_fraction` 调低，不能把示例结果直接当作实盘仓位建议。
- Walk-forward 默认 grid 是手工挑的"合理范围"，超出该范围（如 ATR period=200）
  需要传自定义 `param_grid`。
- Deflated Sharpe Ratio 假设收益分布的偏度 / 峰度估计稳定；样本量 < 100 时
  数值不稳。
- regime 到 stop 的映射是文献启发的候选清单，不是经本数据样本外证明的最优策略。
- MFE 是事后诊断上界，不能解释因果，也可能受 exit bar 内部时序不可见影响。

## 许可

MIT
