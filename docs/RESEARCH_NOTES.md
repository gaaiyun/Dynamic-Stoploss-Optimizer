# Research Notes

整理本仓库每个组件的学界 / 实务界文献来源。读这里之前最好先看 README。

## 1. 8 个停损策略的算法来源

### 1.1 ATR Stop（ATRStop）

- **来源**：Welles Wilder Jr. (1978), *New Concepts in Technical Trading
  Systems*, Trend Research, Greensboro, NC. Chapter 2 定义 True Range +
  ATR；Chapter 3 用 ATR 做 trailing stop。
- **公式**：True Range = max(H-L, |H-C_prev|, |L-C_prev|)；ATR = Wilder
  平滑（不是简单 SMA）：`ATR_n = (ATR_{n-1} × (N-1) + TR_n) / N`。
- **典型参数**：period=14（Wilder 原版），multiplier ∈ [2.0, 3.5]。
- **本仓库实现**：`src/dso/stops/atr.py` + `_wilder_atr` helper。

### 1.2 Chandelier Exit（ChandelierStop）

- **来源**：Charles Le Beau & David W. Lucas (1992), *Computer Analysis of
  the Futures Markets*, McGraw-Hill. Le Beau 后续在 ActiveTrader 杂志
  推广。
- **公式**（long）：`Stop = max(High_n) - multiplier × ATR(N)`
- **典型参数**：period=22（约 1 个月日线），multiplier=3.0（Le Beau 原书推荐）。
- **本仓库实现**：`src/dso/stops/chandelier.py`。
- **TradingView 实现参考**：[TradingView Chandelier Exit](https://www.tradingview.com/support/solutions/43000501980)。

### 1.3 SuperTrend（SuperTrendStop）

- **来源**：Olivier Seban 的法语论坛起源（2008 年前后），后被 TradingView、
  freqtrade、vectorbt 等社区库实现并广泛传播。学界论文：
  Hota & Pal (2024) "SuperTrend indicator efficacy" 等。
- **公式**：
  ```
  HL2 = (high + low) / 2
  Basic_Upper = HL2 + multiplier × ATR
  Basic_Lower = HL2 - multiplier × ATR
  Final_Upper/Lower 用 trailing 规则更新
  long 仓 stop = Final_Lower
  ```
- **典型参数**：period=10, multiplier=3.0。
- **本仓库实现**：`src/dso/stops/supertrend.py`。

### 1.4 Parabolic SAR（ParabolicSARStop）

- **来源**：Welles Wilder Jr. (1978), 同上书 Chapter 5 "The Parabolic
  Time/Price System"。
- **公式**：`SAR_t = SAR_{t-1} + AF × (EP - SAR_{t-1})`
  - EP = Extreme Point（开仓以来 long 的最高 high，short 的最低 low）
  - AF = Acceleration Factor，每次 EP 更新时 +increment，封顶 0.20
- **典型参数**：AF_init=0.02, AF_increment=0.02, AF_max=0.20。
- **本仓库实现**：`src/dso/stops/parabolic_sar.py`。

### 1.5 Donchian Channel Exit（DonchianStop）

- **来源**：Richard Dennis & Bill Eckhardt 在 1983-1988 的 Turtle Traders
  实验中使用；详细规则发布于 Curtis Faith (2007), *Way of the Turtle*,
  McGraw-Hill。
- **公式**：long 仓 stop = 过去 N 根 K 线的最低 low；short = 最高 high。
- **典型参数**：N=10（Turtle 短线退出 S1）或 N=20（与入场对称）。
- **本仓库实现**：`src/dso/stops/donchian.py`。

### 1.6 Moving Average Stop（MovingAverageStop）

- **来源**：经典技术分析。Murphy (1999), *Technical Analysis of the
  Financial Markets*, Chapter 9 详细讨论 MA 作为 dynamic support/resistance
  + trailing stop 的用法。
- **公式**：long 仓 stop = MA(N) × (1 - offset_pct)。MA 类型可选 SMA / EMA。
- **典型参数**：period=20-50。
- **本仓库实现**：`src/dso/stops/moving_average.py`。

### 1.7 Volatility Stop（VolatilityStop）

- **来源**：基于 rolling standard deviation 的 trailing。
  与 Bollinger Bands 思想相通：J. Bollinger (1992), 后续 *Bollinger on
  Bollinger Bands*。
- **公式**：stop = entry_price ± multiplier × σ_annualized × entry_price。
- **本仓库实现**：`src/dso/stops/volatility.py` + `_annualized_vol` helper。

### 1.8 Time-based Stop（TimeStop）

- **来源**：M. Lopez de Prado (2018), *Advances in Financial Machine
  Learning*, Wiley. Chapter 3 "Labeling" 提出 triple-barrier method，其中
  vertical barrier 就是时间停损。
- **本仓库实现**：`src/dso/stops/time_stop.py`。

## 2. Triple-Barrier 回测

- **来源**：Lopez de Prado (2018) Chapter 3。三个 barrier：
  1. **Lower horizontal barrier**（止损）—— `BaseStop` 提供
  2. **Upper horizontal barrier**（止盈）—— `profit_target_pct` 参数
  3. **Vertical barrier**（时间）—— `max_holding_bars` 参数
- 任一 barrier 先到即平仓。这与 LdP 的 "meta-labeling" 一致：每笔交易的
  label = 哪个 barrier 触发了。
- **本仓库实现**：`src/dso/backtest.py::run_backtest`。

## 3. Walk-Forward 优化

- **核心来源**：Robert Pardo (1992, 2008), *The Evaluation and Optimization
  of Trading Strategies*, Wiley.
- **进阶**：Lopez de Prado (2018), Chapter 12 "Backtesting through
  Cross-Validation" 严厉批评全样本 grid search，强调 walk-forward 是诚实
  评估的最低门槛。
- **两种 split 模式**：
  - **Anchored**（扩张窗口）：每折训练段从 0 开始扩展。
  - **Rolling**（滚动窗口）：训练段长度固定，逐折滚动。
- **过拟合诊断**：`mean_train - mean_test = overfitting_gap`。
- **本仓库实现**：`src/dso/walk_forward.py`。

## 4. Deflated Sharpe Ratio

- **核心论文**：D. H. Bailey & M. Lopez de Prado (2014), "The Deflated
  Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and
  Non-Normality", *Journal of Portfolio Management*, Vol. 40, No. 5.
- **问题**：在 N 个策略里选 Sharpe 最高的，相当于 N 次独立检验取最大值。即使
  所有策略本质都是 zero-mean，最大 Sharpe 期望也是正的（Bailey 称之为
  "selection bias under multiple testing"）。
- **公式**（简化）：

  ```
  DSR = Φ( (SR_observed - E[max(SR_null)]) / SE(SR_observed) )

  E[max(SR_null)] ≈ (1 - γ) × Φ⁻¹(1 - 1/N) + γ × Φ⁻¹(1 - 1/(N·e))
                     用 Hartigan-Pearson 近似

  SE(SR) = sqrt( (1 - skew × SR + (kurt-1)/4 × SR²) / (n_obs - 1) )
                考虑收益非正态（偏度 + 峰度）
  ```

- **解读**：DSR 是概率值 ∈ [0, 1]，> 0.95 表示在 95% 置信下确实 outperform。
- **本仓库实现**：`src/dso/metrics.py::deflated_sharpe_ratio`。

## 5. Regime Detection

### 5.1 ADX / +DI / -DI

- **来源**：Welles Wilder Jr. (1978), *New Concepts*, Chapter 13 "Directional
  Movement Index"。
- **判定规则**（业界共识）：
  - ADX > 25：有趋势
  - ADX > 40：强趋势
  - ADX < 20：横盘
  - +DI > -DI：上涨方向；反之下跌
- **本仓库实现**：`src/dso/regime.py::_calc_adx_dmi`（Wilder 平滑）。

### 5.2 Hurst Exponent

- **来源**：Harold E. Hurst (1951), "Long-term storage capacity of
  reservoirs", *Transactions of the ASCE*, Vol. 116, pp. 770-808.
  在金融上由 Mandelbrot 引入（1968 文章）。
- **解读**：
  - H ≈ 0.5：随机游走
  - H > 0.5：趋势 / 持续性
  - H < 0.5：均值回归 / 反持续性
- **估计方法**：经典是 R/S analysis；本仓库用简化版（log-log slope of
  rolling-std vs lag），样本量小时不稳定。
- **本仓库实现**：`src/dso/regime.py::_hurst_exponent`。

### 5.3 Regime → Stop Strategy 映射

| Regime 标签 | 触发条件 | 推荐停损 | 学术依据 |
|---|---|---|---|
| `strong_uptrend` | ADX ≥ 25 + +DI > -DI | Chandelier / SuperTrend / PSAR / ATR | trailing 让趋势走（Wilder 1978, Le Beau 1992） |
| `strong_downtrend` | ADX ≥ 25 + -DI > +DI | 同上（short 仓）| 同上 |
| `weak_uptrend` / `weak_downtrend` | ADX ∈ [20, 25) | MA / ATR / SuperTrend | 趋势不强，trailing 留余地 |
| `ranging` | ADX < 20 | Donchian / MaxDD / MA | 横盘窄止损（Turtle 思路） |
| `high_volatility` | 年化波动 > 40% | Volatility / ATR / MaxDD | 波动自适应 |

## 6. P/L Attribution

- **概念来源**：金融绩效归因（Brinson, Hood & Beebower 1986 框架）应用到
  trade-level 的 entry-vs-exit 拆分。本仓库简化为：

  ```
  perfect_exit_pnl = 假设按"持仓期内最优价"平仓的 P/L
  actual_pnl = 实际策略 P/L
  stop_pnl_loss = actual - perfect ≤ 0  (停损 / 提前平仓让出的潜在利润)
  ```

- 这种分解的局限：忽略了"停损保护亏损"的情形（perfect_exit 是反事实最优，
  不能反映"如果不停损会亏更多"）。但作为"我的停损是不是太紧 / 太晚" 的诊断
  足够。
- **本仓库实现**：`src/dso/attribution.py::attribute_pnl`。

## 7. 现有库横向对比

| 库 | 停损方式 | 我们的优势 |
|---|---|---|
| [freqtrade](https://github.com/freqtrade/freqtrade) | `stoploss` + `trailing_stop` + `custom_stoploss` callback | freqtrade 只关心实盘；我们提供研究层（compare + walk-forward + DSR） |
| [backtrader](https://github.com/mementum/backtrader) | `bt.Order.StopTrail` + 各种 indicators | 单一停损固化，没有比较 / 归因层 |
| [vectorbt](https://github.com/polakowo/vectorbt) | `vbt.OHLCSTX` 灵活但学习曲线陡 | API 集中、文献支撑明确、有 DSR |
| [zipline](https://github.com/quantopian/zipline) | `set_max_position_size` 等 | zipline 不做停损研究 |
| [lumibot](https://github.com/Lumiwealth/lumibot) | `set_stop_loss` + `trail_stop_pct` | lumibot 偏实盘 |
| [pybroker](https://github.com/edtechre/pybroker) | `pos.stop_loss` 各种类型 | pybroker 没有 regime / attribution / DSR |
| QuantConnect / Lean | `TrailingStopOrder` | 闭源 / 平台限制 |

## 8. 进一步阅读

- **Kaufman, Perry J.** (2020), *Trading Systems and Methods*, 6th ed.,
  Wiley. — 停损章节 + KAMA-based adaptive stops。
- **Tomasini, E. & Jaekle, U.** (2009), *Trading Systems: A New Approach
  to System Development and Portfolio Optimisation*, Harriman House. —
  Walk-forward 实操章节。
- **Aronson, D. R.** (2006), *Evidence-Based Technical Analysis*, Wiley.
  — 多重检验偏倚的早期警告（先于 Bailey-LdP）。
- **TradingView 公开实现**：可作为算法 sanity check 的参考点
  - [ATR Trailing Stop](https://www.tradingview.com/script/8r4zp3O3-Average-True-Range-Trailing-Stops/)
  - [SuperTrend](https://www.tradingview.com/script/r6dAP7yi)
  - [Chandelier Exit](https://www.tradingview.com/support/solutions/43000501980)

## 9. 本仓库 NOT 做的事

明确不在 scope 内（避免技术债 + 避免冒充）：

1. **机器学习 / LSTM stop**：学术界还在争议（Borovkova & Tsiamas 2019
   等），加进来会增加依赖 + 增加不确定性，不如规则可解释。
2. **portfolio-level 停损**：本仓库每次只看一个标的，多资产联动停损（如
   regime-based portfolio rebalancing）不做。
3. **实盘下单**：本仓库只做研究 / 回测。实盘走 lumibot / freqtrade /
   pybroker。
4. **K 线生成 / tick 数据处理**：假设输入已经是合法 OHLC，不做 tick 聚合。

---

注：所有书目年份按英文版第一版年份给出（PRC 译本可能略晚）。文献链接如 DOI /
arXiv 在第一作者已能 search 到，本文档不内嵌易过期的 URL。
