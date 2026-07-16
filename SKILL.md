---
name: dynamic-stoploss-optimizer
description: 比较 9 个规则型停损策略，提供 ADX/Hurst regime 诊断、walk-forward、事件回测、MFE 实现差距和 Deflated Sharpe Ratio。用于生成待样本外验证的研究候选，不直接证明实盘有效性。
---

# dynamic-stoploss-optimizer

## 什么时候用

- "AAPL 用 ATR 止损还是 Chandelier 好？"
- "我的策略在 2020 优化的参数到 2024 还有效吗？"（用 walk-forward 验证）
- "现在大盘是趋势还是震荡？哪些停损值得进入下一轮验证？"
- "实际退出捕获了多少持仓期间的最大有利变动？"

## 入口

```bash
python __main__.py list-stops                                # 9 个停损
python __main__.py regime --symbol AAPL --start 2022-01-01   # 状态 + 启发式候选
python __main__.py compare --symbol AAPL --start 2022-01-01  # 9 策略对比 + DSR
python __main__.py backtest --stop chandelier --symbol AAPL  # 单策略 triple-barrier
python __main__.py walk-forward --stop atr --symbol AAPL     # 滚动优化
python __main__.py attribution --stop atr --symbol AAPL      # MFE 实现差距
```

库调用：

- `dso.STOPS` —— 名字 → 类的注册表
- `dso.compare_stops(df, {name: stop})` —— 多策略对比 + Deflated Sharpe
- `dso.detect_regime(df)` —— ADX + Hurst → label + 启发式 stop 候选
- `dso.walk_forward(df, stop_class, param_grid)` —— 滚动训练 / 测试
- `dso.run_backtest(df, stop, profit_target_pct=, max_holding_bars=)` —— triple-barrier 回测
- `dso.attribute_pnl(result, df)` —— 实际退出相对 MFE 的实现差距

## 9 个停损算法与方法来源

详见 `docs/RESEARCH_NOTES.md`：

- **ATR / Parabolic SAR / ADX**：Wilder (1978)
- **Chandelier Exit**：Le Beau (1992)
- **Donchian**：Turtle Traders / Faith (2007)
- **Triple-barrier / Walk-forward CV**：Lopez de Prado (2018)
- **Deflated Sharpe Ratio**：Bailey & Lopez de Prado (2014, JPM)

## 依赖

- 必需：Python 3.10+, pandas, numpy, scipy
- 可选：yfinance（用真数据）

## 注意事项

- OHLC 无法判断同一根 K 线内止盈和止损的先后，当前采用 stop-first；跳空穿越
  止损时按开盘价成交，但仍不模拟盘口深度和冲击成本。
- regime 映射和 MFE 都是诊断工具，不是策略有效性或因果归因结论。
- Deflated Sharpe Ratio 是统计校正，不替代经济学逻辑判断。
- v1 的 4 个停损在 `legacy/` 下保留，API 不再演进。
