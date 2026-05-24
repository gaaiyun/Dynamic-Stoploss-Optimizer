---
name: dynamic-stoploss-optimizer
description: 8 个学界经典停损策略（ATR / Chandelier / SuperTrend / Parabolic SAR / Donchian / MA / Volatility / Time）+ ADX regime 识别 + walk-forward 优化 + triple-barrier 回测 + Deflated Sharpe Ratio。给一段历史 OHLC 数据，回答"哪个停损在这只票上真的好用"。
---

# dynamic-stoploss-optimizer

## 什么时候用

- "AAPL 用 ATR 止损还是 Chandelier 好？"
- "我的策略在 2020 优化的参数到 2024 还有效吗？"（用 walk-forward 验证）
- "现在大盘是趋势还是震荡？该用哪种停损？"
- "我的策略亏钱了，是入场不行还是停损不行？"

## 入口

```bash
python __main__.py list-stops                                # 8 个停损
python __main__.py regime --symbol AAPL --start 2022-01-01   # 状态 + 推荐
python __main__.py compare --symbol AAPL --start 2022-01-01  # 8 策略对比 + DSR
python __main__.py backtest --stop chandelier --symbol AAPL  # 单策略 triple-barrier
python __main__.py walk-forward --stop atr --symbol AAPL     # 滚动优化
python __main__.py attribution --stop atr --symbol AAPL      # P/L 拆分
```

库调用：

- `dso.STOPS` —— 名字 → 类的注册表
- `dso.compare_stops(df, {name: stop})` —— 多策略对比 + Deflated Sharpe
- `dso.detect_regime(df)` —— ADX + Hurst → label + 推荐 stop 列表
- `dso.walk_forward(df, stop_class, param_grid)` —— 滚动训练 / 测试
- `dso.run_backtest(df, stop, profit_target_pct=, max_holding_bars=)` —— triple-barrier 回测
- `dso.attribute_pnl(result, df)` —— 入场 vs 停损贡献拆分

## 8 个停损算法来源

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

- 回测假设按 stop_price 平仓，不模拟 spike-through；高波动场景实际可能更差。
- Deflated Sharpe Ratio 是统计校正，不替代经济学逻辑判断。
- v1 的 4 个停损在 `legacy/` 下保留，API 不再演进。
