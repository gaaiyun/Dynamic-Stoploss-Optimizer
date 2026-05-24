"""Dynamic Stoploss Optimizer CLI（v2）。

子命令：
    list-stops          列 8 个内置停损策略 + 默认参数
    backtest            跑单策略回测
    compare             8 个策略同数据对比 + Deflated Sharpe Ratio
    regime              ADX 市场状态检测 + 推荐停损
    walk-forward        滚动训练 + 样本外评估
    attribution         拆 P/L 入场 vs 停损贡献
    fetch               从 yfinance 抓 OHLCV
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd

from dso import (
    STOPS,
    attribute_pnl,
    compare_stops,
    detect_regime,
    load_csv,
    load_yfinance,
    run_backtest,
    synthetic_ohlcv,
    walk_forward,
)


def _load_data(args) -> pd.DataFrame:
    if args.synthetic:
        return synthetic_ohlcv(n=args.n_bars, seed=args.seed,
                                regime=args.regime)
    if args.csv:
        return load_csv(args.csv)
    if args.symbol:
        return load_yfinance(args.symbol, start=args.start, end=args.end)
    raise SystemExit("[error] 需要 --csv / --symbol / --synthetic 之一")


def cmd_list_stops(args) -> int:
    print(f"{'name':<18} {'default params'}")
    print("-" * 70)
    for name, cls in STOPS.items():
        params = ", ".join(f"{k}={v}" for k, v in cls.DEFAULT_PARAMS.items())
        print(f"{name:<18} {params}")
    return 0


def cmd_backtest(args) -> int:
    df = _load_data(args)
    if args.stop not in STOPS:
        sys.stderr.write(f"[error] 未知 stop {args.stop}，可选 {list(STOPS)}\n")
        return 1
    stop = STOPS[args.stop]()
    result = run_backtest(
        df, stop, side=args.side,
        profit_target_pct=args.profit_target,
        max_holding_bars=args.max_holding,
        commission_pct=args.commission,
        slippage_pct=args.slippage,
        initial_capital=args.initial_cash,
    )
    print(json.dumps({
        "stop": args.stop,
        "stop_params": result.stop_params,
        "n_trades": result.n_trades,
        "total_return": result.total_return,
        "hit_rate": result.hit_rate,
        "avg_pnl_pct": result.avg_pnl_pct,
        "exits": {
            "stopped": result.n_stopped,
            "profit_taken": result.n_profit_taken,
            "timed_out": result.n_timed_out,
            "open_at_end": result.n_open_at_end,
        },
        "final_equity": result.final_equity,
    }, ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
    return 0


def cmd_compare(args) -> int:
    df = _load_data(args)
    stops = {name: cls() for name, cls in STOPS.items()}
    cmp = compare_stops(
        df, stops, side=args.side,
        profit_target_pct=args.profit_target,
        max_holding_bars=args.max_holding,
        commission_pct=args.commission,
        slippage_pct=args.slippage,
        initial_capital=args.initial_cash,
    )
    print(cmp.to_table())
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(cmp.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
    return 0


def cmd_regime(args) -> int:
    df = _load_data(args)
    report = detect_regime(df)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
    return 0


def cmd_walk_forward(args) -> int:
    df = _load_data(args)
    if args.stop not in STOPS:
        sys.stderr.write(f"[error] 未知 stop {args.stop}\n")
        return 1
    stop_class = STOPS[args.stop]
    # 默认 grid：每个策略类型给一个合理的 mini grid
    default_grids = {
        "atr": {"period": [10, 14, 20], "multiplier": [2.0, 2.5, 3.0]},
        "chandelier": {"period": [14, 22, 30], "multiplier": [2.5, 3.0, 3.5]},
        "supertrend": {"period": [7, 10, 14], "multiplier": [2.5, 3.0, 3.5]},
        "parabolic_sar": {"af_init": [0.01, 0.02], "af_max": [0.15, 0.20]},
        "donchian": {"period": [5, 10, 20]},
        "moving_average": {"period": [10, 20, 50],
                            "ma_type": ["sma", "ema"]},
        "volatility": {"window": [10, 20, 30], "multiplier": [1.5, 2.0, 2.5]},
        "max_drawdown": {"max_drawdown_pct": [0.03, 0.05, 0.08]},
        "time": {"max_bars": [5, 10, 20]},
    }
    grid = default_grids.get(args.stop, {})
    result = walk_forward(
        df, stop_class, grid,
        n_folds=args.n_folds, mode=args.mode, side=args.side,
        commission_pct=args.commission, slippage_pct=args.slippage,
        initial_capital=args.initial_cash,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
    return 0


def cmd_attribution(args) -> int:
    df = _load_data(args)
    if args.stop not in STOPS:
        sys.stderr.write(f"[error] 未知 stop {args.stop}\n")
        return 1
    stop = STOPS[args.stop]()
    result = run_backtest(df, stop, side=args.side,
                           commission_pct=args.commission,
                           initial_capital=args.initial_cash)
    report = attribute_pnl(result, df)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8")
    return 0


def cmd_fetch(args) -> int:
    if not args.symbol:
        sys.stderr.write("[error] 需要 --symbol\n")
        return 1
    try:
        df = load_yfinance(args.symbol, start=args.start, end=args.end)
    except ImportError as e:
        sys.stderr.write(f"[error] {e}\n")
        return 2
    df.to_csv(args.output)
    sys.stderr.write(f"[ok] 写入 {args.output}  shape={df.shape}\n")
    return 0


def _add_data_args(sp: argparse.ArgumentParser) -> None:
    g = sp.add_argument_group("data source（任选一个）")
    g.add_argument("--csv")
    g.add_argument("--symbol", help="yfinance ticker")
    g.add_argument("--start", default="2022-01-01")
    g.add_argument("--end")
    g.add_argument("--synthetic", action="store_true")
    g.add_argument("--n-bars", type=int, default=500)
    g.add_argument("--regime", default="mixed",
                   choices=["mixed", "uptrend", "downtrend", "ranging"])
    g.add_argument("--seed", type=int, default=42)


def _add_backtest_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument("--side", default="long", choices=["long", "short"])
    sp.add_argument("--profit-target", type=float,
                    help="止盈百分比，如 0.10 = 10%%")
    sp.add_argument("--max-holding", type=int, help="时间 barrier（bar 数）")
    sp.add_argument("--commission", type=float, default=0.001)
    sp.add_argument("--slippage", type=float, default=0.0)
    sp.add_argument("--initial-cash", type=float, default=100_000.0)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dso", description="Dynamic Stoploss Optimizer v2",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-stops", help="列内置停损策略").set_defaults(func=cmd_list_stops)

    sp = sub.add_parser("backtest", help="跑单策略回测")
    sp.add_argument("--stop", required=True, choices=list(STOPS))
    _add_data_args(sp)
    _add_backtest_args(sp)
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("compare", help="8 策略对比 + Deflated Sharpe")
    _add_data_args(sp)
    _add_backtest_args(sp)
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_compare)

    sp = sub.add_parser("regime", help="ADX 状态检测 + 推荐停损")
    _add_data_args(sp)
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_regime)

    sp = sub.add_parser("walk-forward", help="滚动训练 + 样本外评估")
    sp.add_argument("--stop", required=True, choices=list(STOPS))
    sp.add_argument("--n-folds", type=int, default=4)
    sp.add_argument("--mode", default="anchored",
                    choices=["anchored", "rolling"])
    _add_data_args(sp)
    _add_backtest_args(sp)
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_walk_forward)

    sp = sub.add_parser("attribution", help="拆 P/L 入场 vs 停损贡献")
    sp.add_argument("--stop", required=True, choices=list(STOPS))
    _add_data_args(sp)
    _add_backtest_args(sp)
    sp.add_argument("-o", "--output")
    sp.set_defaults(func=cmd_attribution)

    sp = sub.add_parser("fetch", help="从 yfinance 抓 OHLCV 存 CSV")
    sp.add_argument("--symbol", required=True)
    sp.add_argument("--start", required=True)
    sp.add_argument("--end")
    sp.add_argument("-o", "--output", required=True)
    sp.set_defaults(func=cmd_fetch)

    return p


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
