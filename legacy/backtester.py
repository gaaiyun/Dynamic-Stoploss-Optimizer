"""
回测模块

派蒙的回测引擎！可以测试不同止损策略的历史表现~
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings

try:
    from .stoploss_strategies import (
        BaseStoploss,
        StoplossSignal,
        calculate_sharpe_ratio,
        calculate_max_drawdown,
        calculate_win_rate
    )
except ImportError:
    from stoploss_strategies import (
        BaseStoploss,
        StoplossSignal,
        calculate_sharpe_ratio,
        calculate_max_drawdown,
        calculate_win_rate
    )


@dataclass
class Trade:
    """交易记录数据类"""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    position_type: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    exit_reason: str  # 'stoploss', 'target', 'end'
    stop_type: Optional[str] = None
    holding_period: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entry_date': self.entry_date,
            'exit_date': self.exit_date,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'shares': self.shares,
            'position_type': self.position_type,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'exit_reason': self.exit_reason,
            'stop_type': self.stop_type,
            'holding_period': self.holding_period
        }


@dataclass
class BacktestResult:
    """回测结果数据类"""
    strategy_name: str
    strategy_params: Dict[str, Any]
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    average_trade_pnl: float
    average_holding_period: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = None
    stoploss_signals: List[Dict] = field(default_factory=list)
    
    def summary(self) -> str:
        """生成回测结果摘要"""
        summary = f"""
╔══════════════════════════════════════════════════════════╗
║           回测结果摘要 - {self.strategy_name:^20s}          ║
╠══════════════════════════════════════════════════════════╣
║  初始资金：        ${self.initial_capital:>12,.2f}                      ║
║  最终资金：        ${self.final_capital:>12,.2f}                      ║
╠══════════════════════════════════════════════════════════╣
║  总收益率：        {self.total_return:>12.2%}                      ║
║  年化收益率：      {self.annualized_return:>12.2%}                      ║
║  夏普比率：        {self.sharpe_ratio:>12.2f}                      ║
║  最大回撤：        {self.max_drawdown:>12.2%}                      ║
╠══════════════════════════════════════════════════════════╣
║  总交易次数：      {self.total_trades:>12d}                      ║
║  盈利交易：        {self.winning_trades:>12d}                      ║
║  亏损交易：        {self.losing_trades:>12d}                      ║
║  胜率：            {self.win_rate:>12.2%}                      ║
║  盈亏比：          {self.profit_factor:>12.2f}                      ║
╠══════════════════════════════════════════════════════════╣
║  平均交易盈亏：    ${self.average_trade_pnl:>11,.2f}                      ║
║  平均持仓周期：    {self.average_holding_period:>12.1f} 天                  ║
║  最大连续盈利：    {self.max_consecutive_wins:>12d}                      ║
║  最大连续亏损：    {self.max_consecutive_losses:>12d}                      ║
╚══════════════════════════════════════════════════════════╝
"""
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'strategy_name': self.strategy_name,
            'strategy_params': self.strategy_params,
            'initial_capital': self.initial_capital,
            'final_capital': self.final_capital,
            'total_return': self.total_return,
            'annualized_return': self.annualized_return,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'average_trade_pnl': self.average_trade_pnl,
            'average_holding_period': self.average_holding_period,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses
        }


class StoplossBacktester:
    """
    止损策略回测器
    
    派蒙精心打造的回测引擎！可以测试不同止损策略在历史数据上的表现。
    
    参数:
        data: 包含 OHLC 数据的 DataFrame，必须有 'open', 'high', 'low', 'close' 列
        initial_capital: 初始资金 (默认 100000)
        commission: 交易佣金比例 (默认 0.001 即 0.1%)
        slippage: 滑点比例 (默认 0.001 即 0.1%)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_capital: float = 100000,
        commission: float = 0.001,
        slippage: float = 0.001
    ):
        self.data = data.copy()
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # 验证数据
        required_columns = ['open', 'high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Data must contain columns: {required_columns}")
        
        if 'date' in data.columns:
            self.data = self.data.set_index('date')
        
        self.data.index = pd.to_datetime(self.data.index)
        self.data = self.data.sort_index()
    
    def run(
        self,
        strategy: BaseStoploss,
        position_type: str = 'long',
        entry_signal: Optional[pd.Series] = None,
        exit_target: Optional[float] = None,
        shares_per_trade: int = 100,
        max_positions: int = 1
    ) -> BacktestResult:
        """
        运行回测
        
        Args:
            strategy: 止损策略实例
            position_type: 持仓类型 ('long' 或 'short')
            entry_signal: 入场信号 Series (True 表示入场)
            exit_target: 止盈目标 (百分比，如 0.1 表示 10%)
            shares_per_trade: 每笔交易股数
            max_positions: 最大同时持仓数
            
        Returns:
            BacktestResult: 回测结果
        """
        # 初始化
        capital = self.initial_capital
        positions = []  # 当前持仓列表
        trades = []  # 已完成交易列表
        equity_curve = []  # 权益曲线
        stoploss_signals_log = []  # 止损信号日志
        
        # 如果没有提供入场信号，使用简单的买入持有策略
        if entry_signal is None:
            entry_signal = pd.Series(False, index=self.data.index)
            entry_signal.iloc[0] = True  # 第一天入场
        
        # 遍历每一天
        for i in range(len(self.data)):
            current_date = self.data.index[i]
            current_bar_data = self.data.iloc[:i+1]
            current_price = self.data['close'].iloc[i]
            
            # 检查现有持仓
            positions_to_remove = []
            for pos_idx, position in enumerate(positions):
                entry_price = position['entry_price']
                entry_bar = position['entry_bar']
                current_bar = i - entry_bar
                
                # 计算止损信号
                stop_signal = strategy.calculate_stop(
                    data=current_bar_data,
                    position_type=position_type,
                    entry_price=entry_price,
                    current_bar=current_bar
                )
                
                # 记录止损信号
                stoploss_signals_log.append({
                    'date': current_date,
                    'triggered': stop_signal.triggered,
                    'stop_price': stop_signal.stop_price,
                    'current_price': stop_signal.current_price,
                    'stop_type': stop_signal.stop_type,
                    'metadata': stop_signal.metadata
                })
                
                # 检查是否触发止损
                exit_reason = None
                if stop_signal.triggered:
                    exit_reason = 'stoploss'
                
                # 检查是否达到止盈目标
                if exit_target is not None and exit_reason is None:
                    if position_type == 'long':
                        if current_price >= entry_price * (1 + exit_target):
                            exit_reason = 'target'
                    else:
                        if current_price <= entry_price * (1 - exit_target):
                            exit_reason = 'target'
                
                # 如果触发退出条件
                if exit_reason:
                    # 计算盈亏
                    if position_type == 'long':
                        pnl = (current_price - entry_price) * position['shares']
                    else:
                        pnl = (entry_price - current_price) * position['shares']
                    
                    # 扣除佣金和滑点
                    commission_cost = (entry_price + current_price) * position['shares'] * self.commission
                    slippage_cost = (entry_price + current_price) * position['shares'] * self.slippage
                    net_pnl = pnl - commission_cost - slippage_cost
                    
                    pnl_pct = net_pnl / (entry_price * position['shares'])
                    
                    # 创建交易记录
                    trade = Trade(
                        entry_date=position['entry_date'],
                        exit_date=current_date,
                        entry_price=entry_price,
                        exit_price=current_price,
                        shares=position['shares'],
                        position_type=position_type,
                        pnl=net_pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        stop_type=strategy.name,
                        holding_period=current_bar
                    )
                    trades.append(trade)
                    
                    # 更新资金
                    capital += net_pnl
                    positions_to_remove.append(pos_idx)
            
            # 移除已平仓的持仓
            for idx in sorted(positions_to_remove, reverse=True):
                positions.pop(idx)
            
            # 检查是否有新的入场信号
            if entry_signal.iloc[i] and len(positions) < max_positions:
                # 计算可购买股数
                available_capital = capital / (1 + self.commission + self.slippage)
                shares = min(shares_per_trade, int(available_capital / current_price))
                
                if shares > 0:
                    # 开仓
                    entry_price = current_price * (1 + self.slippage) if position_type == 'long' else current_price * (1 - self.slippage)
                    position = {
                        'entry_date': current_date,
                        'entry_price': entry_price,
                        'entry_bar': i,
                        'shares': shares
                    }
                    positions.append(position)
            
            # 记录当日权益
            position_value = sum(
                (current_price - pos['entry_price']) * pos['shares'] if position_type == 'long'
                else (pos['entry_price'] - current_price) * pos['shares']
                for pos in positions
            )
            total_equity = capital + position_value
            equity_curve.append(total_equity)
        
        # 平仓所有剩余持仓
        if positions:
            final_price = self.data['close'].iloc[-1]
            final_date = self.data.index[-1]
            for position in positions:
                if position_type == 'long':
                    pnl = (final_price - position['entry_price']) * position['shares']
                else:
                    pnl = (position['entry_price'] - final_price) * position['shares']
                
                commission_cost = (position['entry_price'] + final_price) * position['shares'] * self.commission
                slippage_cost = (position['entry_price'] + final_price) * position['shares'] * self.slippage
                net_pnl = pnl - commission_cost - slippage_cost
                pnl_pct = net_pnl / (position['entry_price'] * position['shares'])
                
                trade = Trade(
                    entry_date=position['entry_date'],
                    exit_date=final_date,
                    entry_price=position['entry_price'],
                    exit_price=final_price,
                    shares=position['shares'],
                    position_type=position_type,
                    pnl=net_pnl,
                    pnl_pct=pnl_pct,
                    exit_reason='end',
                    stop_type=strategy.name,
                    holding_period=len(self.data) - position['entry_bar']
                )
                trades.append(trade)
                capital += net_pnl
        
        # 计算回测指标
        final_capital = capital
        total_return = (final_capital - self.initial_capital) / self.initial_capital
        
        # 年化收益率
        n_days = len(self.data)
        n_years = n_days / 252
        if n_years > 0:
            annualized_return = (final_capital / self.initial_capital) ** (1 / n_years) - 1
        else:
            annualized_return = 0.0
        
        # 权益曲线
        equity_series = pd.Series(equity_curve, index=self.data.index)
        daily_returns = equity_series.pct_change().dropna()
        
        # 夏普比率
        sharpe = calculate_sharpe_ratio(daily_returns)
        
        # 最大回撤
        max_dd = calculate_max_drawdown(equity_series)
        
        # 交易统计
        total_trades = len(trades)
        if total_trades > 0:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]
            win_rate = len(winning_trades) / total_trades
            
            gross_profit = sum(t.pnl for t in winning_trades)
            gross_loss = abs(sum(t.pnl for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            average_trade_pnl = np.mean([t.pnl for t in trades])
            average_holding_period = np.mean([t.holding_period for t in trades])
            
            # 连续盈利/亏损
            max_consecutive_wins = self._max_consecutive(trades, True)
            max_consecutive_losses = self._max_consecutive(trades, False)
        else:
            win_rate = 0.0
            profit_factor = 0.0
            average_trade_pnl = 0.0
            average_holding_period = 0.0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            winning_trades = []
            losing_trades = []
        
        # 创建回测结果
        result = BacktestResult(
            strategy_name=strategy.name,
            strategy_params=strategy.get_params(),
            initial_capital=self.initial_capital,
            final_capital=final_capital,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            profit_factor=profit_factor,
            average_trade_pnl=average_trade_pnl,
            average_holding_period=average_holding_period,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            trades=trades,
            equity_curve=equity_series,
            stoploss_signals=stoploss_signals_log
        )
        
        return result
    
    def _max_consecutive(self, trades: List[Trade], winning: bool) -> int:
        """计算最大连续盈利/亏损次数"""
        if not trades:
            return 0
        
        max_count = 0
        current_count = 0
        
        for trade in trades:
            is_winning = trade.pnl > 0
            if is_winning == winning:
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count
    
    def compare_strategies(
        self,
        strategies: List[BaseStoploss],
        **backtest_kwargs
    ) -> pd.DataFrame:
        """
        比较多个止损策略的表现
        
        Args:
            strategies: 止损策略列表
            **backtest_kwargs: 传递给 run() 的参数
            
        Returns:
            DataFrame: 策略对比结果
        """
        results = []
        for strategy in strategies:
            result = self.run(strategy, **backtest_kwargs)
            results.append(result.to_dict())
        
        df = pd.DataFrame(results)
        df = df.set_index('strategy_name')
        return df
    
    def plot_equity_curve(
        self,
        result: BacktestResult,
        title: str = "Equity Curve",
        save_path: Optional[str] = None
    ):
        """
        绘制权益曲线
        
        Args:
            result: 回测结果
            title: 图表标题
            save_path: 保存路径 (可选)
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 绘制权益曲线
            ax.plot(result.equity_curve.index, result.equity_curve.values, 
                   linewidth=2, label='Equity')
            
            # 绘制初始资金线
            ax.axhline(y=self.initial_capital, color='gray', 
                      linestyle='--', linewidth=1, label='Initial Capital')
            
            # 格式化
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Date')
            ax.set_ylabel('Capital ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 格式化 x 轴日期
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            warnings.warn("matplotlib not installed. Install with: pip install matplotlib")
