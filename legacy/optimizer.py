"""
参数优化器模块

派蒙的参数优化工具！帮助找到最优的止损策略参数~
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Callable, Tuple
from itertools import product
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings

try:
    from .stoploss_strategies import BaseStoploss
    from .backtester import StoplossBacktester, BacktestResult
except ImportError:
    from stoploss_strategies import BaseStoploss
    from backtester import StoplossBacktester, BacktestResult


class StoplossOptimizer:
    """
    止损策略参数优化器
    
    通过网格搜索或随机搜索找到最优的止损策略参数。
    
    参数:
        strategy: 止损策略实例
        backtester: 回测器实例 (可选，如不提供则创建新的)
        n_jobs: 并行工作进程数 (默认 -1 表示使用所有 CPU)
    """
    
    def __init__(
        self,
        strategy: BaseStoploss,
        backtester: Optional[StoplossBacktester] = None,
        n_jobs: int = -1
    ):
        self.strategy = strategy
        self.backtester = backtester
        self.n_jobs = n_jobs
        self.optimization_history = []
    
    def optimize(
        self,
        param_grid: Dict[str, List[Any]],
        metric: str = 'sharpe_ratio',
        maximize: bool = True,
        n_folds: int = 1,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        网格搜索优化参数
        
        Args:
            param_grid: 参数网格，如 {'period': [10, 14, 20], 'multiplier': [1.5, 2.0, 2.5]}
            metric: 优化指标 ('sharpe_ratio', 'total_return', 'max_drawdown', 'win_rate' 等)
            maximize: 是否最大化指标 (默认 True)
            n_folds: 交叉验证折数 (默认 1，不做交叉验证)
            verbose: 是否打印进度 (默认 True)
            
        Returns:
            最优参数字典
        """
        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(product(*param_values))
        
        if verbose:
            print(f"派蒙开始优化啦~ 共 {len(all_combinations)} 种参数组合")
        
        results = []
        
        # 单进程或并行执行
        if self.n_jobs == 1:
            for params in all_combinations:
                param_dict = dict(zip(param_names, params))
                result = self._evaluate_params(param_dict, metric, n_folds)
                results.append((param_dict, result))
                if verbose:
                    print(f"  测试参数：{param_dict} -> {metric}: {result:.4f}")
        else:
            # 并行执行
            n_workers = self.n_jobs if self.n_jobs > 0 else None
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                for params in all_combinations:
                    param_dict = dict(zip(param_names, params))
                    future = executor.submit(self._evaluate_params, param_dict, metric, n_folds)
                    futures.append((param_dict, future))
                
                for param_dict, future in futures:
                    result = future.result()
                    results.append((param_dict, result))
                    if verbose:
                        print(f"  测试参数：{param_dict} -> {metric}: {result:.4f}")
        
        # 找到最优参数
        if maximize:
            best_idx = np.argmax([r[1] for r in results])
        else:
            best_idx = np.argmin([r[1] for r in results])
        
        best_params, best_score = results[best_idx]
        
        if verbose:
            print(f"\n✨ 派蒙找到最优参数啦！")
            print(f"  最优参数：{best_params}")
            print(f"  最优 {metric}: {best_score:.4f}")
        
        # 保存优化历史
        self.optimization_history = [
            {'params': params, 'score': score}
            for params, score in results
        ]
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'metric': metric
        }
    
    def _evaluate_params(
        self,
        params: Dict[str, Any],
        metric: str,
        n_folds: int
    ) -> float:
        """评估一组参数"""
        # 创建策略副本并设置参数
        strategy_copy = self.strategy.__class__(**self.strategy.get_params())
        strategy_copy.update_params(**params)
        
        # 如果 backtester 未提供，需要先设置数据
        if self.backtester is None:
            raise ValueError("backtester not provided")
        
        # 运行回测
        result = self.backtester.run(strategy_copy)
        
        # 获取指标值
        if metric == 'sharpe_ratio':
            score = result.sharpe_ratio
        elif metric == 'total_return':
            score = result.total_return
        elif metric == 'annualized_return':
            score = result.annualized_return
        elif metric == 'max_drawdown':
            score = -result.max_drawdown  # 回撤越小越好
        elif metric == 'win_rate':
            score = result.win_rate
        elif metric == 'profit_factor':
            score = result.profit_factor
        elif metric == 'calmar_ratio':
            if result.max_drawdown > 0:
                score = result.annualized_return / result.max_drawdown
            else:
                score = 0
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        return score
    
    def random_search(
        self,
        param_distributions: Dict[str, Tuple[float, float]],
        n_iter: int = 50,
        metric: str = 'sharpe_ratio',
        maximize: bool = True,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        随机搜索优化参数
        
        Args:
            param_distributions: 参数分布，如 {'period': (10, 50), 'multiplier': (1.0, 3.0)}
            n_iter: 迭代次数 (默认 50)
            metric: 优化指标
            maximize: 是否最大化指标
            verbose: 是否打印进度
            
        Returns:
            最优参数字典
        """
        if verbose:
            print(f"派蒙开始随机搜索优化~ 共 {n_iter} 次迭代")
        
        results = []
        
        for i in range(n_iter):
            # 随机采样参数
            params = {}
            for param_name, (low, high) in param_distributions.items():
                if isinstance(low, int) and isinstance(high, int):
                    params[param_name] = np.random.randint(low, high + 1)
                else:
                    params[param_name] = np.random.uniform(low, high)
            
            # 评估参数
            score = self._evaluate_params(params, metric, n_folds=1)
            results.append((params, score))
            
            if verbose and (i + 1) % 10 == 0:
                print(f"  迭代 {i+1}/{n_iter}, 当前最佳 {metric}: {max([r[1] for r in results]):.4f}")
        
        # 找到最优参数
        if maximize:
            best_idx = np.argmax([r[1] for r in results])
        else:
            best_idx = np.argmin([r[1] for r in results])
        
        best_params, best_score = results[best_idx]
        
        if verbose:
            print(f"\n✨ 派蒙找到最优参数啦！")
            print(f"  最优参数：{best_params}")
            print(f"  最优 {metric}: {best_score:.4f}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results,
            'metric': metric
        }
    
    def plot_optimization_results(
        self,
        param1: str,
        param2: Optional[str] = None,
        metric: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        绘制优化结果可视化
        
        Args:
            param1: 第一个参数名 (x 轴)
            param2: 第二个参数名 (y 轴，可选)
            metric: 指标名 (颜色)
            save_path: 保存路径 (可选)
        """
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            if not self.optimization_history:
                warnings.warn("No optimization history available")
                return
            
            # 转换为 DataFrame
            df = pd.DataFrame([
                {**result['params'], 'score': result['score']}
                for result in self.optimization_history
            ])
            
            if param1 not in df.columns:
                warnings.warn(f"Parameter '{param1}' not found in optimization history")
                return
            
            fig = plt.figure(figsize=(10, 6))
            
            if param2 is None:
                # 单参数图
                ax = fig.add_subplot(111)
                ax.scatter(df[param1], df['score'], alpha=0.6, s=50)
                ax.set_xlabel(param1)
                ax.set_ylabel(metric or 'Score')
                ax.set_title(f'{param1} vs {metric or "Score"}')
                ax.grid(True, alpha=0.3)
            else:
                # 双参数热力图
                if param2 not in df.columns:
                    warnings.warn(f"Parameter '{param2}' not found in optimization history")
                    return
                
                ax = fig.add_subplot(111, projection='3d')
                scatter = ax.scatter(df[param1], df[param2], df['score'],
                                   c=df['score'], cmap='viridis', s=50)
                ax.set_xlabel(param1)
                ax.set_ylabel(param2)
                ax.set_zlabel(metric or 'Score')
                ax.set_title(f'Parameter Optimization Surface')
                plt.colorbar(scatter, label=metric or 'Score')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
            
        except ImportError:
            warnings.warn("matplotlib not installed. Install with: pip install matplotlib")
    
    def get_optimization_summary(self) -> pd.DataFrame:
        """获取优化结果摘要"""
        if not self.optimization_history:
            return pd.DataFrame()
        
        df = pd.DataFrame([
            {**result['params'], 'score': result['score']}
            for result in self.optimization_history
        ])
        
        return df.sort_values('score', ascending=False)
