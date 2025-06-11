"""Multi-Armed Bandit implementation for adaptive speculative decoding."""

import math
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, NamedTuple, Optional, Tuple
import time
import threading

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import queue


class MABConfig:
    """Class for handling MAB configuration validation and parsing."""

    @staticmethod
    def validate_configs(config_str: str) -> List[str]:
        """Validate MAB configurations string.

        Args:
            config_str: Comma-separated list of MAB configurations

        Returns:
            List of validated MAB configurations

        Raises:
            ValueError: If any configuration doesn't match the required format
        """
        configs = config_str.split(",")
        pattern = re.compile(r"^\d+_\d+_\d+$")

        for config in configs:
            if not pattern.match(config):
                raise ValueError(
                    f"Invalid MAB configuration format: '{config}'. "
                    f"Each configuration must follow the format '<speculative_num_steps>_<topk>_<draft_tokens>' "
                    f"where each component is a positive integer."
                )

            # Additional validation to ensure each component is a positive integer
            steps, topk, draft_tokens = map(int, config.split("_"))
            if steps <= 0 or topk <= 0 or draft_tokens <= 0:
                raise ValueError(
                    f"Invalid MAB configuration values: '{config}'. "
                    f"Steps, topk, and draft_tokens must all be positive integers."
                )

        return configs

    @staticmethod
    def parse_config(config: str) -> Tuple[int, int, int]:
        """Parse a single MAB configuration string.

        Args:
            config: A string in format '<steps>_<topk>_<draft_tokens>'

        Returns:
            Tuple of (steps, topk, draft_tokens)
        """
        steps, topk, draft_tokens = map(int, config.split("_"))
        return steps, topk, draft_tokens

    @staticmethod
    def format_config(steps: int, topk: int, draft_tokens: int) -> str:
        """Format parameters into a MAB configuration string.

        Args:
            steps: Number of speculative steps
            topk: Top-k value for draft model
            draft_tokens: Number of draft tokens

        Returns:
            Formatted configuration string
        """
        return f"{steps}_{topk}_{draft_tokens}"


@dataclass
class MABStrategyMetrics:
    """A single entry of metrics data for one step."""

    window_size: int
    rewards: Deque[float] = field(default_factory=deque, init=False)
    accept_lengths: Deque[float] = field(default_factory=deque, init=False)

    def __post_init__(self):
        """Initialize deque with window_size."""
        self.rewards = deque(maxlen=self.window_size)
        self.accept_lengths = deque(maxlen=self.window_size)

    def add_metrics(self, reward: float, accept_length: float):
        """Add metrics for a single step in the sliding window."""
        self.rewards.append(reward)
        self.accept_lengths.append(accept_length)


class SpeculativeResources(NamedTuple):
    draft_attn_backend: object
    draft_cuda_graph_runner: object
    target_attn_backend: object
    target_cuda_graph_runner: object


class BaseMAB:
    """Base class for Multi-Armed Bandit implementations.

    Each MAB instance manages strategy selection within a specific group (e.g., batch size group).
    Each instance maintains its own metrics for each strategy and handles its own time window.
    """

    def __init__(self, strategies: List[str], window_size: int = 1000):
        self.strategies = strategies
        # Initialize metrics for each strategy
        self.strategy_metrics = {s: MABStrategyMetrics(window_size) for s in strategies}

    def select_strategy(self, valid_strategies: List[str]) -> str:
        """Choose a strategy based on the algorithm and filtering rules.

        Args:
            valid_strategies: List of valid strategies to choose from

        Returns:
            Selected strategy
        """
        raise NotImplementedError


class EpsilonGreedyMAB(BaseMAB):
    """Epsilon-Greedy implementation of MAB."""

    def __init__(self, strategies: List[str], epsilon: float = 0.1, **kwargs):
        super().__init__(strategies, **kwargs)
        self.epsilon = epsilon

    def select_strategy(self, valid_strategies: List[str]) -> str:
        """Choose strategy using epsilon-greedy approach."""
        # With probability epsilon, choose randomly
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_strategies)

        # Otherwise choose the best strategy based on median reward
        strategy_scores = {}
        for s in valid_strategies:
            rewards = self.strategy_metrics[s].rewards
            strategy_scores[s] = float(np.median(rewards)) if rewards else float("inf")

        return max(strategy_scores, key=strategy_scores.get)


class UCB1MAB(BaseMAB):
    """UCB1(Upper Confidence Bound) implementation of MAB."""

    def __init__(self, strategies: List[str], **kwargs):
        super().__init__(strategies, **kwargs)

    def select_strategy(self, valid_strategies: List[str]) -> str:
        """Choose strategy using UCB1 algorithm."""
        # Calculate total pulls based on current window sizes
        total_pulls = sum(
            len(metrics.rewards) for metrics in self.strategy_metrics.values()
        )

        # Calculate UCB scores for each strategy
        strategy_scores = {
            s: self._ucb_score(self.strategy_metrics[s], total_pulls)
            for s in valid_strategies
        }
        return max(strategy_scores, key=strategy_scores.get)

    def _ucb_score(self, metrics: MABStrategyMetrics, total_pulls: int) -> float:
        """Calculate UCB score for a strategy."""
        pulls_i = len(metrics.rewards)
        if pulls_i == 0:  # force exploration if no recent data
            return float("inf")

        # if trimmed mean is None, use inf to force exploration
        r_hat = self.get_trimmed_mean(metrics.rewards) or float("inf")
        return r_hat + math.sqrt(2 * math.log(total_pulls) / pulls_i)

    @staticmethod
    def get_trimmed_mean(
        values: Deque[float], trim_percent: float = 0.1
    ) -> Optional[float]:
        """Calculate trimmed mean for the values."""
        if not values:
            return None

        sorted_vals = sorted(values)
        n_trim = int(len(sorted_vals) * trim_percent)
        return float(
            np.mean(sorted_vals[n_trim:-n_trim])
            if len(sorted_vals) > 2 * n_trim
            else np.mean(sorted_vals)
        )


class MABGroupManager:
    """Manages groups and their associated MAB instances.

    Each group (e.g., batch size group) has its own MAB instance that learns
    independently which strategies work best for that group.
    """

    # Map algorithm names to their factory functions
    ALGORITHM_FACTORIES = {
        "EG": lambda strategies, window_size: EpsilonGreedyMAB(
            strategies=strategies, window_size=window_size
        ),
        "UCB1": lambda strategies, window_size: UCB1MAB(
            strategies=strategies, window_size=window_size
        ),
    }

    def __init__(
        self,
        strategies: List[str],
        algorithm: str = "EG",
        window_size: int = 1000,
    ):
        # Define batch size groups for MAB
        # TODO: batch_size should be determined more refinely
        groups = (
            list(range(1, 32))  # Fine-grained for small batches
            + list(range(32, 128, 8))  # Medium step size for medium batches
            + list(range(128, 257, 32))  # Large step size for large batches
        )
        self.groups = sorted(groups)
        self.strategies = strategies

        # Validate algorithm type
        algorithm = algorithm.upper()
        if algorithm not in self.ALGORITHM_FACTORIES:
            raise ValueError(
                f"Unsupported MAB algorithm: {algorithm}. "
                f"Must be one of: {', '.join(sorted(self.ALGORITHM_FACTORIES.keys()))}"
            )
        self.algorithm = algorithm
        self.name = f"{algorithm},{','.join(strategies)}"

        self.window_size = window_size

        # Initialize MAB instances
        self.mabs: Dict[int, BaseMAB] = {}
        self._init_mabs()
        self.accept_lengths = {}
        
        # Strategy history tracking
        self.strategy_history = []
        self.batch_size_history = []
        self.timestamp_history = []
        self.reward_history = []
        self.start_time = time.time()
        
        # Real-time plotting
        self.enable_realtime_plot = False
        self.plot_queue = queue.Queue()
        self.plot_thread = None

    def _init_mabs(self):
        """Initialize MAB instances for each group.

        Creates a MAB instance for each group based on the specified algorithm (EG or UCB1).
        Each MAB instance manages its own metrics."""
        factory = self.ALGORITHM_FACTORIES[self.algorithm]
        self.mabs = {
            group: factory(strategies=self.strategies, window_size=self.window_size)
            for group in self.groups
        }

    def _get_valid_strategies(self, batch_size: int) -> List[str]:
        """Get valid strategies for a given batch size to prevent OOM.

        This method filters strategies based on memory constraints to avoid OOM errors.

        Args:
            batch_size: Current batch size for OOM checking

        Returns:
            List of valid strategy names that won't cause OOM
        """
        get_draft_tokens = lambda strategy: MABConfig.parse_config(strategy)[-1]

        # Check heuristic OOM condition
        valid_strategies = [
            s for s in self.strategies if get_draft_tokens(s) * batch_size <= 2048
        ]
        # If all strategies would cause OOM, return the safest one - the one with the least tokens
        if not valid_strategies:
            return [min(self.strategies, key=lambda s: get_draft_tokens(s))]

        return valid_strategies

    def _get_group(self, batch_size: int) -> int:
        """Find the largest group <= batch_size."""
        for group in reversed(self.groups):
            if group <= batch_size:
                return group
        return self.groups[0]  # Fallback to the smallest group

    def select_strategy(self, batch_size: int) -> str:
        """Select strategy for given batch size using appropriate MAB.

        This method handles both filtering strategies to prevent OOM and selecting
        the best strategy using the MAB algorithm.
        """
        # Fast path for single strategy
        if len(self.strategies) == 1:
            selected = self.strategies[0]
        else:
            # Get appropriate group and valid strategies
            group = self._get_group(batch_size)
            valid_strategies = self._get_valid_strategies(batch_size)

            # Select strategy using the MAB algorithm
            selected = self.mabs[group].select_strategy(valid_strategies)
        
        # Track the selection
        self.strategy_history.append(selected)
        self.batch_size_history.append(batch_size)
        self.timestamp_history.append(time.time() - self.start_time)
        
        # Queue for real-time plotting if enabled
        if self.enable_realtime_plot:
            self.plot_queue.put({
                'timestamp': self.timestamp_history[-1],
                'strategy': selected,
                'batch_size': batch_size
            })
        
        return selected

    def record_strategy_metrics(
        self, batch_size: int, strategy: str, reward: float, accept_length: float
    ):
        """Record metrics for a strategy in appropriate group."""
        group = self._get_group(batch_size)
        self.mabs[group].strategy_metrics[strategy].add_metrics(reward, accept_length)
        
        # Track reward history
        self.reward_history.append(reward)

    def get_stable_accept_length(self, strategy: str) -> float:
        """Get stable accept_length across all groups.

        This method calculates a batch-size-independent metric by looking at the
        median accept_length across all batch size groups. This provides a more
        stable metric since accept_length is primarily determined by the strategy
        itself rather than the batch size.

        Args:
            strategy: Strategy to get median accept_length for

        Returns:
            Median accept_length across all groups, or 0.0 if no data is available
        """
        # Only recalculate 10% of the time to reduce overhead
        if np.random.random() < 0.1 or not self.accept_lengths.get(strategy, None):
            all_group_accept_lengths = np.concatenate(
                [
                    self.mabs[group].strategy_metrics[strategy].accept_lengths
                    for group in self.groups
                ]
            )
            self.accept_lengths[strategy] = (
                np.median(all_group_accept_lengths)
                if all_group_accept_lengths.size > 0
                else 0.0
            )

        return self.accept_lengths.get(strategy, 0.0)
    
    def enable_plotting(self, enable: bool = True):
        """Enable or disable real-time plotting."""
        self.enable_realtime_plot = enable
        if enable and self.plot_thread is None:
            self.plot_thread = threading.Thread(target=self._realtime_plot_worker, daemon=True)
            self.plot_thread.start()
    
    def _realtime_plot_worker(self):
        """Worker thread for real-time plotting."""
        plt.ion()
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'MAB Strategy Selection ({self.algorithm})')
        
        while self.enable_realtime_plot:
            try:
                # Get data from queue with timeout
                data = self.plot_queue.get(timeout=0.1)
                self.plot_queue.task_done()
                
                # Update plots
                self._update_plots(fig, ax1, ax2, ax3)
                plt.pause(0.01)
                
            except queue.Empty:
                continue
                
        plt.close(fig)
    
    def _update_plots(self, fig, ax1, ax2, ax3):
        """Update the plot axes with current data."""
        if not self.timestamp_history:
            return
            
        # Clear axes
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # Plot 1: Strategy selection over time
        strategy_indices = {s: i for i, s in enumerate(self.strategies)}
        strategy_nums = [strategy_indices[s] for s in self.strategy_history]
        ax1.scatter(self.timestamp_history, strategy_nums, alpha=0.6, s=10)
        ax1.set_yticks(list(strategy_indices.values()))
        ax1.set_yticklabels(list(strategy_indices.keys()))
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Strategy')
        ax1.set_title('Strategy Selection Over Time')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Batch size over time
        ax2.plot(self.timestamp_history, self.batch_size_history, 'b-', alpha=0.7)
        ax2.fill_between(self.timestamp_history, 0, self.batch_size_history, alpha=0.3)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Batch Size')
        ax2.set_title('Batch Size Over Time')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Reward over time (if available)
        if self.reward_history:
            # Moving average for rewards
            window = min(50, len(self.reward_history))
            if len(self.reward_history) >= window:
                reward_ma = np.convolve(self.reward_history, np.ones(window)/window, mode='valid')
                time_ma = self.timestamp_history[window-1:len(self.reward_history)]
                ax3.plot(time_ma, reward_ma, 'g-', alpha=0.8, label='Moving Avg')
            
            ax3.scatter(self.timestamp_history[:len(self.reward_history)], 
                       self.reward_history, alpha=0.3, s=5, label='Raw')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Reward')
            ax3.set_title('Reward Over Time')
            ax3.grid(True, alpha=0.3)
            ax3.legend()
        
        fig.tight_layout()
    
    def save_history_plot(self, filename: str):
        """Save a plot of the strategy selection history."""
        if not self.timestamp_history:
            print("No history to plot")
            return
            
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'MAB Strategy Selection History ({self.algorithm})')
        
        self._update_plots(fig, ax1, ax2, ax3)
        
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {filename}")
    
    def get_strategy_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each strategy."""
        stats = {}
        for strategy in self.strategies:
            strategy_count = self.strategy_history.count(strategy)
            stats[strategy] = {
                'count': strategy_count,
                'percentage': strategy_count / len(self.strategy_history) * 100 if self.strategy_history else 0,
                'avg_batch_size': np.mean([bs for s, bs in zip(self.strategy_history, self.batch_size_history) if s == strategy]) if strategy_count > 0 else 0
            }
        return stats
