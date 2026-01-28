"""Multi-Armed Bandit implementation for adaptive speculative decoding."""

import logging
import math
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
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


class BucketedEpsilonGreedyMAB(BaseMAB):
    """Hybrid strategy that combines PredefinedMapping and EpsilonGreedy.

    Args:
        bs_threshold: List of batch size thresholds for predefined mappings.

        epsilon: Exploration rate for epsilon-greedy strategy.

    This algorithm uses predefined mappings for specific batch size ranges
    where we have strong priors about optimal configurations, while allowing
    epsilon-greedy learning for other ranges where exploration is beneficial.
    """

    def __init__(self, strategies: List[str], bs_threshold: Optional[List[int]] = None, epsilon: float = 0.1, **kwargs):
        super().__init__(strategies, **kwargs)
        self.epsilon = epsilon

        # Group strategies by draft_tokens for better organization
        self.draft_tokens_groups = self._group_by_draft_tokens()
        self.num_bucket = len(self.draft_tokens_groups)

        # Use default threshold if not provided
        if bs_threshold is None:
            bs_threshold = [1, 2, 5, 21]

        # Validate bs_threshold length matches number of buckets
        if len(bs_threshold) != self.num_bucket:
            raise ValueError(
                f"bs_threshold length ({len(bs_threshold)}) must match the number of "
                f"unique draft_tokens values ({self.num_bucket}) in strategies. "
                f"Found draft_tokens groups: {sorted(self.draft_tokens_groups.keys())}"
            )

        self.bs_threshold = bs_threshold

        self._create_buckets()

        logger.info("BucketedEpsilonGreedyMAB initialized:")
        logger.info(f"  self.batch_strategy_mapping: {self.batch_strategy_mapping}")

    def _group_by_draft_tokens(self) -> Dict[int, List[str]]:
        """Group strategies by their draft_tokens value."""
        groups = {}
        for strategy in self.strategies:
            _, _, draft_tokens = MABConfig.parse_config(strategy)
            if draft_tokens not in groups:
                groups[draft_tokens] = []
            groups[draft_tokens].append(strategy)
        return groups

    def _create_buckets(self) -> List[Tuple[int, int, str]]:
        """Create predefined batch size range mappings."""
        self.bucket_list = list(self.draft_tokens_groups.keys())
        self.bucket_list.sort(reverse=True)  # Sort by draft_tokens descending

        # Create batch size ranges and allocate strategies
        self.batch_bucket_mapping = []  # (min_batch, max_batch, draft_tokens)
        self.batch_strategy_mapping = {}  # (min_batch, max_batch): list of strategies
        for i in range(len(self.bs_threshold)):
            min_batch = self.bs_threshold[i]
            if i + 1 < len(self.bs_threshold):
                max_batch = self.bs_threshold[i + 1] - 1
            else:
                max_batch = float("inf")
            self.batch_bucket_mapping.append((min_batch, max_batch, self.bucket_list[i]))
            self.batch_strategy_mapping[(min_batch, max_batch)] = self.draft_tokens_groups[self.bucket_list[i]]

    def select_strategy(self, batch_size: int = None) -> str:
        """Choose strategy using hybrid approach."""
        if batch_size is None:
            raise ValueError("BucketedEpsilonGreedyMAB requires batch_size parameter")

        for min_batch, max_batch, draft_tokens in self.batch_bucket_mapping:
            if min_batch <= batch_size <= max_batch:
                valid_strategies = self.draft_tokens_groups[draft_tokens]
                # If only one strategy in this bucket, use it directly
                if len(valid_strategies) == 1:
                    return valid_strategies[0]
                # Otherwise, use epsilon-greedy among valid strategies
                return self._epsilon_greedy_select(valid_strategies)
        # Fallback: if no bucket matches, use epsilon-greedy among all strategies
        return self._epsilon_greedy_select(self.strategies)

    def _epsilon_greedy_select(self, valid_strategies: List[str]) -> str:
        """Standard epsilon-greedy selection."""
        # With probability epsilon, choose randomly
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_strategies)

        # Otherwise choose the best strategy based on median reward
        strategy_scores = {}
        for s in valid_strategies:
            rewards = self.strategy_metrics[s].rewards
            strategy_scores[s] = float(np.median(rewards)) if rewards else float("inf")

        return max(strategy_scores, key=strategy_scores.get)

    def get_strategy_bs_range(self, strategy: str) -> Tuple[Optional[int], Optional[int]]:
        """Get batch size range for a given strategy."""
        for min_batch, max_batch, draft_tokens in self.batch_bucket_mapping:
            if strategy in self.draft_tokens_groups[draft_tokens]:
                return min_batch, max_batch
        return None, None


class PredefinedMappingStrategy(BaseMAB):
    """Simple predefined strategy selector based on batch size ranges."""

    def __init__(self, strategies: List[str], bs_threshold: List[int] = [1, 2, 5, 21], **kwargs):
        super().__init__(strategies, **kwargs)
        self.bs_threshold = bs_threshold

        assert len(strategies) == len(self.bs_threshold), (
            "PredefinedMappingStrategy requires exactly same mapping number."
        )

        # Parse and sort strategies by draft_tokens (descending)
        strategy_tokens_pairs = []
        for strategy in strategies:
            _, _, draft_tokens = MABConfig.parse_config(strategy)
            strategy_tokens_pairs.append((strategy, draft_tokens))

        strategy_tokens_pairs.sort(key=lambda x: x[1], reverse=True)
        self.sorted_strategies = [pair[0] for pair in strategy_tokens_pairs]
        self.sorted_tokens = [pair[1] for pair in strategy_tokens_pairs]

        # Create batch size ranges and allocate strategies
        self.batch_ranges = []
        for i in range(len(self.bs_threshold)):
            min_batch = self.bs_threshold[i]
            if i + 1 < len(self.bs_threshold):
                max_batch = self.bs_threshold[i + 1] - 1
            else:
                max_batch = float("inf")
            self.batch_ranges.append((min_batch, max_batch, self.sorted_strategies[i]))

        logger.info("FixedMappingStrategy initialized with sorted strategies (steps_topk_verifyTokens):")
        for min_batch, max_batch, strategy in self.batch_ranges:
            tokens = MABConfig.parse_config(strategy)[2]
            range_str = (
                f"{min_batch}"
                if max_batch == min_batch
                else f"{min_batch}-{max_batch if max_batch != float('inf') else '∞'}"
            )
            logger.info(f"  Batch {range_str}: {strategy} (verify {tokens} tokens)")

    def select_strategy(self, batch_size: int = None) -> str:
        """Choose strategy based on batch size"""
        if batch_size is None:
            raise ValueError("PredefinedMappingStrategy requires batch_size parameter")

        for min_batch, max_batch, strategy in self.batch_ranges:
            if min_batch <= batch_size <= max_batch:
                return strategy

    def get_strategy_bs_range(self, strategy: str) -> Tuple[int, int]:
        """Get batch size range for a given strategy."""
        for min_batch, max_batch, s in self.batch_ranges:
            if s == strategy:
                return min_batch, max_batch
        return None, None


class MABGroupManager:
    """Manages groups and their associated MAB instances.

    Each group (e.g., batch size group) has its own MAB instance that learns
    independently which strategies work best for that group.
    """

    # Map algorithm names to their factory functions
    ALGORITHM_FACTORIES = {
        "EG": lambda strategies, window_size, bs_threshold: EpsilonGreedyMAB(strategies=strategies, window_size=window_size),
        "UCB1": lambda strategies, window_size, bs_threshold: UCB1MAB(strategies=strategies, window_size=window_size),
        "PREDEFINED": lambda strategies, window_size, bs_threshold: PredefinedMappingStrategy(
            strategies=strategies, window_size=window_size, bs_threshold=bs_threshold
        ),
        "BEG": lambda strategies, window_size, bs_threshold: BucketedEpsilonGreedyMAB(strategies=strategies, window_size=window_size, bs_threshold=bs_threshold),
    }

    def __init__(
        self,
        strategies: List[str],
        algorithm: str = "EG",
        window_size: int = 1000,
        bs_threshold: Optional[List[int]] = None,
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
        self.bs_threshold = bs_threshold

        # Initialize MAB instances
        self.mabs: Dict[int, BaseMAB] = {}
        self._init_mabs()
        self.accept_lengths = {}

    def _init_mabs(self):
        """Initialize MAB instances for each group.

        Creates a MAB instance for each group based on the specified algorithm (EG or UCB1).
        Each MAB instance manages its own metrics."""
        factory = self.ALGORITHM_FACTORIES[self.algorithm]
        self.mabs = {
            group: factory(
                strategies=self.strategies,
                window_size=self.window_size,
                bs_threshold=self.bs_threshold,
            )
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
            return [min(self.strategies, key=get_draft_tokens)]

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
            return self.strategies[0]

        # Get appropriate group for all algorithms
        group = self._get_group(batch_size)

        if self.algorithm in ["PREDEFINED", "BEG"]:
            return self.mabs[group].select_strategy(batch_size=batch_size)
        else:
            # Get valid strategies for other algorithms
            valid_strategies = self._get_valid_strategies(batch_size)

            # Select strategy using the MAB algorithm
            return self.mabs[group].select_strategy(valid_strategies)

    def get_strategy_bs_range(self, strategy: str) -> Tuple[int, int]:
        """Get batch size range for a given strategy."""
        if self.algorithm in ["PREDEFINED", "BEG"]:
            return self.mabs[1].get_strategy_bs_range(strategy)
        else:
            return None, None

    def record_strategy_metrics(
        self, batch_size: int, strategy: str, reward: float, accept_length: float
    ):
        """Record metrics for a strategy in appropriate group."""
        group = self._get_group(batch_size)
        self.mabs[group].strategy_metrics[strategy].add_metrics(reward, accept_length)

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