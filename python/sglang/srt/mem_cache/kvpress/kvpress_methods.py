# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Adapted for SGLang by contributors
# SPDX-License-Identifier: Apache-2.0

"""
KV Cache Compression Methods for SGLang.

This module implements various KV cache compression algorithms adapted from
the KVPress library (https://github.com/IsaacRe/kvpress).

Original Work:
    KVPress: Plug-and-play KV Cache Compression for LLMs
    Authors: NVIDIA Corporation & Affiliates
    License: Apache-2.0
    Paper: https://arxiv.org/abs/2410.00161
    Repository: https://github.com/IsaacRe/kvpress
    
Adaptations for SGLang:
    - Simplified score() interface: only requires keys/values for simple methods
    - Integrated with SGLang's two-level KV cache storage (req_to_token + token_to_kv_pool)
    - Added support for per-layer compression (each layer independently selects tokens)
    - In-place compression to avoid memory overhead
    
Compression Methods Implemented:
    - KnormPress: Based on Key L2 Norm (simple and efficient)
    - RandomPress: Random selection (baseline for comparison)
    - StreamingLLMPress: Window-based with attention sink tokens
    - KeyDiffPress: Key similarity-based compression
    - LagKVPress: Lag-relative information (query-free, attention-free)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class BaseCompressionMethod(ABC):
    """
    Base class for KV cache compression methods.
    
    Attributes
    ----------
    compression_ratio : float
        Fraction of tokens to remove (0.0 = no compression, 1.0 = remove all).
    """
    
    compression_ratio: float = 0.0
    
    @abstractmethod
    def score(
        self,
        layer_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute importance scores for each token in the KV cache.
        
        Higher scores indicate more important tokens that should be kept.
        Lower scores indicate less important tokens that can be pruned.
        
        Parameters
        ----------
        layer_id : int
            The transformer layer index (0-indexed).
        keys : torch.Tensor
            Key tensor with shape [num_tokens, num_kv_heads, head_dim].
        values : torch.Tensor
            Value tensor with shape [num_tokens, num_kv_heads, head_dim].
        **kwargs : dict
            Additional method-specific parameters.
            
        Returns
        -------
        torch.Tensor
            Importance scores with shape [num_tokens].
            Higher scores = more important tokens.
        """
        raise NotImplementedError


@dataclass
class KnormPress(BaseCompressionMethod):
    """
    Key norm-based KV cache compression.
    
    Prunes key-value pairs based on L2 norm of key vectors. This is a simple
    and efficient method that only requires computing the norm of keys.
    
    The intuition is that keys with larger norms tend to have more influence
    on the attention weights, making them more important to keep.
    
    Based on: "Effectively Compress KV Heads for LLM" (https://arxiv.org/pdf/2406.11430)
    
    Attributes
    ----------
    compression_ratio : float
        Fraction of tokens to remove (e.g., 0.3 = remove 30% of tokens).
        
    Examples
    --------
    >>> method = KnormPress(compression_ratio=0.3)
    >>> scores = method.score(layer_id=0, keys=k_tensor, values=v_tensor)
    >>> # Higher scores = larger key norms = more important
    """
    
    def score(
        self,
        layer_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute importance scores as the L2 norm of keys.
        
        Parameters
        ----------
        layer_id : int
            Layer index (not used in this method, but kept for interface consistency).
        keys : torch.Tensor
            Key tensor with shape [num_tokens, num_kv_heads, head_dim].
        values : torch.Tensor
            Value tensor (not used in this method).
            
        Returns
        -------
        torch.Tensor
            L2 norms averaged across heads, shape [num_tokens].
            Higher values indicate more important tokens.
        """
        # Compute L2 norm across head_dim, then average across num_kv_heads
        # Shape: [num_tokens, num_kv_heads]
        key_norms = keys.norm(dim=-1)
        
        # Average across heads to get per-token score
        # Shape: [num_tokens]
        scores = key_norms.mean(dim=-1)
        
        return scores


@dataclass
class RandomPress(BaseCompressionMethod):
    """
    Random KV cache compression for baseline comparison.
    
    Randomly selects which tokens to keep. Useful for establishing baseline
    performance metrics and validating that other compression methods actually
    provide benefits over random selection.
    
    Attributes
    ----------
    compression_ratio : float
        Fraction of tokens to remove.
    seed : int, optional
        Random seed for reproducible results. If None, uses PyTorch's default RNG.
        
    Examples
    --------
    >>> method = RandomPress(compression_ratio=0.3, seed=42)
    >>> scores = method.score(layer_id=0, keys=k_tensor, values=v_tensor)
    """
    
    seed: int = None
    
    def score(
        self,
        layer_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate random scores for each token.
        
        Returns
        -------
        torch.Tensor
            Random scores with shape [num_tokens].
        """
        generator = None
        if self.seed is not None:
            generator = torch.Generator(device=keys.device)
            generator.manual_seed(self.seed + layer_id)  # Different seed per layer
        
        # Generate random scores for each token
        # Shape: [num_tokens]
        num_tokens = keys.shape[0]
        scores = torch.rand(num_tokens, generator=generator, device=keys.device)
        
        return scores


@dataclass
class StreamingLLMPress(BaseCompressionMethod):
    """
    StreamingLLM: Window-based compression with attention sink tokens.
    
    Preserves the first few tokens (sink tokens) and most recent tokens,
    while pruning middle tokens. This is based on the observation that LLMs
    often assign high attention to initial tokens regardless of content.
    
    Based on: "Efficient Streaming Language Models with Attention Sinks"
    (https://arxiv.org/abs/2309.17453)
    
    Attributes
    ----------
    compression_ratio : float
        Fraction of tokens to remove.
    n_sink : int
        Number of initial tokens to always preserve (default: 4).
        These "sink tokens" help maintain model stability.
        
    Examples
    --------
    >>> method = StreamingLLMPress(compression_ratio=0.3, n_sink=4)
    >>> scores = method.score(layer_id=0, keys=k_tensor, values=v_tensor)
    # Will keep first 4 tokens + most recent tokens, prune middle
    """
    
    n_sink: int = 4
    
    def score(
        self,
        layer_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Assign high scores to sink tokens and recent tokens, low scores to middle.
        
        Returns
        -------
        torch.Tensor
            Scores with shape [num_tokens].
            Sink tokens and recent tokens get score=1, middle tokens get score=0.
        """
        num_tokens = keys.shape[0]
        
        assert num_tokens > self.n_sink, (
            f"Input must have more than {self.n_sink} tokens (n_sink), got {num_tokens}"
        )
        
        # Calculate how many tokens to prune
        n_kept = max(1, int(num_tokens * (1 - self.compression_ratio)))
        n_pruned = num_tokens - n_kept
        
        # Create scores: 1 for kept, 0 for pruned
        scores = torch.ones(num_tokens, device=keys.device)
        
        # Prune middle tokens (between sink and recent window)
        # Keep: [0:n_sink] + [n_sink+n_pruned:]
        # Prune: [n_sink:n_sink+n_pruned]
        if n_pruned > 0:
            scores[self.n_sink : self.n_sink + n_pruned] = 0
        
        return scores


@dataclass
class LagKVPress(BaseCompressionMethod):
    """
    LagKV: Lag-relative information-based KV cache compression.
    
    Compresses KV cache by leveraging lag-relative information between sequence
    partitions. Divides sequence into partitions and uses subsequent partitions
    as references for scoring tokens in prior partitions.
    
    Key advantages:
    - Query-free: doesn't need query vectors
    - Attention-free: doesn't need attention weights
    - Flash-attention compatible
    
    Based on: "LagKV: Efficient KV Cache Compression via Lag-Relative Information"
    (https://arxiv.org/abs/2504.04704)
    
    Attributes
    ----------
    compression_ratio : float
        Fraction of tokens to remove.
    n_sink : int
        Number of initial tokens to preserve as attention sinks (default: 4).
    lag_size : int
        Size of each partition for lag-relative scoring (default: 128).
        Sequence is divided into partitions of this size.
        
    Examples
    --------
    >>> method = LagKVPress(compression_ratio=0.3, lag_size=128)
    >>> scores = method.score(layer_id=0, keys=k_tensor, values=v_tensor)
    """
    
    n_sink: int = 4
    lag_size: int = 128
    
    def score(
        self,
        layer_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute lag-relative importance scores.
        
        Returns
        -------
        torch.Tensor
            Scores with shape [num_tokens]. Range [0, 1].
        """
        # keys/values shape: [num_tokens, num_kv_heads, head_dim]
        num_tokens, num_kv_heads, head_dim = keys.shape
        
        # Check if sequence is long enough for compression
        if num_tokens < self.n_sink + 2 * self.lag_size:
            # Too short, keep everything with sliding window preference
            scores = torch.ones(num_tokens, dtype=keys.dtype, device=keys.device)
            if num_tokens > self.n_sink:
                # Give higher scores to more recent tokens
                scores[self.n_sink:] = (
                    torch.arange(num_tokens - self.n_sink, device=keys.device) 
                    / (num_tokens - self.n_sink)
                ).to(keys.dtype)
            return scores
        
        # Reshape to [num_kv_heads, num_tokens, head_dim] for processing
        keys_transposed = keys.transpose(0, 1)  # [num_kv_heads, num_tokens, head_dim]
        values_transposed = values.transpose(0, 1)
        
        # Calculate partition boundaries
        end_idx = self.n_sink + ((num_tokens - self.n_sink) // self.lag_size) * self.lag_size
        tail_len = self.lag_size + num_tokens - end_idx
        
        # Extract partition-able region (exclude sink and tail)
        # Shape: [num_kv_heads, num_partitions, lag_size, head_dim]
        num_partitions = (end_idx - self.n_sink) // self.lag_size
        
        keys_partitions = keys_transposed[:, self.n_sink:end_idx].reshape(
            num_kv_heads, num_partitions, self.lag_size, head_dim
        )
        values_partitions = values_transposed[:, self.n_sink:end_idx].reshape(
            num_kv_heads, num_partitions, self.lag_size, head_dim
        )
        
        # Compute scores using lag-relative information
        key_scores = self._get_states_score(keys_partitions)  # [num_kv_heads, num_partitions-1, lag_size]
        value_scores = self._get_states_score(values_partitions)
        
        # Average key and value scores
        scores = (key_scores + value_scores) / 2
        
        # Normalize scores to [0, 1] by ranking within each partition
        # argsort twice gives ranks
        scores = scores.argsort(dim=-1).argsort(dim=-1).float() / self.lag_size
        
        # Average across heads
        scores = scores.mean(dim=0)  # [num_partitions-1, lag_size]
        
        # Build final scores: sink (all 1s) + partition scores + tail (all 1s)
        sink_scores = torch.ones(self.n_sink, dtype=scores.dtype, device=scores.device)
        tail_scores = torch.ones(tail_len, dtype=scores.dtype, device=scores.device)
        
        # Flatten partition scores
        partition_scores = scores.reshape(-1)  # [(num_partitions-1) * lag_size]
        
        # Concatenate all parts
        final_scores = torch.cat([sink_scores, partition_scores, tail_scores])
        
        return final_scores
    
    def _get_states_score(self, partitions: torch.Tensor) -> torch.Tensor:
        """
        Evaluate scores using lag-relative information.
        
        Parameters
        ----------
        partitions : torch.Tensor
            Shape: [num_kv_heads, num_partitions, lag_size, head_dim]
            
        Returns
        -------
        torch.Tensor
            Scores with shape [num_kv_heads, num_partitions-1, lag_size]
        """
        # Use next partition as reference for scoring current partition
        # ref: partitions 1, 2, 3, ... (reference)
        # v:   partitions 0, 1, 2, ... (to be scored)
        ref = partitions[:, 1:, :, :]  # [num_kv_heads, num_partitions-1, lag_size, head_dim]
        v = partitions[:, :-1, :, :]   # [num_kv_heads, num_partitions-1, lag_size, head_dim]
        
        # Compute min/max of reference partition across tokens (dim=-2)
        min_r = ref.min(dim=-2, keepdim=True).values  # [num_kv_heads, num_partitions-1, 1, head_dim]
        max_r = ref.max(dim=-2, keepdim=True).values  # [num_kv_heads, num_partitions-1, 1, head_dim]
        
        # Normalize current partition tokens using reference range
        # This captures how tokens relate to the subsequent partition
        normalized = (v - min_r) / (max_r - min_r + 1e-8)  # Add epsilon for stability
        
        # Compute standard deviation across head_dim, then softmax across tokens
        # Higher std = more distinctive = higher importance
        scores = normalized.std(dim=-1).softmax(dim=-1)  # [num_kv_heads, num_partitions-1, lag_size]
        
        return scores


@dataclass
class KeyDiffPress(BaseCompressionMethod):
    """
    KeyDiff: Key similarity-based KV cache compression.
    
    Evicts tokens based on key vector similarity to average key pattern.
    The intuition is that tokens with similar keys to the average are redundant
    and can be safely pruned, while keeping tokens with distinctive key vectors.
    
    Based on: "KeyDiff: Key Difference-based KV Cache Compression"
    (https://arxiv.org/abs/2504.15364)
    
    Attributes
    ----------
    compression_ratio : float
        Fraction of tokens to remove.
        
    Examples
    --------
    >>> method = KeyDiffPress(compression_ratio=0.3)
    >>> scores = method.score(layer_id=0, keys=k_tensor, values=v_tensor)
    # Tokens with keys most similar to average get lowest scores (pruned first)
    """
    
    def score(
        self,
        layer_id: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute negative cosine similarity to average key pattern.
        
        Returns
        -------
        torch.Tensor
            Negative cosine similarity scores with shape [num_tokens].
            Higher scores (less similar to average) = more important to keep.
        """
        # keys shape: [num_tokens, num_kv_heads, head_dim]
        
        # Normalize keys: [num_tokens, num_kv_heads, head_dim]
        normalized_keys = F.normalize(keys, p=2, dim=-1)
        
        # Compute average key pattern: [1, num_kv_heads, head_dim]
        anchor = normalized_keys.mean(dim=0, keepdim=True)
        
        # Compute cosine similarity: [num_tokens, num_kv_heads]
        similarity = F.cosine_similarity(normalized_keys, anchor, dim=-1)
        
        # Average across heads: [num_tokens]
        similarity = similarity.mean(dim=-1)
        
        # Return negative (lower similarity = higher score = keep)
        return -similarity


# Registry of available compression methods
COMPRESSION_METHODS = {
    "knorm": KnormPress,
    "random": RandomPress,
    "streamingllm": StreamingLLMPress,
    "keydiff": KeyDiffPress,
    "lagkv": LagKVPress,
}


def get_compression_method(method_name: str, compression_ratio: float) -> BaseCompressionMethod:
    """
    Factory function to create a compression method instance.
    
    Parameters
    ----------
    method_name : str
        Name of the compression method (e.g., "knorm").
    compression_ratio : float
        Compression ratio to use.
        
    Returns
    -------
    BaseCompressionMethod
        An instance of the requested compression method.
        
    Raises
    ------
    ValueError
        If the method name is not recognized.
        
    Examples
    --------
    >>> method = get_compression_method("knorm", compression_ratio=0.3)
    >>> isinstance(method, KnormPress)
    True
    """
    if method_name not in COMPRESSION_METHODS:
        available = ", ".join(COMPRESSION_METHODS.keys())
        raise ValueError(
            f"Unknown compression method: {method_name}. "
            f"Available methods: {available}"
        )
    
    method_class = COMPRESSION_METHODS[method_name]
    return method_class(compression_ratio=compression_ratio)

