"""
Manifold-Constrained Hyper-Connection Module (mHC).
"""
from typing import Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.layers.mhc_tilelang_ops import (
    mhc_tilelang_aggregate,
    mhc_tilelang_expand_merge,
    mhc_tilelang_map_sigmoid,
    mhc_tilelang_norm_linear,
    mhc_tilelang_sinkhorn,
)


class HyperConnectionModule(nn.Module):
    """
    Manifold-Constrained Hyper-Connection Module (mHC).
    Manages multiple residual streams and their interaction with transformer blocks.
    """
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n = getattr(config, "num_residual_streams", 4)
        self.sinkhorn_iterations = getattr(config, "sinkhorn_iterations", 20)

        # Projection weights for dynamic mappings
        self.mapping_proj = nn.Linear(
            self.n * self.hidden_size,
            self.n * self.n + 2 * self.n,
            bias=False,
            dtype=torch.float32
        )
        # Learnable scaling factors
        self.alpha_pre = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.alpha_post = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        self.alpha_res = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))
        
        self.alpha = torch.cat([
            self.alpha_pre.expand(self.n),
            self.alpha_post.expand(self.n),
            self.alpha_res.expand(self.n * self.n)
            ], dim=-1).contiguous()

        # Static bias terms
        self.bias = nn.Parameter(torch.zeros(2 * self.n + self.n * self.n, dtype=torch.float32))
        
        self.norm_linear_func = mhc_tilelang_norm_linear
        self.map_sigmoid_func = mhc_tilelang_map_sigmoid
        self.sinkhorn_func = mhc_tilelang_sinkhorn
        self.aggregate_func = mhc_tilelang_aggregate
        self.expand_merge_func = mhc_tilelang_expand_merge

    def compute_mappings(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute mHC mappings from input hidden states using fused operations.

        Args:
            x: [bs, n, d]
        Returns:
            Tuple of (h_pre, h_post, h_res):
                - h_pre: [bs, n] - aggregation weights
                - h_post: [bs, n] - expansion weights
                - h_res: [bs, n, n] - residual mixing weights
        """

        bs, _, _ = x.shape

        # Use mhc_tilelang_norm_linear for normalization + projection
        r, proj = self.norm_linear_func(
            x,
            self.mapping_proj.weight
        )
        
        # Use mhc_tilelang_map_sigmoid for scaling, bias, and sigmoid operations
        h_pre, h_post, h_res = self.map_sigmoid_func(
            r,
            proj,
            self.bias,
            self.alpha,
            self.n
        )

        # apply fused Sinkhorn
        h_res = self.sinkhorn_func(
            h_res, 
            n_iters=self.sinkhorn_iterations
        )

        return h_pre, h_post, h_res

    def forward_in(
        self, 
        residuals: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Phase 1: Aggregation (Pre-Mapping) using fused operations

        Args:
            residuals: [bs, n, dim]

        Returns:
            Tuple of (x_in, h_post, h_res):
                - x_in: [bs, dim] - aggregated input for layer
                - h_post: [bs, n] - expansion weights
                - h_res: [bs, n, n] - residual mixing weights
        """

        # Compute all mappings
        h_pre, h_post, h_res = self.compute_mappings(residuals)
        # Use mhc_tilelang_aggregate for weighted sum aggregation
        # mhc_tilelang_aggregate expects: residuals [bs, n, d], h_pre [bs, n]
        x_in = self.aggregate_func(residuals, h_pre)
        
        return x_in, h_post, h_res

    def forward_out(
        self, 
        residuals: torch.Tensor, 
        layer_output: torch.Tensor,
        h_post: torch.Tensor, 
        h_res: torch.Tensor
    ) -> torch.Tensor:
        """
        Phase 2: Update (Post-Mapping & Residual-Mapping) using fused operations

        Args:
            residuals: [bs, n, dim]
            layer_output: [bs, dim]
            h_post: [bs, n] - from forward_in
            h_res: [bs, n, n] - from forward_in

        Returns:
            residuals_new: [bs, n, dim]
        """

        # Use mhc_tilelang_expand_merge to combine residual mixing and layer output expansion
        # mhc_tilelang_expand_merge expects:
        #   residuals: [bs, n, d]
        #   layer_output: [bs, d]
        #   h_res: [bs, n, n]
        #   h_post: [bs, n]
        residuals_new = self.expand_merge_func(residuals, layer_output, h_res, h_post)
        
        return residuals_new

    @staticmethod
    def input_expand(x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Expand 1-stream to n-stream at TransformerBlock entry.
        """
        bs, d = x.shape
        return x.unsqueeze(1).expand(bs, n, d).contiguous()

    @staticmethod
    def output_contract(x: torch.Tensor, n: int) -> torch.Tensor:
        """
        Contract n-stream to 1-stream at TransformerBlock exit.
        
        Simple averaging strategy: average all streams.
        
        Args:
            x: [bs, n, d] - n-stream hidden states
            n: Number of residual streams
        
        Returns:
            contracted: [bs, d] - single stream hidden states
        """
        return x.mean(dim=-2)