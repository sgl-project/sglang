"""Double Sparsity algorithm — fresh implementation for SGLang.

Skeleton only; lifecycle (K_label writes), selection kernel, and FA3 metadata
adaptation arrive in subsequent milestones. See plan in
`/root/.claude/plans/you-are-claude-code-ethereal-bumblebee.md` for design notes.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch

from sglang.srt.mem_cache.sparsity.algorithms.base_algorithm import BaseSparseAlgorithm
from sglang.srt.mem_cache.sparsity.algorithms.double_sparsity_config import (
    DoubleSparsityCalibration,
    DoubleSparsityRuntimeConfig,
    channel_indices_for_runtime,
    parse_calibration_file,
    torch_dtype_for_klabel,
    validate_against_model,
)

logger = logging.getLogger(__name__)


class DoubleSparsityAlgorithm(BaseSparseAlgorithm):
    """K-channel Double Sparsity for decode-heavy long-context inference.

    v1 scope (see plan):
      - K-channels only.
      - FA3 backend with per-KV-head scoring + GQA reduction → one page table per batch.
      - page_size = 1.
      - Configurable GQA reduction (max_abs / mean / soq), default max_abs.
      - TP-aware: per-rank slice of a global-indexed calibration JSON.

    This skeleton wires up:
      - calibration parsing / TP slicing / runtime-config validation,
      - per-layer K_label allocation hook (no-op until M2),
      - retrieve_topk that raises NotImplementedError until the selection kernel lands in M3.
    """

    def __init__(
        self,
        config,
        device: torch.device,
        *,
        runtime_config: DoubleSparsityRuntimeConfig,
        calibration: DoubleSparsityCalibration,
        tp_size: int = 1,
        tp_rank: int = 0,
        num_kv_heads_local: Optional[int] = None,
        num_q_heads_local: Optional[int] = None,
        head_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(config, device, **kwargs)
        runtime_config.validate()

        if calibration.heavy_channels != runtime_config.heavy_channels:
            raise ValueError(
                f"runtime heavy_channels ({runtime_config.heavy_channels}) does not match "
                f"calibration heavy_channels ({calibration.heavy_channels}); regenerate "
                f"calibration or pass --double-sparsity-heavy-channels="
                f"{calibration.heavy_channels}."
            )

        self.runtime_config = runtime_config
        self.calibration = calibration
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.num_kv_heads_local = (
            num_kv_heads_local
            if num_kv_heads_local is not None
            else calibration.num_kv_heads_global // tp_size
        )
        self.num_q_heads_local = (
            num_q_heads_local
            if num_q_heads_local is not None
            else calibration.num_heads // tp_size
        )
        self.head_dim = head_dim if head_dim is not None else calibration.head_dim
        self.klabel_dtype = torch_dtype_for_klabel(runtime_config.klabel_dtype)

        # Per-layer int32 channel indices, shape [num_kv_heads_local, S].
        self.channel_indices: Dict[int, torch.Tensor] = channel_indices_for_runtime(
            calibration,
            tp_size=tp_size,
            tp_rank=tp_rank,
            device=device,
        )

        # Filled in initialize_representation_pool (M2).
        self.k_label: Dict[int, torch.Tensor] = {}
        self.start_layer: int = 0
        self.end_layer: int = 0

        logger.info(
            "DoubleSparsity init: layers=%d S=%d kv_heads_local=%d q_heads_local=%d "
            "tp_size=%d tp_rank=%d klabel_dtype=%s gqa_reduction=%s",
            calibration.num_layers,
            runtime_config.heavy_channels,
            self.num_kv_heads_local,
            self.num_q_heads_local,
            tp_size,
            tp_rank,
            runtime_config.klabel_dtype,
            runtime_config.gqa_reduction,
        )

    @classmethod
    def from_server_config(
        cls,
        config,
        device: torch.device,
        *,
        calibration_path: str,
        runtime_config: DoubleSparsityRuntimeConfig,
        head_dim: int,
        num_layers: int,
        num_heads: int,
        num_kv_heads_global: int,
        tp_size: int = 1,
        tp_rank: int = 0,
        **kwargs,
    ) -> "DoubleSparsityAlgorithm":
        """Convenience constructor that loads + validates calibration vs the model."""
        calibration = parse_calibration_file(calibration_path)
        validate_against_model(
            calibration,
            head_dim=head_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads_global=num_kv_heads_global,
        )
        return cls(
            config,
            device,
            runtime_config=runtime_config,
            calibration=calibration,
            tp_size=tp_size,
            tp_rank=tp_rank,
            num_kv_heads_local=num_kv_heads_global // tp_size,
            num_q_heads_local=num_heads // tp_size,
            head_dim=head_dim,
            **kwargs,
        )

    def initialize_representation_pool(
        self,
        start_layer: int,
        end_layer: int,
        token_to_kv_pool,
        req_to_token_pool,
        states,
    ) -> None:
        """Allocate K_label side cache. Real allocation arrives in M2."""
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.states = states
        self.start_layer = start_layer
        self.end_layer = end_layer
        # M2 will allocate self.k_label[layer_id] of shape
        # [num_tokens_in_pool, num_kv_heads_local, S] with self.klabel_dtype.

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "DoubleSparsityAlgorithm.retrieve_topk is implemented in M3 "
            "(two-stage selection kernel). The skeleton intentionally raises "
            "so the wiring can be exercised before kernels exist."
        )


def parse_double_sparsity_calibration(server_args) -> DoubleSparsityCalibration:
    """Helper used by tests and the factory."""
    if not server_args.double_sparsity_config:
        raise ValueError(
            "--double-sparsity-config is required when --enable-double-sparsity is set"
        )
    return parse_calibration_file(server_args.double_sparsity_config)


__all__ = [
    "DoubleSparsityAlgorithm",
    "parse_double_sparsity_calibration",
]
