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
    gqa_reduction_id,
    parse_calibration_file,
    torch_dtype_for_klabel,
    validate_against_model,
)
from sglang.srt.mem_cache.sparsity.triton_ops.k_label_kernels import (
    ds_compute_k_label_torch_ref,
    ds_compute_k_label_write,
)
from sglang.srt.mem_cache.sparsity.triton_ops.select_kernels import (
    ds_select_tokens_torch_ref,
    ds_select_tokens_triton,
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

    def effective_sparse_mask(
        self,
        forward_batch,
        req_pool_indices: torch.Tensor,
        default_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gate on current `seq_lens`, not admission-time `prompt_lens`.

        The base coordinator computes `default_mask` from
        `prompt_lens >= min_sparse_prompt_len`. For DS that's wrong:
        a request admitted with prompt_len=100 but generating to
        seq_len=5000 would never become sparse, even though by then
        decode is squarely above `min_seq_len`. We recompute from
        current seq_lens so late-crossing rows flip to sparse, and
        the coordinator threads this same mask into both retrieve_topk
        and the FA3 adaptor.
        """
        seq_lens = forward_batch.seq_lens.to(default_mask.device)
        return seq_lens >= self.runtime_config.min_seq_len

    def initialize_representation_pool(
        self,
        start_layer: int,
        end_layer: int,
        token_to_kv_pool,
        req_to_token_pool,
        states,
    ) -> None:
        """Allocate per-layer K_label side cache.

        Shape per layer: `[num_tokens_in_pool, num_kv_heads_local, S]`. Memory is
        ~12% on top of the KV pool when `S/D ≈ 32/128` and is owned by this
        algorithm (we deliberately do not touch `MHATokenToKVPool`).
        """
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool = token_to_kv_pool
        self.states = states
        self.start_layer = start_layer
        self.end_layer = end_layer

        num_tokens_in_pool = token_to_kv_pool.get_key_buffer(start_layer).shape[0]
        S = self.runtime_config.heavy_channels
        for layer_id in range(start_layer, end_layer):
            self.k_label[layer_id] = torch.zeros(
                (num_tokens_in_pool, self.num_kv_heads_local, S),
                dtype=self.klabel_dtype,
                device=self.device,
            )

        logger.info(
            "DoubleSparsity K_label allocated: layers=%d num_tokens_in_pool=%d "
            "kv_heads_local=%d S=%d dtype=%s mem_per_layer=%.2f MiB",
            end_layer - start_layer,
            num_tokens_in_pool,
            self.num_kv_heads_local,
            S,
            self.klabel_dtype,
            (
                num_tokens_in_pool
                * self.num_kv_heads_local
                * S
                * self.k_label[start_layer].element_size()
            )
            / (1024 * 1024),
        )

    def _write_k_label(
        self,
        layer_id: int,
        k: torch.Tensor,
        out_cache_loc: torch.Tensor,
    ) -> None:
        """Single entry point for K_label writes (extend or decode).

        Uses the Triton kernel on CUDA; falls back to the torch reference on
        CPU. Both paths are byte-equivalent for the test fixtures.
        """
        if k.numel() == 0:
            return
        chan = self.channel_indices[layer_id]
        kl = self.k_label[layer_id]
        if k.is_cuda:
            ds_compute_k_label_write(k, chan, out_cache_loc, kl)
        else:
            ds_compute_k_label_torch_ref(k, chan, out_cache_loc, kl)

    def construct_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        forward_batch,
    ) -> None:
        """Write K_label for new tokens during prefill (extend phase).

        SGLang's coordinator funnels both prefill and decode through
        `attention_end -> construct_representations + update_representations`,
        but only one of the two should fire per call. We dispatch on
        `forward_mode` and use `forward_batch.out_cache_loc` (the same source
        the dense backend uses) so K_label writes target exactly the physical
        token ids that just received K writes.
        """
        if forward_batch.forward_mode.is_extend():
            self._write_k_label_for_new_tokens(layer_id, k_buffer, forward_batch)

    def update_representations(
        self,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        k_buffer: torch.Tensor,
        forward_batch,
    ) -> None:
        """Append K_label for the freshly decoded tokens (decode phase)."""
        if forward_batch.forward_mode.is_decode_or_idle():
            self._write_k_label_for_new_tokens(layer_id, k_buffer, forward_batch)

    def _write_k_label_for_new_tokens(
        self,
        layer_id: int,
        k_buffer: torch.Tensor,
        forward_batch,
    ) -> None:
        """Gather K rows written this step from the KV pool and update K_label.

        Skipped when `save_kv_cache=False` — otherwise the side cache desyncs
        from the KV pool. Used by both extend and decode (the only thing that
        differs between phases is which `forward_mode` predicate gates the call).

        In the piecewise-extend path (`unified_attention_with_output`),
        `forward_batch.out_cache_loc` is full-padded by the time
        `attention_end` runs (the function restores it before returning).
        We narrow to `num_token_non_padded_cpu`, which is set on every
        ForwardBatch and matches the actual token count.
        """
        if not getattr(forward_batch, "save_kv_cache", True):
            return
        out_loc = forward_batch.out_cache_loc
        real = getattr(forward_batch, "num_token_non_padded_cpu", None)
        if real is not None and out_loc.numel() > real:
            out_loc = out_loc[:real]
        if out_loc.numel() == 0:
            return
        k_new = k_buffer[out_loc]
        self._write_k_label(layer_id, k_new, out_loc)

    def retrieve_topk(
        self,
        queries: torch.Tensor,
        layer_id: int,
        req_pool_indices: torch.Tensor,
        sparse_mask: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select per-request logical token positions for FA3 to attend to.

        Returns `(selected_logical[bs, max_selected], valid_lengths[bs])`. The
        adaptor (M4) maps logical → physical via `req_to_token`, preserves
        logical order (no physical sort), and writes the FA3 page-table.

        v1: per-KV-head scoring + GQA reduction → per-batch union; CUDA path
        currently dispatches to the torch reference (correct, parity-tested).
        v1.1 will replace the body of `ds_select_tokens_triton` with the
        block-decomposed Triton kernel once M8 profiling identifies the
        actual hotspot.
        """
        forward_batch = kwargs.get("forward_batch")
        if forward_batch is None:
            raise ValueError(
                "DoubleSparsity retrieve_topk requires forward_batch kwarg"
            )
        seq_lens = forward_batch.seq_lens.to(queries.device)
        # Attention layers commonly hand us `q` as `[bs, H_q*D]` (post-projection
        # flat view). Selection kernels need `[bs, H_q, D]`. Reshape here using
        # the algorithm-known geometry.
        if queries.dim() == 2:
            queries = queries.view(
                queries.shape[0], self.num_q_heads_local, self.head_dim
            )
        select_fn = (
            ds_select_tokens_triton if queries.is_cuda else ds_select_tokens_torch_ref
        )
        return select_fn(
            queries=queries,
            channel_idx=self.channel_indices[layer_id],
            k_label_layer=self.k_label[layer_id],
            req_to_token=self.req_to_token_pool.req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            num_kv_heads=self.num_kv_heads_local,
            token_budget=self.runtime_config.token_budget,
            recent_tokens=self.runtime_config.recent_tokens,
            sink_tokens=self.runtime_config.sink_tokens,
            min_seq_len=self.runtime_config.min_seq_len,
            max_selected=self.runtime_config.max_selected_per_request,
            gqa_reduction_id=gqa_reduction_id(self.runtime_config.gqa_reduction),
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
