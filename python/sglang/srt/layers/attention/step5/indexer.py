# SPDX-License-Identifier: Apache-2.0
"""Step5 index projections, compression and region selection."""

from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

from sglang.kernels.ops.attention.step5 import options as step5_options
from sglang.kernels.ops.attention.step5.indexer_norm import step5_indexer_norm_rope
from sglang.kernels.ops.gemm.tiny_gemm import tiny_gemm_bf16
from sglang.srt.configs.step5 import get_step5_position_padding
from sglang.srt.layers.attention.step5.compressor import compress_index_keys
from sglang.srt.layers.attention.step5.metadata import Step5IndexerMetadata
from sglang.srt.layers.attention.step5.region_selection import select_step5_regions
from sglang.srt.layers.layernorm import GemmaRMSNorm, LayerNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    ReplicatedParallelGroup,
)
from sglang.srt.layers.quantization.unquant import bf16_gemm_dispatch
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.runtime_context import get_exec, get_lora, get_model
from sglang.srt.utils import add_prefix


@lru_cache(maxsize=None)
def _cuda_fp8_saturates(device: torch.device) -> bool:
    """Probe once per CUDA device before capture; Torch cast policy can differ."""
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Warm up the Step5 FP8 cast before CUDA graph capture")
    bits = (
        torch.tensor([1024.0], dtype=torch.bfloat16, device=device)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
        .item()
    )
    if bits not in (0x7E, 0x7F):
        raise RuntimeError(f"Unsupported Torch CUDA E4M3 overflow encoding: {bits}")
    return bits == 0x7E


class Step5Indexer(nn.Module):
    """Provider-sharded Q/W, replicated K/Z, compression and region selection."""

    fold_indexer_scale = False
    fuse_indexer_fp8_cast = False
    fp8_saturate = False
    qkz_tokens = ()
    use_cutedsl_qkz = False

    def __init__(
        self,
        *,
        layer_id: int,
        hidden_size: int,
        index_head_dim: int,
        num_index_heads: int,
        index_tp_size: int,
        tp_rank: int,
        tp_size: int,
        rotary_dim: int,
        rope_theta: float,
        rope_scaling: dict,
        max_position_embeddings: int,
        eps: float,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.index_head_dim = index_head_dim
        self.provider_rank = tp_rank // (tp_size // index_tp_size)
        self.num_heads = num_index_heads // index_tp_size
        self.weight_scale = self.num_heads**-0.5
        self.sparse_indexer_q = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_index_heads * index_head_dim,
            bias=False,
            gather_output=False,
            parallel_group=ReplicatedParallelGroup("attn_tp", tp_size // index_tp_size),
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("sparse_indexer_q", prefix),
        )
        self.sparse_indexer_k = ReplicatedLinear(
            input_size=hidden_size,
            output_size=index_head_dim,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("sparse_indexer_k", prefix),
        )
        self.sparse_indexer_z = ReplicatedLinear(
            input_size=hidden_size,
            output_size=index_head_dim,
            bias=False,
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("sparse_indexer_z", prefix),
        )
        self.sparse_indexer_w = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=num_index_heads,
            bias=False,
            gather_output=False,
            parallel_group=ReplicatedParallelGroup("attn_tp", tp_size // index_tp_size),
            params_dtype=torch.bfloat16,
            quant_config=None,
            prefix=add_prefix("sparse_indexer_w", prefix),
        )
        self.rotary_emb = get_rope(
            head_size=index_head_dim,
            rotary_dim=rotary_dim,
            max_position=max_position_embeddings + get_step5_position_padding(),
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=True,
            dtype=torch.bfloat16,
            cache_dtype=torch.bfloat16,
        )
        weight = self.sparse_indexer_q.weight
        use_b300_fusions = (
            weight.is_cuda
            and torch.version.hip is None
            and torch.cuda.get_device_capability(weight.device) == (10, 3)
            and weight.dtype == torch.bfloat16
            and hidden_size == 4096
            and index_head_dim == 256
            and self.num_heads == 4
            and tp_size in (4, 8)
        )
        self.register_buffer("qkz_weight", None, persistent=False)
        # Inspect the requested backend: auto has already resolved to CuTe at
        # this point. Packed shapes are measured against that actual dispatcher,
        # including its M=256 Q projection.
        if (
            use_b300_fusions
            and get_exec().kernel.bf16_gemm_backend in ("auto", "torch")
            and not get_exec().deterministic.enable_deterministic_inference
            and not get_lora().enable_lora
            and get_model().weight_cache_mode == "off"
            and get_model().load_format
            in ("auto", "pt", "safetensors", "npcache", "dummy")
            and get_exec().offload.cpu_offload_gb == 0
            and get_exec().offload.offload_group_size <= 0
        ):
            self.use_cutedsl_qkz = tp_size == 4
            self.qkz_tokens = (
                (4, 256)
                if tp_size == 8 or get_exec().kernel.bf16_gemm_backend == "auto"
                else (256,)
            )
            self.qkz_weight = torch.empty(
                1536, hidden_size, dtype=weight.dtype, device=weight.device
            )
            # Preserve the parameter objects/loaders, including provider sharding.
            # Their disjoint views also receive dummy initialization and direct
            # in-place weight updates, without a stale packed copy in graphs.
            offset = 0
            for projection in (
                self.sparse_indexer_q,
                self.sparse_indexer_k,
                self.sparse_indexer_z,
            ):
                rows = projection.weight.shape[0]
                projection.weight.data = self.qkz_weight[offset : offset + rows]
                offset += rows
        # Keep existing explicit overrides; the measured B300 shape needs none.
        self.fold_indexer_scale = (
            # TODO(yhyang201): frozen SGLANG_OPT_STEP5_FOLD_INDEXER_SCALE; prune dead paths.
            step5_options.FOLD_INDEXER_SCALE
            if step5_options.FOLD_INDEXER_SCALE is not None
            else use_b300_fusions
        )
        self.fuse_indexer_fp8_cast = (
            # TODO(yhyang201): frozen SGLANG_OPT_STEP5_FUSE_INDEXER_FP8_CAST; prune dead paths.
            step5_options.FUSE_INDEXER_FP8_CAST
            if step5_options.FUSE_INDEXER_FP8_CAST is not None
            else use_b300_fusions
        )
        self.fp8_saturate = (
            _cuda_fp8_saturates(weight.device)
            if self.fuse_indexer_fp8_cast and weight.is_cuda
            else False
        )
        self.eps = eps
        self.sparse_indexer_q_norm = GemmaRMSNorm(
            index_head_dim, eps=eps, weight_dtype=torch.bfloat16
        )
        self.sparse_indexer_k_norm = LayerNorm(
            index_head_dim, eps=eps, dtype=torch.bfloat16
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        if self.qkz_weight is not None:
            offset = 0
            for projection in (
                self.sparse_indexer_q,
                self.sparse_indexer_k,
                self.sparse_indexer_z,
            ):
                weight = projection.weight
                if weight.data_ptr() != self.qkz_weight[offset].data_ptr():
                    # Module transforms can replace each parameter separately.
                    # Use the original projections once their storage diverges.
                    self.qkz_weight = None
                    break
                offset += weight.shape[0]
        return self

    def _project_weights(self, hidden_states):
        tokens = hidden_states.shape[0]
        weight_projection = self.sparse_indexer_w
        if (
            self.use_cutedsl_qkz
            and self.qkz_weight is not None
            and tokens == 4
            and hidden_states.dtype == torch.bfloat16
            and hidden_states.is_contiguous()
            and hidden_states.data_ptr() % 32 == 0
            and weight_projection.weight.is_contiguous()
            and weight_projection.weight.dtype == torch.bfloat16
            and weight_projection.weight.data_ptr() % 32 == 0
            and get_exec().kernel.bf16_gemm_backend == "auto"
            and not weight_projection._forward_hooks
            and not weight_projection._forward_pre_hooks
        ):
            index_weights = tiny_gemm_bf16(
                hidden_states, weight_projection.weight, max_m=4, n_split=1
            )
        else:
            index_weights, _ = weight_projection(hidden_states)
        return index_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        metadata: Step5IndexerMetadata,
        *,
        weights_first: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tokens = hidden_states.shape[0]
        index_weights = self._project_weights(hidden_states) if weights_first else None
        projections = (
            self.sparse_indexer_q,
            self.sparse_indexer_k,
            self.sparse_indexer_z,
        )
        if (
            getattr(self, "qkz_weight", None) is not None
            and hidden_states.dtype == torch.bfloat16
            and tokens in self.qkz_tokens
            and get_exec().kernel.bf16_gemm_backend in ("auto", "torch")
            and not any(p._forward_hooks or p._forward_pre_hooks for p in projections)
        ):
            if (
                self.use_cutedsl_qkz
                and tokens == 4
                and get_exec().kernel.bf16_gemm_backend == "auto"
            ):
                projected = bf16_gemm_dispatch(hidden_states, self.qkz_weight, None)
            else:
                projected = F.linear(hidden_states, self.qkz_weight)
            index_q, index_key, compression_logits = projected.split(
                (1024, 256, 256), dim=-1
            )
            # Both Q/K norm and compression consume padded row strides.
        else:
            index_q, _ = self.sparse_indexer_q(hidden_states)
            index_key, _ = self.sparse_indexer_k(hidden_states)
            compression_logits, _ = self.sparse_indexer_z(hidden_states)
        if index_weights is None:
            index_weights = self._project_weights(hidden_states)
        scoring_weight_scale = None
        if not self.fold_indexer_scale:
            if metadata.selection_workspace.scoring_workspace is not None:
                scoring_weight_scale = self.weight_scale
            else:
                index_weights = index_weights.float() * self.weight_scale
        index_q = index_q.view(tokens, self.num_heads, self.index_head_dim)
        index_key = index_key.view(tokens, 1, self.index_head_dim)
        compression_logits = compression_logits.view(tokens, 1, self.index_head_dim)
        q_dtype = torch.float8_e4m3fn if self.fuse_indexer_fp8_cast else index_q.dtype
        pool = metadata.token_to_kv_pool
        use_ring_pdl = (
            self.use_cutedsl_qkz
            and tokens in (4, 8, 12, 16)
            and not torch.compiler.is_compiling()
            and not get_exec().deterministic.enable_deterministic_inference
            and metadata.request_slots.numel() == tokens // 4
            and metadata.compressed_key_write_indices.shape[0] == 1
            and pool.get_index_key_ring(self.layer_id).shape[1] == 16
        )
        selection_workspace = metadata.selection_workspace
        packed_qw_ready = (
            selection_workspace.scoring_metadata_ready
            and tokens == metadata.query_positions.numel()
            and tokens == metadata.query_request_slots.numel()
            and self.num_heads in (2, 4)
            and selection_workspace.tiles == ((0, tokens),)
        )
        packed_workspace = (
            selection_workspace.scoring_workspace if packed_qw_ready else None
        )
        normalized_q, index_key = step5_indexer_norm_rope(
            index_q,
            index_key,
            q_weight=self.sparse_indexer_q_norm.weight,
            k_weight=self.sparse_indexer_k_norm.weight,
            k_bias=self.sparse_indexer_k_norm.bias,
            cache=self.rotary_emb.cos_sin_cache,
            positions=positions,
            q_out=(
                None
                if packed_qw_ready
                else torch.empty(index_q.shape, dtype=q_dtype, device=index_q.device)
            ),
            k_out=torch.empty_like(index_key),
            eps=self.eps,
            zero_centered=True,
            fp8_saturate=self.fp8_saturate,
            use_pdl=use_ring_pdl,
            packed_workspace=packed_workspace,
            scoring_weights=index_weights if packed_qw_ready else None,
            row_to_req=metadata.query_request_slots if packed_qw_ready else None,
            weight_scale=(
                scoring_weight_scale
                if scoring_weight_scale is not None
                else (0.5 if index_weights.dtype == torch.bfloat16 else 1.0)
            ),
            packed_q_e4m3=self.fuse_indexer_fp8_cast,
        )
        if normalized_q is not None:
            index_q = normalized_q
        # In the packed path index_q is used only for row/head shape metadata.
        compressed_index_keys = pool.get_compressed_index_key_buffer(self.layer_id)
        # Read the old ring before publishing this forward's pending inputs.
        compress_index_keys(
            batch=metadata,
            compressed_index_keys=compressed_index_keys,
            index_key_ring=pool.get_index_key_ring(self.layer_id),
            compression_logits_ring=pool.get_compression_logits_ring(self.layer_id),
            positions=pool.get_index_position_buffer(self.layer_id),
            index_key=index_key,
            compression_logits=compression_logits,
            assert_success=True,
            enabled=metadata.compression_enabled,
            success=metadata.compression_success,
            use_pdl=use_ring_pdl,
        )
        return select_step5_regions(
            index_q=index_q,
            index_weights=index_weights,
            compressed_index_keys=compressed_index_keys,
            req_to_token=metadata.req_to_token,
            row_to_req=metadata.query_request_slots,
            positions=metadata.query_positions,
            score_width=metadata.score_width,
            score_group_rows=metadata.score_group_rows,
            scoring_weight_scale=scoring_weight_scale,
            packed_qw_ready=packed_qw_ready,
            workspace=metadata.selection_workspace,
        )
