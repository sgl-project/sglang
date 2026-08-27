# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
import functools
import os
from collections.abc import Sequence
from contextlib import nullcontext
from typing import Type

import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from sglang.kernels.ops.diffusion import (
    build_inv_indices,
    fused_pack_qkv,
    fused_pack_segmented_qkv,
    fused_scatter_to_padded,
)
from sglang.multimodal_gen.runtime.breakable_cuda_graph.replay_token import (
    get_current_replay_token,
)
from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_rank,
    get_ring_parallel_world_size,
    get_sequence_parallel_world_size,
    get_sp_group,
    get_sp_parallel_rank,
    get_sp_world_size,
    get_ulysses_parallel_rank,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.backends import (
    flash_attn as _fa_backend,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionImpl,
    wrap_attention_impl_forward,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.layers.attention.turbo_layer import (
    async_a2a_communicate,
)
from sglang.multimodal_gen.runtime.layers.usp import (
    _ipc_input_a2a_qkv,
    _merge_attention_partials,
    _ring_attention_varlen,
    _usp_input_all_to_all,
    _usp_input_all_to_all_qkv,
    _usp_input_all_to_all_varlen,
    _usp_output_all_to_all,
    _usp_output_all_to_all_varlen,
    ring_attn,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    ForwardContext,
    get_forward_context,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.utils import get_compute_dtype
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    eager_on_graph,
    is_in_breakable_cuda_graph,
)

_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS = [
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
]

# Set ``SGLANG_VARLEN_FA=0`` to disable the varlen FA fast path in
# USPAttention masked branch and fall back to SDPA.
_VARLEN_FA_ENABLED = os.environ.get("SGLANG_VARLEN_FA", "1") != "0"


def _resolve_sp_attention_mode(
    *, causal: bool, sparse_backend: bool
) -> tuple[str, bool]:
    """Resolve one layer's SP exchange; returns (mode, is_auto).

    ``kv_gather_degree > 1`` selects the gather exchange for the SP rows. When
    the degree was auto-assigned, layers the gather path cannot serve fall
    back to Ulysses; an explicit degree fails closed instead of degrading.
    """
    from sglang.multimodal_gen.runtime.server_args import get_global_server_args

    args = get_global_server_args()
    if args.kv_gather_degree <= 1:
        return "ulysses", False
    if causal or sparse_backend:
        if args.sp_split_auto:
            return "ulysses", True
        if causal:
            raise ValueError("K/V-gather SP does not support causal attention.")
        raise NotImplementedError(
            "K/V-gather SP does not support sparse attention backends."
        )
    return "kv_gather", args.sp_split_auto


def _kv_gather_unsupported_reason(
    *,
    qkv_pre_all_to_all: bool,
    replicated_mode_count: int,
    attn_mask: torch.Tensor | None,
    num_replicated_kv_prefix: int,
) -> str | None:
    """Call shapes the gather path does not take; explicit mode fails closed
    on these, auto falls back to the Ulysses exchange for the call."""
    if qkv_pre_all_to_all:
        return (
            "K/V-gather SP expects sequence-sharded Q/K/V; "
            "caller-side pre-all-to-all is Ulysses-only."
        )
    if replicated_mode_count > 1:
        return "K/V-gather SP supports at most one replicated-token mode per call."
    if attn_mask is not None:
        if num_replicated_kv_prefix:
            return "K/V-gather SP masked attention does not support a KV-only prefix."
        if attn_mask.dim() != 2:
            return "K/V-gather SP masked attention expects a [B, S_local] mask."
        if torch.is_floating_point(attn_mask):
            return "K/V-gather SP supports boolean or integer padding masks."
    return None


def _count_active_replicated_modes(
    num_replicated_prefix: int,
    num_replicated_suffix: int,
    num_replicated_kv_prefix: int,
) -> int:
    """Count active replicated-token modes without adding symbolic booleans."""
    return sum(
        int(value > 0)
        for value in (
            num_replicated_prefix,
            num_replicated_suffix,
            num_replicated_kv_prefix,
        )
    )


def build_varlen_mask_meta(
    key_mask: torch.Tensor,
) -> dict:
    """Build varlen FA metadata from a ``[B, S]`` key mask.

    Returns ``cu_seqlens``, ``indices``, ``inv_indices``, ``max_seqlen``.
    Passing the result via ``joint_attention_kwargs`` opts the caller into
    ``USPAttention``'s varlen FA fast path, which zero-fills masked query
    rows on output; only use when those rows are dropped or ignored
    downstream.
    """
    assert key_mask.dim() == 2, "key_mask must be [B, S]"
    bs, seq = key_mask.shape
    bool_mask = key_mask.to(dtype=torch.bool)
    valid_lens = bool_mask.sum(dim=1, dtype=torch.int32)
    indices = bool_mask.reshape(-1).nonzero(as_tuple=False).flatten()
    cu_seqlens = torch.zeros(bs + 1, dtype=torch.int32, device=key_mask.device)
    cu_seqlens[1:] = torch.cumsum(valid_lens, dim=0)
    inv_indices = build_inv_indices(indices, bs * seq)

    return {
        "cu_seqlens": cu_seqlens,
        "indices": indices,
        "inv_indices": inv_indices,
        "max_seqlen": seq,  # upper bound; FA varlen uses cu_seqlens for actual ranges
    }


def build_varlen_mask_meta_from_lengths(
    lengths: Sequence[int],
    max_seqlen: int,
    device: torch.device,
) -> dict:
    """Build varlen FA metadata for prefix-valid masks without CUDA nonzero.

    This is equivalent to ``build_varlen_mask_meta`` for masks where row ``i`` is
    true on ``[:lengths[i]]`` and false afterwards. Keeping the lengths on the
    host lets callers avoid a GPU ``nonzero``/dynamic-shape path while still
    producing the same packed indices.
    """

    return build_varlen_mask_meta_from_ranges(
        [[(0, int(length))] for length in lengths],
        max_seqlen=max_seqlen,
        device=device,
    )


def build_varlen_mask_meta_from_ranges(
    valid_ranges: Sequence[Sequence[tuple[int, int]]],
    max_seqlen: int,
    device: torch.device,
) -> dict:
    """Build varlen FA metadata from host-side valid token ranges.

    ``valid_ranges[i]`` contains half-open intervals in row-local coordinates.
    The intervals are packed in the provided order, matching the flattened
    ``nonzero`` order for ordinary left-to-right masks.
    """

    range_values = [
        [(int(start), int(end)) for start, end in row_ranges]
        for row_ranges in valid_ranges
    ]
    if any(
        start < 0 or end < start or end > max_seqlen
        for row_ranges in range_values
        for start, end in row_ranges
    ):
        raise ValueError(
            f"All ranges must be within [0, {max_seqlen}], got {range_values}"
        )

    bs = len(range_values)
    length_values = [
        sum(end - start for start, end in row_ranges) for row_ranges in range_values
    ]
    valid_lens = torch.as_tensor(length_values, dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(valid_lens, dim=0)

    index_parts = [
        torch.arange(
            row * max_seqlen + start,
            row * max_seqlen + end,
            dtype=torch.long,
            device=device,
        )
        for row, row_ranges in enumerate(range_values)
        for start, end in row_ranges
        if end > start
    ]
    if index_parts:
        indices = torch.cat(index_parts, dim=0)
    else:
        indices = torch.empty((0,), dtype=torch.long, device=device)
    inv_indices = build_inv_indices(indices, bs * max_seqlen)

    return {
        "cu_seqlens": cu_seqlens,
        "indices": indices,
        "inv_indices": inv_indices,
        "max_seqlen": max_seqlen,
    }


class DynamicVarlenMaskMeta:
    """Replay-local builder for varlen attention metadata.

    BCG attention break points capture Python kwargs once. Passing a plain
    ``attn_mask_meta`` dict would replay stale cu_seqlens/indices when the same
    graph bucket is reused for a different prompt length. This helper keeps only
    replay-local metadata and rebuilds it from the current ``attn_mask`` tensor
    on the first attention block of each graph replay.
    """

    def __init__(self) -> None:
        self._cache_key = None
        self._meta = None

    def resolve(self, attn_mask: torch.Tensor | None) -> dict | None:
        if attn_mask is None:
            self._cache_key = None
            self._meta = None
            return None

        replay_token = get_current_replay_token()
        if replay_token is None:
            cache_key = ("capture", id(attn_mask), tuple(attn_mask.shape))
        else:
            cache_key = ("replay", replay_token, tuple(attn_mask.shape))

        if cache_key != self._cache_key:
            self._meta = build_varlen_mask_meta(attn_mask)
            self._cache_key = cache_key
        return self._meta


def prepare_attention_backend_override(
    layer: nn.Module, target: AttentionBackendEnum
) -> None:
    """Build and cache the impl for ``target``; may raise, mutates nothing."""
    if target in layer._attn_impl_by_backend:
        return
    backend_cls = get_attn_backend(
        layer.head_size,
        layer.dtype,
        supported_attention_backends=layer._supported_attention_backends,
        selected_attention_backend=target,
    )
    resolved = backend_cls.get_enum()
    if resolved is not target:
        raise ValueError(
            f"Attention backend override '{target}' resolved to '{resolved}' on "
            f"{type(layer).__name__}; refusing the request instead of silently "
            "falling back."
        )
    impl = backend_cls.get_impl_cls()(**layer._attn_impl_ctor_kwargs)
    wrap_attention_impl_forward(impl)
    layer._attn_impl_by_backend[target] = impl


def apply_attention_backend_override(
    layer: nn.Module, target: AttentionBackendEnum | None
) -> None:
    """Flip to a prepared impl (None = construction default); cannot fail."""
    target = target or layer._default_attn_backend
    if target is layer.backend:
        return
    layer.attn_impl = layer._attn_impl_by_backend[target]
    layer.backend = target


class UlyssesAttention(nn.Module):
    """Ulysses-style SequenceParallelism attention layer."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        super().__init__()
        if get_ring_parallel_world_size() > 1:
            raise NotImplementedError(
                "UlyssesAttention's all-to-all spans the combined sequence "
                "parallel group and is not ring-aware; it would silently "
                "shuffle across ring ranks instead of rotating KV within "
                "them. Ring parallelism is not supported for models still "
                "using UlyssesAttention -- use USPAttention instead."
            )
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale

        if num_kv_heads is None:
            num_kv_heads = num_heads

        dtype = get_compute_dtype()
        attn_backend = get_attn_backend(
            head_size, dtype, supported_attention_backends=supported_attention_backends
        )
        impl_cls = attn_backend.get_impl_cls()

        self._attn_impl_ctor_kwargs = dict(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=self.softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=f"{prefix}.impl",
            **extra_impl_args,
        )
        self.attn_impl = impl_cls(**self._attn_impl_ctor_kwargs)
        wrap_attention_impl_forward(self.attn_impl)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = attn_backend.get_enum()
        self._default_attn_backend = self.backend
        self._attn_impl_by_backend = {self.backend: self.attn_impl}
        self._supported_attention_backends = supported_attention_backends
        self.dtype = dtype
        self.causal = causal
        self.sp_attention_mode, self.sp_attention_mode_is_auto = (
            _resolve_sp_attention_mode(
                causal=causal, sparse_backend=self.backend.is_sparse
            )
        )

    def _forward_with_kv_gather(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx_attn_metadata,
        replicated_q: torch.Tensor | None,
        replicated_k: torch.Tensor | None,
        replicated_v: torch.Tensor | None,
        seq_lens: list[int] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if seq_lens is not None:
            raise NotImplementedError(
                "K/V-gather SP does not support varlen UlyssesAttention."
            )
        if any(x is not None for x in (replicated_q, replicated_k, replicated_v)):
            if any(x is None for x in (replicated_q, replicated_k, replicated_v)):
                raise ValueError("Replicated Q, K, and V must be provided together.")

        k = sequence_model_parallel_all_gather(k, dim=1)
        v = sequence_model_parallel_all_gather(v, dim=1)

        local_query_len = q.shape[1]
        if replicated_q is not None:
            q = torch.cat([q, replicated_q], dim=1)
            k = torch.cat([k, replicated_k], dim=1)
            v = torch.cat([v, replicated_v], dim=1)

        output = self.attn_impl.forward(q, k, v, ctx_attn_metadata)
        if replicated_q is None:
            return output, None
        return output[:, :local_query_len], output[:, local_query_len:]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
        seq_lens: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass for distributed attention.

        Args:
            q (torch.Tensor): Query tensor [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor [batch_size, seq_len, num_heads, head_dim]
            replicated_q (Optional[torch.Tensor]): Replicated query tensor, typically for text tokens
            replicated_k (Optional[torch.Tensor]): Replicated key tensor
            replicated_v (Optional[torch.Tensor]): Replicated value tensor

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - o (torch.Tensor): Output tensor after attention for the main sequence
                - replicated_o (Optional[torch.Tensor]): Output tensor for replicated tokens, if provided
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"
        batch_size, seq_len, num_heads, head_dim = q.shape

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        if self.sp_attention_mode == "kv_gather" and not (
            self.sp_attention_mode_is_auto and seq_lens is not None
        ):
            return self._forward_with_kv_gather(
                q,
                k,
                v,
                ctx_attn_metadata,
                replicated_q,
                replicated_k,
                replicated_v,
                seq_lens,
            )

        local_rank = get_sp_parallel_rank()
        world_size = get_sp_world_size()
        if seq_lens is not None:
            assert (
                replicated_q is None and replicated_k is None and replicated_v is None
            ), "Varlen Ulysses attention does not support replicated QKV"

        # Stack QKV
        qkv = torch.cat([q, k, v], dim=0)  # [3, seq_len, num_heads, head_dim]

        # Redistribute heads across sequence dimension
        if seq_lens is None:
            qkv = sequence_model_parallel_all_to_all_4D(
                qkv, scatter_dim=2, gather_dim=1
            )
        else:
            qkv = _usp_input_all_to_all_varlen(qkv, seq_lens, head_dim=2)
        # Apply backend-specific preprocess_qkv
        qkv = self.attn_impl.preprocess_qkv(qkv, ctx_attn_metadata)

        # Concatenate with replicated QKV if provided
        if replicated_q is not None:
            assert replicated_k is not None and replicated_v is not None
            replicated_qkv = torch.cat(
                [replicated_q, replicated_k, replicated_v], dim=0
            )  # [3, seq_len, num_heads, head_dim]
            heads_per_rank = num_heads // world_size
            replicated_qkv = replicated_qkv[
                :, :, local_rank * heads_per_rank : (local_rank + 1) * heads_per_rank
            ]
            qkv = torch.cat([qkv, replicated_qkv], dim=1)

        q, k, v = qkv.chunk(3, dim=0)

        output = self.attn_impl.forward(q, k, v, ctx_attn_metadata)

        # Redistribute back if using sequence parallelism
        replicated_output = None
        if replicated_q is not None:
            replicated_output = output[:, seq_len * world_size :]
            output = output[:, : seq_len * world_size]
            # TODO: make this asynchronous
            replicated_output = sequence_model_parallel_all_gather(
                replicated_output.contiguous(), dim=2
            )
        # Apply backend-specific postprocess_output
        output = self.attn_impl.postprocess_output(output, ctx_attn_metadata)

        if seq_lens is None:
            output = sequence_model_parallel_all_to_all_4D(
                output, scatter_dim=1, gather_dim=2
            )
        else:
            output = _usp_output_all_to_all_varlen(output, seq_lens, head_dim=2)
        return output, replicated_output


class UlyssesAttention_VSA(UlyssesAttention):
    """Distributed attention layer with VSA support."""

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
        gate_compress: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for distributed attention.

        Args:
            q (torch.Tensor): Query tensor [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor [batch_size, seq_len, num_heads, head_dim]
            gate_compress (torch.Tensor): Gate compress tensor [batch_size, seq_len, num_heads, head_dim]
            replicated_q (Optional[torch.Tensor]): Replicated query tensor, typically for text tokens
            replicated_k (Optional[torch.Tensor]): Replicated key tensor
            replicated_v (Optional[torch.Tensor]): Replicated value tensor

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple containing:
                - o (torch.Tensor): Output tensor after attention for the main sequence
                - replicated_o (Optional[torch.Tensor]): Output tensor for replicated tokens, if provided
        """
        if self.sp_attention_mode == "kv_gather":
            raise NotImplementedError(
                "K/V-gather SP does not support video sparse attention."
            )
        # Check text tokens are not supported for VSA now
        assert (
            replicated_q is None and replicated_k is None and replicated_v is None
        ), "Replicated QKV is not supported for VSA now"
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        # Stack QKV
        qkvg = torch.cat(
            [q, k, v, gate_compress], dim=0
        )  # [3, seq_len, num_heads, head_dim]

        # Redistribute heads across sequence dimension
        qkvg = sequence_model_parallel_all_to_all_4D(qkvg, scatter_dim=2, gather_dim=1)

        qkvg = self.attn_impl.preprocess_qkv(qkvg, ctx_attn_metadata)

        q, k, v, gate_compress = qkvg.chunk(4, dim=0)
        output = self.attn_impl.forward(
            q, k, v, gate_compress=gate_compress, attn_metadata=ctx_attn_metadata
        )  # type: ignore[call-arg]

        # Apply backend-specific postprocess_output
        output = self.attn_impl.postprocess_output(output, ctx_attn_metadata)

        output = sequence_model_parallel_all_to_all_4D(
            output, scatter_dim=1, gather_dim=2
        )

        return output


class LocalAttention(nn.Module):
    """Attention layer."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        default_attention_backend: AttentionBackendEnum | None = None,
        is_cross_attention: bool = False,
        compute_dtype: torch.dtype | None = None,
        **extra_impl_args,
    ) -> None:
        super().__init__()
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale
        if num_kv_heads is None:
            num_kv_heads = num_heads

        dtype = compute_dtype or get_compute_dtype()
        attn_backend = get_attn_backend(
            head_size,
            dtype,
            supported_attention_backends=supported_attention_backends,
            default_attention_backend=default_attention_backend,
            is_cross_attention=is_cross_attention,
        )
        impl_cls = attn_backend.get_impl_cls()
        self.allow_cudnn_sdp = bool(extra_impl_args.get("allow_cudnn_sdp", False))
        self._attn_impl_ctor_kwargs = dict(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=self.softmax_scale,
            num_kv_heads=num_kv_heads,
            causal=causal,
            **extra_impl_args,
        )
        self.attn_impl = impl_cls(**self._attn_impl_ctor_kwargs)
        wrap_attention_impl_forward(self.attn_impl)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = attn_backend.get_enum()
        self._default_attn_backend = self.backend
        self._attn_impl_by_backend = {self.backend: self.attn_impl}
        self._supported_attention_backends = supported_attention_backends
        self.dtype = dtype

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Apply local attention between query, key and value tensors.

        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len, num_heads, head_dim]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len, num_heads, head_dim]
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len, num_heads, head_dim]

        Returns:
            torch.Tensor: Output tensor after local attention
        """
        # Check input shapes
        assert q.dim() == 4 and k.dim() == 4 and v.dim() == 4, "Expected 4D tensors"

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        if attn_mask is not None:
            q_ = q.transpose(1, 2)
            k_ = k.transpose(1, 2)
            v_ = v.transpose(1, 2)

            if torch.is_floating_point(attn_mask):
                mask = attn_mask.to(dtype=q_.dtype, device=q_.device)
                if mask.dim() == 2:
                    mask = mask[:, None, None, :]
                elif mask.dim() == 3:
                    mask = mask[:, None, :, :]
            else:
                mask = attn_mask.to(dtype=q_.dtype, device=q_.device)
                if mask.dim() == 2:
                    mask = mask[:, None, None, :]
                elif mask.dim() == 3:
                    mask = mask[:, None, :, :]
                mask = (mask - 1.0) * torch.finfo(q_.dtype).max

            if q_.shape[1] != k_.shape[1]:
                repeat_factor = q_.shape[1] // k_.shape[1]
                k_ = k_.repeat_interleave(repeat_factor, dim=1)
                v_ = v_.repeat_interleave(repeat_factor, dim=1)

            sdpa_context = (
                sdpa_kernel(_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS)
                if self.allow_cudnn_sdp and q_.device.type == "cuda"
                else nullcontext()
            )
            attn_kwargs = {
                "attn_mask": mask,
                "dropout_p": 0.0,
                "is_causal": False,
                "scale": self.softmax_scale,
            }
            with sdpa_context:
                return torch.nn.functional.scaled_dot_product_attention(
                    q_,
                    k_,
                    v_,
                    **attn_kwargs,
                ).transpose(1, 2)

        output = self.attn_impl.forward(q, k, v, attn_metadata=ctx_attn_metadata)
        return output


class USPAttention(nn.Module):
    """
    Sequence-parallel attention with Ulysses, K/V gather, and Ring Attention.

    The default path implements USP, which combines Ulysses-style all-to-all
    communication for sequence-head dimension sharding with Ring Attention
    inside subgroups. The K/V-gather path keeps queries sequence-sharded and
    gathers keys and values within the SP group.
    """

    _usp_a2a_stream = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        default_attention_backend: AttentionBackendEnum | None = None,
        prefix: str = "",
        dropout_rate: float = 0.0,
        skip_sequence_parallel: bool = False,
        enable_packed_qkv_input_a2a: bool = False,
        is_cross_attention: bool = False,
        **extra_impl_args,
    ) -> None:
        """
        Args:
            skip_sequence_parallel:
              when KV is replicated across all SP ranks (e.g. cross-attention to
              text/image encoder outputs), the full USP pipeline is redundant:
              each rank's local Q shard can attend directly to the locally-held
              full KV without any collective communication.
            default_attention_backend:
              preferred fallback when the global backend is incompatible with
              this layer. Explicit component overrides otherwise remain strict.
            is_cross_attention:
              sparse backend preferences may select a compatible dense backend
              for cross-attention while remaining strict for self-attention.
        """
        super().__init__()
        if softmax_scale is None:
            self.softmax_scale = head_size**-0.5
        else:
            self.softmax_scale = softmax_scale

        if num_kv_heads is None:
            num_kv_heads = num_heads

        dtype = get_compute_dtype()
        attn_backend = get_attn_backend(
            head_size,
            dtype,
            supported_attention_backends=supported_attention_backends,
            default_attention_backend=default_attention_backend,
            is_cross_attention=is_cross_attention,
        )
        if not skip_sequence_parallel and get_ring_parallel_world_size() > 1:
            if not attn_backend.supports_ring_rotation():
                raise RuntimeError(
                    f"Ring Attention requires a backend whose kernel exposes the "
                    f"softmax LSE for the per-hop merge; "
                    f"{attn_backend.get_enum().name} does not declare support "
                    f"(see AttentionBackend.supports_ring_rotation)."
                )
        impl_cls: Type[AttentionImpl] = attn_backend.get_impl_cls()
        self.allow_cudnn_sdp = bool(extra_impl_args.get("allow_cudnn_sdp", False))
        self._attn_impl_ctor_kwargs = dict(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=self.softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=f"{prefix}.impl",
            **extra_impl_args,
        )
        self.attn_impl = impl_cls(**self._attn_impl_ctor_kwargs)
        wrap_attention_impl_forward(self.attn_impl)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = attn_backend.get_enum()
        self._default_attn_backend = self.backend
        self._attn_impl_by_backend = {self.backend: self.attn_impl}
        self._supported_attention_backends = supported_attention_backends
        self.dtype = dtype
        self.causal = causal
        self.dropout_p = dropout_rate

        self.skip_sequence_parallel = skip_sequence_parallel
        self.enable_packed_qkv_input_a2a = bool(enable_packed_qkv_input_a2a)
        self.sp_attention_mode, self.sp_attention_mode_is_auto = (
            _resolve_sp_attention_mode(
                causal=causal, sparse_backend=self.backend.is_sparse
            )
        )

    def _get_usp_a2a_stream(self):
        if USPAttention._usp_a2a_stream is None:
            USPAttention._usp_a2a_stream = torch.get_device_module().Stream()
        return USPAttention._usp_a2a_stream

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        num_replicated_prefix: int = 0,
        num_replicated_suffix: int = 0,
        num_replicated_kv_prefix: int = 0,
        skip_sequence_parallel_override: bool = False,
        attn_mask_meta: dict | None = None,
        qkv_pre_all_to_all: bool = False,
        seq_lens: list[int] | None = None,
        q_prefix: torch.Tensor | None = None,
        k_prefix: torch.Tensor | None = None,
        v_prefix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for USPAttention.

            q, k, v: [B, S_local, H, D]
            num_replicated_prefix: number of leading tokens in q/k/v that are
                replicated (identical) across all SP ranks, e.g. text tokens
                in FLUX joint attention.  These tokens are excluded from the
                Ulysses all-to-all so they appear exactly once in the gathered
                sequence, preserving correct attention weights.
            num_replicated_suffix: number of trailing tokens in q/k/v that are
                replicated across all SP ranks, e.g. caption tokens appended
                after image tokens in Z-Image joint attention.
            num_replicated_kv_prefix: number of leading tokens in k/v only
                (not q) that are replicated across all SP ranks. Used for
                cross-attention where the keys/values include a fully-replicated
                conditioning prefix (e.g. cached text K/V) followed by a
                sequence-sharded suffix (image tokens). Q has no replicated
                portion and is fully sequence-sharded.
            attn_mask_meta: optional metadata for the varlen FA fast path.
                Callers may pass ``build_varlen_mask_meta(attn_mask)`` or a
                known contiguous padding gap. Masked query rows are zero-filled
                on output (differs from SDPA semantics).

        Note: Replicated tensors are not supported in this implementation.
        When skip_sequence_parallel=True (set at construction time), all SP
        communication is bypassed — use this for cross-attention where KV
        content is replicated across ranks (distinct from replicated_k/v args).
        """
        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata
        effective_skip_sp = (
            self.skip_sequence_parallel or skip_sequence_parallel_override
        )
        if seq_lens is not None:
            assert (
                attn_mask is None
                and attn_mask_meta is None
                and not num_replicated_prefix
                and not num_replicated_suffix
                and not num_replicated_kv_prefix
            ), "Varlen USPAttention does not support masks or replicated tokens"
            if effective_skip_sp or get_sequence_parallel_world_size() == 1:
                return self.attn_impl.forward(q, k, v, ctx_attn_metadata)
            if get_ring_parallel_world_size() > 1:
                # The varlen all-to-all spans the combined SP group and is not
                # ring-aware; it would shuffle rows across ring ranks instead
                # of rotating KV, corrupting the output silently.
                raise NotImplementedError(
                    "Varlen USPAttention does not support ring parallelism yet."
                )
            qkv = torch.cat([q, k, v], dim=0)
            qkv = _usp_input_all_to_all_varlen(qkv, seq_lens, head_dim=2)
            qkv = self.attn_impl.preprocess_qkv(qkv, ctx_attn_metadata)
            q, k, v = qkv.chunk(3, dim=0)
            out = self.attn_impl.forward(q, k, v, ctx_attn_metadata)
            out = self.attn_impl.postprocess_output(out, ctx_attn_metadata)
            return _usp_output_all_to_all_varlen(out, seq_lens, head_dim=2)

        if isinstance(attn_mask_meta, DynamicVarlenMaskMeta):
            attn_mask_meta = attn_mask_meta.resolve(attn_mask)

        segmented_prefix = q_prefix is not None
        if segmented_prefix != (k_prefix is not None) or segmented_prefix != (
            v_prefix is not None
        ):
            raise ValueError("q_prefix, k_prefix, and v_prefix must be set together")
        if segmented_prefix and not (
            effective_skip_sp or get_sequence_parallel_world_size() == 1
        ):
            raise NotImplementedError(
                "Segmented QKV input currently supports only the local attention path."
            )

        # Tail-pad meta alone (sp_shard.tail_attn_meta; mask derivable from the
        # pad span) also opts into the masked SP branch. gap_* = legacy alias.
        meta_pad_start = meta_pad_end = None
        if attn_mask_meta is not None:
            meta_pad_start = attn_mask_meta.get(
                "pad_start", attn_mask_meta.get("gap_start")
            )
            meta_pad_end = attn_mask_meta.get("pad_end", attn_mask_meta.get("gap_end"))
        meta_only_pad = (
            attn_mask is None
            and meta_pad_start is not None
            and not effective_skip_sp
            and get_sequence_parallel_world_size() > 1
        )
        replicated_mode_count = _count_active_replicated_modes(
            num_replicated_prefix,
            num_replicated_suffix,
            num_replicated_kv_prefix,
        )
        if (
            self.sp_attention_mode == "kv_gather"
            and not effective_skip_sp
            and get_sequence_parallel_world_size() > 1
        ):
            unsupported = _kv_gather_unsupported_reason(
                qkv_pre_all_to_all=qkv_pre_all_to_all,
                replicated_mode_count=replicated_mode_count,
                attn_mask=attn_mask,
                num_replicated_kv_prefix=num_replicated_kv_prefix,
            )
            if unsupported is None:
                return self._forward_with_kv_gather(
                    q,
                    k,
                    v,
                    ctx_attn_metadata,
                    attn_mask,
                    attn_mask_meta,
                    num_replicated_prefix,
                    num_replicated_suffix,
                    num_replicated_kv_prefix,
                )
            if not self.sp_attention_mode_is_auto:
                raise NotImplementedError(unsupported)

        if attn_mask is not None or meta_only_pad:
            if (
                (
                    num_replicated_prefix
                    or num_replicated_suffix
                    or num_replicated_kv_prefix
                )
                and not effective_skip_sp
                and get_sequence_parallel_world_size() > 1
            ):
                # Under SP this path shards every row through the all-to-all;
                # a replicated prefix/suffix would be duplicated across ranks
                # and silently corrupt the output, so refuse loudly instead.
                # On a single rank the mask already describes the full
                # sequence and the replicated counts are meaningless, so the
                # call is legal.
                raise NotImplementedError(
                    "USPAttention's masked path does not support replicated "
                    "prefix/suffix tokens under sequence parallelism; drop "
                    "attn_mask/attn_mask_meta or the replicated segment."
                )

            def _prepare_sdpa_mask(
                mask: torch.Tensor, *, dtype: torch.dtype, device: torch.device
            ) -> torch.Tensor:
                mask = mask.to(device=device)
                if torch.is_floating_point(mask):
                    mask = mask.to(dtype=dtype)
                    if mask.dim() == 2:
                        mask = mask[:, None, None, :]
                    elif mask.dim() == 3:
                        mask = mask[:, None, :, :]
                    return mask

                mask = mask.to(dtype=dtype)
                if mask.dim() == 2:
                    mask = mask[:, None, None, :]
                elif mask.dim() == 3:
                    mask = mask[:, None, :, :]
                return (mask - 1.0) * torch.finfo(dtype).max

            sp_world_size = get_sequence_parallel_world_size()
            if effective_skip_sp or sp_world_size == 1:
                # Varlen FA fast path: SDPA with a non-None mask falls back
                # to cutlassF. Meta-gated to opt in callers that drop masked
                # query rows downstream (zero-filled on output, differs from
                # SDPA semantics). Without meta, fall through to SDPA.
                if (
                    _VARLEN_FA_ENABLED
                    and attn_mask_meta is not None
                    and self.backend == AttentionBackendEnum.FA
                    and attn_mask.dim() == 2
                    and attn_mask.dtype
                    in (torch.bool, torch.uint8, torch.int32, torch.int64)
                    and q.device.type == "cuda"
                    and attn_mask.device == q.device
                    and q.dtype in (torch.float16, torch.bfloat16)
                    and (
                        (q.shape[0], q.shape[1] + q_prefix.shape[1])
                        if segmented_prefix
                        else q.shape[:2]
                    )
                    == attn_mask.shape
                    and q.shape == k.shape == v.shape
                    and (
                        not segmented_prefix
                        or q_prefix.shape == k_prefix.shape == v_prefix.shape
                    )
                ):
                    bs = q.shape[0]
                    seq = q.shape[1] + (q_prefix.shape[1] if segmented_prefix else 0)
                    indices = attn_mask_meta["indices"]
                    cu_seqlens = attn_mask_meta["cu_seqlens"]
                    max_seqlen = attn_mask_meta["max_seqlen"]
                    inv_indices = attn_mask_meta["inv_indices"]
                    # Guard against a caller passing meta from a different
                    # mask shape (silent corruption otherwise).
                    assert (
                        inv_indices.shape[0] == bs * seq
                    ), "attn_mask_meta shape does not match attn_mask"
                    # All-False mask: FA varlen rejects zero-length input.
                    # Fall through to SDPA which handles it via broadcast.
                    # (Joint attention with an image side is always non-empty
                    # in practice, so this only guards malformed inputs.)
                    if indices.shape[0] > 0:
                        all_valid = indices.shape[0] == bs * seq
                        if segmented_prefix:
                            q_unpad, k_unpad, v_unpad = fused_pack_segmented_qkv(
                                q_prefix,
                                k_prefix,
                                v_prefix,
                                q,
                                k,
                                v,
                                indices,
                            )
                        else:
                            if all_valid:
                                q_unpad, k_unpad, v_unpad = q, k, v
                            else:
                                q_unpad, k_unpad, v_unpad = fused_pack_qkv(
                                    q, k, v, indices
                                )
                        if bs == 1 or all_valid:
                            # Empty cu_seqlens selects FA3's faster static
                            # persistent scheduler. A single packed sequence is
                            # dense even when its BCG bucket contains padding.
                            dense_seq = indices.shape[0] if bs == 1 else seq
                            out_dense = flash_attn_varlen_func(
                                q=q_unpad.reshape(bs, dense_seq, *q_unpad.shape[-2:]),
                                k=k_unpad.reshape(bs, dense_seq, *k_unpad.shape[-2:]),
                                v=v_unpad.reshape(bs, dense_seq, *v_unpad.shape[-2:]),
                                cu_seqlens_q=None,
                                cu_seqlens_k=None,
                                max_seqlen_q=dense_seq,
                                max_seqlen_k=dense_seq,
                                softmax_scale=self.softmax_scale,
                                causal=False,
                                ver=_fa_backend.fa_ver,
                            )
                            if all_valid:
                                return out_dense
                            return fused_scatter_to_padded(
                                out_dense.flatten(0, 1), inv_indices, bs, seq
                            )
                        out_unpad = flash_attn_varlen_func(
                            q=q_unpad,
                            k=k_unpad,
                            v=v_unpad,
                            cu_seqlens_q=cu_seqlens,
                            cu_seqlens_k=cu_seqlens,
                            max_seqlen_q=max_seqlen,
                            max_seqlen_k=max_seqlen,
                            softmax_scale=self.softmax_scale,
                            causal=False,
                            ver=_fa_backend.fa_ver,
                        )
                        return fused_scatter_to_padded(out_unpad, inv_indices, bs, seq)

                if segmented_prefix:
                    q = torch.cat([q_prefix, q], dim=1)
                    k = torch.cat([k_prefix, k], dim=1)
                    v = torch.cat([v_prefix, v], dim=1)

                q_ = q.transpose(1, 2)
                k_ = k.transpose(1, 2)
                v_ = v.transpose(1, 2)
                mask = _prepare_sdpa_mask(attn_mask, dtype=q_.dtype, device=q_.device)
                sdpa_context = (
                    sdpa_kernel(_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS)
                    if self.allow_cudnn_sdp and q_.device.type == "cuda"
                    else nullcontext()
                )
                with sdpa_context:
                    return torch.nn.functional.scaled_dot_product_attention(
                        q_,
                        k_,
                        v_,
                        attn_mask=mask,
                        dropout_p=0.0,
                        is_causal=False,
                        scale=self.softmax_scale,
                    ).transpose(1, 2)

            if get_ring_parallel_world_size() > 1:
                if (
                    meta_only_pad
                    and q.shape[0] == 1
                    and self.backend == AttentionBackendEnum.FA
                ):
                    return self._forward_ring_tail_pad(q, k, v, attn_mask_meta)
                raise NotImplementedError(
                    "USPAttention masked path supports ring parallelism only "
                    "for batch-1 tail-pad metadata on the FA backend."
                )
            if attn_mask is not None and attn_mask.dim() != 2:
                raise NotImplementedError(
                    "USPAttention masked SP path currently expects a [B, S_local] key mask."
                )

            sp_size = get_ulysses_parallel_world_size()
            if sp_size > 1 and not qkv_pre_all_to_all:
                qkv_fast = _ipc_input_a2a_qkv(q, k, v)
                if qkv_fast is not None:
                    q, k, v = qkv_fast
                else:
                    q, k, v = _usp_input_all_to_all_qkv(q, k, v)

            if (
                _VARLEN_FA_ENABLED
                and self.backend == AttentionBackendEnum.FA
                and meta_pad_start is not None
                and meta_pad_end is not None
                and meta_pad_end > meta_pad_start
                and q.device.type == "cuda"
                and q.dtype in (torch.float16, torch.bfloat16)
            ):
                bs, seq = q.shape[0], q.shape[1]
                assert 0 <= meta_pad_start < meta_pad_end <= seq
                cu_tail = attn_mask_meta.get("cu_seqlens_tail")
                if cu_tail is not None and meta_pad_end == seq:
                    # Zero-copy tail path: run varlen FA straight over the
                    # padded layout, each row split into [valid | pad] segments
                    # (contiguous reshapes only, no repacking).
                    assert (
                        cu_tail.numel() == 2 * bs + 1
                    ), "cu_seqlens_tail does not match the batch size"
                    out = flash_attn_varlen_func(
                        q=q.reshape(bs * seq, *q.shape[2:]),
                        k=k.reshape(bs * seq, *k.shape[2:]),
                        v=v.reshape(bs * seq, *v.shape[2:]),
                        cu_seqlens_q=cu_tail,
                        cu_seqlens_k=cu_tail,
                        max_seqlen_q=attn_mask_meta["max_seqlen_tail"],
                        max_seqlen_k=attn_mask_meta["max_seqlen_tail"],
                        softmax_scale=self.softmax_scale,
                        causal=False,
                        ver=_fa_backend.fa_ver,
                    ).reshape(bs, seq, *q.shape[2:])
                    # Match the packed paths: masked query rows read as zeros.
                    out[:, meta_pad_start:].zero_()
                    if sp_size > 1:
                        out = _usp_output_all_to_all(out, head_dim=2)
                    return out
                valid_seq = seq - (meta_pad_end - meta_pad_start)
                q_dense = torch.cat([q[:, :meta_pad_start], q[:, meta_pad_end:]], dim=1)
                k_dense = torch.cat([k[:, :meta_pad_start], k[:, meta_pad_end:]], dim=1)
                v_dense = torch.cat([v[:, :meta_pad_start], v[:, meta_pad_end:]], dim=1)
                cu_seqlens = torch.arange(
                    0,
                    (bs + 1) * valid_seq,
                    valid_seq,
                    dtype=torch.int32,
                    device=q.device,
                )
                out_dense = flash_attn_varlen_func(
                    q=q_dense.reshape(bs * valid_seq, *q.shape[2:]),
                    k=k_dense.reshape(bs * valid_seq, *k.shape[2:]),
                    v=v_dense.reshape(bs * valid_seq, *v.shape[2:]),
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=valid_seq,
                    max_seqlen_k=valid_seq,
                    softmax_scale=self.softmax_scale,
                    causal=False,
                    ver=_fa_backend.fa_ver,
                ).reshape(bs, valid_seq, *q.shape[2:])
                gap_out = out_dense.new_zeros(
                    bs,
                    meta_pad_end - meta_pad_start,
                    out_dense.shape[2],
                    out_dense.shape[3],
                )
                out = torch.cat(
                    [
                        out_dense[:, :meta_pad_start],
                        gap_out,
                        out_dense[:, meta_pad_start:],
                    ],
                    dim=1,
                )
                if sp_size > 1:
                    out = _usp_output_all_to_all(out, head_dim=2)
                return out

            # If NCCL timeout/deadlock occurs here, check whether
            # attn_mask is inconsistent across SP ranks (None on some, Tensor on
            # others), which causes all_gather participant mismatch. Upstream
            # mask builders must ensure all ranks produce the same mask type.
            if attn_mask is None:
                # Meta-only tail-pad caller on a non-FA fallback: the gathered
                # mask is fully determined by the pad span, no collective needed.
                gathered_mask = torch.ones(
                    q.shape[0], q.shape[1], dtype=torch.bool, device=q.device
                )
                gathered_mask[:, meta_pad_start:meta_pad_end] = False
            else:
                gathered_mask = sequence_model_parallel_all_gather(
                    attn_mask.contiguous(), dim=1
                )
            if (
                _VARLEN_FA_ENABLED
                and self.backend == AttentionBackendEnum.FA
                and gathered_mask.dtype
                in (torch.bool, torch.uint8, torch.int32, torch.int64)
                and q.device.type == "cuda"
                and gathered_mask.device == q.device
                and q.dtype in (torch.float16, torch.bfloat16)
                and q.shape[:2] == gathered_mask.shape == k.shape[:2] == v.shape[:2]
            ):
                bs, seq = q.shape[0], q.shape[1]
                gathered_mask_meta = build_varlen_mask_meta(gathered_mask)
                indices = gathered_mask_meta["indices"]
                inv_indices = gathered_mask_meta["inv_indices"]
                assert (
                    inv_indices.shape[0] == bs * seq
                ), "gathered attn_mask shape does not match q/k/v"
                if indices.shape[0] > 0:
                    q_unpad, k_unpad, v_unpad = fused_pack_qkv(q, k, v, indices)
                    out_unpad = flash_attn_varlen_func(
                        q=q_unpad,
                        k=k_unpad,
                        v=v_unpad,
                        cu_seqlens_q=gathered_mask_meta["cu_seqlens"],
                        cu_seqlens_k=gathered_mask_meta["cu_seqlens"],
                        max_seqlen_q=gathered_mask_meta["max_seqlen"],
                        max_seqlen_k=gathered_mask_meta["max_seqlen"],
                        softmax_scale=self.softmax_scale,
                        causal=False,
                        ver=_fa_backend.fa_ver,
                    )
                    out = fused_scatter_to_padded(out_unpad, inv_indices, bs, seq)
                    if sp_size > 1:
                        out = _usp_output_all_to_all(out, head_dim=2)
                    return out

            q_ = q.transpose(1, 2)
            k_ = k.transpose(1, 2)
            v_ = v.transpose(1, 2)
            mask = _prepare_sdpa_mask(gathered_mask, dtype=q_.dtype, device=q_.device)
            sdpa_context = (
                sdpa_kernel(_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS)
                if self.allow_cudnn_sdp and q_.device.type == "cuda"
                else nullcontext()
            )
            with sdpa_context:
                out = torch.nn.functional.scaled_dot_product_attention(
                    q_,
                    k_,
                    v_,
                    attn_mask=mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.softmax_scale,
                ).transpose(1, 2)
            if sp_size > 1:
                out = _usp_output_all_to_all(out, head_dim=2)
            return out

        if effective_skip_sp or get_sequence_parallel_world_size() == 1:
            # No sequence parallelism, just run local attention.
            out = self.attn_impl.forward(q, k, v, ctx_attn_metadata)
            return out

        sp_size = get_ulysses_parallel_world_size()
        if replicated_mode_count > 1:
            raise ValueError(
                "USPAttention supports at most one replicated-token mode per call."
            )
        # Replicated-token handling is keyed on the full SP group: with u=1,
        # r>1 the plain ring path would rotate the replicated tokens as if
        # they were sharded rows, double-counting them.
        sp_ws = get_sequence_parallel_world_size()
        if sp_ws > 1 and num_replicated_prefix > 0:
            return self._forward_with_replicated_prefix(
                q, k, v, ctx_attn_metadata, num_replicated_prefix
            )
        if sp_ws > 1 and num_replicated_suffix > 0:
            return self._forward_with_replicated_suffix(
                q, k, v, ctx_attn_metadata, num_replicated_suffix
            )
        if sp_ws > 1 and num_replicated_kv_prefix > 0:
            return self._forward_with_replicated_kv_prefix(
                q, k, v, ctx_attn_metadata, num_replicated_kv_prefix
            )

        # Ulysses-style All-to-All for sequence/head sharding
        if sp_size > 1 and not qkv_pre_all_to_all:
            # -> [B, S, H_local, D]
            if self.enable_packed_qkv_input_a2a and q.device.type == "cuda":
                q, k, v = async_a2a_communicate(
                    [q, k, v],
                    sp_size,
                    get_sp_group().ulysses_group,
                    self._get_usp_a2a_stream(),
                    local_seq_2_local_head=True,
                )
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            else:
                q, k, v = _usp_input_all_to_all_qkv(q, k, v)

        # Ring Attention within subgroups or local attention
        if get_ring_parallel_world_size() > 1:
            out = ring_attn(
                q,
                k,
                v,
                attn_impl=self.attn_impl,
                is_causal=self.causal,
                dropout_p=self.dropout_p,
            )
        else:
            # -> [B, S, H_local, D]
            out = self.attn_impl.forward(q, k, v, ctx_attn_metadata)

        # Ulysses-style All-to-All to restore original sharding
        if sp_size > 1:
            # -> [B, S_local, H, D]
            out = _usp_output_all_to_all(out, head_dim=2)

        return out

    def _forward_ring_tail_pad(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask_meta: dict,
    ) -> torch.Tensor:
        """Ring attention for a tail-padded shard.

        After the Ulysses all-to-all each ring rank holds one contiguous block
        of the gathered sequence, and the tail-pad invariant keeps all padding
        at the global tail — exactly the ring kernel's real-length clamp, so
        no masks or repacking are needed. Pad rows receive garbage output,
        which the tail-pad consumers already trim.
        """
        q, k, v = _usp_input_all_to_all_qkv(q, k, v)
        out = _ring_attention_varlen(
            q.squeeze(0),
            k.squeeze(0),
            v.squeeze(0),
            attn_impl=self.attn_impl,
            real_seq_len=int(attn_mask_meta["pad_start"]),
            ring_ws=get_ring_parallel_world_size(),
        )
        # Match the Ulysses tail path: masked query rows read as zeros. This
        # rank's chunk covers global rows [rank*chunk, (rank+1)*chunk).
        pad_from = (
            int(attn_mask_meta["pad_start"]) - get_ring_parallel_rank() * out.shape[0]
        )
        if pad_from < out.shape[0]:
            out[max(pad_from, 0) :].zero_()
        return _usp_output_all_to_all(out.unsqueeze(0), head_dim=2)

    @staticmethod
    def _gather_sharded_sequence(
        tensor: torch.Tensor,
        num_replicated_prefix: int = 0,
        num_replicated_suffix: int = 0,
    ) -> torch.Tensor:
        if num_replicated_prefix and num_replicated_suffix:
            raise ValueError(
                "Replicated prefix and suffix cannot be used at the same time."
            )

        if num_replicated_prefix:
            replicated = tensor[:, :num_replicated_prefix]
            sharded = tensor[:, num_replicated_prefix:]
            gathered = sequence_model_parallel_all_gather(sharded, dim=1)
            return torch.cat([replicated, gathered], dim=1)

        if num_replicated_suffix:
            replicated = tensor[:, -num_replicated_suffix:]
            sharded = tensor[:, :-num_replicated_suffix]
            gathered = sequence_model_parallel_all_gather(sharded, dim=1)
            return torch.cat([gathered, replicated], dim=1)

        return sequence_model_parallel_all_gather(tensor, dim=1)

    def _forward_with_kv_gather(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx_attn_metadata,
        attn_mask: torch.Tensor | None,
        attn_mask_meta: dict | None,
        num_replicated_prefix: int,
        num_replicated_suffix: int,
        num_replicated_kv_prefix: int,
    ) -> torch.Tensor:
        if attn_mask is not None and num_replicated_kv_prefix:
            raise NotImplementedError(
                "K/V-gather SP masked attention does not support a KV-only prefix."
            )

        kv_prefix = num_replicated_prefix or num_replicated_kv_prefix
        k = self._gather_sharded_sequence(
            k,
            num_replicated_prefix=kv_prefix,
            num_replicated_suffix=num_replicated_suffix,
        )
        v = self._gather_sharded_sequence(
            v,
            num_replicated_prefix=kv_prefix,
            num_replicated_suffix=num_replicated_suffix,
        )

        if attn_mask is None and attn_mask_meta is None:
            return self.attn_impl.forward(q, k, v, ctx_attn_metadata)

        explicit_mask = attn_mask is not None
        if attn_mask is None:
            local_pad = int(attn_mask_meta.get("local_pad", 0))
            attn_mask = torch.ones(q.shape[:2], dtype=torch.bool, device=q.device)
            if local_pad:
                attn_mask[:, -local_pad:] = False
        elif attn_mask.dim() != 2:
            raise NotImplementedError(
                "K/V-gather SP masked attention expects a [B, S_local] mask."
            )
        else:
            if attn_mask.dtype not in (
                torch.bool,
                torch.uint8,
                torch.int32,
                torch.int64,
            ):
                raise NotImplementedError(
                    "K/V-gather SP supports boolean or integer padding masks."
                )
            attn_mask = attn_mask.to(dtype=torch.bool)

        cache_key = (
            tuple(attn_mask.shape),
            num_replicated_prefix,
            num_replicated_suffix,
            explicit_mask,
        )
        mask_cache = None
        if attn_mask_meta is not None:
            mask_cache = attn_mask_meta.get("_kv_gather_cache")
        cached = mask_cache.get(cache_key) if mask_cache is not None else None
        if cached is None:
            key_mask = self._gather_sharded_sequence(
                attn_mask,
                num_replicated_prefix=num_replicated_prefix,
                num_replicated_suffix=num_replicated_suffix,
            )
            cached = {"key_mask": key_mask}
            if attn_mask_meta is not None:
                if mask_cache is None:
                    mask_cache = {}
                    attn_mask_meta["_kv_gather_cache"] = mask_cache
                mask_cache[cache_key] = cached
        else:
            key_mask = cached["key_mask"]

        if (
            _VARLEN_FA_ENABLED
            and self.backend == AttentionBackendEnum.FA
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            if (
                explicit_mask
                and attn_mask_meta is not None
                and all(
                    key in attn_mask_meta
                    for key in (
                        "indices",
                        "cu_seqlens",
                        "max_seqlen",
                        "inv_indices",
                    )
                )
            ):
                query_meta = attn_mask_meta
            else:
                query_meta = cached.get("query_meta")
                if query_meta is None:
                    query_meta = build_varlen_mask_meta(attn_mask)
                    cached["query_meta"] = query_meta
            key_meta = cached.get("key_meta")
            if key_meta is None:
                key_meta = build_varlen_mask_meta(key_mask)
                cached["key_meta"] = key_meta

            batch_size, query_len = q.shape[:2]
            q_unpad = q.reshape(-1, *q.shape[2:]).index_select(0, query_meta["indices"])
            k_unpad = k.reshape(-1, *k.shape[2:]).index_select(0, key_meta["indices"])
            v_unpad = v.reshape(-1, *v.shape[2:]).index_select(0, key_meta["indices"])
            out_unpad = flash_attn_varlen_func(
                q=q_unpad,
                k=k_unpad,
                v=v_unpad,
                cu_seqlens_q=query_meta["cu_seqlens"],
                cu_seqlens_k=key_meta["cu_seqlens"],
                max_seqlen_q=query_meta["max_seqlen"],
                max_seqlen_k=key_meta["max_seqlen"],
                softmax_scale=self.softmax_scale,
                causal=False,
                ver=_fa_backend.fa_ver,
            )
            return fused_scatter_to_padded(
                out_unpad,
                query_meta["inv_indices"],
                batch_size,
                query_len,
            )

        q_ = q.transpose(1, 2)
        k_ = k.transpose(1, 2)
        v_ = v.transpose(1, 2)
        if q_.shape[1] != k_.shape[1]:
            if q_.shape[1] % k_.shape[1] != 0:
                raise ValueError(
                    f"Query heads ({q_.shape[1]}) must be divisible by "
                    f"KV heads ({k_.shape[1]})."
                )
            repeat_factor = q_.shape[1] // k_.shape[1]
            k_ = k_.repeat_interleave(repeat_factor, dim=1)
            v_ = v_.repeat_interleave(repeat_factor, dim=1)

        sdpa_context = (
            sdpa_kernel(_PYTORCH_DEFAULT_CUDA_SDP_BACKENDS)
            if self.allow_cudnn_sdp and q_.device.type == "cuda"
            else nullcontext()
        )
        with sdpa_context:
            out = torch.nn.functional.scaled_dot_product_attention(
                q_,
                k_,
                v_,
                attn_mask=key_mask[:, None, None, :],
                dropout_p=0.0,
                is_causal=False,
                scale=self.softmax_scale,
            ).transpose(1, 2)
        return out * attn_mask[:, :, None, None]

    def _forward_with_replicated_prefix(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx_attn_metadata,
        num_rep: int,
    ) -> torch.Tensor:
        """Ulysses attention where the first *num_rep* tokens are replicated
        across SP ranks (e.g. text tokens) and should NOT be duplicated by the
        all-to-all.

        Strategy:
        1. Split q/k/v into replicated prefix and SP-sharded suffix.
        2. All-to-all only the sharded suffix (gathers sequence, shards heads).
        3. Locally slice the replicated prefix to the same head shard.
        4. Concatenate [prefix_h_local, gathered_suffix] and run attention.
        5. Split output, all-to-all back the suffix, all-gather prefix heads.
        """
        sp_size = get_ulysses_parallel_world_size()
        u_rank = get_ulysses_parallel_rank()

        q_rep, q_shard = q[:, :num_rep], q[:, num_rep:]
        k_rep, k_shard = k[:, :num_rep], k[:, num_rep:]
        v_rep, v_shard = v[:, :num_rep], v[:, num_rep:]

        q_shard = _usp_input_all_to_all(q_shard, head_dim=2)
        k_shard = _usp_input_all_to_all(k_shard, head_dim=2)
        v_shard = _usp_input_all_to_all(v_shard, head_dim=2)

        # Q and KV can have different head counts (GQA), so slice each replicated
        # prefix by its own per-rank head shard to match the all-to-all'd suffix.
        # For MHA (kv heads == q heads) this is identical to the q shard.
        h_local = q_shard.shape[2]
        kv_h_local = k_shard.shape[2]
        h_start = u_rank * h_local
        kv_h_start = u_rank * kv_h_local
        q_rep = q_rep[:, :, h_start : h_start + h_local, :].contiguous()
        k_rep = k_rep[:, :, kv_h_start : kv_h_start + kv_h_local, :].contiguous()
        v_rep = v_rep[:, :, kv_h_start : kv_h_start + kv_h_local, :].contiguous()

        q = torch.cat([q_rep, q_shard], dim=1)
        out = self._replicated_kv_attention(
            q, k_shard, v_shard, k_rep, v_rep, ctx_attn_metadata
        )

        out_rep = out[:, :num_rep]
        out_shard = out[:, num_rep:]

        out_shard = _usp_output_all_to_all(out_shard, head_dim=2)

        if sp_size > 1:
            gathered = [torch.empty_like(out_rep) for _ in range(sp_size)]
            torch.distributed.all_gather(
                gathered,
                out_rep.contiguous(),
                group=get_sp_group().ulysses_group,
            )
            out_rep = torch.cat(gathered, dim=2)

        return torch.cat([out_rep, out_shard], dim=1)

    def _replicated_kv_attention(
        self,
        q: torch.Tensor,
        k_shard: torch.Tensor,
        v_shard: torch.Tensor,
        k_rep: torch.Tensor,
        v_rep: torch.Tensor,
        ctx_attn_metadata,
        rep_first: bool = True,
    ) -> torch.Tensor:
        """Attention of q against replicated + ring-sharded KV.

        Without ring parallelism the two KV parts concatenate into one local
        kernel call, replicated part first unless `rep_first=False` (the
        suffix path keeps KV in tail order for bitwise stability). Under ring
        parallelism the sharded KV rotates around the ring while the
        replicated KV contributes one extra local partial, LSE-merged with
        the ring result (exact up to float reordering).
        """
        if get_ring_parallel_world_size() > 1:
            out_ring, lse_ring = ring_attn(
                q, k_shard, v_shard, self.attn_impl, return_softmax_lse=True
            )
            out_rep, lse_rep, *_ = self.attn_impl.forward(
                q, k_rep, v_rep, attn_metadata=None, return_softmax_lse=True
            )
            merged = _merge_attention_partials(out_ring, lse_ring, out_rep, lse_rep)
            return merged.to(q.dtype)
        kv_parts = (
            ([k_rep, k_shard], [v_rep, v_shard])
            if rep_first
            else (
                [k_shard, k_rep],
                [v_shard, v_rep],
            )
        )
        k = torch.cat(kv_parts[0], dim=1)
        v = torch.cat(kv_parts[1], dim=1)
        return self.attn_impl.forward(q, k, v, ctx_attn_metadata)

    def forward_with_replicated_kv_prefix(
        self,
        q: torch.Tensor,
        k_prefix: torch.Tensor,
        v_prefix: torch.Tensor,
        k_suffix: torch.Tensor,
        v_suffix: torch.Tensor,
    ) -> torch.Tensor:
        """attention with replicated K/V prefix supplied separately"""
        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        if self.skip_sequence_parallel or get_sequence_parallel_world_size() == 1:
            k = torch.cat([k_prefix, k_suffix], dim=1)
            v = torch.cat([v_prefix, v_suffix], dim=1)
            return self.attn_impl.forward(q, k, v, ctx_attn_metadata)

        if self.sp_attention_mode == "kv_gather":
            k_suffix = sequence_model_parallel_all_gather(k_suffix, dim=1)
            v_suffix = sequence_model_parallel_all_gather(v_suffix, dim=1)
            k = torch.cat([k_prefix, k_suffix], dim=1)
            v = torch.cat([v_prefix, v_suffix], dim=1)
            return self.attn_impl.forward(q, k, v, ctx_attn_metadata)

        if (
            get_ulysses_parallel_world_size() == 1
            and get_ring_parallel_world_size() == 1
        ):
            k = torch.cat([k_prefix, k_suffix], dim=1)
            v = torch.cat([v_prefix, v_suffix], dim=1)
            return self(q, k, v)

        return self._forward_with_replicated_kv_prefix_split(
            q, k_prefix, v_prefix, k_suffix, v_suffix, ctx_attn_metadata
        )

    def _forward_with_replicated_kv_prefix(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx_attn_metadata,
        num_rep: int,
    ) -> torch.Tensor:
        """Ulysses cross-attention where only K/V have a replicated prefix.

        Q is sequence-sharded across SP ranks with no replicated portion. K/V
        carry a fully-replicated prefix (``[:num_rep]``, same on every rank,
        e.g. cached text K/V) followed by a sequence-sharded suffix (e.g.
        image tokens) that aligns with Q's sharding.

        Strategy:
        1. All-to-all Q and the sharded K/V suffix (seq → head shard).
        2. Locally slice the replicated K/V prefix to the same head shard.
        3. Concatenate prefix + suffix on the sequence dim and attend.
        4. All-to-all the output back (head shard → seq shard).
        """
        k_rep, k_shard = k[:, :num_rep], k[:, num_rep:]
        v_rep, v_shard = v[:, :num_rep], v[:, num_rep:]

        return self._forward_with_replicated_kv_prefix_split(
            q, k_rep, v_rep, k_shard, v_shard, ctx_attn_metadata
        )

    def _forward_with_replicated_kv_prefix_split(
        self,
        q: torch.Tensor,
        k_rep: torch.Tensor,
        v_rep: torch.Tensor,
        k_shard: torch.Tensor,
        v_shard: torch.Tensor,
        ctx_attn_metadata,
    ) -> torch.Tensor:
        """split form avoids materializing full K/V before Ulysses all-to-all"""
        u_rank = get_ulysses_parallel_rank()

        if q.device.type == "cuda" and get_ulysses_parallel_world_size() > 1:
            q, k_shard, v_shard = async_a2a_communicate(
                [q, k_shard, v_shard],
                get_ulysses_parallel_world_size(),
                get_sp_group().ulysses_group,
                self._get_usp_a2a_stream(),
                local_seq_2_local_head=True,
            )
            q = q.contiguous()
            k_shard = k_shard.contiguous()
            v_shard = v_shard.contiguous()
        else:
            q = _usp_input_all_to_all(q, head_dim=2)
            k_shard = _usp_input_all_to_all(k_shard, head_dim=2)
            v_shard = _usp_input_all_to_all(v_shard, head_dim=2)

        h_kv_local = k_shard.shape[2]
        h_start = u_rank * h_kv_local
        h_end = h_start + h_kv_local
        k_rep = k_rep[:, :, h_start:h_end, :].contiguous()
        v_rep = v_rep[:, :, h_start:h_end, :].contiguous()

        out = self._replicated_kv_attention(
            q, k_shard, v_shard, k_rep, v_rep, ctx_attn_metadata
        )
        return _usp_output_all_to_all(out, head_dim=2)

    def _forward_with_replicated_suffix(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx_attn_metadata,
        num_rep: int,
    ) -> torch.Tensor:
        """Ulysses attention where the last num_rep tokens are replicated
        across SP ranks and should not be duplicated by the all-to-all.

        The suffix stays at the sequence tail so every query scans K/V in the
        same order as a single rank (bitwise-stable across SP degrees);
        rotating it to the front reorders the reduction, and few-step models
        amplify that into visible drift.
        """
        if num_rep <= 0:
            raise ValueError("num_rep must be positive for replicated suffix.")
        u_rank = get_ulysses_parallel_rank()

        q_shard, q_rep = q[:, :-num_rep], q[:, -num_rep:]
        k_shard, k_rep = k[:, :-num_rep], k[:, -num_rep:]
        v_shard, v_rep = v[:, :-num_rep], v[:, -num_rep:]

        q_shard = _usp_input_all_to_all(q_shard, head_dim=2)
        k_shard = _usp_input_all_to_all(k_shard, head_dim=2)
        v_shard = _usp_input_all_to_all(v_shard, head_dim=2)

        h_local = q_shard.shape[2]
        kv_h_local = k_shard.shape[2]
        h_start = u_rank * h_local
        kv_h_start = u_rank * kv_h_local
        q_rep = q_rep[:, :, h_start : h_start + h_local, :].contiguous()
        k_rep = k_rep[:, :, kv_h_start : kv_h_start + kv_h_local, :].contiguous()
        v_rep = v_rep[:, :, kv_h_start : kv_h_start + kv_h_local, :].contiguous()

        q = torch.cat([q_shard, q_rep], dim=1)
        out = self._replicated_kv_attention(
            q, k_shard, v_shard, k_rep, v_rep, ctx_attn_metadata, rep_first=False
        )

        out_shard = out[:, :-num_rep]
        out_rep = out[:, -num_rep:]

        out_shard = _usp_output_all_to_all(out_shard, head_dim=2)

        sp_size = get_ulysses_parallel_world_size()
        if sp_size > 1:
            gathered = [torch.empty_like(out_rep) for _ in range(sp_size)]
            torch.distributed.all_gather(
                gathered,
                out_rep.contiguous(),
                group=get_sp_group().ulysses_group,
            )
            out_rep = torch.cat(gathered, dim=2)

        return torch.cat([out_shard, out_rep], dim=1)


class _BCGBoxedTupleOutput:
    """Box a tuple-returning break-point output as tensor attributes.

    ``_copy_output`` copies tensors and objects-with-tensor-attributes in
    place across replays but ignores tuples, so tuple-returning attention
    forwards (``UlyssesAttention``) are boxed for the break point and
    unboxed after.
    """

    def __init__(self, values: tuple) -> None:
        self.num_values = len(values)
        for i, value in enumerate(values):
            setattr(self, f"value_{i}", value)

    def astuple(self) -> tuple:
        return tuple(getattr(self, f"value_{i}") for i in range(self.num_values))


def _make_breakable_attention_forward(forward_method):
    """Wrap a DiT attention module's ``forward`` so it becomes a breakable
    CUDA graph (BCG) break point.

    During BCG capture the whole attention forward runs eagerly between
    captured graph segments -- the sequence-parallel all-to-all collectives,
    varlen packing, and dynamic/sparse attention kernels that live here
    cannot (or should not) be captured into a static CUDA graph. When BCG is
    disabled this is a transparent pass-through to the original method.
    """

    def _forward_boxing_tuples(*args, **kwargs):
        out = forward_method(*args, **kwargs)
        return _BCGBoxedTupleOutput(out) if isinstance(out, tuple) else out

    bcg_forward = eager_on_graph(True)(_forward_boxing_tuples)

    @functools.wraps(forward_method)
    def forward(self, *args, **kwargs):
        if is_in_breakable_cuda_graph():
            out = bcg_forward(self, *args, **kwargs)
            return out.astuple() if isinstance(out, _BCGBoxedTupleOutput) else out
        return forward_method(self, *args, **kwargs)

    return forward


# Install the break points on every DiT attention entry point. All diffusion
# models route attention through one of these modules (e.g. FLUX -> USPAttention),
# so wrapping here gives universal, model-agnostic BCG break points without
# touching individual model files.
for _attn_cls in (
    UlyssesAttention,
    UlyssesAttention_VSA,
    LocalAttention,
    USPAttention,
):
    _attn_cls.forward = _make_breakable_attention_forward(_attn_cls.forward)
del _attn_cls
