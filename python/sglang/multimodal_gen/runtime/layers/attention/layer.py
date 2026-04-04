# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
from typing import Type

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.distributed.communication_op import (
    sequence_model_parallel_all_gather,
    sequence_model_parallel_all_to_all_4D,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_ring_parallel_world_size,
    get_sequence_parallel_world_size,
    get_sp_group,
    get_sp_parallel_rank,
    get_sp_world_size,
    get_ulysses_parallel_world_size,
)
from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionImpl,
    wrap_attention_impl_forward,
)
from sglang.multimodal_gen.runtime.layers.attention.selector import get_attn_backend
from sglang.multimodal_gen.runtime.layers.usp import (
    _usp_input_all_to_all,
    _usp_output_all_to_all,
    ring_attn,
)
from sglang.multimodal_gen.runtime.managers.forward_context import (
    ForwardContext,
    get_forward_context,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.utils import get_compute_dtype


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

        self.attn_impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=self.softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=f"{prefix}.impl",
            **extra_impl_args,
        )
        wrap_attention_impl_forward(self.attn_impl)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = attn_backend.get_enum()
        self.dtype = dtype

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        replicated_q: torch.Tensor | None = None,
        replicated_k: torch.Tensor | None = None,
        replicated_v: torch.Tensor | None = None,
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
        local_rank = get_sp_parallel_rank()
        world_size = get_sp_world_size()

        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata

        # Stack QKV
        qkv = torch.cat([q, k, v], dim=0)  # [3, seq_len, num_heads, head_dim]

        # Redistribute heads across sequence dimension
        qkv = sequence_model_parallel_all_to_all_4D(qkv, scatter_dim=2, gather_dim=1)
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

        output = sequence_model_parallel_all_to_all_4D(
            output, scatter_dim=1, gather_dim=2
        )
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
        **extra_impl_args,
    ) -> None:
        super().__init__()
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
        self.attn_impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            softmax_scale=self.softmax_scale,
            num_kv_heads=num_kv_heads,
            causal=causal,
            **extra_impl_args,
        )
        wrap_attention_impl_forward(self.attn_impl)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = attn_backend.get_enum()
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

            return torch.nn.functional.scaled_dot_product_attention(
                q_,
                k_,
                v_,
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
                scale=self.softmax_scale,
            ).transpose(1, 2)

        output = self.attn_impl.forward(q, k, v, attn_metadata=ctx_attn_metadata)
        return output


class USPAttention(nn.Module):
    """
    Ulysses Sequence Parallelism with Ring Attention.

    This class implements the USP algorithm, which is a combination of
    Ulysses-style all-to-all communication for sequence-head dimension sharding
    and Ring Attention for fine-grained sequence parallelism within subgroups.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        num_kv_heads: int | None = None,
        softmax_scale: float | None = None,
        causal: bool = False,
        supported_attention_backends: set[AttentionBackendEnum] | None = None,
        prefix: str = "",
        dropout_rate: float = 0.0,
        skip_sequence_parallel: bool = False,
        **extra_impl_args,
    ) -> None:
        """
        Args:
            skip_sequence_parallel:
              when KV is replicated across all SP ranks (e.g. cross-attention to
              text/image encoder outputs), the full USP pipeline is redundant:
              each rank's local Q shard can attend directly to the locally-held
              full KV without any collective communication.
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
            head_size, dtype, supported_attention_backends=supported_attention_backends
        )
        impl_cls: Type["AttentionImpl"] = attn_backend.get_impl_cls()
        self.attn_impl = impl_cls(
            num_heads=num_heads,
            head_size=head_size,
            causal=causal,
            softmax_scale=self.softmax_scale,
            num_kv_heads=num_kv_heads,
            prefix=f"{prefix}.impl",
            **extra_impl_args,
        )
        wrap_attention_impl_forward(self.attn_impl)
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads
        self.backend = attn_backend.get_enum()
        self.dtype = dtype
        self.causal = causal
        self.dropout_p = dropout_rate

        self.skip_sequence_parallel = skip_sequence_parallel

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_replicated_prefix: int = 0,
        num_replicated_suffix: int = 0,
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

        Note: Replicated tensors are not supported in this implementation.
        When skip_sequence_parallel=True (set at construction time), all SP
        communication is bypassed — use this for cross-attention where KV
        content is replicated across ranks (distinct from replicated_k/v args).
        """
        forward_context: ForwardContext = get_forward_context()
        ctx_attn_metadata = forward_context.attn_metadata
        if self.skip_sequence_parallel or get_sequence_parallel_world_size() == 1:
            # No sequence parallelism, just run local attention.
            out = self.attn_impl.forward(q, k, v, ctx_attn_metadata)
            return out

        sp_size = get_ulysses_parallel_world_size()
        if num_replicated_prefix > 0 and num_replicated_suffix > 0:
            raise ValueError(
                "USPAttention does not support replicated prefix and suffix at the same time."
            )
        if sp_size > 1 and num_replicated_prefix > 0:
            return self._forward_with_replicated_prefix(
                q, k, v, ctx_attn_metadata, num_replicated_prefix
            )
        if sp_size > 1 and num_replicated_suffix > 0:
            return self._forward_with_replicated_suffix(
                q, k, v, ctx_attn_metadata, num_replicated_suffix
            )

        # Ulysses-style All-to-All for sequence/head sharding
        if sp_size > 1:
            # -> [B, S, H_local, D]
            q = _usp_input_all_to_all(q, head_dim=2)
            k = _usp_input_all_to_all(k, head_dim=2)
            v = _usp_input_all_to_all(v, head_dim=2)

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
        sp_rank = get_sp_parallel_rank()

        q_rep, q_shard = q[:, :num_rep], q[:, num_rep:]
        k_rep, k_shard = k[:, :num_rep], k[:, num_rep:]
        v_rep, v_shard = v[:, :num_rep], v[:, num_rep:]

        q_shard = _usp_input_all_to_all(q_shard, head_dim=2)
        k_shard = _usp_input_all_to_all(k_shard, head_dim=2)
        v_shard = _usp_input_all_to_all(v_shard, head_dim=2)

        h_local = q_shard.shape[2]
        h_start = sp_rank * h_local
        h_end = h_start + h_local
        q_rep = q_rep[:, :, h_start:h_end, :].contiguous()
        k_rep = k_rep[:, :, h_start:h_end, :].contiguous()
        v_rep = v_rep[:, :, h_start:h_end, :].contiguous()

        q = torch.cat([q_rep, q_shard], dim=1)
        k = torch.cat([k_rep, k_shard], dim=1)
        v = torch.cat([v_rep, v_shard], dim=1)

        out = self.attn_impl.forward(q, k, v, ctx_attn_metadata)

        out_rep = out[:, :num_rep]
        out_shard = out[:, num_rep:]

        out_shard = _usp_output_all_to_all(out_shard, head_dim=2)

        gathered = [torch.empty_like(out_rep) for _ in range(sp_size)]
        torch.distributed.all_gather(
            gathered,
            out_rep.contiguous(),
            group=get_sp_group().ulysses_group,
        )
        out_rep = torch.cat(gathered, dim=2)

        return torch.cat([out_rep, out_shard], dim=1)

    def _forward_with_replicated_suffix(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        ctx_attn_metadata,
        num_rep: int,
    ) -> torch.Tensor:
        """Ulysses attention where the last num_rep tokens are replicated
        across SP ranks and should not be duplicated by the all-to-all."""
        if num_rep <= 0:
            raise ValueError("num_rep must be positive for replicated suffix.")

        q_shard, q_rep = q[:, :-num_rep], q[:, -num_rep:]
        k_shard, k_rep = k[:, :-num_rep], k[:, -num_rep:]
        v_shard, v_rep = v[:, :-num_rep], v[:, -num_rep:]

        # dense self-attention is permutation equivariant for non-causal use.
        # 1. rotate the replicated suffix to the front
        # 2. reuse the validated replicated-prefix path, then
        # 3. rotate the output back
        out = self._forward_with_replicated_prefix(
            torch.cat([q_rep, q_shard], dim=1),
            torch.cat([k_rep, k_shard], dim=1),
            torch.cat([v_rep, v_shard], dim=1),
            ctx_attn_metadata,
            num_rep,
        )
        out_rep, out_shard = out[:, :num_rep], out[:, num_rep:]
        return torch.cat([out_shard, out_rep], dim=1)
