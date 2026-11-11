# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Gated DeltaProduct (GDP) linear-attention backend (Nemotron Shortcut).

Each token applies one per-head decay and then ``n`` beta-gated Householder
updates to an FP32 ``[H, V, K]`` state; its query reads the state after the
last update. The conv input row is packed Householder-major as
``[V_0..V_{n-1}, K_0..K_{n-1}, Q]``; ``b`` is ``[b_0..b_{n-1}]`` per head.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
import torch.nn.functional as F

from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import mamba_cache_chunk_size


def gdp_split_conv_output(
    mixed: torch.Tensor,
    *,
    num_householder: int,
    num_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Views of a ``[T, conv_dim]`` conv output as q ``[T, H, K]``,
    k ``[T, n, H, K]`` and v ``[T, n, H, V]``."""
    num_tokens = mixed.shape[0]
    value_dim = num_householder * num_heads * head_v_dim
    key_dim = num_householder * num_heads * head_k_dim
    value = mixed[:, :value_dim].view(
        num_tokens, num_householder, num_heads, head_v_dim
    )
    key = mixed[:, value_dim : value_dim + key_dim].view(
        num_tokens, num_householder, num_heads, head_k_dim
    )
    query = mixed[:, value_dim + key_dim :].view(num_tokens, num_heads, head_k_dim)
    return query, key, value


def gdp_gates(
    *,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    num_householder: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear-space decay alpha ``[T, H]`` and post-sigmoid beta ``[T * n, H]``,
    both FP32, matching the HF reference's FP32 gate math."""
    log_alpha = -A_log.float().exp() * F.softplus(a.float() + dt_bias.float())
    beta = b.float().sigmoid().reshape(-1, A_log.shape[0])
    assert beta.shape[0] == a.shape[0] * num_householder
    return log_alpha.exp(), beta


def gdp_prefill_flashinfer(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    """FlashInfer ``chunk_gated_delta_product``; reads and writes the pool
    slots in place."""
    from flashinfer import chunk_gated_delta_product

    num_tokens, n, num_heads, head_k_dim = key.shape
    out = chunk_gated_delta_product(
        q=query,
        k=key.reshape(num_tokens * n, num_heads, head_k_dim),
        v=value.reshape(num_tokens * n, num_heads, value.shape[-1]),
        g=alpha,
        beta=beta,
        num_householder=n,
        initial_state=ssm_states,
        output_final_state=True,
        output_state=ssm_states,
        cu_seqlens=query_start_loc.to(torch.int32),
        use_qk_l2norm_in_kernel=True,
        state_indices=cache_indices.to(torch.int32).contiguous(),
        backend=backend,
    )
    return out[0] if isinstance(out, tuple) else out


def gdp_decode_flashinfer(
    *,
    layer: RadixLinearAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
) -> torch.Tensor:
    """One token per row through FlashInfer's Cake GDP decode kernel, which
    fuses the gates; padding rows carry a negative slot and are skipped."""
    from flashinfer import cake_gated_delta_product_mtp

    batch, n, num_heads, _ = key.shape
    out, _ = cake_gated_delta_product_mtp(
        q=query.unsqueeze(1),
        k=key.unsqueeze(1),
        v=value.unsqueeze(1),
        initial_state=ssm_states,
        initial_state_indices=cache_indices,
        A_log=layer.A_log,
        a=a.view(batch, 1, num_heads),
        dt_bias=layer.dt_bias,
        b=b.view(batch, 1, n, num_heads),
        use_qk_l2norm=True,
    )
    return out.squeeze(1)


def _gdp_prefill_backend(device: torch.device) -> str:
    """FlashInfer's GDP prefill engine for this GPU: cuDNN on SM100-family
    parts, Cake on SM90; the Cake decode kernel covers both."""
    major, _ = torch.cuda.get_device_capability(device)
    if major == 10:
        return "cudnn"
    if major == 9:
        return "cake"
    raise NotImplementedError(
        f"Nemotron Shortcut GDP needs an SM90 or SM100-family GPU, got SM{major}x."
    )


class GDPTrackPlan(msgspec.Struct, frozen=True):
    """Extend-time state snapshots for the extra_buffer radix cache. Each tracked
    row snapshots its state after ``T`` extend tokens; a row with 0 < T < len
    runs as two prefill passes split at T, with the snapshot copied in between.
    Slot tensors are pool indices; token tensors index the flattened extend."""

    conv_dst: torch.Tensor
    pre_src: Optional[torch.Tensor]  # T == 0: snapshot is the incoming state
    pre_dst: Optional[torch.Tensor]
    post_src: Optional[torch.Tensor]  # T == len: snapshot is the final state
    post_dst: Optional[torch.Tensor]
    first_tokens: Optional[torch.Tensor]  # None: no split row, one pass
    first_cu_seqlens: Optional[torch.Tensor]
    split_src: Optional[torch.Tensor]
    split_dst: Optional[torch.Tensor]
    second_tokens: Optional[torch.Tensor]
    second_cu_seqlens: Optional[torch.Tensor]


def build_gdp_track_plan(
    *,
    track_rows: list[int],
    track_seqlens: list[int],
    extend_lens: list[int],
    prefix_lens: list[int],
    chunk_size: int,
    cache_indices: torch.Tensor,
    track_dst: torch.Tensor,
) -> GDPTrackPlan:
    device = cache_indices.device

    def rows_tensor(rows: list[int]) -> Optional[torch.Tensor]:
        return torch.tensor(rows, dtype=torch.int64).to(device) if rows else None

    def slots(rows):
        idx = rows_tensor(rows)
        if idx is None:
            return None, None
        return cache_indices.index_select(0, idx), track_dst.index_select(0, idx)

    starts = [0]
    for length in extend_lens:
        starts.append(starts[-1] + length)
    pre, post, split = [], [], {}
    for row in track_rows:
        # Same floor as ForwardBatch.mamba_track_aligned_lens.
        cut = (track_seqlens[row] - prefix_lens[row]) // chunk_size * chunk_size
        if cut <= 0:
            pre.append(row)
        elif cut >= extend_lens[row]:
            post.append(row)
        else:
            split[row] = cut

    first_tokens = first_cu = second_tokens = second_cu = None
    if split:
        first, first_lens, second, second_lens = [], [], [], []
        for row, length in enumerate(extend_lens):
            cut = split.get(row, length)
            first.append(torch.arange(starts[row], starts[row] + cut))
            first_lens.append(cut)
            if row in split:
                second.append(torch.arange(starts[row] + cut, starts[row + 1]))
                second_lens.append(length - cut)

        def cu(lens):
            return torch.tensor([0] + lens, dtype=torch.int32).cumsum(0).to(device)

        first_tokens = torch.cat(first).to(device)
        first_cu = cu(first_lens).to(torch.int32)
        second_tokens = torch.cat(second).to(device)
        second_cu = cu(second_lens).to(torch.int32)
    split_rows = list(split)
    pre_src, pre_dst = slots(pre)
    post_src, post_dst = slots(post)
    split_src, split_dst = slots(split_rows)
    return GDPTrackPlan(
        conv_dst=track_dst.index_select(0, rows_tensor(track_rows)),
        pre_src=pre_src,
        pre_dst=pre_dst,
        post_src=post_src,
        post_dst=post_dst,
        first_tokens=first_tokens,
        first_cu_seqlens=first_cu,
        split_src=split_src,
        split_dst=split_dst,
        second_tokens=second_tokens,
        second_cu_seqlens=second_cu,
    )


class GDPAttnBackend(MambaAttnBackendBase):
    """Attention backend for Nemotron Shortcut's Gated DeltaProduct layers."""

    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        text_config = model_runner.model_config.hf_text_config
        self.num_householder = text_config.gdp_num_householder
        self.conv_states_shape = self.state_pool.mamba_cache.conv[0].shape
        self.prefill_backend = _gdp_prefill_backend(torch.device(self.device))
        self.track_plan: Optional[GDPTrackPlan] = None

    def init_forward_metadata_out_graph(
        self, forward_batch: ForwardBatch, in_capture: bool = False
    ):
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        self.track_plan = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        self.track_plan = None
        metadata = self.forward_metadata
        if not (
            metadata.has_mamba_track_mask and forward_batch.forward_mode.is_extend()
        ):
            return
        if forward_batch.mamba_prefill_track_mask_cpu is not None:
            mask = forward_batch.mamba_prefill_track_mask_cpu
        else:
            mask = forward_batch.mamba_track_mask.tolist()
        self.track_plan = build_gdp_track_plan(
            track_rows=[i for i, tracked in enumerate(mask) if tracked],
            track_seqlens=(
                forward_batch.mamba_track_seqlens_cpu
                if forward_batch.mamba_track_seqlens_cpu is not None
                else forward_batch.mamba_track_seqlens.tolist()
            ),
            extend_lens=forward_batch.extend_seq_lens_cpu,
            prefix_lens=forward_batch.extend_prefix_lens_cpu,
            chunk_size=mamba_cache_chunk_size(),
            cache_indices=metadata.mamba_cache_indices,
            track_dst=forward_batch.mamba_track_indices,
        )

    def _layer_dims(self, layer: RadixLinearAttention) -> dict:
        return dict(
            num_householder=self.num_householder,
            num_heads=layer.num_v_heads,
            head_k_dim=layer.head_k_dim,
            head_v_dim=layer.head_v_dim,
        )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        layer_cache = self.state_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        cache_indices = self.forward_metadata.mamba_cache_indices

        mixed = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )
        query, key, value = gdp_split_conv_output(mixed, **self._layer_dims(layer))
        out = gdp_decode_flashinfer(
            layer=layer,
            query=query,
            key=key,
            value=value,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
        )
        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
        )
        return out.unsqueeze(0)

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        if forward_batch.forward_mode.is_target_verify():
            raise NotImplementedError("GDP does not support speculative verify yet.")
        metadata = self.forward_metadata
        plan = self.track_plan
        layer_cache = self.state_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        cache_indices = metadata.mamba_cache_indices
        query_start_loc = metadata.query_start_loc

        if plan is not None:
            # The conv window ending at each snapshot point, from raw inputs.
            conv_states[plan.conv_dst] = mixed_qkv[
                metadata.track_conv_indices
            ].transpose(1, 2)
            if plan.pre_src is not None:
                # T == 0: no extend token precedes the snapshot, so the window is
                # the incoming one, read before causal_conv1d_fn updates it.
                conv_states[plan.pre_dst] = conv_states[plan.pre_src]
        mixed = causal_conv1d_fn(
            mixed_qkv.transpose(0, 1),
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=forward_batch.extend_prefix_lens > 0,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)[: mixed_qkv.shape[0]]
        query, key, value = gdp_split_conv_output(mixed, **self._layer_dims(layer))
        alpha, beta = gdp_gates(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            a=a,
            b=b,
            num_householder=self.num_householder,
        )
        if plan is not None and plan.pre_src is not None:
            ssm_states[plan.pre_dst] = ssm_states[plan.pre_src]
        if plan is None or plan.first_tokens is None:
            out = self._prefill(
                query,
                key,
                value,
                alpha,
                beta,
                ssm_states,
                cache_indices,
                query_start_loc,
            )
        else:
            out = self._prefill_split(
                query, key, value, alpha, beta, ssm_states, cache_indices, plan
            )
        if plan is not None and plan.post_src is not None:
            ssm_states[plan.post_dst] = ssm_states[plan.post_src]
        return out.unsqueeze(0)

    def _prefill(
        self, query, key, value, alpha, beta, ssm_states, cache_indices, cu_seqlens
    ) -> torch.Tensor:
        return gdp_prefill_flashinfer(
            query=query,
            key=key,
            value=value,
            alpha=alpha,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=cu_seqlens,
            backend=self.prefill_backend,
        )

    def _prefill_split(
        self, query, key, value, alpha, beta, ssm_states, cache_indices, plan
    ) -> torch.Tensor:
        beta = beta.view(query.shape[0], self.num_householder, -1)

        def run(tokens, slots, cu_seqlens):
            return self._prefill(
                query.index_select(0, tokens),
                key.index_select(0, tokens),
                value.index_select(0, tokens),
                alpha.index_select(0, tokens),
                beta.index_select(0, tokens).flatten(0, 1),
                ssm_states,
                slots,
                cu_seqlens,
            )

        out = query.new_empty(query.shape[0], value.shape[2], value.shape[3])
        out[plan.first_tokens] = run(
            plan.first_tokens, cache_indices, plan.first_cu_seqlens
        )
        ssm_states[plan.split_dst] = ssm_states[plan.split_src]
        out[plan.second_tokens] = run(
            plan.second_tokens, plan.split_src, plan.second_cu_seqlens
        )
        return out
