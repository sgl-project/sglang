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


class GDPAttnBackend(MambaAttnBackendBase):
    """Attention backend for Nemotron Shortcut's Gated DeltaProduct layers."""

    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        text_config = model_runner.model_config.hf_text_config
        self.num_householder = text_config.gdp_num_householder
        self.conv_states_shape = self.state_pool.mamba_cache.conv[0].shape
        self.prefill_backend = _gdp_prefill_backend(torch.device(self.device))

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
        if metadata.has_mamba_track_mask:
            raise NotImplementedError(
                "GDP does not support mamba state tracking (extra_buffer) yet."
            )
        layer_cache = self.state_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        cache_indices = metadata.mamba_cache_indices
        query_start_loc = metadata.query_start_loc

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
        out = gdp_prefill_flashinfer(
            query=query,
            key=key,
            value=value,
            alpha=alpha,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            backend=self.prefill_backend,
        )
        return out.unsqueeze(0)
