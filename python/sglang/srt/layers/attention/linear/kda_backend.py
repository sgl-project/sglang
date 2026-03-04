from typing import Tuple, Union

import torch
from einops import rearrange

from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.utils import is_cpu, is_npu
from sglang.srt.utils.common import rank0_log

# KDA always uses the triton causal_conv1d_fn (no CUDA override).
# Only causal_conv1d_update needs platform-specific overrides for decode.
if is_npu():
    from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_update_npu

    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_update_cpu

    causal_conv1d_update = causal_conv1d_update_cpu

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


class KDAKernelDispatcher:
    """Dispatches KDA kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonKDAKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        else:
            raise ValueError(
                f"Unsupported KDA decode backend: {decode_backend}. "
                "KDA currently only supports 'triton'."
            )

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        else:
            raise ValueError(
                f"Unsupported KDA prefill backend: {prefill_backend}. "
                "KDA currently only supports 'triton'."
            )

        rank0_log(
            f"KDA kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}"
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class KDAAttnBackend(MambaAttnBackendBase):
    """Attention backend for KDA (Kimi Delta Attention) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = KDAKernelDispatcher(decode_backend, prefill_backend)

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        q_proj_states, k_proj_states, v_proj_states = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        q_conv_weights, k_conv_weights, v_conv_weights = layer.conv_weights
        q_conv_bias, k_conv_bias, v_conv_bias = layer.bias

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        q_conv_state, k_conv_state, v_conv_state = layer_cache.conv
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        q_conv_state = q_conv_state.transpose(-1, -2)
        k_conv_state = k_conv_state.transpose(-1, -2)
        v_conv_state = v_conv_state.transpose(-1, -2)

        q = causal_conv1d_update(
            q_proj_states,
            q_conv_state,
            q_conv_weights,
            q_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        k = causal_conv1d_update(
            k_proj_states,
            k_conv_state,
            k_conv_weights,
            k_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )
        v = causal_conv1d_update(
            v_proj_states,
            v_conv_state,
            v_conv_weights,
            v_conv_bias,
            activation="silu",
            conv_state_indices=cache_indices,
        )

        q = rearrange(q, "n (h d) -> 1 n h d", d=layer.head_q_dim)
        k = rearrange(k, "n (h d) -> 1 n h d", d=layer.head_k_dim)
        v = rearrange(v, "n (h d) -> 1 n h d", d=layer.head_v_dim)

        return self.kernel_dispatcher.decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        q_proj_states, k_proj_states, v_proj_states = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        q_conv_weights, k_conv_weights, v_conv_weights = layer.conv_weights
        q_conv_bias, k_conv_bias, v_conv_bias = layer.bias

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_state_q, conv_state_k, conv_state_v = mamba_cache_params.conv
        # deal with strides
        conv_state_q = conv_state_q.transpose(-1, -2)
        conv_state_k = conv_state_k.transpose(-1, -2)
        conv_state_v = conv_state_v.transpose(-1, -2)

        ssm_states = mamba_cache_params.temporal

        has_initial_state = forward_batch.extend_prefix_lens > 0

        q_proj_states = q_proj_states.transpose(0, 1)
        k_proj_states = k_proj_states.transpose(0, 1)
        v_proj_states = v_proj_states.transpose(0, 1)

        q = causal_conv1d_fn(
            q_proj_states,
            q_conv_weights,
            q_conv_bias,
            activation="silu",
            conv_states=conv_state_q,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        k = causal_conv1d_fn(
            k_proj_states,
            k_conv_weights,
            k_conv_bias,
            activation="silu",
            conv_states=conv_state_k,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        v = causal_conv1d_fn(
            v_proj_states,
            v_conv_weights,
            v_conv_bias,
            activation="silu",
            conv_states=conv_state_v,
            has_initial_state=has_initial_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)

        q = rearrange(q, "n (h d) -> 1 n h d", d=layer.head_q_dim)
        k = rearrange(k, "n (h d) -> 1 n h d", d=layer.head_k_dim)
        v = rearrange(v, "n (h d) -> 1 n h d", d=layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.extend(
            q=q,
            k=k,
            v=v,
            g=a,
            beta=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        return core_attn_out
