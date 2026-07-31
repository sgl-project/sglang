from typing import Optional

import torch
from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_fn_npu

from sglang.srt.hardware_backend.npu.kernels.causal_conv1d_verify import (
    causal_conv1d_linear_verify_npu,
)
from sglang.srt.hardware_backend.npu.kernels.kda_target_verify import (
    kda_target_verify_npu,
)
from sglang.srt.layers.attention.linear.kda_backend import (
    KDAAttnBackend,
    ragged_verify_dense_scatter_indices,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class AscendKDAAttnBackend(KDAAttnBackend):
    """Ascend implementation of Kimi Delta Attention.

    The model, scheduler, metadata, and non-operator control flow stay in the
    shared KDA backend. This class contains only the layout and operator
    differences required by Ascend.
    """

    @staticmethod
    def _channel_first_conv_states(conv_states: torch.Tensor) -> torch.Tensor:
        # The NPU pool is allocated directly as [pool, channels, window].
        return conv_states

    def _causal_conv1d_extend(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        state: torch.Tensor,
        *,
        has_initial_state: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
    ) -> torch.Tensor:
        # The Ascend varlen kernel pads in the weight dtype. K3 keeps its
        # weights in FP32 and its persistent convolution cache in BF16, so use
        # a compact FP32 working set for the active rows and cast it back.
        local_indices = torch.arange(
            cache_indices.shape[0],
            device=cache_indices.device,
            dtype=cache_indices.dtype,
        )
        state_work = state.index_select(0, cache_indices.to(torch.int64))
        state_work = state_work.to(weight.dtype).contiguous()
        out = causal_conv1d_fn_npu(
            x.to(weight.dtype),
            weight,
            bias,
            activation="silu",
            conv_states=state_work,
            has_initial_state=has_initial_state,
            cache_indices=local_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=seq_lens_cpu,
        )
        state.index_copy_(0, cache_indices.to(torch.int64), state_work.to(state.dtype))
        return out.to(x.dtype).transpose(0, 1)

    def _forward_target_verify(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Run fixed-width DSpark verify with Ascend-native state snapshots."""
        metadata = self.forward_metadata
        seq_len = mixed_qkv.shape[0]
        query_start_loc = metadata.query_start_loc
        cache_indices = metadata.mamba_cache_indices

        cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        intermediate_state = cache.intermediate_ssm
        if intermediate_state is None:
            raise RuntimeError(
                "Ascend KDA target verify requires speculative Mamba scratch."
            )

        draft_token_num = forward_batch.spec_info.draft_token_num
        batch_size = query_start_loc.shape[0] - 1
        num_dense_tokens = batch_size * draft_token_num
        ragged_layout = forward_batch.spec_info.ragged_verify_layout
        if ragged_layout is None and seq_len == num_dense_tokens:
            dense_token_indices = None
            dense_qkv = mixed_qkv.view(batch_size, draft_token_num, -1)
            dense_a = a
            dense_b = b
        else:
            dense_token_indices = ragged_verify_dense_scatter_indices(
                query_start_loc=query_start_loc,
                seq_len=seq_len,
                draft_token_num=draft_token_num,
            )
            dense_qkv = self._scatter_tokens_to_dense(
                mixed_qkv, dense_token_indices, num_dense_tokens
            ).view(batch_size, draft_token_num, -1)
            dense_a = self._scatter_gate_to_dense(
                a, dense_token_indices, num_dense_tokens
            )
            dense_b = self._scatter_gate_to_dense(
                b, dense_token_indices, num_dense_tokens
            )

        intermediate_indices = self.verify_intermediate_state_indices[:batch_size]
        processed_qkv = causal_conv1d_linear_verify_npu(
            dense_qkv.transpose(1, 2).contiguous(),
            cache.conv[0],
            layer.conv_weights,
            layer.bias,
            cache_indices[:batch_size],
            cache.intermediate_conv_window[0],
            intermediate_indices,
            activation="silu",
            update_persistent_state=False,
        )
        processed_qkv = processed_qkv.transpose(1, 2).reshape(
            num_dense_tokens, -1
        )
        q, k, v = processed_qkv.split(
            [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)

        out = kda_target_verify_npu(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=q,
            k=k,
            v=v,
            a=dense_a,
            b=dense_b,
            initial_state_source=cache.temporal,
            initial_state_indices=cache_indices[:batch_size],
            intermediate_states_buffer=intermediate_state,
            intermediate_state_indices=intermediate_indices,
            cache_steps=draft_token_num,
            lower_bound=layer.lower_bound,
            gates_are_preactivated=False,
        )
        if dense_token_indices is None:
            return out
        padded_out = out.new_zeros(
            1, num_dense_tokens + 1, *out.shape[2:]
        )
        padded_out[:, :num_dense_tokens] = out
        return padded_out[:, dense_token_indices]

    @staticmethod
    def _scatter_tokens_to_dense(
        value: torch.Tensor,
        dense_token_indices: torch.Tensor,
        num_dense_tokens: int,
    ) -> torch.Tensor:
        dense = value.new_zeros((num_dense_tokens + 1, *value.shape[1:]))
        dense.index_copy_(0, dense_token_indices, value)
        return dense[:num_dense_tokens]

    @classmethod
    def _scatter_gate_to_dense(
        cls,
        value: torch.Tensor,
        dense_token_indices: torch.Tensor,
        num_dense_tokens: int,
    ) -> torch.Tensor:
        has_leading_singleton = value.ndim >= 2 and value.shape[0] == 1
        token_value = value.squeeze(0) if has_leading_singleton else value
        dense = cls._scatter_tokens_to_dense(
            token_value, dense_token_indices, num_dense_tokens
        )
        return dense.unsqueeze(0) if has_leading_singleton else dense
