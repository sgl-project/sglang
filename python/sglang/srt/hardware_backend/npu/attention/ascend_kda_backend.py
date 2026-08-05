from typing import Optional

import torch
from sgl_kernel_npu.fla.kda_chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h_npu,
)
from sgl_kernel_npu.fla.kda_gate import fused_kda_gate_npu
from sgl_kernel_npu.fla.kda_prefill import (
    chunk_gla_fwd_o_gk_npu,
    recompute_w_u_fwd_npu,
)
from sgl_kernel_npu.fla.kda_target_verify import kda_target_verify_npu
from sgl_kernel_npu.fla.l2norm import l2norm_fwd
from sgl_kernel_npu.fla.solve_tril import solve_tril_npu
from sgl_kernel_npu.fla.utils import prepare_chunk_indices
from sgl_kernel_npu.mamba.causal_conv1d import causal_conv1d_fn_npu
from sgl_kernel_npu.mamba.causal_conv1d_verify import (
    causal_conv1d_linear_verify_npu,
)

from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.kda import chunk_kda_scaled_dot_kkt_fwd
from sglang.srt.layers.attention.linear.kda_backend import (
    KDAAttnBackend,
    ragged_verify_dense_scatter_indices,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class _AscendKDAExtendKernel:
    """Ascend-only KDA prefill decomposition backed by sgl-kernel-npu."""

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
        return_intermediate_states: bool = False,
        **kwargs,
    ):
        chunk_size = 64
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())
        v = v.contiguous()
        beta = beta.contiguous()
        chunk_indices = prepare_chunk_indices(query_start_loc, chunk_size)
        g = chunk_local_cumsum(
            g.contiguous(),
            chunk_size=chunk_size,
            scale=1.4426950408889634,
            cu_seqlens=query_start_loc,
            chunk_indices=chunk_indices,
        )

        triangular, query_key = chunk_kda_scaled_dot_kkt_fwd(
            q=q,
            k=k,
            gk=g,
            beta=beta,
            scale=k.shape[-1] ** -0.5,
            cu_seqlens=query_start_loc,
            output_dtype=torch.float32,
        )
        triangular = solve_tril_npu(
            A=triangular,
            cu_seqlens=query_start_loc,
            output_dtype=k.dtype,
        )
        w, u, gated_k = recompute_w_u_fwd_npu(
            k=k,
            v=v,
            beta=beta,
            A=triangular,
            gk=g,
            cu_seqlens=query_start_loc,
            chunk_indices=chunk_indices,
        )
        del triangular
        chunk_states, new_values = chunk_gated_delta_rule_fwd_h_npu(
            k=gated_k,
            w=w,
            u=u,
            gk=g,
            initial_state=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            chunk_indices=chunk_indices,
            use_exp2=True,
        )
        del w, u, gated_k
        out = chunk_gla_fwd_o_gk_npu(
            q=q,
            v=new_values,
            g=g,
            A=query_key,
            h=chunk_states,
            out=v,
            scale=k.shape[-1] ** -0.5,
            cu_seqlens=query_start_loc,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
        )
        del query_key, new_values
        if return_intermediate_states:
            return out, chunk_states.transpose(-1, -2).contiguous()
        return out


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

    def __init__(self, model_runner):
        super().__init__(model_runner)
        self.kernel_dispatcher.extend_kernel = _AscendKDAExtendKernel()

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

    def _prepare_extend_gate_inputs(
        self,
        layer: RadixLinearAttention,
        g: torch.Tensor,
        beta: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Restore the 0728 Ascend prefill gate contract.

        The checkpoint was validated with FP32 gate activation before
        ``chunk_kda``. Keeping this platform override here leaves the shared
        GPU model/backend paths unchanged.
        """
        preactivated_g = fused_kda_gate_npu(
            g.flatten(-2),
            layer.A_log,
            layer.head_k_dim,
            gate_bias=layer.dt_bias,
            lower_bound=layer.lower_bound,
        )
        return preactivated_g, beta, None, None

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
        processed_qkv = processed_qkv.transpose(1, 2).reshape(num_dense_tokens, -1)
        q, k, v = processed_qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)

        # Match the proven 0728 target-verify contract exactly: activate the
        # forget gate and beta in FP32 before entering the recurrent kernel.
        # This stays in the Ascend backend so shared/GPU model code is unchanged.
        preactivated_a = fused_kda_gate_npu(
            dense_a.flatten(-2),
            layer.A_log,
            layer.head_k_dim,
            gate_bias=layer.dt_bias,
            lower_bound=layer.lower_bound,
        )
        preactivated_b = dense_b.float().sigmoid()
        out = kda_target_verify_npu(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=q,
            k=k,
            v=v,
            a=preactivated_a,
            b=preactivated_b,
            initial_state_source=cache.temporal,
            initial_state_indices=cache_indices[:batch_size],
            intermediate_states_buffer=intermediate_state,
            intermediate_state_indices=intermediate_indices,
            cache_steps=draft_token_num,
            lower_bound=None,
            gates_are_preactivated=True,
        )
        if dense_token_indices is None:
            return out
        padded_out = out.new_zeros(1, num_dense_tokens + 1, *out.shape[2:])
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
