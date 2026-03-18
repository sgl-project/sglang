from typing import Tuple, Union, Optional
import torch
import torch.nn.functional as F
import torch_npu

from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNKernelDispatcher,
    GDNAttnBackend,
)
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.utils import is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

def npu_causal_conv1d_update(
    hidden_state: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    silu_activation: bool = True,
):
    bsz, hidden_size, seq_len = hidden_state.shape
    kernel_size = weight.shape[-1]
    target_state_len = (kernel_size - 1) + (seq_len - 1)
    full_context = torch.cat([conv_state, hidden_state], dim=-1).to(weight.dtype)
    computation_input = full_context[:, :, -(kernel_size - 1 + seq_len):]
    windows = computation_input.unfold(-1, kernel_size, 1)
    out = (windows * weight[None, :, None, :]).sum(dim=-1)

    if bias is not None:
        out = out + bias[None, :, None]
    if silu_activation:
        out = F.silu(out)
    out = out.to(hidden_state.dtype)

    if target_state_len > 0:
        new_conv_state = full_context[:, :, -target_state_len:]
    else:
        new_conv_state = torch.empty(bsz, hidden_size, 0, device=hidden_state.device, dtype=hidden_state.dtype)
    return out, new_conv_state

class AscendGDNKernelDispatcher(GDNKernelDispatcher):
    pass

class AscendGDNAttnBackend(GDNAttnBackend):

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        # self.graph_mode = False

    def prepare_gdn_inputs(
        self,
        bs: int,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        cache_indices = self.forward_metadata.mamba_cache_indices
        self.num_accepted_tokens = torch.ones(
            [bs], dtype=torch.int32, device=cache_indices.device
        )
        self.actual_seq_lengths = torch.ones(
            [bs], dtype=torch.int32, device=cache_indices.device
        )
        if forward_mode.is_target_verify():
            seq_len = spec_info.draft_token_num
            self.actual_seq_lengths = self.actual_seq_lengths * seq_len
            start_indices = cache_indices * seq_len
            offset = torch.arange(seq_len, device=start_indices.device)
            ranges = start_indices.unsqueeze(1) + offset
            self.ssm_state_indices = ranges.flatten().to(torch.int32)
            self.forward_metadata.ssm_state_indices = self.ssm_state_indices
        else:
            self.ssm_state_indices = cache_indices
            self.forward_metadata.ssm_state_indices = self.ssm_state_indices

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_draft_extend(True):
            return
        super().init_forward_metadata(forward_batch)
        self.prepare_gdn_inputs(
            forward_batch.batch_size,
            forward_batch.forward_mode,
            forward_batch.spec_info
        )
        self.graph_mode = False

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        if forward_mode.is_draft_extend(True):
            return
        super().init_forward_metadata_capture_cuda_graph(
            bs, num_tokens, req_pool_indices, seq_lens, encoder_lens, forward_mode, spec_info
        )
        self.prepare_gdn_inputs(bs, forward_mode, spec_info)
        self.graph_mode = True

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        if forward_mode.is_draft_extend(True):
            return
        super().init_forward_metadata_replay_cuda_graph(
            bs, req_pool_indices, seq_lens, seq_lens_sum, encoder_lens,
            forward_mode, spec_info, seq_lens_cpu
        )
        self.prepare_gdn_inputs(bs, forward_mode, spec_info)
        self.graph_mode = True

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        assert isinstance(mixed_qkv, torch.Tensor)
        conv_states_tmp = conv_states.transpose(1, 2).clone()
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states_tmp,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )
        conv_states[:] = conv_states_tmp.transpose(1, 2)

        query, key, value = torch.split(
            mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
        )
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query, k=key, v=value, a=a, b=b,
            A_log=layer.A_log, dt_bias=layer.dt_bias,
            ssm_states=ssm_states, cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )
        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]
        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        intermediate_state_cache = None

        if not forward_batch.spec_algorithm.is_none():
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_conv_window_cache = mamba_cache_params.intermediate_conv_window[0]
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool, device=forward_batch.input_ids.device
            )
            intermediate_state_indices = torch.arange(
                cache_indices.shape[0], dtype=torch.int32, device=cache_indices.device
            )
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            num_token_padding = mixed_qkv.shape[0]
            batch_size = cache_indices.shape[0]

            if (not self.graph_mode) and forward_batch.num_token_non_padded_cpu != num_token_padding:
                mixed_qkv = mixed_qkv[: forward_batch.num_token_non_padded_cpu]
                a = a[: forward_batch.num_token_non_padded_cpu]
                b = b[: forward_batch.num_token_non_padded_cpu]
                seq_len = forward_batch.num_token_non_padded_cpu

            mixed_qkv_reshaped = mixed_qkv.view(batch_size, draft_token_num, -1)
            conv_states_to_use = conv_states[cache_indices]
            mixed_qkv_processed, new_conv_state = npu_causal_conv1d_update(
                mixed_qkv_reshaped.transpose(1, 2).contiguous(),
                layer.conv_weights,
                conv_states_to_use.transpose(1, 2).contiguous(),
                layer.bias, True
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).contiguous().view(seq_len, -1)
            conv_states[cache_indices] = new_conv_state.transpose(1, 2).contiguous()
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if forward_batch.mamba_track_mask is not None and forward_batch.mamba_track_mask.any():
                conv_dst = forward_batch.mamba_track_indices
                mixed_qkv_to_track = mixed_qkv[:, forward_metadata.track_conv_indices].transpose(0, 1)
                mask_indices = forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
                conv_states[conv_dst[mask_indices]] = mixed_qkv_to_track

            kernel_size = layer.conv_weights.shape[-1]
            conv_states_for_prefill = conv_states[:, -(kernel_size - 1):, :]
            conv_states_tmp = conv_states_for_prefill.transpose(1, 2).contiguous()

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv, layer.conv_weights, layer.bias, activation=layer.activation,
                conv_states=conv_states_tmp, has_initial_state=has_initial_states,
                cache_indices=cache_indices, query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]
            conv_states[:, -(kernel_size - 1):, :] = conv_states_tmp.transpose(1, 2).contiguous()

        query, key, value = torch.split(mixed_qkv, [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        actual_seq_len = query.shape[0]
        num_heads, head_k_dim = layer.num_q_heads, layer.head_q_dim
        num_value_heads, head_v_dim = layer.num_v_heads, layer.head_v_dim

        query = query.view(1, actual_seq_len, num_heads, head_k_dim)
        key = key.view(1, actual_seq_len, num_heads, head_k_dim)
        value = value.view(1, actual_seq_len, num_value_heads, head_v_dim)
        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

        if is_target_verify:
            query = query.view(-1, num_heads, head_k_dim)
            key = key.view(-1, num_heads, head_k_dim)
            value = value.view(-1, num_value_heads, head_v_dim)
            query = l2norm_fwd(query.contiguous(), eps=1e-6, output_dtype=torch.bfloat16)
            key = l2norm_fwd(key.contiguous(), eps=1e-6, output_dtype=torch.bfloat16)

            core_attn_out = self.fused_recurrent_gated_delta_rule_update_npu(
                query, key, value, recurrent_state=ssm_states,
                beta=beta, g=g, cache_indices=cache_indices,
                intermediate_state=intermediate_state_cache,
            )
            core_attn_out = core_attn_out.view(-1, num_value_heads, head_v_dim)

            if (not self.graph_mode) and core_attn_out.shape[0] < num_token_padding:
                core_attn_out = torch.cat([
                    core_attn_out,
                    core_attn_out.new_zeros(num_token_padding - core_attn_out.shape[0], *core_attn_out.shape[1:])
                ], dim=0)
        else:
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query, k=key, v=value, g=g, beta=beta,
                ssm_states=ssm_states, cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )
            if not forward_batch.spec_algorithm.is_none() and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.transpose(-1, -2).to(ssm_states.dtype, copy=False)
                intermediate_state_cache[cache_indices, 0] = last_recurrent_state
            elif last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(ssm_states.dtype, copy=False)
                ssm_states[cache_indices] = last_recurrent_state

            if h is not None:
                self._track_mamba_state_extend(forward_batch, h, ssm_states, forward_metadata)
        return core_attn_out

    def fused_recurrent_gated_delta_rule_update_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        recurrent_state: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        cache_indices: torch.Tensor,
        intermediate_state: Optional[torch.Tensor] = None,
    ):
        _, num_heads, head_k_dim = query.shape
        _, num_value_heads, head_v_dim = value.shape
        beta = beta.view(-1, num_value_heads).to(torch.bfloat16)
        g = g.view(-1, num_value_heads).to(torch.float32)
        batch_size = cache_indices.shape[0]
        seq_len = query.shape[0] // batch_size
        scale = 1 / (head_k_dim ** 0.5)

        if intermediate_state is not None:
            ssm_state = intermediate_state.view(-1, num_value_heads, head_k_dim, head_v_dim)
        else:
            ssm_state = recurrent_state[cache_indices]

        if self.graph_mode:
            num_accepted_tokens = torch.ones([batch_size], dtype=torch.int32, device=cache_indices.device)
            actual_seq_lengths = torch.ones([batch_size], dtype=torch.int32, device=cache_indices.device) * seq_len
            ssm_state_indices = self.forward_metadata.mamba_cache_indices_gdn
        else:
            num_accepted_tokens = self.num_accepted_tokens
            actual_seq_lengths = self.actual_seq_lengths
            ssm_state_indices = self.forward_metadata.mamba_cache_indices

        attn_core_out = torch_npu.npu_recurrent_gated_delta_rule(
            query, key, value, ssm_state, beta=beta, scale=scale,
            actual_seq_lengths=actual_seq_lengths, ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens, g=g,
        )

        if intermediate_state is not None:
            intermediate_state = ssm_state.view(-1, seq_len, num_value_heads, head_k_dim, head_v_dim)
        return attn_core_out