from typing import Tuple, Union, Optional

import torch
import torch.nn.functional as F
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
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
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
    fused_gdn_gating = fused_gdn_gating_npu

    try: # todo: move to sgl_kernel_npu
        import vllm_ascend
        from vllm_ascend.utils import enable_custom_op
        import torch_npu
    except ModuleNotFoundError as e:
        raise ValueError(
            "Npu ascend conv1d ops not found, Please intall package. "
        ) from e
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


def vllm_causal_conv1d_update(
        hidden_state: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        silu_activation: bool = True,
    ):
        bsz, hidden_size, seq_len = hidden_state.shape
        kernel_size = weight.shape[-1]

        # 逻辑: (K-1) + (L-1). 丢弃最老的历史 (H1), 保留最新的输入 (C)
        target_state_len = (kernel_size - 1) + (seq_len - 1)

        full_context = torch.cat([conv_state, hidden_state], dim=-1).to(weight.dtype)

        # 计算 output
        computation_input = full_context[:, :, -(kernel_size - 1 + seq_len):]
        windows = computation_input.unfold(-1, kernel_size, 1)

        # 同样假设 weight 是 2D [H, K]
        out = (windows * weight[None, :, None, :]).sum(dim=-1)

        if bias is not None:
            out = out + bias[None, :, None]

        if silu_activation:
            out = F.silu(out)

        out = out.to(hidden_state.dtype)

        # 更新 State: 保留最后 target_state_len 长度
        if target_state_len > 0:
            new_conv_state = full_context[:, :, -target_state_len:]
        else:
            new_conv_state = torch.empty(bsz, hidden_size, 0, device=hidden_state.device, dtype=hidden_state.dtype)

        return out, new_conv_state
        

class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            self.decode_kernel = CuteDSLGDNKernel()
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            raise ValueError(
                "CuTe DSL backend only supports decode, not prefill. "
                "Use --linear-attn-prefill-backend triton instead."
            )
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel: use FlashInfer if either decode or prefill selected it
        if decode_backend.is_flashinfer() or prefill_backend.is_flashinfer():
            self.verify_kernel = flashinfer_kernel
        else:
            self.verify_kernel = triton_kernel

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__}"
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
    ) -> tuple:
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

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert (
                self.conv_states_shape[-1] < FLA_CHUNK_SIZE
            ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)

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
        if is_npu() and enable_custom_op():
            conv_weights = layer.conv_weights.transpose(-1, -2).contiguous()
            mixed_qkv = torch.ops._C_ascend.npu_causal_conv1d_update(
                    mixed_qkv,
                    conv_weights,
                    conv_states,
                    cache_indices,
                    layer.bias,
                    None,
                    None,
                    layer.activation,
                    self.pad_slot_id,
            )
        else:
            mixed_qkv = causal_conv1d_update(
                mixed_qkv,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices,
            )

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
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
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            has_initial_states = torch.ones(
                seq_len // forward_batch.spec_info.draft_token_num,
                dtype=torch.bool,
                device=forward_batch.input_ids.device,
            )
            intermediate_state_indices = torch.arange(
                cache_indices.shape[0], dtype=torch.int32, device=cache_indices.device
            )
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            if is_npu() and enable_custom_op():
                num_token_padding = mixed_qkv.shape[0]
                if (
                    not self.graph_mode
                    and forward_batch.num_token_non_padded_cpu != num_token_padding
                ):
                    mixed_qkv = mixed_qkv[: forward_batch.num_token_non_padded_cpu]
                    a = a[: forward_batch.num_token_non_padded_cpu]
                    b = b[: forward_batch.num_token_non_padded_cpu]
                    seq_len = forward_batch.num_token_non_padded_cpu
                    batch_size = cache_indices.shape[0] # remove padding bs
                num_accepted_tokens = torch.full((batch_size,), draft_token_num, dtype=torch.int32, device=mixed_qkv.device)
                mixed_qkv_reshaped = mixed_qkv.view(
                    batch_size, draft_token_num, -1
                )
                # when bs=2 cache_indices.shape=[1], remove mixed_qkv_reshaped padding
                mixed_qkv = torch.ops._C_ascend.npu_causal_conv1d_update(
                    mixed_qkv_reshaped,
                    layer.conv_weights.transpose(-1, -2).contiguous(),
                    conv_states,
                    cache_indices,
                    layer.bias,
                    num_accepted_tokens,
                    None,
                    layer.activation,
                    self.pad_slot_id,
                ).view(seq_len, -1)

                # for qwen3.5 conv1d_update
                # conv_states_to_use = conv_states[cache_indices]
                # mixed_qkv_processed, new_conv_state = vllm_causal_conv1d_update(
                #     mixed_qkv_reshaped.transpose(1, 2).contiguous(),
                #     layer.conv_weights,
                #     conv_states_to_use.transpose(1, 2).contiguous(),
                #     layer.bias,
                #     True,
                # )
                # mixed_qkv =mixed_qkv_processed.transpose(1, 2).contiguous().view(seq_len, -1)
                # conv_states[cache_indices] = new_conv_state.transpose(1, 2).contiguous()
            else:
                mixed_qkv_reshaped = mixed_qkv.view(
                    batch_size, draft_token_num, -1
                ).transpose(1, 2)
                mixed_qkv_processed = causal_conv1d_update(
                    mixed_qkv_reshaped,
                    conv_states,
                    layer.conv_weights,
                    layer.bias,
                    layer.activation,
                    conv_state_indices=cache_indices[:batch_size],
                    intermediate_conv_window=intermediate_conv_window_cache,
                    intermediate_state_indices=intermediate_state_indices[:batch_size],
                    retrieve_next_token=retrieve_next_token,
                    retrieve_next_sibling=retrieve_next_sibling,
                    retrieve_parent_token=retrieve_parent_token,
                )
                mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if (
                forward_batch.mamba_track_mask is not None
                and forward_batch.mamba_track_mask.any()
            ):
                conv_dst = forward_batch.mamba_track_indices
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                mask_indices = forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
                conv_states[conv_dst[mask_indices]] = mixed_qkv_to_track

            if is_npu() and enable_custom_op():
                x_origin=mixed_qkv.transpose(-1, -2).contiguous()
                weight_origin=layer.conv_weights.transpose(-1, -2).contiguous()
                kernel_size = layer.conv_weights.shape[-1]
                conv_states = conv_states[:, -(kernel_size - 1):, :]  # fix mtp prefill, kernel_size = 4
                mixed_qkv = torch.ops._C_ascend.causal_conv1d_fn(
                    x_origin,
                    weight_origin,
                    layer.bias,
                    activation=layer.activation,
                    conv_state=conv_states,
                    has_initial_state=has_initial_states,
                    non_spec_state_indices_tensor=cache_indices,
                    non_spec_query_start_loc=query_start_loc,
                    pad_slot_id=self.pad_slot_id,
                )
            else:
                mixed_qkv = causal_conv1d_fn(
                    mixed_qkv,
                    layer.conv_weights,
                    layer.bias,
                    activation=layer.activation,
                    conv_states=conv_states,
                    has_initial_state=has_initial_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
                ).transpose(0, 1)[:seq_len]

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            if is_npu():
                g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
                num_heads, head_k_dim = layer.num_q_heads,  layer.head_q_dim
                num_value_heads, head_v_dim = layer.num_v_heads, layer.head_v_dim
                query = query.view(-1, num_heads, head_k_dim)
                key = key.view(-1, num_heads, head_k_dim)
                value = value.view(-1, num_value_heads, head_v_dim)
                query = l2norm_fwd(
                    query.contiguous(), eps=1e-6, output_dtype=torch.bfloat16
                )
                key = l2norm_fwd(key.contiguous(), eps=1e-6, output_dtype=torch.bfloat16)
    
                core_attn_out = self.fused_recurrent_gated_delta_rule_update_npu(
                    query,
                    key,
                    value,
                    recurrent_state=ssm_states,
                    beta=beta,
                    g=g,
                    cache_indices=cache_indices,
                    intermediate_state=intermediate_state_cache,
                )
                if (not self.graph_mode) and core_attn_out.shape[0] < num_token_padding:
                    core_attn_out = torch.cat(
                        [
                            core_attn_out,
                            core_attn_out.new_zeros(
                                num_token_padding - core_attn_out.shape[0],
                                *core_attn_out.shape[1:],
                            ),
                        ],
                        dim=0,
                    )
            else:
                core_attn_out = self.kernel_dispatcher.target_verify(
                    A_log=layer.A_log,
                    dt_bias=layer.dt_bias,
                    q=query,
                    k=key,
                    v=value,
                    a=a,
                    b=b,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    intermediate_states_buffer=intermediate_state_cache,
                    intermediate_state_indices=intermediate_state_indices,
                    cache_steps=forward_batch.spec_info.draft_token_num,
                    retrieve_parent_token=retrieve_parent_token,
                )
        else:
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )
            if (is_npu() or is_cpu()) and last_recurrent_state is not None:
                if is_npu() and not forward_batch.spec_algorithm.is_none():
                    # AscendC GDN fusion operator requires the shape of the recurrent_state to be [b, s, Dv, Dk]
                    last_recurrent_state = last_recurrent_state.transpose(-1, -2).to(ssm_states.dtype, copy=False)
                else:
                    last_recurrent_state = last_recurrent_state.to(
                        ssm_states.dtype, copy=False
                    )
                ssm_states[cache_indices] = last_recurrent_state

            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

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
            _, num_heads, head_k_dim = query.shape  # T, N, D
            _, num_value_heads, head_v_dim = value.shape
            beta = beta.view(-1, num_value_heads).to(torch.bfloat16)
            g = g.view(-1, num_value_heads).to(torch.float32)
            batch_size = cache_indices.shape[0]
            seq_len = query.shape[0] // batch_size
            scale = 1 / (head_k_dim**0.5)
    
            if intermediate_state is not None:
                # MTP intermediate_state
                intermediate_state[cache_indices, 0] = recurrent_state[cache_indices] # update indexput slow
                ssm_state = intermediate_state.view(
                    -1, num_value_heads, head_k_dim, head_v_dim
                )
            else:
                ssm_state = recurrent_state

            if self.graph_mode:
                num_accepted_tokens = torch.ones(
                    [batch_size], dtype=torch.int32, device=cache_indices.device
                )
                actual_seq_lengths = torch.ones(
                    [batch_size], dtype=torch.int32, device=cache_indices.device
                )
                # actual_seq_lengths = self.forward_metadata.actual_seq_lengths
                ssm_state_indices = self.forward_metadata.mamba_cache_indices_mtp
                # num_accepted_tokens = self.forward_metadata.num_accepted_tokens
            else:
                actual_seq_lengths = self.actual_seq_lengths
                ssm_state_indices = self.ssm_state_indices
                num_accepted_tokens = self.num_accepted_tokens
            attn_core_out = torch_npu.npu_recurrent_gated_delta_rule(
                query,
                key,
                value,
                ssm_state,  # for shape: (BlockNum, Nv, Dv, Dk)
                beta=beta,
                scale=scale,
                actual_seq_lengths=actual_seq_lengths,
                ssm_state_indices=ssm_state_indices,
                num_accepted_tokens=num_accepted_tokens,
                g=g,
            )
    
            if intermediate_state is not None:
                intermediate_state = ssm_state.view(
                    -1, seq_len, num_value_heads, head_k_dim, head_v_dim
                )
            return attn_core_out