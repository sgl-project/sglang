import math
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
from sgl_kernel_npu.fla.solve_tril import solve_tril_npu
from sgl_kernel_npu.fla.utils import prepare_chunk_indices

from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.kda import chunk_kda_scaled_dot_kkt_fwd
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.linear.kda_backend import (
    KDAAttnBackend,
    ragged_verify_dense_scatter_indices,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

_LOG2_E = math.log2(math.e)


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
            scale=_LOG2_E,
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

    Conv states use the GDN-style [layers, pool, window, channels] layout
    (transposed from the shared KDA backend's [channels, window]). The
    speculative window is extended by draft_tokens - 1 so that verify
    writes all draft token conv states directly into conv_states; after
    verify, conv_state_rollback reverts unaccepted tokens. This replaces
    the previous intermediate_conv_window snapshot + scatter scheme.
    """

    supports_speculative_conv_state_snapshots: bool = False

    def __init__(self, model_runner):
        super().__init__(model_runner)
        # The NPU pool is allocated as [layers, pool, window, channels]
        # (transposed from the shared KDA [channels, window]). Expose the
        # transposed shape so _init_track_conv_indices reads
        # conv_states_shape[-1] as the conv window length.
        conv_pool_shape = model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[
            0
        ].shape
        self.conv_states_shape = torch.Size(
            (
                *conv_pool_shape[:-2],
                conv_pool_shape[-1],
                conv_pool_shape[-2],
            )
        )
        self.kernel_dispatcher.extend_kernel = _AscendKDAExtendKernel()

    def _get_conv_weights_t(
        self, layer: RadixLinearAttention, dtype: torch.dtype
    ) -> torch.Tensor:
        """Transposed conv weights [width, dim], cached on the layer.

        The NPU causal_conv1d CANN op expects weight as [width, dim]
        (transposed from layer.conv_weights [dim, width]) and requires
        weight dtype to match the input. KDA keeps conv_weights in FP32
        while inputs/conv_states are BF16, so the cached FP32 transpose is
        cast to the caller's dtype here.
        """
        w = getattr(layer, "_conv_weights_t", None)
        if w is None:
            w = layer.conv_weights.transpose(0, 1).contiguous().to(dtype)
            layer._conv_weights_t = w
        return w

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        """Run KDA decode against the native channel-first Ascend cache."""
        assert isinstance(mixed_qkv, torch.Tensor)
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        # setting activation_mode to 1 means using SiLU activation after conv.
        qkv = torch.ops.npu.causal_conv1d(
            mixed_qkv.contiguous(),
            self._get_conv_weights_t(layer, mixed_qkv.dtype),
            conv_states=conv_states,
            bias=layer.bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )

        if self.kernel_dispatcher.supports_packed_decode:
            assert qkv.shape[0] == cache_indices.shape[0], (
                "KDA packed decode requires one token per sequence (T=1): "
                f"got {qkv.shape[0]} tokens for {cache_indices.shape[0]} requests."
            )
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
                lower_bound=layer.lower_bound,
                replayssm_d=layer_cache.replayssm_d,
                replayssm_k=layer_cache.replayssm_k,
                replayssm_g=layer_cache.replayssm_g,
                replayssm_write_pos=getattr(
                    self.forward_metadata, "replayssm_write_pos", None
                ),
                replayssm_force_flush=getattr(
                    self.forward_metadata, "replayssm_force_flush", None
                ),
            )
        else:
            q, k, v = qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
            q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
            k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
            v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)
            core_attn_out = self.kernel_dispatcher.decode(
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
                lower_bound=layer.lower_bound,
            )

        self._track_mamba_state_decode(
            forward_batch,
            conv_states,
            ssm_states,
            cache_indices,
            layer.layer_id,
        )
        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        """Run Ascend prefill without changing the shared KDA backend."""
        assert isinstance(mixed_qkv, torch.Tensor)
        if forward_batch.forward_mode.is_target_verify():
            return self._forward_target_verify(layer, forward_batch, mixed_qkv, a, b)

        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices
        cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = cache.conv[0]
        ssm_states = cache.temporal

        if forward_batch.extend_prefix_lens is None:
            raise RuntimeError(
                "extend_prefix_lens cannot be None in non-TARGET_VERIFY mode."
            )
        has_initial_state = forward_batch.extend_prefix_lens > 0

        if self.forward_metadata.has_mamba_track_mask:
            mixed_qkv_to_track = mixed_qkv[self.forward_metadata.track_conv_indices]
            conv_states[self.forward_metadata.conv_states_mask_indices] = (
                mixed_qkv_to_track
            )

        kernel_size = layer.conv_weights.shape[-1]
        conv_states_for_prefill = conv_states[:, -(kernel_size - 1) :, :].contiguous()
        mixed_qkv = torch.ops.npu.causal_conv1d(
            mixed_qkv.contiguous(),
            self._get_conv_weights_t(layer, mixed_qkv.dtype),
            conv_states=conv_states_for_prefill,
            bias=layer.bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            has_initial_state=has_initial_state,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=0,
        )
        conv_states[:, -(kernel_size - 1) :, :] = conv_states_for_prefill
        q, k, v = mixed_qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)
        g, beta, extend_A_log, extend_dt_bias = self._prepare_extend_gate_inputs(
            layer, a, b
        )
        track_ssm = self.forward_metadata.has_mamba_track_mask
        core_attn_out = self.kernel_dispatcher.extend(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            A_log=extend_A_log,
            dt_bias=extend_dt_bias,
            lower_bound=layer.lower_bound,
            extend_seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            is_spec_decode=forward_batch.forward_mode.is_draft_extend_v2(),
            return_intermediate_states=track_ssm,
            track_ssm_h_src=(
                self.forward_metadata.track_ssm_h_src if track_ssm else None
            ),
        )
        if track_ssm:
            core_attn_out, h = core_attn_out
            self._track_mamba_state_extend(
                forward_batch, h, ssm_states, self.forward_metadata
            )
        return core_attn_out

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
        """Apply the Ascend prefill gate contract.

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
        conv_states = cache.conv[0]
        num_accepted_tokens = torch.full(
            (batch_size,),
            draft_token_num,
            dtype=torch.int32,
            device=mixed_qkv.device,
        )
        dense_query_start_loc = torch.arange(
            0,
            num_dense_tokens + 1,
            step=draft_token_num,
            dtype=torch.int32,
            device=mixed_qkv.device,
        )
        processed_qkv = torch.ops.npu.causal_conv1d(
            dense_qkv.reshape(num_dense_tokens, -1).contiguous(),
            self._get_conv_weights_t(layer, mixed_qkv.dtype),
            conv_states=conv_states,
            bias=layer.bias,
            query_start_loc=dense_query_start_loc,
            cache_indices=cache_indices[:batch_size],
            num_accepted_tokens=num_accepted_tokens,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )
        q, k, v = processed_qkv.split([layer.q_dim, layer.k_dim, layer.v_dim], dim=-1)
        q = q.unflatten(-1, (-1, layer.head_q_dim)).unsqueeze(0)
        k = k.unflatten(-1, (-1, layer.head_k_dim)).unsqueeze(0)
        v = v.unflatten(-1, (-1, layer.head_v_dim)).unsqueeze(0)

        # Activate the forget gate and beta in FP32 before entering the
        # recurrent kernel to match the checkpoint's verify contract.
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


class AscendKDAHybridLinearAttnBackend:
    """KDA-specific hybrid backend with strided destination state mover.

    ``AscendHybridLinearAttnBackend`` uses ``move_intermediate_cache`` which
    assumes a contiguous destination layout. KDA's temporal SSM state on NPU
    is transposed (-1, -2) and requires the strided variant
    ``move_intermediate_cache_kda`` to preserve correct (V, K) indexing.

    This class overrides only ``update_mamba_state_after_mtp_verify`` to
    substitute the KDA-aware mover; the rest of the hybrid behaviour is
    inherited unchanged.
    """

    def __new__(cls, *args, **kwargs):
        # Delay importing AscendHybridLinearAttnBackend to avoid circular deps.
        from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
            AscendHybridLinearAttnBackend as _Base,
        )

        # Dynamically create a subclass of _Base with our override.
        class _AscendKDAHybrid(_Base):
            def update_mamba_state_after_mtp_verify(
                self,
                last_correct_step_indices,
                mamba_track_indices,
                mamba_steps_to_track,
                model,
                req_pool_indices=None,
            ):
                from sgl_kernel_npu.mamba.mamba_state_update_triton import (
                    conv_state_rollback,
                    move_intermediate_cache_kda,
                )
                from sgl_kernel_npu.mamba.speculative_state_scatter import (
                    speculative_state_scatter_npu,
                )

                del req_pool_indices
                request_number = last_correct_step_indices.shape[0]

                state_indices_tensor = (
                    self.linear_attn_backend.forward_metadata.mamba_cache_indices[
                        :request_number
                    ]
                )

                mamba_caches = (
                    self.linear_attn_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
                )

                conv_states = mamba_caches.conv[0]
                ssm_states = mamba_caches.temporal
                intermediate_state_cache = mamba_caches.intermediate_ssm
                dst_indices_tensor = state_indices_tensor.to(torch.int32)
                src_indices_tensor = torch.arange(
                    dst_indices_tensor.shape[0],
                    device=dst_indices_tensor.device,
                    dtype=torch.int32,
                )
                last_steps = last_correct_step_indices.to(torch.int32)

                move_intermediate_cache_kda(
                    ssm_states,
                    intermediate_state_cache,
                    dst_indices_tensor,
                    src_indices_tensor,
                    last_steps,
                    h_block_size=1,
                )
                draft_token_num = intermediate_state_cache.shape[2]
                has_conv_snapshots = getattr(
                    self.linear_attn_backend,
                    "supports_speculative_conv_state_snapshots",
                    False,
                )
                if has_conv_snapshots:
                    intermediate_conv_window_cache = (
                        mamba_caches.intermediate_conv_window[0]
                    )
                    speculative_state_scatter_npu(
                        conv_states,
                        intermediate_conv_window_cache,
                        dst_indices_tensor,
                        src_indices_tensor,
                        last_steps,
                    )
                if mamba_track_indices is not None:
                    assert mamba_steps_to_track is not None
                    mamba_track_indices = mamba_track_indices.to(torch.int32)
                    mamba_steps_to_track = mamba_steps_to_track.to(torch.int32)

                    move_intermediate_cache_kda(
                        ssm_states,
                        intermediate_state_cache,
                        mamba_track_indices,
                        src_indices_tensor,
                        mamba_steps_to_track,
                        h_block_size=1,
                    )

                    if has_conv_snapshots:
                        speculative_state_scatter_npu(
                            conv_states,
                            intermediate_conv_window_cache,
                            mamba_track_indices,
                            src_indices_tensor,
                            mamba_steps_to_track,
                        )
                    else:
                        track_mask = mamba_steps_to_track >= 0
                        src_slots = torch.where(
                            track_mask, dst_indices_tensor, mamba_track_indices
                        )
                        conv_states[:, mamba_track_indices] = conv_states[:, src_slots]

                if not has_conv_snapshots:
                    if dst_indices_tensor.numel() > 0:
                        conv_state_rollback(
                            conv_states,
                            dst_indices_tensor,
                            last_steps,
                            draft_token_num,
                        )

                    if (
                        mamba_track_indices is not None
                        and mamba_track_indices.numel() > 0
                    ):
                        conv_state_rollback(
                            conv_states,
                            mamba_track_indices,
                            mamba_steps_to_track,
                            draft_token_num,
                        )

                return

        return _AscendKDAHybrid(*args, **kwargs)
