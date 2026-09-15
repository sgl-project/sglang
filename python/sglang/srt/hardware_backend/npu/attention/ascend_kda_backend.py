import math
from typing import Optional

import torch

from sglang.srt.environ import envs

# from sgl_kernel_npu.fla.kda_chunk_delta_h import (
#     chunk_gated_delta_rule_fwd_h_npu,
# )
from sgl_kernel_npu.fla.kda_gate import fused_kda_gate_npu

_USE_TRITON_KDA = (
    getattr(envs, "SGLANG_NPU_KDA_PREFILL_BACKEND", None) is not None
    and envs.SGLANG_NPU_KDA_PREFILL_BACKEND.get() == "triton"
)
if _USE_TRITON_KDA:
    from sglang.srt.hardware_backend.npu.attention.triton_kda_prefill import (
        chunk_kda_fwd_npu,
    )
# from sgl_kernel_npu.fla.kda_prefill import (
#     chunk_gla_fwd_o_gk_npu,
#     recompute_w_u_fwd_npu,
# )
from sgl_kernel_npu.fla.kda_target_verify import kda_target_verify_npu
# from sgl_kernel_npu.fla.solve_tril import solve_tril_npu
# from sgl_kernel_npu.fla.utils import prepare_chunk_indices
from sgl_kernel_npu.mamba.causal_conv1d import (
    causal_conv1d_fn_npu,
    causal_conv1d_update_npu,
)
from sgl_kernel_npu.mamba.causal_conv1d_verify import (
    causal_conv1d_linear_verify_npu,
)
from sgl_kernel_npu.fla.solve_tril import solve_tril_npu
from sgl_kernel_npu.fla.utils import prepare_chunk_indices

# from cann_ops_transformer.ops import chunk_kda_fwd
# from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
# from sglang.kernels.ops.attention.fla.kda import chunk_kda_scaled_dot_kkt_fwd
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.kernels.ops.attention.fla.fused_recurrent_linear_replayssm import (
    fused_recurrent_linear_replayssm_decode,
)
from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
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
        v = v.contiguous()
        g = g.contiguous()
        beta = beta.contiguous()
        # chunk_kda_fwd accepts one initial state per logical sequence, while
        # SGLang owns a slot-indexed persistent pool. Gather the active slots in
        # canonical contiguous [N, H, V, K] layout and scatter final_state back.
        num_sequences = query_start_loc.shape[0] - 1
        source_indices = cache_indices[:num_sequences].to(torch.long)
        valid_state_mask = source_indices >= 0
        # Forward metadata may use -1 for a padded request.  index_select would
        # otherwise read the last cache slot and index_copy_ would overwrite it.
        # Slot 0 is a gather placeholder for that padded row; its computed result
        # is irrelevant because padded rows are filtered before state writeback.
        gather_indices = source_indices.clamp_min(0)
        initial_state = (
            ssm_states.index_select(0, gather_indices)
            .to(dtype=torch.float32)
            .contiguous()
        )
        scale = k.shape[-1] ** -0.5
        query_start_loc = (
            query_start_loc
            .to(dtype=torch.int64)
            .contiguous()
        )

        if _USE_TRITON_KDA:
            out, final_state, chunk_states = self._extend_triton(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                query_start_loc=query_start_loc,
                chunk_size=chunk_size,
                return_intermediate_states=return_intermediate_states,
                **kwargs,
            )
        else:
            out, final_state, chunk_states = self._extend_cann(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                query_start_loc=query_start_loc,
                chunk_size=chunk_size,
                return_intermediate_states=return_intermediate_states,
            )

        valid_positions = valid_state_mask.nonzero(as_tuple=False).flatten()
        ssm_states.index_copy_(
            0,
            source_indices.index_select(0, valid_positions),
            final_state.index_select(0, valid_positions).to(dtype=ssm_states.dtype),
        )

        if return_intermediate_states:
            return out, chunk_states
        return out

    @staticmethod
    def _extend_cann(
        *,
        q, k, v, g, beta,
        scale, initial_state, query_start_loc, chunk_size,
        return_intermediate_states,
    ):
        q = l2norm_fwd(q.contiguous())
        k = l2norm_fwd(k.contiguous())
        outputs = torch.ops.npu.chunk_kda_fwd(
            q, k, v, g, beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=query_start_loc,
            chunk_size=chunk_size,
            layout="BSND",
            safe_gate=True,
            use_gate_in_kernel=False,
            state_v_first=True,
            output_h=return_intermediate_states,
        )
        return outputs[0], outputs[1], outputs[10]

    @staticmethod
    def _extend_triton(
        *,
        q, k, v, g, beta,
        scale, initial_state, query_start_loc, chunk_size,
        return_intermediate_states,
        **kwargs,
    ):
        A_log = kwargs.get("A_log")
        dt_bias = kwargs.get("dt_bias")
        lower_bound = kwargs.get("lower_bound", -5.0)
        chunk_indices = kwargs.get("chunk_indices")
        with torch.inference_mode():
            o, final_state, h = chunk_kda_fwd_npu(
                q=q.contiguous(),
                k=k.contiguous(),
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=query_start_loc,
                chunk_indices=chunk_indices,
                chunk_size=chunk_size,
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=lower_bound,
                state_v_first=True,
                A_log=A_log,
                dt_bias=dt_bias,
                return_intermediate_states=return_intermediate_states,
            )
        return o, final_state, h


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
            replayssm_d = layer_cache.replayssm_d
            replayssm_k = layer_cache.replayssm_k
            replayssm_g = layer_cache.replayssm_g
            replayssm_write_pos = getattr(
                self.forward_metadata, "replayssm_write_pos", None
            )
            replayssm_force_flush = getattr(
                self.forward_metadata, "replayssm_force_flush", None
            )
            if (
                replayssm_d is not None
                and replayssm_k is not None
                and replayssm_g is not None
                and replayssm_write_pos is not None
            ):
                if layer.lower_bound is not None:
                    raise NotImplementedError(
                        "KDA safe gate (lower_bound) is not implemented in the "
                        "ReplaySSM decode kernel; disable --enable-linear-replayssm."
                    )
                B = qkv.shape[0]
                K = ssm_states.shape[-1]
                out = qkv.new_empty(B, 1, layer.num_v_heads, layer.head_v_dim)
                fused_recurrent_linear_replayssm_decode(
                    mixed_qkv=qkv,
                    a=a.reshape(B, layer.num_v_heads, K).contiguous(),
                    b=b.reshape(B, layer.num_v_heads).contiguous(),
                    A_log=layer.A_log.reshape(-1),
                    dt_bias=layer.dt_bias.reshape(layer.num_v_heads, K).contiguous(),
                    scale=layer.head_k_dim**-0.5,
                    initial_state=ssm_states,
                    d_cache=replayssm_d,
                    k_cache=replayssm_k,
                    g_cache=replayssm_g,
                    out=out,
                    ssm_state_indices=cache_indices,
                    write_pos=replayssm_write_pos,
                    force_flush=replayssm_force_flush,
                    use_qk_l2norm_in_kernel=True,
                    is_kda=True,
                )
                core_attn_out = out.transpose(0, 1)
            else:
                q, k, v = qkv.split(
                    [layer.q_dim, layer.k_dim, layer.v_dim], dim=-1
                )
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
            chunk_indices=self.forward_metadata.kda_chunk_indices,
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

        CANN path: the checkpoint was validated with FP32 gate activation
        before ``chunk_kda``, so the gate is pre-activated here.

        Triton path: the optimized kernel fuses gate activation, l2norm,
        and beta sigmoid in-kernel, so we pass through the raw gate and
        hand off ``A_log`` / ``dt_bias`` for in-kernel activation.
        """
        if _USE_TRITON_KDA:
            return g, beta, layer.A_log, layer.dt_bias
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
        """Run fixed-width DSpark verify with Ascend-native state snapshots.

        When ReplaySSM spec-verify is enabled (replayssm_spec_fold), the per-draft
        full-state snapshots are replaced by a ring-writing Triton verify kernel
        (fused_sigmoid_gating_delta_rule_update with cache_ring=True). The commit
        fold replays the accepted prefix into the fp32 checkpoint.
        """
        metadata = self.forward_metadata
        seq_len = mixed_qkv.shape[0]
        query_start_loc = metadata.query_start_loc
        cache_indices = metadata.mamba_cache_indices

        cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)

        mamba_pool = self.req_to_token_pool.mamba_pool
        replayssm_spec_fold = getattr(mamba_pool, "replayssm_spec_fold", False)

        intermediate_state = cache.intermediate_ssm
        if intermediate_state is None and not replayssm_spec_fold:
            raise RuntimeError(
                "Ascend KDA target verify requires speculative Mamba scratch "
                "(or --enable-linear-replayssm-spec)."
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

        if replayssm_spec_fold:
            dense_cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * draft_token_num,
                draft_token_num,
                device=q.device,
                dtype=torch.int32,
            )
            out = fused_sigmoid_gating_delta_rule_update(
                A_log=layer.A_log,
                a=dense_a,
                dt_bias=layer.dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=q,
                k=k,
                v=v,
                b=dense_b,
                initial_state_source=cache.temporal,
                initial_state_indices=cache_indices[:batch_size],
                cu_seqlens=dense_cu_seqlens,
                use_qk_l2norm_in_kernel=True,
                is_kda=True,
                lower_bound=layer.lower_bound,
                disable_state_update=True,
                cache_ring=True,
                replayssm_rawv=cache.replayssm_rawv,
                replayssm_rawk=cache.replayssm_rawk,
                replayssm_g=cache.replayssm_g,
                replayssm_beta=cache.replayssm_beta,
                num_warps=4,
            )
            out = out[0] if isinstance(out, tuple) else out
            if dense_token_indices is None:
                return out
            padded_out = out.new_zeros(
                1, num_dense_tokens + 1, *out.shape[2:]
            )
            padded_out[:, :num_dense_tokens] = out
            return padded_out[:, dense_token_indices]

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

                mamba_pool = self.linear_attn_backend.req_to_token_pool.mamba_pool
                replayssm_spec_fold = getattr(
                    mamba_pool, "replayssm_spec_fold", False
                )

                conv_states = mamba_caches.conv[0]
                ssm_states = mamba_caches.temporal
                dst_indices_tensor = state_indices_tensor.to(torch.int32)
                src_indices_tensor = torch.arange(
                    dst_indices_tensor.shape[0],
                    device=dst_indices_tensor.device,
                    dtype=torch.int32,
                )
                last_steps = last_correct_step_indices.to(torch.int32)

                if replayssm_spec_fold:
                    from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (
                        commit_kda_replayssm_spec_all_layers,
                    )

                    L = mamba_caches.replayssm_rawv.shape[-2]
                    num_k_heads = mamba_caches.replayssm_rawk.shape[2]
                    accept_lens = (last_steps + 1).to(torch.int32)

                    track_idx = (
                        mamba_track_indices.to(torch.int32)
                        if mamba_track_indices is not None
                        else None
                    )
                    track_steps = (
                        mamba_steps_to_track.to(torch.int32)
                        if mamba_steps_to_track is not None
                        else None
                    )

                    commit_kda_replayssm_spec_all_layers(
                        checkpoint_state=ssm_states,
                        rawv_cache=mamba_caches.replayssm_rawv,
                        rawk_cache=mamba_caches.replayssm_rawk,
                        gk_cache=mamba_caches.replayssm_g,
                        beta_cache=mamba_caches.replayssm_beta,
                        ssm_state_indices=dst_indices_tensor,
                        accept_lens=accept_lens,
                        max_cache_len=L,
                        num_k_heads=num_k_heads,
                        mamba_track_indices=track_idx,
                        mamba_steps_to_track=track_steps,
                    )

                    draft_token_num = L
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
                            mamba_track_indices.to(torch.int32),
                            mamba_steps_to_track.to(torch.int32),
                            draft_token_num,
                        )
                    return

                intermediate_state_cache = mamba_caches.intermediate_ssm

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
                        # No-op self-copy for non-tracked entries so we never run
                        # bool-mask indexing (aten::nonzero) or a host numel check.
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
