from typing import Optional, Tuple, Union

import torch
from sgl_kernel_npu.fla.fused_gdn_gating import (
    fused_gdn_gating_kernel_without_sigmoid,
    fused_gdn_gating_npu,
)
from sgl_kernel_npu.fla.l2norm import l2norm_fwd
from sgl_kernel_npu.fla.utils import prepare_chunk_indices

from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
    AscendMambaAttnBackendBase,
)
from sglang.srt.layers.attention.linear.gdn_backend import GDNKernelDispatcher
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput

fused_gdn_gating = fused_gdn_gating_npu


class AscendGDNAttnBackend(AscendMambaAttnBackendBase):
    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = torch.Size(
            (
                *model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape[
                    :-2
                ],
                model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape[-1],
                model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape[-2],
            )
        )
        backends = model_runner.linear_attn_backends
        decode_backend = backends.decode
        prefill_backend = backends.prefill
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)
        # Scratch buffers for the non-persisting (guiding-tail) extend pass.
        self._tail_conv_cache_indices = torch.zeros(
            1, dtype=torch.int32, device=model_runner.device
        )
        self._tail_has_initial_state = torch.ones(
            1, dtype=torch.bool, device=model_runner.device
        )

    def _prepare_mamba_track_metadata(self, forward_batch: ForwardBatch):
        if self.forward_metadata.has_mamba_track_mask:
            mamba_track_mask_indices = forward_batch.mamba_track_mask.nonzero(
                as_tuple=True
            )[0]
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[mamba_track_mask_indices]
            )

    def prepare_gdn_inputs(
        self,
        bs: int,
        forward_mode: ForwardMode,
        spec_info: Optional[Union[EagleDraftInput, EagleVerifyInput]],
    ):
        cache_indices = self.forward_metadata.mamba_cache_indices
        self.num_accept_tokens = torch.ones(
            [bs], dtype=torch.int32, device=cache_indices.device
        )
        self.actual_seq_lengths = torch.ones(
            [bs], dtype=torch.int32, device=cache_indices.device
        )
        if forward_mode.is_target_verify():
            seq_len = spec_info.draft_token_num
            self.actual_seq_lengths = self.actual_seq_lengths * seq_len
            # indices
            self.ssm_state_indices = torch.arange(
                cache_indices.shape[0] * seq_len,
                dtype=torch.int32,
                device=cache_indices.device,
            )
        else:
            self.ssm_state_indices = cache_indices
        if forward_mode.is_prefill():
            self.actual_seq_lengths = torch.diff(
                self.forward_metadata.query_start_loc
            ).to(torch.int32)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        if forward_batch.forward_mode.is_draft_extend_v2():
            return
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        self.prepare_gdn_inputs(
            forward_batch.batch_size,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )
        self._prepare_mamba_track_metadata(forward_batch)
        self.graph_mode = True

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if forward_batch.forward_mode.is_draft_extend_v2():
            return
        super().init_forward_metadata(forward_batch)
        self.prepare_gdn_inputs(
            forward_batch.batch_size,
            forward_batch.forward_mode,
            forward_batch.spec_info,
        )
        self._prepare_mamba_track_metadata(forward_batch)
        self.graph_mode = False

    def _get_conv_weights_t(self, layer: RadixLinearAttention) -> torch.Tensor:
        w = getattr(layer, "_conv_weights_t", None)
        if w is None:
            w = layer.conv_weights.transpose(0, 1).contiguous()
            layer._conv_weights_t = w
        return w

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
        mixed_qkv = torch.ops.npu.causal_conv1d(
            mixed_qkv,
            self._get_conv_weights_t(layer),
            conv_states=conv_states,
            bias=layer.bias,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            activation_mode=1,
            pad_slot_id=-1,
            run_mode=1,
        )

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
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
        """Route the GDN extend scan. Batches without a guiding tail run a
        single pass; chunked-prefill KV-cache-pruning batches carry a guiding
        tail and run two passes, where only the main pass persists mamba
        state (the tail tokens are computed but never cached).

        The split is decided here, from the extend batch itself, so that no
        other forward path (decode, graph replay, verify) has to know about
        the tail."""
        if (
            forward_batch.uncached_suffix_len <= 0
            or not forward_batch.forward_mode.is_prefill()
            or forward_batch.forward_mode.is_target_verify()
        ):
            return self._forward_extend_internal(
                layer, forward_batch, mixed_qkv, a, b, **kwargs
            )

        if self.forward_metadata.has_mamba_track_mask:
            # Track indices refer to the unsplit batch, which the two-pass
            # split no longer is.
            raise RuntimeError(
                "mamba track mask is not supported together with the "
                "uncached suffix (guiding tail) extend"
            )
        if forward_batch.batch_size != 1:
            raise RuntimeError(
                "uncached suffix (guiding tail) requires batch_size == 1, "
                f"got {forward_batch.batch_size}"
            )
        tail_len = forward_batch.uncached_suffix_len
        seq_len = mixed_qkv.shape[0]
        main_len = seq_len - tail_len
        if main_len <= 0:
            raise RuntimeError(
                f"uncached suffix ({tail_len}) must be shorter than the "
                f"extend batch ({seq_len})"
            )
        query_start_loc = self.forward_metadata.query_start_loc
        qsl_main = torch.tensor(
            [0, main_len], dtype=query_start_loc.dtype, device=query_start_loc.device
        )
        qsl_tail = torch.tensor(
            [0, tail_len], dtype=query_start_loc.dtype, device=query_start_loc.device
        )

        core_attn_out_main = self._forward_extend_internal(
            layer,
            forward_batch,
            mixed_qkv[:main_len],
            a[:main_len],
            b[:main_len],
            query_start_loc=qsl_main,
        )
        # The tail continues from the state the main pass just persisted
        # into the pools, but leaves the pools untouched.
        core_attn_out_tail = self._forward_extend_internal(
            layer,
            forward_batch,
            mixed_qkv[main_len:],
            a[main_len:],
            b[main_len:],
            query_start_loc=qsl_tail,
            persist_state=False,
        )
        return torch.cat([core_attn_out_main, core_attn_out_tail], dim=0)

    def _forward_extend_internal(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        query_start_loc: Optional[torch.Tensor] = None,
        persist_state: bool = True,
        **kwargs,
    ):
        """The GDN extend scan (causal conv + chunked gated delta rule).
        By default it uses the forward-metadata/pool state and persists the
        resulting mamba state. With ``persist_state=False`` (the guiding-tail
        pass) it still reads the current pool state but leaves the pools
        untouched: causal_conv1d updates its state buffer in place (and in
        the plain NPU pool layout that buffer is the pool itself), so that
        pass runs the conv on a scratch copy of the used conv row(s) and
        skips the SSM scatter.
        """
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]
        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        if query_start_loc is None:
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
        elif not persist_state:
            # Non-persisting (guiding-tail) pass: the conv scratch below
            # always starts from the state the previous pass persisted.
            has_initial_states = self._tail_has_initial_state
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0
        if is_target_verify:
            num_token_padding = mixed_qkv.shape[0]
            if (
                not self.graph_mode
                and forward_batch.global_num_token_non_padded_cpu != num_token_padding
            ):
                mixed_qkv = mixed_qkv[: forward_batch.global_num_token_non_padded_cpu]
                a = a[: forward_batch.global_num_token_non_padded_cpu]
                b = b[: forward_batch.global_num_token_non_padded_cpu]
                seq_len = forward_batch.global_num_token_non_padded_cpu

            batch_size = cache_indices.shape[0]
            draft_token_num = forward_batch.spec_info.draft_token_num
            num_accepted_tokens = torch.full(
                (batch_size,),
                draft_token_num,
                dtype=torch.int32,
                device=mixed_qkv.device,
            )
            mixed_qkv = torch.ops.npu.causal_conv1d(
                mixed_qkv,
                self._get_conv_weights_t(layer),
                conv_states=conv_states,
                bias=layer.bias,
                query_start_loc=query_start_loc,
                cache_indices=cache_indices,
                num_accepted_tokens=num_accepted_tokens,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=1,
            )
        else:
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[forward_metadata.track_conv_indices]
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )
            kernel_size = layer.conv_weights.shape[-1]
            if persist_state:
                conv_states_for_prefill = conv_states[
                    :, -(kernel_size - 1) :, :
                ].contiguous()
                conv_cache_indices = cache_indices
            else:
                # Shield the pools from the in-place conv state update: run
                # on a scratch copy of the used conv row(s) only.
                conv_states_for_prefill = conv_states[
                    cache_indices, -(kernel_size - 1) :, :
                ].clone()
                conv_cache_indices = self._tail_conv_cache_indices
            mixed_qkv = torch.ops.npu.causal_conv1d(
                mixed_qkv,
                self._get_conv_weights_t(layer),
                conv_states=conv_states_for_prefill,
                bias=layer.bias,
                query_start_loc=query_start_loc,
                cache_indices=conv_cache_indices,
                has_initial_state=has_initial_states,
                activation_mode=1,
                pad_slot_id=-1,
                run_mode=0,
            )
            if persist_state:
                conv_states[:, -(kernel_size - 1) :, :] = conv_states_for_prefill
        if is_target_verify:
            g, beta = fused_gdn_gating_kernel_without_sigmoid(
                layer.A_log, a, b, layer.dt_bias
            )
            beta = beta.unsqueeze(0)
            num_heads, head_k_dim = layer.num_q_heads, layer.head_q_dim
            num_value_heads, head_v_dim = layer.num_v_heads, layer.head_v_dim

            mixed_qkv_last_dim = mixed_qkv.shape[-1]

            mixed_qkv = mixed_qkv.view(batch_size, -1, mixed_qkv_last_dim)
            beta = beta.view(batch_size, -1, num_value_heads)
            g = g.view(batch_size, -1, num_value_heads)

            core_attn_out = self.fused_recurrent_gated_delta_rule_update(
                mixed_qkv,
                num_heads,
                num_value_heads,
                head_k_dim,
                head_v_dim,
                recurrent_state=ssm_states,
                beta=beta,
                g=g,
                cache_indices=cache_indices,
                intermediate_state=intermediate_state_cache,
            )
            core_attn_out = core_attn_out.view(-1, num_value_heads, head_v_dim)
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
            query, key, value = torch.split(
                mixed_qkv,
                [layer.q_dim, layer.k_dim, layer.v_dim],
                dim=-1,
            )

            actual_seq_len = query.shape[0]
            query = query.view(
                actual_seq_len, layer.num_q_heads, layer.head_q_dim
            ).contiguous()
            key = key.view(
                actual_seq_len, layer.num_k_heads, layer.head_k_dim
            ).contiguous()
            value = value.view(
                actual_seq_len, layer.num_v_heads, layer.head_v_dim
            ).contiguous()
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

            core_attn_out, last_recurrent_state, h = self.chunk_gdn(
                query, key, value, g, beta, ssm_states[cache_indices], query_start_loc
            )

            if persist_state:
                if last_recurrent_state is not None:
                    last_recurrent_state = last_recurrent_state.to(
                        ssm_states.dtype, copy=False
                    )
                    ssm_states[cache_indices] = last_recurrent_state
                if h is not None:
                    self._track_mamba_state_extend(
                        forward_batch, h, ssm_states, forward_metadata
                    )

        return core_attn_out

    def chunk_gdn(
        self,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens,
        actual_seq_lengths: Optional[torch.Tensor] = None,
    ):
        q = l2norm_fwd(q)  # [T, Nk, Dk]
        k = l2norm_fwd(k)  # [T, Nk, Dk]
        g = g.to(torch.float32)  # [T, Nv]
        scale = k.shape[-1] ** -0.5

        chunk_size = 64
        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, chunk_size)
            if cu_seqlens is not None
            else None
        )
        h = k.new_empty(
            len(chunk_indices), v.shape[-2], v.shape[-1], k.shape[-1]
        )  # [NT, NH, DV, Dk]

        if actual_seq_lengths is None:
            actual_seq_lengths = torch.diff(cu_seqlens).to(torch.int32)

        core_attn_out, last_recurrent_state = torch.ops.npu.chunk_gated_delta_rule(
            q,
            k,
            v,
            beta=beta,
            initial_state=initial_state,
            actual_seq_lengths=actual_seq_lengths,
            scale=scale,
            chunk_state=h,
            g=g,
        )
        return core_attn_out, last_recurrent_state, h

    def fused_recurrent_gated_delta_rule_update(
        self,
        mix_qkv: torch.Tensor,
        num_heads,
        num_value_heads,
        head_k_dim,
        head_v_dim,
        recurrent_state: torch.Tensor,
        beta: torch.Tensor,
        g: torch.Tensor,
        cache_indices: torch.Tensor,
        intermediate_state: Optional[torch.Tensor] = None,
    ):
        beta = beta.to(torch.bfloat16)
        g = g.to(torch.float32)
        batch_size = mix_qkv.shape[0]
        seq_len = mix_qkv.shape[1]
        scale = 1 / (head_k_dim**0.5)

        if intermediate_state is not None:
            intermediate_state = intermediate_state.view(
                -1, num_value_heads, head_k_dim, head_v_dim
            )

        if self.graph_mode:
            num_accept_tokens = torch.full(
                [batch_size], 1, dtype=torch.int32, device=cache_indices.device
            )
            actual_seq_lengths = torch.full(
                [batch_size], seq_len, dtype=torch.int32, device=cache_indices.device
            )
            ssm_state_indices = self.forward_metadata.mamba_cache_indices_gdn
        else:
            num_accept_tokens = self.num_accept_tokens
            actual_seq_lengths = self.actual_seq_lengths
            ssm_state_indices = self.ssm_state_indices

        attn_core_out = torch.ops.npu.recurrent_gated_delta_rule(
            mix_qkv,
            recurrent_state,
            beta=beta,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices.view(batch_size, seq_len),
            nk=num_heads,
            nv=num_value_heads,
            intermediate_state=intermediate_state,
            cache_indices=cache_indices,
            num_accepted_tokens=num_accept_tokens,
            g=g,
        )

        if intermediate_state is not None:
            intermediate_state = intermediate_state.view(
                -1, seq_len, num_value_heads, head_k_dim, head_v_dim
            )
        return attn_core_out
