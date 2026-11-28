from typing import Optional, Tuple, Union

import torch
from sgl_kernel_npu.operators.attention.gdn import (
    chunk_gated_delta_rule,
    fused_gdn_gating,
    fused_gdn_gating_without_sigmoid,
    recurrent_gated_delta_rule,
)
from sgl_kernel_npu.operators.attention.gdn.gdn_utils import prepare_chunk_indices
from sgl_kernel_npu.operators.mamba import causal_conv1d
from sgl_kernel_npu.operators.norm import l2norm_fwd

try:
    from sgl_kernel_npu.operators.attention.gdn import (
        build_gdn_prefill_chunk_meta,
        fused_conv1d_decode_gating,
        fused_gdn_prefill,
        fused_rmsnorm_gated_strided_z,
    )
except ImportError:
    build_gdn_prefill_chunk_meta = None
    fused_conv1d_decode_gating = None
    fused_gdn_prefill = None
    fused_rmsnorm_gated_strided_z = None

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.ascend_hybrid_linear_attn_backend import (
    AscendMambaAttnBackendBase,
)
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.layers.attention.linear.gdn_backend import GDNKernelDispatcher
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput


def can_use_fused_gdn_prefill_950pr(
    layer,
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    ssm_states: torch.Tensor,
    chunk_meta,
) -> bool:
    if (
        fused_gdn_prefill is None
        or not envs.SGLANG_NPU_ENABLE_FUSED_GDN_PREFILL.get()
        or not is_npu_arch35()
        or chunk_meta is None
    ):
        return False
    nk = layer.num_k_heads
    nv = layer.num_v_heads
    dk = layer.head_k_dim
    return (
        dk == 128
        and layer.head_q_dim == 128
        and layer.head_v_dim == 128
        and layer.num_q_heads == nk
        and nv > 0
        and nk > 0
        and nv % nk == 0
        and (nk & (nk - 1)) == 0
        and mixed_qkv.dtype == torch.bfloat16
        and mixed_qkv.is_contiguous()
        and a.is_contiguous()
        and b.is_contiguous()
        and ssm_states.is_contiguous()
        and layer.q_dim == nk * 128
        and layer.k_dim == nk * 128
        and layer.v_dim == nv * 128
        and mixed_qkv.shape[-1] == layer.q_dim + layer.k_dim + layer.v_dim
    )


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
        # Persistent per-bs static ones buffers safe for NPU decode graph capture/replay.
        self._decode_ones_cache: dict[int, torch.Tensor] = {}

    def _prepare_mamba_track_metadata(self, forward_batch: ForwardBatch):
        if self.forward_metadata.has_mamba_track_mask:
            mamba_track_mask_indices = forward_batch.mamba_track_mask.nonzero(
                as_tuple=True
            )[0]
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[mamba_track_mask_indices]
            )
        self._gdn_2piece_spec = None
        self._gdn_950pr_chunk_meta = None
        self._gdn_950pr_ckpt_idx = None
        self._gdn_950pr_ckpt_dst = None
        if (
            build_gdn_prefill_chunk_meta is not None
            and forward_batch.forward_mode.is_prefill()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            if seq_lens_cpu is not None and len(seq_lens_cpu) > 0:
                dev = self.actual_seq_lengths.device
                self._gdn_950pr_chunk_meta = build_gdn_prefill_chunk_meta(
                    seq_lens_cpu, dev, chunk_size=self.mamba_chunk_size
                )
                if (
                    self.forward_metadata.has_mamba_track_mask
                    and self.forward_metadata.track_chunk_idx is not None
                    and self.forward_metadata.mamba_track_indices is not None
                    and self.forward_metadata.track_ssm_h_src is not None
                    and self.forward_metadata.track_ssm_h_src.numel() > 0
                ):
                    self._gdn_950pr_ckpt_idx = self.forward_metadata.track_chunk_idx.to(
                        torch.int32
                    )
                    self._gdn_950pr_ckpt_dst = (
                        self.forward_metadata.mamba_track_indices.to(torch.int32)
                    )
        if (
            forward_batch.forward_mode.is_prefill()
            and not forward_batch.forward_mode.is_target_verify()
            and self.forward_metadata.has_mamba_track_mask
            and self.forward_metadata.track_ssm_h_src is not None
            and self.forward_metadata.track_ssm_h_src.numel() > 0
            and self.forward_metadata.track_ssm_h_batch_src is not None
            and hasattr(torch.ops.npu, "npu_chunk_gated_delta_rule")
        ):
            seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            track_mask_cpu = getattr(
                forward_batch, "mamba_prefill_track_mask_cpu", None
            )
            track_seqlens_cpu = getattr(forward_batch, "mamba_track_seqlens_cpu", None)
            prefix_lens_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
            if (
                seq_lens_cpu is not None
                and track_mask_cpu is not None
                and track_seqlens_cpu is not None
                and prefix_lens_cpu is not None
                and len(seq_lens_cpu) > 0
                and len(seq_lens_cpu) == self.forward_metadata.track_ssm_h_dst.numel()
            ):
                bs = len(seq_lens_cpu)
                L0 = int(seq_lens_cpu[0])
                chunk_size = self.mamba_chunk_size
                if all(track_mask_cpu) and all(int(x) == L0 for x in seq_lens_cpu):
                    deltas = [
                        int(t) - int(p)
                        for t, p in zip(track_seqlens_cpu, prefix_lens_cpu)
                    ]
                    cuts = [(d // chunk_size) * chunk_size for d in deltas]
                    cut0 = cuts[0]
                    if (
                        0 < cut0 < L0
                        and all(d % chunk_size != 0 for d in deltas)
                        and all(c == cut0 for c in cuts)
                    ):
                        dev = self.actual_seq_lengths.device
                        lens1 = torch.full((bs,), cut0, dtype=torch.int32, device=dev)
                        lens2 = torch.full(
                            (bs,), L0 - cut0, dtype=torch.int32, device=dev
                        )
                        self._gdn_2piece_spec = (bs, L0, cut0, lens1, lens2)

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
            self._cached_chunk_indices = None
        else:
            self._cached_chunk_indices = None
            ones_cache = getattr(self, "_decode_ones_cache", None)
            if ones_cache is None:
                ones_cache = {}
                self._decode_ones_cache = ones_cache
            if bs not in ones_cache or ones_cache[bs].device != cache_indices.device:
                ones_cache[bs] = torch.ones(
                    (bs,), dtype=torch.int32, device=cache_indices.device
                )

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
        layer_cache = self.state_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices
        bs = forward_batch.batch_size

        if isinstance(mixed_qkv, tuple):
            qkvz, ba = mixed_qkv
            qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
            nv = layer.num_v_heads
            if (
                fused_conv1d_decode_gating is not None
                and layer.bias is None
                and layer.conv_weights.shape[-1] == 4
                and conv_states.shape[1] == 3
                and conv_states.is_contiguous()
                and qkvz.is_contiguous()
                and ba.is_contiguous()
                and qkv_dim % 512 == 0
            ):
                mixed_out, g_out, beta_out = fused_conv1d_decode_gating(
                    qkvz,
                    ba,
                    self._get_conv_weights_t(layer),
                    conv_states,
                    cache_indices,
                    layer.A_log,
                    layer.dt_bias,
                    nv,
                )
                ones_cache = getattr(self, "_decode_ones_cache", None)
                if ones_cache is None:
                    ones_cache = {}
                    self._decode_ones_cache = ones_cache
                ones = ones_cache.get(bs)
                if ones is None or ones.device != cache_indices.device:
                    ones = torch.ones(
                        (bs,), dtype=torch.int32, device=cache_indices.device
                    )
                    ones_cache[bs] = ones
                scale = layer.head_k_dim**-0.5
                core_attn_out = recurrent_gated_delta_rule(
                    mixed_out.view(bs, 1, qkv_dim),
                    ssm_states,
                    beta=beta_out.view(bs, 1, nv),
                    scale=scale,
                    actual_seq_lengths=ones,
                    ssm_state_indices=cache_indices.view(bs, 1),
                    nk=layer.num_q_heads,
                    nv=nv,
                    cache_indices=cache_indices,
                    num_accepted_tokens=ones,
                    g=g_out.view(bs, 1, nv),
                )
                self._track_mamba_state_decode(
                    forward_batch,
                    conv_states,
                    ssm_states,
                    cache_indices,
                    layer_id=layer.layer_id,
                )
                return core_attn_out
            # Fallback if shapes don't match fused kernel requirements
            mixed_qkv = qkvz[:, :qkv_dim].contiguous()
            b = ba[:, :nv].contiguous()
            a = ba[:, nv:].contiguous()

        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = causal_conv1d(
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
            forward_batch,
            conv_states,
            ssm_states,
            cache_indices,
            layer_id=layer.layer_id,
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

        mamba_cache_params = self.state_layer_cache(layer.layer_id)
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
            mixed_qkv = causal_conv1d(
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
            if conv_states.shape[1] == kernel_size - 1 and conv_states.is_contiguous():
                # In-place update avoids a 30MB HBM read+writeback copy per layer.
                mixed_qkv = causal_conv1d(
                    mixed_qkv,
                    self._get_conv_weights_t(layer),
                    conv_states=conv_states,
                    bias=layer.bias,
                    query_start_loc=query_start_loc,
                    cache_indices=cache_indices,
                    has_initial_state=has_initial_states,
                    activation_mode=1,
                    pad_slot_id=-1,
                    run_mode=0,
                )
            else:
                conv_states_for_prefill = conv_states[
                    :, -(kernel_size - 1) :, :
                ].contiguous()
                mixed_qkv = causal_conv1d(
                    mixed_qkv,
                    self._get_conv_weights_t(layer),
                    conv_states=conv_states_for_prefill,
                    bias=layer.bias,
                    query_start_loc=query_start_loc,
                    cache_indices=cache_indices,
                    has_initial_state=has_initial_states,
                    activation_mode=1,
                    pad_slot_id=-1,
                    run_mode=0,
                )
                conv_states[:, -(kernel_size - 1) :, :] = conv_states_for_prefill
        if is_target_verify:
            g, beta = fused_gdn_gating_without_sigmoid(layer.A_log, a, b, layer.dt_bias)
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
            chunk_meta = getattr(self, "_gdn_950pr_chunk_meta", None)
            if can_use_fused_gdn_prefill_950pr(
                layer, mixed_qkv, a, b, ssm_states, chunk_meta
            ):
                chunk_indices, chunk_offsets, _, total_valid = chunk_meta
                core_attn_out = fused_gdn_prefill(
                    mixed_qkv,
                    a,
                    b,
                    layer.A_log,
                    layer.dt_bias,
                    ssm_states,
                    cache_indices,
                    query_start_loc,
                    chunk_indices,
                    chunk_offsets,
                    layer.num_k_heads,
                    layer.num_v_heads,
                    total_valid_tokens=total_valid,
                    ckpt_chunk_idx=getattr(self, "_gdn_950pr_ckpt_idx", None),
                    ckpt_dst_slots=getattr(self, "_gdn_950pr_ckpt_dst", None),
                )
                if (
                    forward_metadata.has_mamba_track_mask
                    and forward_metadata.track_ssm_final_src is not None
                    and forward_metadata.track_ssm_final_src.numel() > 0
                ):
                    ssm_states[forward_metadata.track_ssm_final_dst] = ssm_states[
                        forward_metadata.track_ssm_final_src
                    ]
                return core_attn_out

            needs_chunk_state = (
                forward_metadata.has_mamba_track_mask
                and forward_metadata.track_ssm_h_src is not None
                and forward_metadata.track_ssm_h_src.numel() > 0
            )
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

            core_attn_out, last_recurrent_state, h, h_track_buf = self.chunk_gdn(
                query,
                key,
                value,
                g,
                beta,
                ssm_states[cache_indices],
                query_start_loc,
                needs_chunk_state=needs_chunk_state,
            )

            if last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state
            if forward_metadata.has_mamba_track_mask:
                self._track_mamba_state_extend(
                    forward_batch,
                    h,
                    ssm_states,
                    forward_metadata,
                    h_track_buf=h_track_buf,
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
        *,
        qk_already_normed: bool = False,
        needs_chunk_state: bool = True,
    ):
        if not qk_already_normed:
            q = l2norm_fwd(q)  # [T, Nk, Dk]
            k = l2norm_fwd(k)  # [T, Nk, Dk]
        g = g.to(torch.float32)  # [T, Nv]
        scale = k.shape[-1] ** -0.5

        if not needs_chunk_state and hasattr(
            torch.ops.npu, "npu_chunk_gated_delta_rule"
        ):
            core_attn_out, last_recurrent_state = (
                torch.ops.npu.npu_chunk_gated_delta_rule(
                    q,
                    k,
                    v,
                    beta=beta,
                    initial_state=initial_state,
                    actual_seq_lengths=self.actual_seq_lengths,
                    scale=scale,
                    g=g,
                )
            )
            return core_attn_out, last_recurrent_state, None, None

        spec = getattr(self, "_gdn_2piece_spec", None)
        if spec is not None and needs_chunk_state:
            bs, L, cut, lens1, lens2 = spec
            T = q.shape[0]
            if bs == 1 and T >= L:
                o1, mid = torch.ops.npu.npu_chunk_gated_delta_rule(
                    q[:cut],
                    k[:cut],
                    v[:cut],
                    beta=beta[:cut],
                    initial_state=initial_state,
                    actual_seq_lengths=lens1,
                    scale=scale,
                    g=g[:cut],
                )
                mid_state = mid.to(initial_state.dtype, copy=False)
                o2, last_recurrent_state = torch.ops.npu.npu_chunk_gated_delta_rule(
                    q[cut:L],
                    k[cut:L],
                    v[cut:L],
                    beta=beta[cut:L],
                    initial_state=mid_state,
                    actual_seq_lengths=lens2,
                    scale=scale,
                    g=g[cut:L],
                )
                if T == L:
                    core_attn_out = torch.cat([o1, o2], dim=0)
                else:
                    core_attn_out = torch.empty_like(v)
                    core_attn_out[:cut] = o1
                    core_attn_out[cut:L] = o2
                return core_attn_out, last_recurrent_state, None, mid_state
            elif bs > 1 and T == bs * L:
                Nk, Dk = q.shape[1], q.shape[2]
                Nv, Dv = v.shape[1], v.shape[2]
                rem = L - cut
                q1 = (
                    q.view(bs, L, Nk, Dk)[:, :cut]
                    .reshape(bs * cut, Nk, Dk)
                    .contiguous()
                )
                k1 = (
                    k.view(bs, L, Nk, Dk)[:, :cut]
                    .reshape(bs * cut, Nk, Dk)
                    .contiguous()
                )
                v1 = (
                    v.view(bs, L, Nv, Dv)[:, :cut]
                    .reshape(bs * cut, Nv, Dv)
                    .contiguous()
                )
                g1 = g.view(bs, L, Nv)[:, :cut].reshape(bs * cut, Nv).contiguous()
                b1 = beta.view(bs, L, Nv)[:, :cut].reshape(bs * cut, Nv).contiguous()
                o1, mid = torch.ops.npu.npu_chunk_gated_delta_rule(
                    q1,
                    k1,
                    v1,
                    beta=b1,
                    initial_state=initial_state,
                    actual_seq_lengths=lens1,
                    scale=scale,
                    g=g1,
                )
                mid_state = mid.to(initial_state.dtype, copy=False)
                q2 = (
                    q.view(bs, L, Nk, Dk)[:, cut:]
                    .reshape(bs * rem, Nk, Dk)
                    .contiguous()
                )
                k2 = (
                    k.view(bs, L, Nk, Dk)[:, cut:]
                    .reshape(bs * rem, Nk, Dk)
                    .contiguous()
                )
                v2 = (
                    v.view(bs, L, Nv, Dv)[:, cut:]
                    .reshape(bs * rem, Nv, Dv)
                    .contiguous()
                )
                g2 = g.view(bs, L, Nv)[:, cut:].reshape(bs * rem, Nv).contiguous()
                b2 = beta.view(bs, L, Nv)[:, cut:].reshape(bs * rem, Nv).contiguous()
                o2, last_recurrent_state = torch.ops.npu.npu_chunk_gated_delta_rule(
                    q2,
                    k2,
                    v2,
                    beta=b2,
                    initial_state=mid_state,
                    actual_seq_lengths=lens2,
                    scale=scale,
                    g=g2,
                )
                core_attn_out = torch.empty_like(v)
                ov = core_attn_out.view(bs, L, Nv, Dv)
                ov[:, :cut] = o1.view(bs, cut, Nv, Dv)
                ov[:, cut:] = o2.view(bs, rem, Nv, Dv)
                return core_attn_out, last_recurrent_state, None, mid_state

        chunk_size = 64
        if cu_seqlens is not None:
            if getattr(self, "_cached_chunk_indices", None) is None:
                self._cached_chunk_indices = prepare_chunk_indices(
                    cu_seqlens, chunk_size
                )
            chunk_indices = self._cached_chunk_indices
        else:
            chunk_indices = None
        h = k.new_empty(
            len(chunk_indices), v.shape[-2], v.shape[-1], k.shape[-1]
        )  # [NT, NH, DV, Dk]

        core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
            q,
            k,
            v,
            beta=beta,
            initial_state=initial_state,
            actual_seq_lengths=self.actual_seq_lengths,
            scale=scale,
            chunk_state=h,
            g=g,
        )
        return core_attn_out, last_recurrent_state, h, None

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

        attn_core_out = recurrent_gated_delta_rule(
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
