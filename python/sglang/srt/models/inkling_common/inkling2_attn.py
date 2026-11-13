from functools import partial
from typing import NamedTuple

import torch
from torch import nn

from sglang.kernels.ops.inkling2.attention import (
    combine_gate_sink,
    compute_log_scaling_tau,
    gate_sink,
    pad_head_groups,
)
from sglang.kernels.ops.inkling2.comm import (
    all_reduce,
    oneshot_all_reduce,
    oneshot_all_reduce_input,
)
from sglang.kernels.ops.inkling2.linear import (
    linear,
    linear_k_group_partials_pair,
    uses_k_groups,
)
from sglang.kernels.ops.inkling2.prologue import attention_prologue
from sglang.srt.environ import envs
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.utils.common import add_prefix, set_weight_attrs
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.models.inkling_common.attn import (
    RelLogitsProj,
    get_inkling_relative_attention_score_mod,
    serving_attention_backend,
)
from sglang.srt.models.inkling_common.inkling2_sconv import ShortConvolution
from sglang.srt.models.inkling_common.sconv import SconvType
from sglang.srt.runtime_context import get_model, get_parallel, get_platform
from sglang.srt.utils import get_device_core_count

PADDED_DECODE_MIN_TOKENS = 8
# linear() sums split-K partials up to this many rows (its tiles < 256 branch).
PAIRED_PARTIALS_MAX_TOKENS = 2048
VERIFY_REL_BIAS_MAX_TOKENS = 32


def wave_fill_splits(tiles, sms, max_splits=128, min_fill=0.9):
    """Smallest KV split count whose CTAs fill at least ``min_fill`` of the last
    SM wave; the fullest count up to ``max_splits`` otherwise."""
    best, best_fill = 1, 0.0
    for splits in range(1, max_splits + 1):
        ctas = tiles * splits
        fill = ctas / (-(-ctas // sms) * sms)
        if fill >= min_fill:
            return splits
        if fill > best_fill:
            best, best_fill = splits, fill
    return best


class AttentionSources(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


def _stacked(sources):
    """torch.cat(sources) without a copy when the sources are adjacent in memory."""
    q = sources.q
    storage = q.untyped_storage().data_ptr()
    adjacent = all(
        x.is_contiguous()
        and x.shape == q.shape
        and x.dtype == q.dtype
        and x.untyped_storage().data_ptr() == storage
        and x.data_ptr() == q.data_ptr() + i * q.nbytes
        for i, x in enumerate(sources)
    )
    if adjacent and q.ndim == 2:
        return q.as_strided((3 * q.shape[0], q.shape[1]), q.stride())
    return torch.cat(sources)


class QKL2Clip(nn.Module):
    def __init__(self, dim, threshold, eps, num_heads, learnable_scale, *, tp_rank=0):
        super().__init__()
        self.threshold = threshold
        self.eps = eps
        self.tp_rank = tp_rank
        if learnable_scale:
            self.scale = nn.Parameter(torch.empty(num_heads, dim), requires_grad=False)
            set_weight_attrs(self.scale, {"weight_loader": self.weight_loader})
        else:
            self.register_parameter("scale", None)

    def weight_loader(self, param, loaded_weight):
        if loaded_weight.shape != param.shape:
            loaded_weight = loaded_weight.narrow(
                0, self.tp_rank * param.shape[0], param.shape[0]
            )
        assert loaded_weight.shape == param.shape
        param.data.copy_(loaded_weight)

    def forward(self, x, *, output_scale=None):
        from sglang.kernels.ops.inkling2.qk_l2_clip import qk_l2_clip_fwd_cutedsl

        output = qk_l2_clip_fwd_cutedsl(
            x, self.threshold, self.eps, scale=self.scale, output_scale=output_scale
        )
        return output.view(x.shape)


class Inkling2Attention(nn.Module):
    def __init__(self, config, layer_id, is_local, prefix=""):
        super().__init__()
        parallel = get_parallel()
        self.tp_rank, self.tp_size = parallel.attn_tp_rank, parallel.attn_tp_size
        self.hidden_size = config.hidden_size
        self.num_total_heads = (
            config.swa_num_attention_heads if is_local else config.num_attention_heads
        )
        self.num_total_kv_heads = (
            config.swa_num_key_value_heads if is_local else config.num_key_value_heads
        )
        self.head_dim = config.swa_head_dim if is_local else config.head_dim
        assert self.num_total_heads % self.tp_size == 0
        assert self.num_total_kv_heads % self.tp_size == 0, (
            "Attention TP must divide the number of KV heads"
        )
        self.num_tp_heads = self.num_total_heads // self.tp_size
        self.num_tp_kv_heads = self.num_total_kv_heads // self.tp_size
        self.d_rel = config.d_rel
        self.is_local = is_local
        self.layer_id = layer_id
        self.scaling = self.head_dim**-0.5
        self.rel_extent = config.sliding_window_size if is_local else config.rel_extent
        self.log_scaling_n_floor = config.log_scaling_n_floor
        self.log_scaling_alpha = config.log_scaling_alpha
        self.split_sizes = [
            self.num_tp_heads * self.head_dim,
            self.num_tp_kv_heads * self.head_dim,
            self.num_tp_kv_heads * self.head_dim,
            self.num_tp_heads * self.d_rel,
            self.num_tp_heads * self.head_dim,
            self.num_tp_heads,
        ]
        qkvr_pad_size = -sum(self.split_sizes) % 8
        if qkvr_pad_size:
            self.split_sizes.append(qkvr_pad_size)
        self.qkvr = MergedColumnParallelLinear(
            config.hidden_size,
            [size * self.tp_size for size in self.split_sizes],
            bias=config.q_bias,
            parallel_group="attn_tp",
            prefix=add_prefix("qkvr", prefix),
        )
        if self.qkvr.bias is not None:
            self.qkvr.bias.data.zero_()
        self.wo_ud = RowParallelLinear(
            self.head_dim * self.num_total_heads,
            config.hidden_size,
            bias=config.o_bias,
            parallel_group="attn_tp",
            reduce_results=False,
            use_dp_attention_reduce=True,
            prefix=add_prefix("wo_ud", prefix),
        )
        self.rel_logits_proj = RelLogitsProj(self.d_rel, self.rel_extent)
        self.q_l2_clip = QKL2Clip(
            self.head_dim,
            config.qk_l2_clip_threshold,
            1e-6,
            self.num_tp_heads,
            True,
            tp_rank=self.tp_rank,
        )
        self.k_l2_clip = QKL2Clip(
            self.head_dim,
            config.qk_l2_clip_threshold,
            1e-6,
            self.num_tp_kv_heads,
            False,
        )
        self.rel_l2_clip = QKL2Clip(
            self.d_rel,
            config.rel_l2_clip_threshold,
            1e-6,
            self.num_tp_heads,
            True,
            tp_rank=self.tp_rank,
        )
        if config.use_value_norm_no_gamma:
            from sglang.srt.models.inkling_common.inkling2_norm import RMSNorm

            self.v_norm = RMSNorm(
                self.head_dim, config.rms_norm_eps, elementwise_affine=False
            )
        else:
            self.v_norm = None
        conv_args = {
            "hidden_size": self.head_dim * self.num_tp_kv_heads,
            "kernel_size": config.sconv_kernel_size,
            "layer_id": layer_id,
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "is_dynamic": not is_local,
            "dynamic_input_size": config.hidden_size,
            "n_heads": self.num_tp_kv_heads,
        }
        self.k_sconv = ShortConvolution(
            sconv_type=SconvType.K_LOCAL if is_local else SconvType.K_FULL, **conv_args
        )
        self.v_sconv = ShortConvolution(
            sconv_type=SconvType.V_LOCAL if is_local else SconvType.V_FULL, **conv_args
        )
        self.attn = RadixAttention(
            self.num_tp_heads,
            self.head_dim,
            self.scaling,
            self.num_tp_kv_heads,
            layer_id,
            sliding_window_size=config.sliding_window_size - 1 if is_local else -1,
            prefix=add_prefix("attn", prefix),
            out_dtype=torch.float32,
            num_splits=1,
            pack_gqa=False,
            strided_lse=True,
        )
        # FA4 packs GQA heads only for group sizes dividing 128, so decode pads
        # each KV group with zero query heads up to the next power of two.
        group = self.num_tp_heads // self.num_tp_kv_heads
        self.padded_group = 1 << (group - 1).bit_length()
        kv_heads = self.num_tp_kv_heads
        pad_rows = kv_heads * (self.padded_group - group)
        self.fused_decode_prologue = (
            self.head_dim == 128
            and self.d_rel == 16
            and self.rel_extent % 128 == 0
            and kv_heads & (kv_heads - 1) == 0
            and pad_rows & (pad_rows - 1) == 0
            and self.v_norm is not None
            and self.v_norm.weight is None
            and self.v_norm.pre_weight is None
            and self.q_l2_clip.threshold == self.k_l2_clip.threshold
            and self.q_l2_clip.eps == self.k_l2_clip.eps == self.rel_l2_clip.eps
            and self.k_sconv.use_residual
            and self.v_sconv.use_residual
        )
        self.attn_padded = (
            RadixAttention(
                self.num_tp_kv_heads * self.padded_group,
                self.head_dim,
                self.scaling,
                self.num_tp_kv_heads,
                layer_id,
                sliding_window_size=config.sliding_window_size - 1 if is_local else -1,
                prefix=add_prefix("attn_padded", prefix),
                out_dtype=torch.float32,
                num_splits=0 if get_platform().is_sm90 else 1,
                pack_gqa=True,
                strided_lse=True,
            )
            if group > 1 and 128 % group
            else None
        )
        self.verify_wave_splits = not get_platform().is_sm90
        self.verify_max_splits = (
            -(-config.sliding_window_size // 128) + 1 if is_local else 128
        )

    def _pad_heads(self, x):
        return pad_head_groups(
            x, self.num_tp_heads // self.num_tp_kv_heads, self.padded_group
        )

    def _prologue(self, q, k, v, r, sources, positions, forward_batch):
        tokens = q.shape[0]
        k = self.k_sconv(
            k,
            positions,
            forward_batch,
            dynamic_input=None if self.is_local else sources.k,
        )
        v = self.v_sconv(
            v,
            positions,
            forward_batch,
            dynamic_input=None if self.is_local else sources.v,
        )
        tau = (
            compute_log_scaling_tau(
                positions, self.log_scaling_n_floor, self.log_scaling_alpha
            )
            if not self.is_local and self.log_scaling_n_floor is not None
            else None
        )
        q = self.q_l2_clip(
            q.reshape(tokens, self.num_tp_heads, self.head_dim), output_scale=tau
        )
        k = self.k_l2_clip(k.reshape(tokens, self.num_tp_kv_heads, self.head_dim))
        v = v.reshape(tokens, self.num_tp_kv_heads, self.head_dim)
        if self.v_norm is not None:
            v = self.v_norm(v)
        r = self.rel_l2_clip(
            r.reshape(tokens, self.num_tp_heads, self.d_rel), output_scale=tau
        )
        return q, k, v, r

    def _decode_prologue(
        self,
        q,
        k,
        v,
        r,
        sources,
        positions,
        forward_batch,
        padded,
        *,
        verify=False,
        shear=None,
    ):
        """_prologue and rel_logits_proj in one kernel; q and the rel bias come out
        padded when padded. With verify (MTP verify tokens), the K/V conv windows
        are saved instead of updating the conv cache; shear: see attention_prologue."""
        meta = self.k_sconv._conv_state(forward_batch)
        if self.is_local:
            k_weight, v_weight = self.k_sconv._weight_2d(), self.v_sconv._weight_2d()
            k_dynamic = v_dynamic = None
        else:
            k_weight, v_weight = self.k_sconv.static_w, self.v_sconv.static_w
            k_source, v_source = sources.k.to(k.dtype), sources.v.to(v.dtype)
            k_proj = self.k_sconv.weight_proj.to(k.dtype)
            v_proj = self.v_sconv.weight_proj.to(v.dtype)
            if (
                uses_k_groups(k_source, k_proj)
                and k_source.shape[0] <= PAIRED_PARTIALS_MAX_TOKENS
                and k_source.stride() == v_source.stride()
                and k_proj.stride() == v_proj.stride()
            ):
                k_dynamic, v_dynamic = linear_k_group_partials_pair(
                    k_source, k_proj, v_source, v_proj
                )
            else:
                k_dynamic = linear(k_source, k_proj)
                v_dynamic = linear(v_source, v_proj)
        tau = not self.is_local and self.log_scaling_n_floor is not None
        group = self.num_tp_heads // self.num_tp_kv_heads
        return attention_prologue(
            q,
            k,
            v,
            r,
            self.num_tp_heads,
            self.num_tp_kv_heads,
            self.d_rel,
            self.padded_group if padded else group,
            k_weight,
            v_weight,
            self.k_sconv._sconv_cache(),
            self.v_sconv._sconv_cache(),
            meta.precomputed,
            self.q_l2_clip.scale,
            self.rel_l2_clip.scale,
            self.rel_logits_proj.proj,
            forward_batch.seq_lens,
            self.is_local,
            self.q_l2_clip.threshold,
            self.rel_l2_clip.threshold,
            self.q_l2_clip.eps,
            self.v_norm.eps,
            k_dynamic=k_dynamic,
            v_dynamic=v_dynamic,
            positions=positions if tau else None,
            tau_floor=self.log_scaling_n_floor,
            tau_alpha=self.log_scaling_alpha,
            track_mask=None if verify else forward_batch.mamba_track_mask,
            track_indices=None if verify else meta.track_cache_indices,
            shear=shear,
            verify_windows=(
                (
                    self.k_sconv._intermediate_window(),
                    self.v_sconv._intermediate_window(),
                )
                if verify
                else None
            ),
        )

    def _project_qkvr(self, x):
        projected, _ = self.qkvr(x)
        return projected.split(self.split_sizes, dim=-1)[:6]

    def _project_sources(self, sources):
        if len({id(sources.q), id(sources.k), id(sources.v)}) == 3:
            q_proj, k_proj, v_proj = (
                projected.split(self.split_sizes, dim=-1)[:6]
                for projected in self.qkvr(_stacked(sources))[0].chunk(3)
            )
            q, _, _, r, gate, sinks = q_proj
            return q, k_proj[1], v_proj[2], r, gate, sinks
        q_proj = self._project_qkvr(sources.q)
        k_proj = q_proj if sources.k is sources.q else self._project_qkvr(sources.k)
        v_proj = (
            q_proj
            if sources.v is sources.q
            else k_proj
            if sources.v is sources.k
            else self._project_qkvr(sources.v)
        )
        q, _, _, r, gate, sinks = q_proj
        return q, k_proj[1], v_proj[2], r, gate, sinks

    def forward(self, hidden_states, positions, forward_batch):
        assert serving_attention_backend(forward_batch) == "fa4", (
            "Inkling2 requires FA4 attention"
        )
        mxfp8 = get_model().kv_cache_dtype == "mxfp8"
        sources = (
            AttentionSources(hidden_states, hidden_states, hidden_states)
            if isinstance(hidden_states, torch.Tensor)
            else hidden_states
        )
        q, k, v, r, gate, sinks = self._project_sources(sources)
        tokens = q.shape[0]
        mode = forward_batch.forward_mode
        verify = mode.is_target_verify()
        compact = (
            not mxfp8
            and self.fused_decode_prologue
            and (
                mode.is_decode()
                or verify
                and (
                    get_platform().is_sm90
                    or envs.SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL.get()
                    and tokens <= VERIFY_REL_BIAS_MAX_TOKENS
                )
            )
        )
        padded = (
            self.attn_padded is not None
            and not compact
            and (get_platform().is_sm90 or tokens >= PADDED_DECODE_MIN_TOKENS)
            and (mode.is_decode() or verify)
        )
        presheared = False
        if compact:
            q, k, v, rel_bias = self._decode_prologue(
                q, k, v, r, sources, positions, forward_batch, False, verify=verify
            )
        elif padded and verify and self.fused_decode_prologue:
            fa_meta = get_attn_backend().forward_metadata
            # The MXFP8 score modifier consumes the unsheared relative-bias row.
            presheared = (
                not mxfp8
                and not get_platform().is_sm90
                and fa_meta.swa_spec_metadata is None
                and fa_meta.max_seq_len_q * self.padded_group <= 128
            )
            shear = (
                (
                    fa_meta.cache_seqlens_int32,
                    self.attn.sliding_window_size if self.is_local else None,
                )
                if presheared
                else None
            )
            q, k, v, rel_bias = self._decode_prologue(
                q,
                k,
                v,
                r,
                sources,
                positions,
                forward_batch,
                True,
                verify=True,
                shear=shear,
            )
        else:
            q, k, v, r = self._prologue(q, k, v, r, sources, positions, forward_batch)
            rel_bias = self.rel_logits_proj(r)
            if padded:
                q, rel_bias = self._pad_heads(q), self._pad_heads(rel_bias)
        bias_kwargs = {"rel_bias_presheared": True} if presheared else {}
        output_dtype = q.dtype
        if mxfp8:
            from sglang.kernels.ops.quantization.mxfp8_quant import to_mxfp8

            quantized_q = to_mxfp8(q.reshape(tokens, -1, self.head_dim))
            q = quantized_q.data
            # MXFP8 dequant buffers leave no room for shared-memory bias staging.
            attention_kwargs = dict(
                score_mod=get_inkling_relative_attention_score_mod(self.rel_extent),
                aux_tensors=[rel_bias],
                q_descale=quantized_q.scale.view(torch.float8_e8m0fnu),
            )
        else:
            attention_kwargs = dict(
                rel_bias=rel_bias, rel_bias_scale_after_add=True, **bias_kwargs
            )
        if compact:
            output = self.attn(
                q,
                k,
                v,
                forward_batch,
                rel_bias=rel_bias,
                rel_bias_compact_decode=True,
                rel_bias_decode_epilogue=partial(
                    combine_gate_sink, gate=gate, sinks=sinks
                ),
            )
        elif padded:
            if self.verify_wave_splits:
                num_splits = 1
                if verify and (not self.is_local or forward_batch.batch_size == 1):
                    num_splits = wave_fill_splits(
                        forward_batch.batch_size
                        * self.num_tp_kv_heads
                        * -(
                            -tokens
                            * self.padded_group
                            // (128 * forward_batch.batch_size)
                        ),
                        get_device_core_count(q.device.index),
                        self.verify_max_splits,
                    )
                self.attn_padded.num_splits = num_splits
            output, lse = self.attn_padded(
                q,
                k,
                v,
                forward_batch,
                return_lse=True,
                **attention_kwargs,
            )
            head_groups = (self.num_tp_heads // self.num_tp_kv_heads, self.padded_group)
        else:
            output, lse = self.attn(
                q,
                k,
                v,
                forward_batch,
                return_lse=True,
                **attention_kwargs,
            )
            head_groups = None
        if not compact:
            output = gate_sink(
                output.reshape(tokens, -1),
                gate,
                lse,
                sinks,
                output_dtype=output_dtype,
                head_groups=head_groups,
            )
        parallel = get_parallel()
        multicast = parallel.moe_ep_size > 1
        if (
            multicast
            and (mode.is_decode() or verify)
            and envs.SGLANG_OPT_INKLING2_ATTN_PUSH_ALL_REDUCE.get()
        ):
            output, _ = self.wo_ud(output)
            return parallel.attn_tp_group.all_reduce(output)
        buffer = (
            oneshot_all_reduce_input(
                parallel.attn_tp_group, tokens, self.hidden_size, output.dtype
            )
            if multicast
            else None
        )
        if buffer is not None:
            self.wo_ud(output, output_tensor=buffer)
            return oneshot_all_reduce(buffer, parallel.attn_tp_group)
        output, _ = self.wo_ud(output)
        return all_reduce(output, parallel.attn_tp_group, multicast=multicast)
