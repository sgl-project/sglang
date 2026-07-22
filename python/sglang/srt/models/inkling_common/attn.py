from __future__ import annotations

from collections.abc import Callable
from functools import cache

import torch
from torch import nn

from sglang.kernels.ops.attention.log_scaling_tau import (
    apply_log_scaling_tau as _apply_log_scaling_tau,
)
from sglang.kernels.ops.attention.score_mod import (
    relative_bias_score_mod as triton_relative_bias_score_mod,
)
from sglang.kernels.ops.model.inkling.inkling_rel_proj import rel_proj_small_t
from sglang.kernels.ops.model.inkling.inkling_row_scale import row_compact_bf16
from sglang.srt.environ import envs
from sglang.srt.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode
from sglang.srt.models.inkling_common.kernels.comm import (
    get_ar_buffer,
    reduce_scatter_hidden,
    symm_mem_all_reduce,
)
from sglang.srt.models.inkling_common.norm import RMSNorm
from sglang.srt.models.inkling_common.sconv import SconvType, ShortConvolution
from sglang.srt.models.utils import apply_qk_norm
from sglang.srt.runtime_context import get_exec, get_parallel, get_server_args
from sglang.srt.utils import add_prefix, get_current_device_stream_fast

try:
    import cutlass.cute as cute
    from cutlass.cute import Float32

    from sglang.kernels.ops.attention.flash_attn.cute.seqlen_info import SeqlenInfoQK
except Exception as _import_error:
    cute = None
    Float32 = None
    SeqlenInfoQK = None
    _cute_import_error = _import_error
else:
    _cute_import_error = None


@cache
def get_inkling_relative_attention_score_mod(rel_extent: int) -> Callable:
    if cute is None or Float32 is None or SeqlenInfoQK is None:
        raise ImportError(
            "Inkling relative attention requires the vendored FA4 CUTE interface."
        ) from _cute_import_error

    @cute.jit
    def score_mod_rel_bias(
        scores: cute.TensorSSA,
        b_idx: cute.TensorSSA,
        h_idx: cute.TensorSSA,
        q_idx: cute.TensorSSA,
        kv_idx: cute.TensorSSA,
        seqlen_info: SeqlenInfoQK,
        aux_tensors: list[cute.Tensor],
    ) -> cute.TensorSSA:
        rel_logits = aux_tensors[0]

        seqlen_local_offset = seqlen_info.seqlen_k - seqlen_info.seqlen_q
        rel_dist = (q_idx + seqlen_local_offset) - kv_idx
        global_q_idx = seqlen_info.offset_q + q_idx

        rel_dist_0 = rel_dist[0]
        rel_idx = rel_dist_0 if rel_dist_0 >= 0 else 0
        rel_idx = rel_idx if rel_idx < rel_extent else (rel_extent - 1)

        rel_bias = rel_logits[global_q_idx[0], h_idx[0], rel_idx]
        rel_bias = Float32(rel_bias) if rel_dist_0 == rel_idx else Float32(0.0)
        return scores + rel_bias

    return score_mod_rel_bias


def compute_log_scaling_tau(
    positions: torch.Tensor, n_floor: int, alpha: float
) -> torch.Tensor:
    effective_n = (positions + 1).to(torch.float32)
    return 1.0 + alpha * torch.log(torch.clamp(effective_n / float(n_floor), min=1.0))


# Strided-matmul band of RelLogitsProj._project (bf16, h=16, d_rel=16,
# extent=1024, r strided from the packed qkvr row).
# t<=48: the zero-copy strided-batched GEMM (r @ proj) wins over einsum's
# hidden-copy chain; t>=64: {JIT row-compact + einsum} wins; the batched
# GEMM degrades past the band.
_REL_PROJ_MATMUL_MAX_T = 48
# tau-ON small-t band: the single rel_proj_small_t launch (tau folded in
# registers) beats the {row_scale -> einsum} chain up to t=32 (loses from
# t=48). tau-OFF stays on cuBLAS everywhere: at t=1 the bare GEMM wins and
# the kernel's launch floor loses -- there is no hidden copy to remove at
# t=1 (r is contiguous).
_REL_PROJ_TAU_KERNEL_MAX_T = 32


def _rel_proj_kernel_eligible(r: torch.Tensor) -> bool:
    """rel_proj_small_t input contract: bf16 CUDA, [t, h, d_rel] with a
    contiguous (h*d_rel) inner block (token rows may be strided), d_rel a
    vector multiple, 16B-aligned base and token stride."""
    return (
        r.is_cuda
        and r.dtype == torch.bfloat16
        and r.stride(-1) == 1
        and r.stride(-2) == r.shape[-1]
        and r.shape[2] % 8 == 0
        and r.data_ptr() % 16 == 0
        and (r.stride(0) * 2) % 16 == 0
    )


class RelLogitsProj(nn.Module):
    def __init__(self, d_rel: int, rel_extent: int):
        super().__init__()
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.proj = nn.Parameter(torch.empty(d_rel, rel_extent), requires_grad=False)
        # Fold the optional log-scaling tau into the einsum's r OPERAND: the
        # per-token diagonal scale commutes through the linear projection, so
        # the scale pass runs over [t, h, d_rel] instead of the
        # rel_extent/d_rel-times-larger output (64x at d_rel=16/extent=1024).
        # The einsum itself stays cuBLAS -- the K=16 expansion is tensor-core
        # territory. Rounding moves with the fold (r*tau rounds to bf16 before
        # the GEMM instead of after); flag-off keeps the exact legacy post-scale.
        self._prescale_tau = envs.SGLANG_OPT_USE_INKLING_FUSED_LOG_TAU.get()
        self._proj_dispatch = envs.SGLANG_OPT_USE_INKLING_REL_PROJ_DISPATCH.get()

    def _project(self, r: torch.Tensor) -> torch.Tensor:
        """``einsum("thd,de->the", r, proj)`` -- but dispatched: in production
        ``r`` is a strided view into the packed qkvr projection output, and
        einsum's reshape then hides a slow TensorIterator compaction copy.
        Both replacements are bit-identical to the einsum (same GEMM reduction
        order; asserted in test_inkling_attn_prologue_tau.py)."""
        if not self._proj_dispatch or r.is_contiguous():
            return torch.einsum("thd,de->the", r, self.proj)
        if r.shape[0] <= _REL_PROJ_MATMUL_MAX_T:
            return r @ self.proj  # zero-copy strided-batched GEMM over t
        rows, inner = r.shape[0], r.shape[1] * r.shape[2]
        if (
            r.is_cuda
            and r.dtype == torch.bfloat16
            and r.stride(-1) == 1
            and r.stride(-2) == r.shape[-1]
            and inner % 8 == 0
            and r.data_ptr() % 16 == 0
            and (r.stride(0) * 2) % 16 == 0
        ):
            r2d = torch.as_strided(r, (rows, inner), (r.stride(0), 1))
            r = row_compact_bf16(r2d).view(r.shape)
        return torch.einsum("thd,de->the", r, self.proj)

    def forward(
        self, r_out: torch.Tensor, log_scaling_tau: torch.Tensor | None = None
    ) -> torch.Tensor:
        """``log_scaling_tau``: optional per-token scale applied to the
        projected logits (the conditional long-context log-scaling); folded
        into the projection's r operand when enabled."""
        if log_scaling_tau is not None and self._prescale_tau:
            if (
                self._proj_dispatch
                and r_out.shape[0] <= _REL_PROJ_TAU_KERNEL_MAX_T
                and _rel_proj_kernel_eligible(r_out)
            ):
                # Single launch: tau prescale (same round-before-dot
                # semantics) + projection, replacing the two-kernel
                # {row_scale -> einsum} chain in its band.
                tau_flat = log_scaling_tau.reshape(-1)
                if tau_flat.dtype != torch.float32:
                    tau_flat = tau_flat.float()
                return rel_proj_small_t(r_out, self.proj, tau_flat)
            # The prescale already compacts r (row-scale writes contiguous),
            # so the einsum runs on a contiguous operand via _project.
            r_out = _apply_log_scaling_tau(r_out, log_scaling_tau.view(-1, 1, 1))
            log_scaling_tau = None
        out = self._project(r_out)
        if log_scaling_tau is not None:
            out = _apply_log_scaling_tau(out, log_scaling_tau.view(-1, 1, 1))
        return out


class InklingQKVRLinear(MergedColumnParallelLinear):
    """Fused q/k/v/r projection that is KV-replication-aware for LoRA.

    The base K/V weights are replicated at load when attn_tp_size > num_kv_heads
    (via ``_kv_total_for_sizing``). This subclass carries the head geometry so
    the LoRA wrapper (``InklingQKVRLinearWithLoRA``) can replicate the K/V slices of the
    adapter LoRA-B the same way. ``is_inkling_qkvr`` lets ``get_lora_layer`` route it
    without importing Inkling into ``srt/lora``.
    """

    is_inkling_qkvr = True

    def __init__(
        self,
        *args,
        inkling_num_kv_heads: int,
        inkling_head_dim: int,
        inkling_num_heads: int,
        inkling_d_rel: int,
        inkling_tp_size: int,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.inkling_num_kv_heads = inkling_num_kv_heads
        self.inkling_head_dim = inkling_head_dim
        self.inkling_num_heads = inkling_num_heads
        self.inkling_d_rel = inkling_d_rel
        self.inkling_tp_size = inkling_tp_size


class InklingAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int | None,
        d_rel: int,
        rel_extent: int,
        local_extent: int,
        norm_eps: float,
        is_local: bool,
        layer_id: int,
        q_bias: bool = False,
        o_bias: bool = False,
        kv_conv: bool = False,
        sconv_kernel_size: int | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        alt_stream: torch.cuda.Stream | None = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.alt_stream = alt_stream

        attn_tp_rank = get_parallel().attn_tp_rank
        attn_tp_size = get_parallel().attn_tp_size

        self.tp_size = attn_tp_size
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.d_rel = d_rel
        self.num_total_heads = num_heads
        self.num_total_kv_heads = num_kv_heads
        self.scaling = 1.0 / self.head_dim

        assert self.num_total_heads % self.tp_size == 0
        self.num_tp_heads = self.num_total_heads // self.tp_size

        if self.num_total_kv_heads >= self.tp_size:
            assert self.num_total_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.num_total_kv_heads == 0
        self.num_tp_kv_heads = max(1, self.num_total_kv_heads // self.tp_size)
        self.layer_id = layer_id
        self.is_local = is_local
        self._kv_total_for_sizing = max(self.num_total_kv_heads, self.tp_size)

        output_sizes = [
            self.head_dim * self.num_total_heads,
            self.head_dim * self._kv_total_for_sizing,
            self.head_dim * self._kv_total_for_sizing,
            self.d_rel * self.num_total_heads,
        ]

        self.qkvr = InklingQKVRLinear(
            input_size=self.hidden_size,
            output_sizes=output_sizes,
            bias=q_bias,
            prefix=add_prefix("qkvr", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            quant_config=quant_config,
            inkling_num_kv_heads=self.num_total_kv_heads,
            inkling_head_dim=self.head_dim,
            inkling_num_heads=self.num_total_heads,
            inkling_d_rel=self.d_rel,
            inkling_tp_size=attn_tp_size,
        )
        self.wo_ud = RowParallelLinear(
            input_size=self.head_dim * self.num_total_heads,
            output_size=self.hidden_size,
            bias=o_bias,
            prefix=add_prefix("wo_ud", prefix),
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            use_dp_attention_reduce=True,
            quant_config=quant_config,
        )
        # --enable-scattered-sconv: the output reduction becomes a hidden-dim
        # reduce-scatter (the consumer attn_sconv runs on the [T, H/P] shard).
        self.scattered_sconv = get_exec().comm.enable_scattered_sconv

        if is_local:
            self.rel_extent = local_extent
            self.local_extent = local_extent
        else:
            self.rel_extent = rel_extent
            self.local_extent = None

        self.rel_logits_proj = RelLogitsProj(self.d_rel, self.rel_extent)
        # Fold the conditional log-scaling tau into the fused prologue's q
        # path (deletes the external scale kernel; bit-exact rounding).
        self._fused_log_tau = envs.SGLANG_OPT_USE_INKLING_FUSED_LOG_TAU.get()
        self.q_norm = RMSNorm(self.head_dim, eps=norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=norm_eps)

        self.kv_conv = kv_conv
        self.sconv_kernel_size = sconv_kernel_size
        self.k_sconv = (
            ShortConvolution(
                hidden_size=self.head_dim * self.num_tp_kv_heads,
                kernel_size=self.sconv_kernel_size,
                sconv_type=SconvType.K_LOCAL if is_local else SconvType.K_FULL,
                layer_id=layer_id,
            )
            if self.kv_conv
            else None
        )
        self.v_sconv = (
            ShortConvolution(
                hidden_size=self.head_dim * self.num_tp_kv_heads,
                kernel_size=self.sconv_kernel_size,
                sconv_type=SconvType.V_LOCAL if is_local else SconvType.V_FULL,
                layer_id=layer_id,
            )
            if self.kv_conv
            else None
        )

        self.attn = RadixAttention(
            self.num_tp_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_tp_kv_heads,
            layer_id=self.layer_id,
            sliding_window_size=self.local_extent - 1 if self.is_local else -1,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def _project_qkvr(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_tokens = hidden_states.size(0)
        qkvr, _ = self.qkvr(hidden_states)
        qkvr = qkvr.view(num_tokens, -1)
        split_sizes = [
            self.head_dim * self.num_tp_heads,
            self.head_dim * self.num_tp_kv_heads,
            self.head_dim * self.num_tp_kv_heads,
            self.d_rel * self.num_tp_heads,
        ]
        q, k, v, r = qkvr.split(split_sizes, dim=-1)
        return q, k, v, r

    def _fused_attn_prologue_verify(self, q, k, v, forward_batch, log_scaling_tau=None):
        """Fused target-verify {k/v sconv + save_windows + qk-norm (+ KV store)}
        (kernels/ops/model/inkling/inkling_attn_prologue.py); returns
        ``(q, k, v, did_store)``.

        The fused kernel writes raw bf16 KV, so it only does the store when the
        KV pool is bf16: full layers at ``out_cache_loc`` in the full pool,
        local (SWA) layers at the backend's pre-translated ``swa_out_cache_loc``
        in the SWA sub-pool (get_key_buffer dispatches by layer). For an FP8 /
        MXFP8 KV pool the fused store is skipped (``did_store=False``) and the
        caller keeps ``save_kv_cache=True`` so the backend quantizes the
        already-normed/conv'd K/V and writes the block scales; conv + windows +
        qk-norm stay fused either way. For the FA4 MXFP8 pool, the prologue can
        quantize Q and directly fill the fp8 K/V cache plus interleaved scale
        buffers, returning Q's per-token scales as ``q_descale``/``sfq``."""
        from sglang.kernels.ops.model.inkling.inkling_attn_prologue import (
            inkling_attn_prologue_verify,
        )
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            get_token_to_kv_pool,
        )

        if not hasattr(self, "_qk_gamma_bf16"):
            self._qk_gamma_bf16 = (
                self.q_norm.weight.to(torch.bfloat16),
                self.k_norm.weight.to(torch.bfloat16),
            )
        k_cache, ci, cm, kw, k_inter = self.k_sconv.verify_fused_ar_inputs(
            forward_batch
        )
        v_cache, _, _, vw, v_inter = self.v_sconv.verify_fused_ar_inputs(forward_batch)
        pool = get_token_to_kv_pool()
        k_buf = pool.get_key_buffer(self.layer_id)
        v_buf = pool.get_value_buffer(self.layer_id)
        # The fused store writes raw bf16 into an NHD [slot, head, head_dim]
        # buffer indexed by loc. Take it only for that exact layout: FP8/MXFP8
        # (non-bf16) and HND/vectorized_5d (4D/5D, paged (page, head) index)
        # pools keep the backend store, which owns their quant + layout. conv +
        # windows + qk-norm stay fused regardless.
        do_bf16_store = (
            k_buf.dtype == torch.bfloat16
            and k_buf.dim() == 3
            and k_buf.shape[-1] == self.head_dim
            and k_buf.is_contiguous()
        )
        sfk = sfv = None
        do_mxfp8_store = False
        server_args = get_server_args()
        if server_args.kv_cache_dtype == "mxfp8" and hasattr(
            pool, "get_kv_scale_buffer"
        ):
            sfk, sfv = pool.get_kv_scale_buffer(self.layer_id)
            do_mxfp8_store = (
                k_buf.dtype == torch.float8_e4m3fn
                and v_buf.dtype == torch.float8_e4m3fn
                and k_buf.dim() == 3
                and v_buf.dim() == 3
                and k_buf.shape[-1] == self.head_dim
                and v_buf.shape[-1] == self.head_dim
                and k_buf.is_contiguous()
                and v_buf.is_contiguous()
                and sfk.dim() == 5
                and sfv.dim() == 5
                and getattr(pool, "page_size", 0) == 128
            )
        do_store = do_bf16_store or do_mxfp8_store
        metadata = get_attn_backend().forward_metadata
        if do_store and self.is_local:
            # SWA sub-pool + the backend's full->SWA translated write location.
            loc = metadata.swa_out_cache_loc
        elif do_store:
            loc = getattr(metadata, "out_cache_loc_full_physical", None)
            if loc is None:
                loc = forward_batch.out_cache_loc
        else:
            loc = forward_batch.out_cache_loc
        es = q.element_size()
        q, k, v, q_descale = inkling_attn_prologue_verify(
            q,
            k_cache,
            v_cache,
            ci,
            cm,
            kw,
            vw,
            k_inter,
            v_inter,
            self._qk_gamma_bf16[0],
            self._qk_gamma_bf16[1],
            self.q_norm.variance_epsilon,
            loc,
            k_buf,
            v_buf,
            0,
            (k.data_ptr() - q.data_ptr()) // es,
            (v.data_ptr() - q.data_ptr()) // es,
            q.shape[1],
            k.shape[1],
            forward_batch.spec_info.draft_token_num,
            activation=self.k_sconv.activation,
            use_residual=self.k_sconv.use_residual,
            do_store=do_store,
            mxfp8_quant=do_mxfp8_store,
            sfk=sfk,
            sfv=sfv,
            page_size=getattr(pool, "page_size", 128),
            log_scaling_tau=log_scaling_tau,
        )
        return q, k, v, do_store, q_descale

    def _fused_attn_prologue_extend(self, q, k, v, forward_batch, log_scaling_tau=None):
        """Extend analog of _fused_attn_prologue_verify: {k/v varlen sconv +
        qk-norm (+ KV store)} in the main kernel plus a tiny trailing k/v
        conv-cache update (+ prefix-cache track) kernel -- replacing
        2x causal_conv1d + apply_qk_norm + 2x update_sconv_cache (+ track) +
        the backend store. Store gating (bf16 NHD / FA4 MXFP8 pools, SWA loc
        translation) is identical to the verify prologue. Returns
        (q, k, v, did_store, q_descale)."""
        from sglang.kernels.ops.model.inkling.inkling_attn_prologue import (
            inkling_attn_prologue_extend,
        )
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            get_token_to_kv_pool,
        )

        if not hasattr(self, "_qk_gamma_bf16"):
            self._qk_gamma_bf16 = (
                self.q_norm.weight.to(torch.bfloat16),
                self.k_norm.weight.to(torch.bfloat16),
            )
        (
            k_cache,
            _safe_idx,
            cm,
            cu,
            si,
            kw,
            _qsl,
            ci,
            has_init,
            trows,
            tmask,
            tdst,
        ) = self.k_sconv.extend_fused_ar_inputs(forward_batch)
        v_inputs = self.v_sconv.extend_fused_ar_inputs(forward_batch)
        v_cache, vw = v_inputs[0], v_inputs[5]
        pool = get_token_to_kv_pool()
        k_buf = pool.get_key_buffer(self.layer_id)
        v_buf = pool.get_value_buffer(self.layer_id)
        do_bf16_store = (
            k_buf.dtype == torch.bfloat16
            and v_buf.dtype == torch.bfloat16
            and k_buf.dim() == 3
            and v_buf.dim() == 3
            and k_buf.shape[-1] == self.head_dim
            and v_buf.shape[-1] == self.head_dim
            and k_buf.is_contiguous()
            and v_buf.is_contiguous()
        )
        sfk = sfv = None
        do_mxfp8_store = False
        server_args = get_server_args()
        if server_args.kv_cache_dtype == "mxfp8" and hasattr(
            pool, "get_kv_scale_buffer"
        ):
            sfk, sfv = pool.get_kv_scale_buffer(self.layer_id)
            do_mxfp8_store = (
                k_buf.dtype == torch.float8_e4m3fn
                and v_buf.dtype == torch.float8_e4m3fn
                and k_buf.dim() == 3
                and v_buf.dim() == 3
                and k_buf.shape[-1] == self.head_dim
                and v_buf.shape[-1] == self.head_dim
                and k_buf.is_contiguous()
                and v_buf.is_contiguous()
                and sfk.dim() == 5
                and sfv.dim() == 5
                and getattr(pool, "page_size", 0) == 128
            )
        do_store = do_bf16_store or do_mxfp8_store
        metadata = get_attn_backend().forward_metadata
        if do_store and self.is_local:
            loc = metadata.swa_out_cache_loc
        elif do_store:
            loc = getattr(metadata, "out_cache_loc_full_physical", None)
            if loc is None:
                loc = forward_batch.out_cache_loc
        else:
            loc = forward_batch.out_cache_loc
        is_v2 = forward_batch.forward_mode.is_draft_extend_v2()
        if is_v2:
            # The seq-end-window trailing update and the extend prefix-cache
            # track are WRONG for DRAFT_EXTEND_V2: the conv state must
            # reflect only num_accept_tokens. Run the accept-gated update
            # (which also handles accept-aware tracking) after the kernel.
            dev = trows.device
            trows = torch.empty((0, trows.shape[1]), dtype=torch.int64, device=dev)
            tmask = torch.empty((0,), dtype=torch.bool, device=dev)
            tdst = torch.empty((0,), dtype=torch.int64, device=dev)
        k_pre, v_pre = k, v
        es = q.element_size()
        q, k, v, q_descale = inkling_attn_prologue_extend(
            q,
            k_cache,
            v_cache,
            ci,
            cm,
            has_init,
            cu,
            si,
            kw,
            vw,
            trows,
            tmask,
            tdst,
            self._qk_gamma_bf16[0],
            self._qk_gamma_bf16[1],
            self.q_norm.variance_epsilon,
            loc,
            k_buf,
            v_buf,
            0,
            (k.data_ptr() - q.data_ptr()) // es,
            (v.data_ptr() - q.data_ptr()) // es,
            q.shape[1],
            k.shape[1],
            activation=self.k_sconv.activation,
            use_residual=self.k_sconv.use_residual,
            do_store=do_store,
            mxfp8_quant=do_mxfp8_store,
            sfk=sfk,
            sfv=sfv,
            page_size=getattr(pool, "page_size", 128),
            do_cache_update=not is_v2,
            log_scaling_tau=log_scaling_tau,
        )
        if is_v2:
            self.k_sconv._update_sconv_cache_for_draft_extend(
                forward_batch, k_cache, ci, k_pre
            )
            self.v_sconv._update_sconv_cache_for_draft_extend(
                forward_batch, v_cache, ci, v_pre
            )
        return q, k, v, do_store, q_descale

    def _fused_attn_prologue_decode(self, q, k, v, forward_batch, log_scaling_tau=None):
        """Decode analog of _fused_attn_prologue_verify: {k/v decode-conv +
        conv-cache shift-update (+ prefix track) + qk-norm (+ KV store)} in one
        kernel. Decode is one token/seq so the conv taps come from the working
        cache (no cross-token reads, no barrier). Returns
        (q, k, v, did_store, q_descale)."""
        from sglang.kernels.ops.model.inkling.inkling_attn_prologue import (
            inkling_attn_prologue_decode,
        )
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            get_token_to_kv_pool,
        )

        if not hasattr(self, "_qk_gamma_bf16"):
            self._qk_gamma_bf16 = (
                self.q_norm.weight.to(torch.bfloat16),
                self.k_norm.weight.to(torch.bfloat16),
            )
        k_cache, ci, cm, kw = self.k_sconv.decode_fused_ar_inputs(forward_batch)
        v_cache, _, _, vw = self.v_sconv.decode_fused_ar_inputs(forward_batch)
        pool = get_token_to_kv_pool()
        k_buf = pool.get_key_buffer(self.layer_id)
        v_buf = pool.get_value_buffer(self.layer_id)
        do_bf16_store = (
            k_buf.dtype == torch.bfloat16
            and v_buf.dtype == torch.bfloat16
            and k_buf.dim() == 3
            and v_buf.dim() == 3
            and k_buf.shape[-1] == self.head_dim
            and v_buf.shape[-1] == self.head_dim
            and k_buf.is_contiguous()
            and v_buf.is_contiguous()
        )
        sfk = sfv = None
        do_mxfp8_store = False
        server_args = get_server_args()
        if server_args.kv_cache_dtype == "mxfp8" and hasattr(
            pool, "get_kv_scale_buffer"
        ):
            sfk, sfv = pool.get_kv_scale_buffer(self.layer_id)
            do_mxfp8_store = (
                k_buf.dtype == torch.float8_e4m3fn
                and v_buf.dtype == torch.float8_e4m3fn
                and k_buf.dim() == 3
                and v_buf.dim() == 3
                and k_buf.shape[-1] == self.head_dim
                and v_buf.shape[-1] == self.head_dim
                and k_buf.is_contiguous()
                and v_buf.is_contiguous()
                and sfk.dim() == 5
                and sfv.dim() == 5
                and getattr(pool, "page_size", 0) == 128
            )
        do_store = do_bf16_store or do_mxfp8_store
        metadata = get_attn_backend().forward_metadata
        if do_store and self.is_local:
            loc = metadata.swa_out_cache_loc
        elif do_store:
            loc = getattr(metadata, "out_cache_loc_full_physical", None)
            if loc is None:
                loc = forward_batch.out_cache_loc
        else:
            loc = forward_batch.out_cache_loc
        es = q.element_size()
        q, k, v, q_descale = inkling_attn_prologue_decode(
            q,
            k_cache,
            v_cache,
            ci,
            cm,
            kw,
            vw,
            self._qk_gamma_bf16[0],
            self._qk_gamma_bf16[1],
            self.q_norm.variance_epsilon,
            loc,
            k_buf,
            v_buf,
            0,
            (k.data_ptr() - q.data_ptr()) // es,
            (v.data_ptr() - q.data_ptr()) // es,
            q.shape[1],
            k.shape[1],
            activation=self.k_sconv.activation,
            use_residual=self.k_sconv.use_residual,
            track_mask=forward_batch.mamba_track_mask,
            track_indices=forward_batch.mamba_track_indices,
            do_store=do_store,
            mxfp8_quant=do_mxfp8_store,
            sfk=sfk,
            sfv=sfv,
            page_size=getattr(pool, "page_size", 128),
            log_scaling_tau=log_scaling_tau,
        )
        return q, k, v, do_store, q_descale

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        *,
        log_scaling_tau: torch.Tensor | None = None,
        reduce: bool = True,
    ) -> torch.Tensor:
        """With ``reduce=False`` the TP all-reduce of the wo_ud output is
        skipped and the LOCAL partial sums are returned (bias already folded in
        exactly once, on tp_rank 0) -- the caller takes over the reduction via
        the fused decode {AR -> attn_sconv -> mlp_norm} kernel."""
        assert hidden_states.ndim == 2
        num_tokens = hidden_states.size(0)
        q, k, v, r = self._project_qkvr(hidden_states)
        r = r.view(num_tokens, -1, self.d_rel)

        apply_log_scaling = log_scaling_tau is not None and not self.is_local

        server_args = get_server_args()
        assert server_args.attention_backend in ("fa4", "triton")
        # The overlap threads a CUDA event into the FA4 sheared-bias kernel, so it
        # is FA4-only for now.
        # TODO(triton): plumb rel_bias_event through the triton attn path too.
        fa4 = server_args.attention_backend == "fa4"

        rel_event = None
        prologue_did_store = False
        prologue_q_descale = None
        _fm = forward_batch.forward_mode
        _prologue_verify = _fm.is_target_verify()
        _prologue_decode = _fm.is_decode()
        # Extend (prefill): varlen conv + trailing cache-update kernel.
        # DRAFT_EXTEND_V2 also fuses the main kernel; only the conv-cache
        # update differs (accept-gated), handled inside the extend helper.
        _prologue_extend = (
            _fm.is_extend(include_draft_extend_v2=True) and not _prologue_verify
        )
        fused_prologue = (
            fa4
            and self.kv_conv
            and self.alt_stream is not None
            and num_tokens > 0
            and self.head_dim == 128
            and (_prologue_verify or _prologue_decode or _prologue_extend)
            and envs.SGLANG_OPT_USE_INKLING_FUSED_ATTN_PROLOGUE.get()
        )
        # Fold tau into the prologue's q path (in-kernel, bit-exact vs the
        # external scale; lands BEFORE MXFP8 quantization there).
        fold_tau_q = apply_log_scaling and self._fused_log_tau
        if fused_prologue:
            # rel_logits overlaps on the alt stream with the fused prologue
            # {k/v sconv + (save_windows | cache-update) + qk-norm (+ KV store)}
            # on the current stream. Same overlap for verify, decode and extend.
            current_stream = get_current_device_stream_fast()
            self.alt_stream.wait_stream(current_stream)
            _tau_arg = log_scaling_tau if fold_tau_q else None
            if _prologue_verify:
                (
                    q,
                    k,
                    v,
                    prologue_did_store,
                    prologue_q_descale,
                ) = self._fused_attn_prologue_verify(
                    q, k, v, forward_batch, log_scaling_tau=_tau_arg
                )
            elif _prologue_decode:
                (
                    q,
                    k,
                    v,
                    prologue_did_store,
                    prologue_q_descale,
                ) = self._fused_attn_prologue_decode(
                    q, k, v, forward_batch, log_scaling_tau=_tau_arg
                )
            else:
                (
                    q,
                    k,
                    v,
                    prologue_did_store,
                    prologue_q_descale,
                ) = self._fused_attn_prologue_extend(
                    q, k, v, forward_batch, log_scaling_tau=_tau_arg
                )
            with torch.cuda.stream(self.alt_stream):
                rel_logits = self.rel_logits_proj(
                    r, log_scaling_tau if apply_log_scaling else None
                )
                rel_event = torch.cuda.Event()
                rel_event.record()
        use_alt = (
            not fused_prologue
            and fa4
            and self.alt_stream is not None
            and hidden_states.is_cuda
            and num_tokens > 0
            and get_is_capture_mode()
        )
        if use_alt:
            # Alt stream runs v_sconv then rel_logits_proj, each fenced by its own
            # event, while the current stream runs k_sconv + apply_qk_norm. v_event
            # gates attn on v_sconv (joined below); rel_event is deferred into the
            # FA4 backend so rel_logits_proj also overlaps the KV-write. No
            # record_stream: graph-unsafe under capture and unneeded since v
            # aliases the qkvr buffer q keeps alive past the join.
            current_stream = get_current_device_stream_fast()
            self.alt_stream.wait_stream(current_stream)
            with torch.cuda.stream(self.alt_stream):
                if self.kv_conv:
                    assert self.v_sconv is not None
                    v = self.v_sconv(v, positions, forward_batch)
                v_event = torch.cuda.Event()
                v_event.record()
                rel_logits = self.rel_logits_proj(
                    r, log_scaling_tau if apply_log_scaling else None
                )
                rel_event = torch.cuda.Event()
                rel_event.record()
            if self.kv_conv:
                assert self.k_sconv is not None
                k = self.k_sconv(k, positions, forward_batch)
        elif not fused_prologue:
            if self.kv_conv:
                assert self.k_sconv is not None
                assert self.v_sconv is not None
                k = self.k_sconv(k, positions, forward_batch)
                v = self.v_sconv(v, positions, forward_batch)

        # apply_qk_norm runs on the current stream now -- the alt stream is busy
        # with v_sconv + rel_logits_proj. (The fused prologue already normed.)
        if fused_prologue:
            pass
        else:
            q, k = apply_qk_norm(
                q=q,
                k=k,
                q_norm=self.q_norm,
                k_norm=self.k_norm,
                head_dim=self.head_dim,
                alt_stream=None,
            )

        if apply_log_scaling and not (fused_prologue and fold_tau_q):
            # (When the fused prologue ran with fold_tau_q, tau was already
            # folded into its q path -- including BEFORE the MXFP8 quant.)
            # q is 2D [num_tokens, heads*head_dim]; tau is per-token, so broadcast
            # over the whole row with [num_tokens, 1] -- identical to scaling the
            # (num_tokens, heads, head_dim) view by tau.view(-1, 1, 1), but with no
            # 3D view (the strided split is not collapsible back to [-1, head_dim]).
            q = _apply_log_scaling_tau(q, log_scaling_tau.view(-1, 1))

        if use_alt:
            # v (produced on the alt stream) must be ready before attn reads it.
            current_stream.wait_event(v_event)
        elif not fused_prologue:
            rel_logits = self.rel_logits_proj(
                r, log_scaling_tau if apply_log_scaling else None
            )

        extra_attn_kwargs = {}
        if server_args.kv_cache_dtype == "mxfp8":
            # Must run AFTER v is joined above (wait_event(v_event)): v (and k)
            # may be produced by sconv on the alt stream, and quantizing them on
            # the main stream before the join reads half-written buffers under
            # load. The bf16 path only touches k/v inside self.attn, past the join.
            #
            # Block-scaled QK + in-kernel V dequant (FA4 downloads contract):
            # Q/K/V all quantize to fp8. Q's per-token scales ride q_descale
            # (backend passes them as sfq); K/V scales ride k/v_descale into
            # set_kv_buffer, which stores them interleaved as sfk/sfv.
            if prologue_q_descale is not None:
                extra_attn_kwargs["q_descale"] = prologue_q_descale
            else:
                from sglang.srt.layers.quantization.mxfp8_quant import to_mxfp8

                q_mxfp = to_mxfp8(q.view(num_tokens, self.num_tp_heads, self.head_dim))
                q = q_mxfp.data.view(num_tokens, -1)
                extra_attn_kwargs["q_descale"] = q_mxfp.scale.view(torch.float8_e8m0fnu)
            if (
                not prologue_did_store
                and not envs.SGLANG_OPT_INKLING_MXFP8_FUSED_QUANT_STORE.get()
            ):
                k_mxfp = to_mxfp8(
                    k.view(num_tokens, self.num_tp_kv_heads, self.head_dim)
                )
                v_mxfp = to_mxfp8(
                    v.view(num_tokens, self.num_tp_kv_heads, self.head_dim)
                )
                k = k_mxfp.data.view(num_tokens, -1)
                v = v_mxfp.data.view(num_tokens, -1)
                extra_attn_kwargs["k_descale"] = k_mxfp.scale.view(torch.float8_e8m0fnu)
                extra_attn_kwargs["v_descale"] = v_mxfp.scale.view(torch.float8_e8m0fnu)
            # Else either the prologue already filled the MXFP8 KV cache, or K/V
            # stay bf16 here and the MXFP8 pool's set_kv_buffer quantizes and
            # stores them in one fused kernel (absent descales signal it).

        if envs.SGLANG_OPT_USE_INKLING_SHEARED_BIAS.get() and fa4:
            # FA4 sheared-bias kernel: pass rel_logits directly; the kernel shears
            # it into a column-aligned pre-softmax bias.
            attn_output = self.attn(
                q,
                k,
                v,
                forward_batch,
                save_kv_cache=not prologue_did_store,
                rel_bias=rel_logits,
                rel_bias_event=rel_event,
                **extra_attn_kwargs,
            )
        else:
            # The score_mod / aux_tensors path can't carry the event into the
            # kernel, so join rel_logits on the current stream before self.attn.
            if rel_event is not None:
                get_current_device_stream_fast().wait_event(rel_event)
            if fa4:
                attn_output = self.attn(
                    q,
                    k,
                    v,
                    forward_batch,
                    save_kv_cache=not prologue_did_store,
                    score_mod=get_inkling_relative_attention_score_mod(self.rel_extent),
                    aux_tensors=[rel_logits],
                    **extra_attn_kwargs,
                )
            else:
                attn_output = self.attn(
                    q,
                    k,
                    v,
                    forward_batch,
                    save_kv_cache=not prologue_did_store,
                    score_mod=triton_relative_bias_score_mod,
                    aux_tensors=[rel_logits],
                    **extra_attn_kwargs,
                )
        attn_output = attn_output.view(num_tokens, -1)

        # Fuse wo_ud's local output GEMM with the all-reduce: write the GEMM
        # straight into the symm-mem AR buffer so the reduce is fully in place
        # (no stage-in, no copy-out). Only valid for the bf16 (unquantized)
        # wo_ud; the fp4 path (fp4_gemm has no out=) falls back to the plain
        # call. Bias is fused only on tp_rank 0 -- matching RowParallelLinear --
        # so it is added exactly once after the reduce.
        tp = get_parallel().attn_tp_group
        wo_ud = self.wo_ud
        # A LoRA-wrapped wo_ud must take the plain call below: the fused GEMM->AR-buffer
        # path drops the LoRA delta (and the wrapper doesn't forward quant_method).
        is_lora_wrapped = not isinstance(wo_ud, RowParallelLinear)
        buf = (
            None
            if is_lora_wrapped
            else get_ar_buffer(tp, num_tokens, self.hidden_size, attn_output.dtype)
        )
        if buf is not None and type(wo_ud.quant_method) is UnquantizedLinearMethod:
            torch.matmul(attn_output, wo_ud.weight.t(), out=buf)
            bias_ = None if (wo_ud.tp_rank > 0 or wo_ud.skip_bias_add) else wo_ud.bias
            if bias_ is not None:
                buf.add_(bias_)
            if not reduce:
                return buf
            if self.scattered_sconv:
                # Scattered sconv: reduce + scatter hidden -> the [T, H/P] shard
                # that attn_sconv consumes; the layer all-gathers after the sconv.
                return reduce_scatter_hidden(buf, tp, input_is_ar_buffer=True)
            return symm_mem_all_reduce(buf, tp, input_is_ar_buffer=True)

        result, _ = wo_ud(attn_output)
        if not reduce:
            return result
        if self.scattered_sconv:
            return reduce_scatter_hidden(result, tp)
        return symm_mem_all_reduce(result, tp)
