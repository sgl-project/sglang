import inspect
import logging

import torch

from sglang.kernels.ops.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from sglang.kernels.ops.attention.fla.chunk_o import chunk_fwd_o
from sglang.kernels.ops.attention.fla.cumsum import chunk_local_cumsum
from sglang.kernels.ops.attention.fla.index import prepare_chunk_indices
from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

logger = logging.getLogger(__name__)

CHUNK_SIZE = 64

# aiter FlyDSL GDN prefill state kernel. Guarded: if aiter (or its flydsl GDN
# op) is unavailable, ``FlyDSLGDNKernel.extend`` transparently falls back to the
# Triton pipeline it inherits from ``TritonGDNKernel``.
try:
    from aiter.ops.flydsl.linear_attention_prefill_kernels import (
        chunk_gated_delta_rule_fwd_h_flydsl,
    )

    _FLYDSL_AVAILABLE = True
except Exception as e:  # pragma: no cover - import-time capability probe
    chunk_gated_delta_rule_fwd_h_flydsl = None
    _FLYDSL_AVAILABLE = False
    logger.warning(
        "FlyDSL GDN kernel unavailable (%s); FlyDSLGDNKernel.extend will fall "
        "back to the Triton chunk_gated_delta_rule path.",
        repr(e),
    )

# aiter FlyDSL GDN decode (recurrent) kernel. Separately guarded so a build with
# only the prefill kernel still gets the Phase-A prefill win; decode then falls
# back to the inherited Triton packed/split decode.
try:
    from aiter.ops.flydsl.linear_attention_kernels import flydsl_gdr_decode

    _FLYDSL_DECODE_AVAILABLE = True
    # ``qkv_contiguous=False`` lets the kernel read the strided slices of the
    # packed mixed_qkv projection in place instead of making three contiguous
    # copies (ROCm/aiter#4550). Older aiter has no such argument; probe for it
    # so this backend works on both, and picks up the zero-copy path for free
    # once the installed aiter is new enough.
    _FLYDSL_DECODE_STRIDED = (
        "qkv_contiguous" in inspect.signature(flydsl_gdr_decode).parameters
    )
except Exception as e:  # pragma: no cover - import-time capability probe
    flydsl_gdr_decode = None
    _FLYDSL_DECODE_AVAILABLE = False
    _FLYDSL_DECODE_STRIDED = False
    logger.warning(
        "FlyDSL GDN decode kernel unavailable (%s); FlyDSLGDNKernel.decode will "
        "fall back to the Triton recurrent decode path.",
        repr(e),
    )


class FlyDSLGDNKernel(TritonGDNKernel):
    """GDN linear-attention kernel with the FlyDSL prefill state (``fwd_h``) core.

    Only the prefill ``extend`` path is overridden: it reproduces the Triton
    ``chunk_gated_delta_rule`` pipeline (cumsum -> intra(kkt+solve+recompute w/u)
    -> fwd_h -> chunk_fwd_o) but swaps the ``chunk_gated_delta_rule_fwd_h`` step
    for aiter's FlyDSL ``chunk_gated_delta_rule_fwd_h_flydsl``, which is ~1.8-3x
    faster at the model's TP-sharded shapes and bit-identical to Triton (verified
    across head counts, sequence lengths, and initial-state configurations).

    The ``decode`` path is also overridden to use aiter's FlyDSL recurrent
    decode (``flydsl_gdr_decode``). It is CUDA-graph-safe: ``need_shuffle_state``
    is False (the sglang ssm pool is already ``[N, H, K, V]``, so no whole-pool
    permute/copy), q/k/v/a/b are reshaped with pure ``view`` (no copies), and the
    kernel is ``@lru_cache``-compiled during the graph runner's pre-capture
    warmup forwards. ``supports_packed_decode`` is False so the backend routes to
    this split-input decode instead of the Triton packed fast path.

    ``target_verify`` is inherited unchanged from ``TritonGDNKernel``.
    """

    supports_flydsl_extend: bool = _FLYDSL_AVAILABLE
    supports_flydsl_decode: bool = _FLYDSL_DECODE_AVAILABLE
    # Route the GDN backend to the split-input ``decode`` path (below) rather
    # than the Triton packed_decode fast path, so we can call flydsl_gdr_decode.
    supports_packed_decode: bool = False

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
        if not _FLYDSL_AVAILABLE:
            return super().extend(
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

        cu_seqlens = query_start_loc

        # use_qk_l2norm_in_kernel=True in the Triton path -> l2-normalize q/k.
        q = l2norm_fwd(q)
        k = l2norm_fwd(k)

        scale = q.shape[-1] ** -0.5

        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
            if cu_seqlens is not None
            else None
        )

        # Cumulative log-decay per chunk (natural-log space).
        g = chunk_local_cumsum(
            g,
            chunk_size=CHUNK_SIZE,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

        # fused kkt + solve_tril + recompute_w_u (Triton, unchanged).
        w, u, A = chunk_gated_delta_rule_fwd_intra(
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

        # Gather per-sequence initial states from the pool. FlyDSL fwd_h takes
        # already-gathered [N, H, K, V] states (no initial_state_indices arg)
        # and does not update the pool in place, so we scatter the result back
        # ourselves after the call.
        init_state = ssm_states[cache_indices]

        # FlyDSL fwd_h expects head-major (VK) layouts for w/u/g; k stays
        # [B, T, Hg, K]. g is passed in natural-log space (use_exp2=False),
        # matching chunk_local_cumsum output. Bit-identical to Triton fwd_h.
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h_flydsl(
            k=k,
            w=w.permute(0, 2, 1, 3).contiguous(),
            u=u.permute(0, 2, 1, 3).contiguous(),
            g=g.permute(0, 2, 1).contiguous(),
            initial_state=init_state,
            output_final_state=True,
            chunk_size=CHUNK_SIZE,
            cu_seqlens=cu_seqlens,
            use_exp2=False,
        )

        # v_new comes back head-major [B, H, T, V] -> [B, T, H, V] for chunk_fwd_o.
        v_new = v_new.permute(0, 2, 1, 3).contiguous()

        o = chunk_fwd_o(
            q=q,
            k=k,
            v=v_new,
            h=h,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
        )

        # In-place state write-back: the Triton fwd_h uses INPLACE_UPDATE and the
        # GPU caller only scatters last_recurrent_state on NPU, so mirror the
        # in-place contract by writing the final state into the pool here.
        ssm_states[cache_indices] = final_state.to(ssm_states.dtype, copy=False)

        # Match TritonGDNKernel.extend's return contract: (o, last_state, h).
        # last_state is None on GPU (state already written in place); h is the
        # per-chunk state tensor consumed by mamba-state radix tracking.
        return o.to(q.dtype), None, h

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
        if not _FLYDSL_DECODE_AVAILABLE:
            return super().decode(
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

        # sglang decode inputs: q/k [1, bs, Hk, Dk], v [1, bs, Hv, Dv],
        # a/b [bs, Hv]. flydsl_gdr_decode wants [bs, 1, H, D] and [bs, 1, Hv].
        # dim0 == 1, so these reshapes are pure views (no copy) -- q/k/v stay in
        # their packed-mixed_qkv strided layout (batch stride = qkv_dim).
        bs = q.shape[1]
        Hk, Dk = q.shape[2], q.shape[3]
        Hv, Dv = v.shape[2], v.shape[3]
        query = q.reshape(bs, 1, Hk, Dk)
        key = k.reshape(bs, 1, Hk, Dk)
        value = v.reshape(bs, 1, Hv, Dv)
        a_ = a.reshape(bs, 1, Hv)
        b_ = b.reshape(bs, 1, Hv)

        out = value.new_empty(bs, 1, Hv, Dv)

        # flydsl_gdr_decode indexes/updates ``state`` by ``indices`` internally,
        # so we pass the full pool + slot indices (no gather/scatter). The pool is
        # already [N, H, K, V] (== flydsl's expected VK layout), so
        # need_shuffle_state=False -> no whole-pool permute/copy (fast + graph
        # safe). Where the installed aiter supports it, qkv_contiguous=False
        # reads the strided (split-of-mixed_qkv) q/k/v directly -- this "fuses"
        # the split (no .contiguous() copies) and is CUDA-graph safe. Older
        # aiter copies instead; both paths are numerically identical.
        # dt_bias must match the activation dtype (kernel assertion).
        flydsl_gdr_decode(
            query=query,
            key=key,
            value=value,
            a=a_,
            b=b_,
            dt_bias=dt_bias.to(value.dtype),
            A_log=A_log,
            indices=cache_indices.to(torch.int32),
            state=ssm_states,
            out=out,
            use_qk_l2norm=True,
            need_shuffle_state=False,
            **({"qkv_contiguous": False} if _FLYDSL_DECODE_STRIDED else {}),
        )

        # [bs, 1, Hv, Dv] -> [1, bs, Hv, Dv] (pure view) to match the Triton
        # decode output layout consumed by the caller.
        return out.reshape(1, bs, Hv, Dv)
