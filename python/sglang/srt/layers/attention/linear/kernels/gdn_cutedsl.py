"""CuTe DSL kernels for GDN (Gated Delta Network) linear attention.

Decode path uses the existing ``cutedsl_fused_sigmoid_gating_delta_rule_update``
(works on SM90+).

Prefill (extend) path uses the ported vLLM SM100 chunkwise kernel
(``chunk_gated_delta_rule_cutedsl``). Requires SM100+ and ``head_k_dim == 128``.
"""

import logging
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.jit_kernel.cutedsl_gdn import cutedsl_fused_sigmoid_gating_delta_rule_update
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

logger = logging.getLogger(__name__)


def _is_blackwell() -> bool:
    """True iff running on SM100+ (Blackwell) where the ported kernel is valid."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 10


@triton.jit
def _l2norm_row_block(
    x, y, eps, i_t, T: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr
):
    # Byte-identical to fla.l2norm.l2norm_fwd_kernel (the D<=512 branch): fp32
    # reduction, division form (not the * rstd reciprocal), round-to-nearest store
    # in the input dtype. Kept 0-ULP so the fused q/k prologue is a pure launch
    # merge with no numeric change vs the two separate l2norm_fwd calls.
    p_x = tl.make_block_ptr(x, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    b_x = tl.load(p_x, boundary_check=(0, 1)).to(tl.float32)
    b_var = tl.sum(b_x * b_x, axis=1)
    b_y = b_x / tl.sqrt(b_var + eps)[:, None]
    p_y = tl.make_block_ptr(y, (T, D), (D, 1), (i_t * BT, 0), (BT, BD), (1, 0))
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), boundary_check=(0, 1))


@triton.jit
def l2norm_fwd_qk_kernel(
    xq, yq, xk, yk, eps, T: tl.constexpr, D: tl.constexpr, BT: tl.constexpr,
    BD: tl.constexpr,
):
    i_t = tl.program_id(0)
    # program_id(1) is CTA-uniform (0 -> q, 1 -> k) so there is no warp divergence:
    # each CTA does exactly one row-block of one tensor, same work as the original.
    if tl.program_id(1) == 0:
        _l2norm_row_block(xq, yq, eps, i_t, T, D, BT, BD)
    else:
        _l2norm_row_block(xk, yk, eps, i_t, T, D, BT, BD)


def l2norm_fwd_qk(xq: torch.Tensor, xk: torch.Tensor, eps: float = 1e-6):
    """L2-normalize q and k (last dim) in a single Triton launch.

    Replaces two ``fla.l2norm.l2norm_fwd`` calls in the CuteDSL GDN prefill
    prologue. q and k share shape/dtype (GDN uses ``num_q_heads == num_k_heads``
    heads of ``head_k_dim``), so both fold into one grid whose second axis selects
    the tensor. Result is bit-identical to the two separate calls.
    """
    assert xq.shape == xk.shape and xq.dtype == xk.dtype
    shape_og = xq.shape
    xq2 = xq.view(-1, shape_og[-1])
    xk2 = xk.view(-1, shape_og[-1])
    T, D = xq2.shape
    assert D <= 512, "only the l2norm block-ptr (D<=512) branch is fused here"
    yq = torch.empty_like(xq2)
    yk = torch.empty_like(xk2)
    assert yq.stride(-1) == 1 and yk.stride(-1) == 1
    BD = min(65536 // xq2.element_size(), triton.next_power_of_2(D))
    grid = (triton.cdiv(T, 16), 2)  # axis0 = row-block (== original i_t), axis1 = q/k
    l2norm_fwd_qk_kernel[grid](
        xq2, yq, xk2, yk, eps, T=T, D=D, BT=16, BD=BD, num_warps=8, num_stages=3
    )
    return yq.view(shape_og), yk.view(shape_og)


class CuteDSLGDNKernel(LinearAttnKernelBase):
    """CuTe DSL kernel for GDN.

    Decode: ``cutedsl_fused_sigmoid_gating_delta_rule_update`` (SM90+).
    Extend (prefill): chunkwise ``chunk_gated_delta_rule_cutedsl``
    (SM100+ only, ``head_k_dim`` must be 128). On SM90 the prefill path is
    unsupported; callers should query :attr:`supports_prefill` and fall back
    to another backend (e.g. Triton).
    """

    def __init__(self):
        # The Blackwell extend kernel uses tcgen05/TMA-bulk-swizzle features
        # that don't exist on SM90. The decode kernel does work on SM90+.
        self.supports_prefill = _is_blackwell()

        # Heavy CuteDSL imports are deferred to extend() so SM90 boxes can
        # still construct the kernel just for decode.
        self._extend_fn: Optional[callable] = None
        self._prepare_meta_fn: Optional[callable] = None

    def _ensure_extend_loaded(self, head_k_dim: int) -> None:
        if self._extend_fn is not None:
            return
        if not self.supports_prefill:
            major = (
                torch.cuda.get_device_capability()[0]
                if torch.cuda.is_available()
                else -1
            )
            raise RuntimeError(
                f"CuTe DSL GDN prefill requires SM100+ (Blackwell); got SM{major}."
            )
        if head_k_dim != 128:
            raise RuntimeError(
                f"CuTe DSL GDN prefill requires head_k_dim=128, got {head_k_dim}."
            )
        from sglang.srt.layers.attention.linear.kernels.gdn_blackwell import (
            chunk_gated_delta_rule_cutedsl,
            prepare_metadata_cutedsl,
        )

        self._extend_fn = chunk_gated_delta_rule_cutedsl
        self._prepare_meta_fn = prepare_metadata_cutedsl
        logger.info("Using CuTe DSL GDN prefill (Blackwell)")

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
        return cutedsl_fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
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
        out: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> tuple:
        head_k_dim = k.shape[-1]
        self._ensure_extend_loaded(head_k_dim)

        total_seq_len = q.shape[1]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        # L2 norm Q/K outside the kernel (same as flashinfer path), fused into a
        # single launch (was two separate l2norm_fwd calls).
        q_norm, k_norm = l2norm_fwd_qk(q[0].contiguous(), k[0].contiguous())
        q_norm = q_norm.unsqueeze(0)
        k_norm = k_norm.unsqueeze(0)
        v_in = v[0].contiguous().unsqueeze(0)
        # Kernel expects log-space float32 gate per (token, v-head).
        g_in = g[0].to(torch.float32).unsqueeze(0)
        beta_in = beta[0].to(torch.float32).unsqueeze(0)

        cu_seqlens = query_start_loc.to(torch.int32)

        # Pool gather: remap padding (-1) to the last (sentinel) slot.
        ssm_cache_indices = torch.where(
            cache_indices >= 0,
            cache_indices,
            ssm_states.shape[0] - 1,
        ).to(torch.long)
        initial_state = ssm_states[ssm_cache_indices].contiguous()

        chunk_indices, chunk_offsets = self._prepare_meta_fn(
            cu_seqlens, total_seq_len, chunk_size=64
        )

        # Direct-write epilogue: when the caller supplies its final output slice
        # (shape [1, T, Hv, V]), the o-kernel writes [T, Hv, V] straight into it,
        # eliminating the caller's device-to-device copy.
        core_attn_out = None
        if out is not None:
            assert out.shape == (1, total_seq_len, num_v_heads, head_v_dim), (
                f"direct-write out buffer {tuple(out.shape)} != expected "
                f"{(1, total_seq_len, num_v_heads, head_v_dim)}"
            )
            core_attn_out = out.squeeze(0)

        output, final_state = self._extend_fn(
            q=q_norm,
            k=k_norm,
            v=v_in,
            g=g_in,
            beta=beta_in,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            core_attn_out=core_attn_out,
        )

        ssm_states.index_copy_(
            0,
            ssm_cache_indices,
            final_state.to(ssm_states.dtype),
        )

        # Match Triton extend interface: (output, last_recurrent_state, h).
        # We've already written state back, so no need to return it.
        return output, None, None

    def target_verify(self, *args, **kwargs):
        raise NotImplementedError("CuteDSLGDNKernel does not support target_verify")
