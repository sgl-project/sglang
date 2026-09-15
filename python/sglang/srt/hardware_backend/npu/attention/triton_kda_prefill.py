import logging
from typing import Optional

import torch

from triton_ascend_kernels.attention.fla.kda.l2norm_kda import l2norm_fwd
from triton_ascend_kernels.attention.fla.kda.beta_sigmoid_kda import (
    fused_beta_sigmoid,
)
from triton_ascend_kernels.attention.fla.kda.chunk_delta_h_kda import (
    chunk_gated_delta_rule_fwd_h_o_fused,
)
from triton_ascend_kernels.attention.fla.kda.chunk_intra import (
    chunk_kda_fwd_intra_fused,
)
from triton_ascend_kernels.attention.fla.kda.gate import kda_gate_chunk_cumsum
from triton_ascend_kernels.attention.fla.kda.utils import prepare_chunk_indices
from triton_ascend_kernels.attention.fla.utils import RCP_LN2

logger = logging.getLogger(__name__)


def chunk_kda_fwd_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    chunk_indices: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    safe_gate: bool = True,
    lower_bound: Optional[float] = -5.0,
    state_v_first: bool = True,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    return_intermediate_states: bool = False,
):
    r"""Forward-only chunk KDA kernel for NPU (no autograd).

    Accepts raw inputs (un-normed q/k, un-activated g, un-sigmoided beta) and
    performs l2norm, beta sigmoid, and gate activation inside the kernel
    pipeline, then returns output / final_state / optional per-chunk states.

    Args:
        q: (1, T, H, K) bfloat16 — raw (pre-l2norm).
        k: (1, T, H, K) bfloat16 — raw (pre-l2norm).
        v: (1, T, H, V) bfloat16.
        g: (1, T, HV, K) bfloat16 — raw forget gate (pre-activation).
        beta: (1, T, HV) bfloat16 — raw beta logits (pre-sigmoid).
        scale: query scaling factor, usually K ** -0.5.
        initial_state: (N, H, V, K) float32 or None.
        output_final_state: whether to compute and return final recurrent state.
        cu_seqlens: (N+1,) int64 varlen sequence boundaries, or None.
        chunk_size: chunk size for the recurrence (64 or 128).
        use_qk_l2norm_in_kernel: apply l2norm to q and k before the kernel.
        use_gate_in_kernel: compute gate activation inside the kernel.
        use_beta_sigmoid_in_kernel: apply sigmoid to beta inside the kernel.
        safe_gate: enable bounded gate (clamped to [lower_bound, 0)).
        lower_bound: safe gate lower bound.
        state_v_first: state tensors have (V, K) as last two dims.
        A_log: (HV,) float32 — gate decay parameter, required when
            use_gate_in_kernel=True.
        dt_bias: (HV*K,) float32 — gate bias.
        return_intermediate_states: materialize per-chunk recurrent states h.

    Returns:
        tuple of:
            o: (1, T, H, V) bfloat16 — attention output.
            final_state: (N, H, V, K) float32 or None.
            h: (1, NT, H, V, K) bfloat16 or None — per-chunk states
               (only when return_intermediate_states=True).
    """
    if use_qk_l2norm_in_kernel:
        q, _ = l2norm_fwd(q)
        k, _ = l2norm_fwd(k)

    if use_beta_sigmoid_in_kernel:
        beta = fused_beta_sigmoid(beta, scale=1.0)

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

    # ------------------------------------------------------------------
    # Inlined chunk_kda_fwd with early tensor frees to minimise HBM peak.
    #
    # The original chunk_kda_fwd() keeps all intermediates alive until the
    # very end (line 122-129), which raises the caching-allocator
    # high-water mark by ~3 × T × HV × K × 2 bytes vs the CANN path.
    #
    # By inlining we can del tensors the moment their last consumer
    # finishes, letting the allocator reuse those blocks for downstream
    # allocations (o, h, final_state).
    # ------------------------------------------------------------------

    # Phase 1: gate activation (cumsum)
    g_org = g
    g = kda_gate_chunk_cumsum(
        g=g_org,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=RCP_LN2,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
    )
    del g_org

    # Phase 2: intra-chunk (Aqk, Akk) + WY repr (w, u, kg)
    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra_fused(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=False,
    )

    # beta and Akk are no longer needed by any downstream kernel.
    del beta, Akk

    # k is no longer read after intra phase; kg supersedes it.
    # v is no longer read after intra phase; u supersedes it.
    del k, v

    # Phase 3: fused h + o kernel
    store_h = return_intermediate_states
    o, h, v_new, final_state = chunk_gated_delta_rule_fwd_h_o_fused(
        k=kg,
        w=w,
        u=u,
        q=q,
        Aqk=Aqk,
        gk=g,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=None,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
        store_h=store_h,
        save_new_value=False,
        state_v_first=state_v_first,
    )

    # Phase 4: cleanup intermediates
    del w, u, qg, kg, Aqk, g

    if return_intermediate_states:
        return o.type_as(q), final_state, h
    return o.type_as(q), final_state, None
