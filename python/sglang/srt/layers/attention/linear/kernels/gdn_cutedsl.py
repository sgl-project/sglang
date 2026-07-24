"""CuTe DSL kernels for GDN (Gated Delta Network) linear attention.

Decode path uses the existing ``cutedsl_fused_sigmoid_gating_delta_rule_update``
(works on SM90+).

Prefill (extend) path uses the ported vLLM SM100 chunkwise kernel
(``chunk_gated_delta_rule_cutedsl``). Requires SM100+ and ``head_k_dim == 128``.
"""

import logging
from typing import Optional

import torch

from sglang.kernels.ops.attention.cutedsl_gdn import (
    cutedsl_fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.linear.kernels.kernel_backend import (
    LinearAttnKernelBase,
)

logger = logging.getLogger(__name__)


def _is_blackwell() -> bool:
    """True iff running on the SM100/SM103 family supported by this prefill kernel."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 10


class CuteDSLGDNKernel(LinearAttnKernelBase):
    """CuTe DSL kernel for GDN.

    Decode: ``cutedsl_fused_sigmoid_gating_delta_rule_update`` (SM90+).
    Extend (prefill): chunkwise ``chunk_gated_delta_rule_cutedsl``
    (SM100/SM103 only, ``head_k_dim`` must be 128). On other architectures the
    prefill path is unsupported; callers should query :attr:`supports_prefill`
    and fall back to another backend (e.g. Triton).
    """

    def __init__(self):
        # The Blackwell extend kernel uses tcgen05/TMA-bulk-swizzle features
        # that don't exist on SM90. The decode kernel does work on SM90+.
        self.supports_prefill = _is_blackwell()

        # Heavy CuteDSL imports are deferred to extend() so SM90 boxes can
        # still construct the kernel just for decode.
        self._extend_fn: Optional[callable] = None
        self._prepare_meta_fn: Optional[callable] = None
        self._l2norm_fn: Optional[callable] = None

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
                f"CuTe DSL GDN prefill requires SM100/SM103; got SM{major}."
            )
        if head_k_dim != 128:
            raise RuntimeError(
                f"CuTe DSL GDN prefill requires head_k_dim=128, got {head_k_dim}."
            )
        from sglang.kernels.ops.attention.fla.l2norm import l2norm_fwd
        from sglang.kernels.ops.attention.linear.gdn_blackwell import (
            chunk_gated_delta_rule_cutedsl,
            prepare_metadata_cutedsl,
        )

        self._extend_fn = chunk_gated_delta_rule_cutedsl
        self._prepare_meta_fn = prepare_metadata_cutedsl
        self._l2norm_fn = l2norm_fwd
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
        **kwargs,
    ) -> tuple:
        head_k_dim = k.shape[-1]
        self._ensure_extend_loaded(head_k_dim)

        total_seq_len = q.shape[1]
        num_v_heads = v.shape[2]
        head_v_dim = v.shape[3]

        # L2 norm Q/K outside the kernel (same as flashinfer path).
        q_norm = self._l2norm_fn(q[0].contiguous()).unsqueeze(0)
        k_norm = self._l2norm_fn(k[0].contiguous()).unsqueeze(0)
        v_in = v[0].contiguous().unsqueeze(0)
        # Kernel expects log-space float32 gate per (token, v-head).
        g_in = g[0].to(torch.float32).unsqueeze(0)
        beta_in = beta[0].to(torch.float32).unsqueeze(0)

        cu_seqlens = query_start_loc.to(torch.int32)

        # Pool state I/O is fused into the h kernel's TMA load/store: pass the
        # pool + per-seq slots and the kernel reads h0/writes ht in place at
        # those rows (no gather/scatter kernels, no [N, Hv, V, K] intermediates).
        # Remap padding (-1) to the last (sentinel) slot.
        ssm_cache_indices = torch.where(
            cache_indices >= 0,
            cache_indices,
            ssm_states.shape[0] - 1,
        ).to(torch.int32)

        chunk_indices, chunk_offsets = self._prepare_meta_fn(
            cu_seqlens, total_seq_len, chunk_size=64
        )

        output, _ = self._extend_fn(
            q=q_norm,
            k=k_norm,
            v=v_in,
            g=g_in,
            beta=beta_in,
            initial_state=ssm_states,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            initial_state_indices=ssm_cache_indices,
        )

        # Match Triton extend interface: (output, last_recurrent_state, h).
        # The kernel already wrote state back into the pool in place.
        return output, None, None

    def target_verify(self, *args, **kwargs):
        raise NotImplementedError("CuteDSLGDNKernel does not support target_verify")
