"""GDN chunked prefill on ROCm through AITER.

Two paths: the opt_vk stages (prepare, hidden-state kernel, output kernel),
mirroring ``aiter.ops.triton.gated_delta_net.chunk_gated_delta_rule_opt_vk``
but called stage by stage because that entry point drops the per-chunk states
``h`` that radix state tracking reads; and a single fused FlyDSL kernel that
keeps the state on chip and writes the tracked chunk-boundary state itself.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

logger = logging.getLogger(__name__)

H_KERNELS = ("hip", "flydsl", "triton", "fused")
FUSED = "fused"


class AiterGDNPrefill:
    """AITER GDN prefill: the opt_vk stages, or the single fused FlyDSL kernel."""

    def __init__(self, h_kernel: str):
        if h_kernel not in H_KERNELS:
            raise ValueError(
                f"SGLANG_AITER_GDN_PREFILL_H must be one of {H_KERNELS}, "
                f"got {h_kernel!r}"
            )
        self._h_kernel = h_kernel
        # Only the fused kernel writes the fp32 chunk-boundary snapshot itself.
        self.writes_track_state = h_kernel == FUSED
        if h_kernel == FUSED:
            from aiter.ops.flydsl.gdn_fused_prefill import (
                gdn_fused_prefill_fwd,
                gdn_fused_prefill_supported,
            )

            self._fused = gdn_fused_prefill_fwd
            self._fused_supported = gdn_fused_prefill_supported
            logger.info("Using AITER fused FlyDSL GDN chunked prefill")
            return
        from aiter.ops.flydsl.linear_attention_prefill_kernels import (
            gdn_prepare_flydsl_supported,
            gdn_prepare_fwd_flydsl,
        )
        from aiter.ops.triton._triton_kernels.gated_delta_net.prefill.chunk import (
            chunk_fwd_o_opt_vk,
            chunk_gated_delta_rule_fwd_h_opt_vk,
            fused_chunk_local_cumsum_scaled_dot_kkt_fwd,
            fused_solve_tril_recompute_w_u,
        )
        from aiter.ops.triton._triton_kernels.gated_delta_net.utils import (
            build_gated_delta_rule_prefill_metadata,
            l2norm_fwd,
        )

        self._prepare_supported = gdn_prepare_flydsl_supported
        self._prepare_flydsl = gdn_prepare_fwd_flydsl
        self._cumsum_kkt = fused_chunk_local_cumsum_scaled_dot_kkt_fwd
        self._solve_w_u = fused_solve_tril_recompute_w_u
        self._fwd_o = chunk_fwd_o_opt_vk
        self._build_metadata = build_gated_delta_rule_prefill_metadata
        self._l2norm = l2norm_fwd
        if h_kernel == "hip":
            from aiter.ops.chunk_gated_delta_rule_fwd_h import (
                chunk_gated_delta_rule_fwd_h_hip_fn,
            )

            self._fwd_h = chunk_gated_delta_rule_fwd_h_hip_fn
        elif h_kernel == "flydsl":
            from aiter.ops.flydsl.linear_attention_prefill_kernels import (
                chunk_gated_delta_rule_fwd_h_flydsl_opt,
            )

            self._fwd_h = chunk_gated_delta_rule_fwd_h_flydsl_opt
        else:
            self._fwd_h = chunk_gated_delta_rule_fwd_h_opt_vk
        self._metadata_key: tuple[int, ...] | None = None
        self._metadata = None
        self._metadata_cu_seqlens: torch.Tensor | None = None
        logger.info("Using AITER opt_vk GDN prefill (hidden state: %s)", h_kernel)

    def _prefill_metadata(self, seq_lens_cpu: Sequence[int], cu_seqlens: torch.Tensor):
        # Built from host lengths only, so one build serves every GDN layer of a
        # forward without a device-to-host read. AITER binds the metadata to the
        # exact cu_seqlens tensor object, hence the identity check.
        key = tuple(int(n) for n in seq_lens_cpu)
        if key != self._metadata_key or cu_seqlens is not self._metadata_cu_seqlens:
            self._metadata = self._build_metadata(
                key, cu_seqlens=cu_seqlens, chunk_size=64
            )
            self._metadata_key = key
            self._metadata_cu_seqlens = cu_seqlens
        return self._metadata

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
        cu_seqlens: torch.Tensor,
        seq_lens_cpu: Sequence[int],
        track_state: torch.Tensor | None = None,
        track_chunk_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """Run prefill with the pool state updated in place; returns (o, h).

        q/k: [1, T, Hg, K]; v: [1, T, H, V]; g (natural-log decay), beta:
        [1, T, H]; ssm_states: [slots, H, V, K]; h: [1, num_chunks, H, V, K],
        or None from the fused kernel, which instead fills ``track_state``
        (fp32 [N, H, V, K]) at ``track_chunk_idx`` when those are given.
        Returns None, touching nothing, when the fused kernel declines the
        operands.
        """
        if self._h_kernel == FUSED:
            if not self._fused_supported(q, k, v, g, beta, ssm_states):
                return None
            o = self._fused(
                q,
                k,
                v,
                g,
                beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                cu_seqlens=cu_seqlens,
                track_state=track_state,
                track_chunk_idx=track_chunk_idx,
            )
            return o, None
        metadata = self._prefill_metadata(seq_lens_cpu, cu_seqlens)
        # q/k may be strided views of the packed projection; AITER's l2norm
        # flattens with view().
        q, _ = self._l2norm(q.contiguous())
        k, _ = self._l2norm(k.contiguous())
        v = v.contiguous()
        # Padded rows carry -1; slot 0 is the reserved padding slot, and the HIP
        # kernel does not guard negative slots.
        state_indices = cache_indices.clamp(min=0).to(torch.int32)

        if self._prepare_supported(k, v):
            w, u, g_cumsum = self._prepare_flydsl(
                k=k,
                v=v,
                g=g,
                beta=beta,
                cu_seqlens=cu_seqlens,
                use_exp2=True,
                prefill_metadata=metadata,
            )
        else:
            g_cumsum, A_raw = self._cumsum_kkt(
                k=k,
                beta=beta,
                g=g,
                cu_seqlens=cu_seqlens,
                use_exp2=True,
                prefill_metadata=metadata,
            )
            w, u = self._solve_w_u(
                A_raw=A_raw,
                k=k,
                v=v,
                beta=beta,
                g_cumsum=g_cumsum,
                cu_seqlens=cu_seqlens,
                use_exp2=True,
                prefill_metadata=metadata,
            )

        h_kwargs = dict(
            k=k,
            w=w,
            u=u,
            g=g_cumsum,
            initial_state=ssm_states,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            state_dtype=ssm_states.dtype,
            use_exp2=True,
            prefill_metadata=metadata,
            initial_state_indices=state_indices,
            inplace_final_state=True,
        )
        if self._h_kernel != "triton":
            h_kwargs["g_head_major"] = True
        h, v_new, _ = self._fwd_h(**h_kwargs)

        o = self._fwd_o(
            q=q,
            k=k,
            v=v_new,
            o=v.new_empty(v.shape),
            h=h,
            g=g_cumsum,
            scale=k.shape[-1] ** -0.5,
            cu_seqlens=cu_seqlens,
            use_exp2=True,
            prefill_metadata=metadata,
        )
        return o, h
