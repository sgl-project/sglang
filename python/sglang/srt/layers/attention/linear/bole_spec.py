"""Bole speculative target verify for GDN linear attention.

The default recurrent verify snapshots the full SSM state after every draft
step into ``MambaPool.intermediate_ssm`` (max_running x D x state per layer)
so the commit can copy back the accepted step. Bole instead keeps the low-rank
factors (p, k, u) of each step in a small per-batch-size workspace and rebuilds
the accepted state on commit, so ``intermediate_ssm`` is not allocated and the
memory goes to the KV pool. Linear draft chains (DFLASH / MTP topk=1) only.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Tuple

import torch
import triton

from sglang.kernels.ops.attention.linear.bole_spec_verify import (
    bole_commit_kernel,
    bole_factor_kernel,
    bole_solve_kernel,
)
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Relative-L2 bounds checked by SGLANG_BOLE_AUDIT against the recurrent kernel.
_AUDIT_OUTPUT_TOL = 2e-3
_AUDIT_STATE_TOL = {torch.float32: 2e-4, torch.bfloat16: 1e-2}


class BoleWorkspace:
    """Factor buffers for one (batch size, draft length, HK, DK) verify shape.

    One workspace exists per CUDA graph batch size; it is allocated on the
    first verify of that shape and reused by every layer and every round.
    """

    def __init__(
        self,
        temporal: torch.Tensor,
        bs: int,
        t: int,
        hk: int,
        dk: int,
        out_dtype: torch.dtype,
    ):
        self.layers, _, self.hv, self.dv, self.dk = temporal.shape
        assert self.dk == dk, (self.dk, dk)
        self.bs, self.t, self.hk = bs, t, hk
        self.bt = max(16, triton.next_power_of_2(t))
        device = temporal.device
        f32 = torch.float32
        self.mask = torch.tril(torch.ones((t, t), dtype=torch.int32)).to(device)
        self.g = torch.empty((bs, self.hv, self.bt, self.bt), device=device, dtype=f32)
        self.c = torch.empty_like(self.g)
        self.beta = torch.empty((bs, self.hv, t), device=device, dtype=f32)
        self.p = torch.empty((self.layers, bs, self.hv, t), device=device, dtype=f32)
        self.k = torch.empty((self.layers, bs, hk, t, dk), device=device, dtype=f32)
        self.u = torch.empty(
            (self.layers, bs, self.hv, t, self.dv), device=device, dtype=f32
        )
        self.output = torch.empty(
            (self.layers, bs * t, self.hv, self.dv), device=device, dtype=out_dtype
        )

    def verify(
        self,
        layer: int,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
    ) -> torch.Tensor:
        qs, ks, vs = q.stride(1), k.stride(1), v.stride(1)
        bk = triton.next_power_of_2(self.dk)
        bole_factor_kernel[(self.bs, self.hv)](
            q, k, a, b, A_log, dt_bias, self.mask, self.g, self.c,
            self.p[layer], self.beta, self.k[layer],
            self.t, self.hk, self.hv, self.dk, self.bt, bk,
            qs, ks, a.stride(-2), b.stride(-2),
            num_warps=4, num_stages=1,
        )  # fmt: skip
        bole_solve_kernel[(triton.cdiv(self.dv, 32), self.bs, self.hv)](
            q, k, v, ssm_states, cache_indices, self.g, self.c,
            self.p[layer], self.beta, self.u[layer], self.output[layer],
            self.t, self.hk, self.hv, self.dk, self.dv, self.bt, bk, 32, self.t - 1,
            qs, ks, vs, ssm_states.stride(0),
            num_warps=4, num_stages=1,
        )  # fmt: skip
        return self.output[layer].view_as(v)

    def commit(
        self,
        temporal: torch.Tensor,
        slots: torch.Tensor,
        steps: torch.Tensor,
        src_slots: Optional[torch.Tensor] = None,
    ):
        """Rebuild step ``steps[i]`` from ``src_slots[i]``'s prefix state into
        ``slots[i]`` for every layer; rows with a negative step/slot are skipped."""
        src_slots = slots if src_slots is None else src_slots
        grid = (
            triton.cdiv(self.dv, 32),
            triton.cdiv(self.dk, 32),
            self.layers * self.bs * self.hv,
        )
        bole_commit_kernel[grid](
            temporal, self.k, self.p, self.u, src_slots, slots, steps, self.mask,
            self.t, self.hk, self.hv, self.dk, self.dv, self.bt, 32, 32,
            self.bs, steps.shape[0], temporal.stride(0), temporal.stride(1),
            num_warps=4, num_stages=1,
        )  # fmt: skip


class BoleSpecVerifier:
    """Owns the Bole workspaces of one MambaPool's temporal (SSM) state."""

    def __init__(self, temporal: torch.Tensor):
        assert temporal.dtype in (
            torch.float32,
            torch.bfloat16,
        ), f"Bole spec-verify supports fp32/bf16 SSM state, got {temporal.dtype}"
        self.temporal = temporal
        self.workspaces: Dict[Tuple[int, int, int, int], BoleWorkspace] = {}
        self.audit = envs.SGLANG_BOLE_AUDIT.get()
        self._audit_snapshots: Dict[int, torch.Tensor] = {}
        self.audit_max_output_err = 0.0
        self.audit_max_state_err = 0.0
        logger.info(
            "Bole spec-verify enabled: per-draft SSM snapshots are not allocated "
            f"(state shape {list(temporal.shape)}, audit={self.audit})."
        )

    def target_verify(
        self,
        layer: int,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
    ) -> torch.Tensor:
        bs = query_start_loc.shape[0] - 1
        t = q.shape[1] // bs
        hk, dk = q.shape[-2:]
        key = (bs, t, hk, dk)
        workspace = self.workspaces.get(key)
        if workspace is None:
            workspace = BoleWorkspace(self.temporal, bs, t, hk, dk, v.dtype)
            self.workspaces[key] = workspace
        output = workspace.verify(
            layer,
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
        )
        if self.audit and not torch.cuda.is_current_stream_capturing():
            self._audit_verify(
                layer, workspace, output, A_log, dt_bias, q, k, v, a, b,
                ssm_states, cache_indices, query_start_loc,
            )  # fmt: skip
        return output

    def commit(
        self,
        buffer_bs: int,
        slots: torch.Tensor,
        steps: torch.Tensor,
        track_slots: Optional[torch.Tensor] = None,
        track_steps: Optional[torch.Tensor] = None,
    ):
        """Write the accepted state of each request back into its slot.

        ``buffer_bs`` is the (padded) batch size the verify ran with, which
        selects the workspace holding this round's factors. With mamba radix
        tracking, the interval-crossing step is first rebuilt into the track
        slot from the request slot's prefix state, before that is overwritten.
        """
        candidates = [w for key, w in self.workspaces.items() if key[0] == buffer_bs]
        assert len(candidates) == 1, (
            "Bole commit workspace selection",
            buffer_bs,
            list(self.workspaces),
        )
        workspace = candidates[0]
        if track_slots is not None:
            workspace.commit(self.temporal, track_slots, track_steps, src_slots=slots)
        workspace.commit(self.temporal, slots, steps)
        if self._audit_snapshots:
            targets = [(slots, steps)]
            if track_slots is not None:
                targets.append((track_slots, track_steps))
            self._audit_commit(targets)

    def _audit_verify(
        self, layer, workspace, output, A_log, dt_bias, q, k, v, a, b,
        ssm_states, cache_indices, query_start_loc,
    ):  # fmt: skip
        from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )

        bs, t = workspace.bs, workspace.t
        snapshots = torch.empty(
            (bs, t, workspace.hv, workspace.dv, workspace.dk),
            device=output.device,
            dtype=ssm_states.dtype,
        )
        rows = torch.arange(bs, device=output.device, dtype=torch.int32)
        expected = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            disable_state_update=True,
            intermediate_states_buffer=snapshots,
            intermediate_state_indices=rows,
            cache_steps=t,
        )
        err = float(
            (output.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-15)
        )
        self.audit_max_output_err = max(self.audit_max_output_err, err)
        assert err < _AUDIT_OUTPUT_TOL, ("Bole verify output audit", layer, err)
        self._audit_snapshots[layer] = snapshots

    def _audit_commit(self, targets):
        tol = _AUDIT_STATE_TOL[self.temporal.dtype]
        max_err = 0.0
        for slots, steps in targets:
            pairs = list(zip(slots.tolist(), steps.tolist()))
            for layer, snapshots in self._audit_snapshots.items():
                for row, (slot, step) in enumerate(pairs):
                    if slot < 0 or step < 0:
                        continue
                    expected = snapshots[row, step].float()
                    actual = self.temporal[layer, slot].float()
                    err = float(
                        (actual - expected).norm() / expected.norm().clamp_min(1e-15)
                    )
                    max_err = max(max_err, err)
                    assert err < tol, (
                        "Bole commit state audit",
                        layer,
                        slot,
                        step,
                        err,
                    )
        self._audit_snapshots.clear()
        self.audit_max_state_err = max(self.audit_max_state_err, max_err)
        logger.info(
            f"Bole audit: max output rel-L2 {self.audit_max_output_err:.3e}, "
            f"max state rel-L2 {self.audit_max_state_err:.3e}"
        )
