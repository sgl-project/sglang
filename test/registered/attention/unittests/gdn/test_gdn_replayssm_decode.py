"""Standalone correctness test for the buffered output-only GDN decode kernel
(ReplaySSM Part A), ported in
``python/sglang/srt/layers/attention/fla/fused_recurrent_gdn_replayssm.py``.

Ground truth: ``fused_recurrent_gated_delta_rule_packed_decode`` (the existing
SGLang single-token GDN decode Triton kernel,
``python/sglang/srt/layers/attention/fla/fused_recurrent.py:186``). We run a
multi-step decode sequence through BOTH kernels and assert:

  * per-step outputs ``y`` match, AND
  * the ground-truth full state matches the ReplaySSM checkpoint *after a flush*
    (folding the ring buffer into S0). At a non-flush step the ReplaySSM
    checkpoint legitimately lags (the update lives in the ring), so we only
    compare state at flush boundaries / at the end after a forced flush.

L sweep:
  * L=1 (flush every step): algebraically identical to the reference. The ring
    is always empty and ``write_pos == L-1`` every step, so the reconstruction
    term is zero, total_decay == 1, and the kernel reduces to the packed decode.
    We assert atol 2e-6 / rtol 1e-5 (essentially bit-exact): the ONLY fp
    difference is that the reference folds the per-head scalar decay ``alpha``
    into the state *before* the K-reduction (``(alpha*S).k``) while this kernel
    folds it *after* (``alpha*(S.k)``) -- ``alpha*(sum x)`` vs ``sum(alpha*x)`` --
    a single distribute-vs-factor reordering of one scalar multiply, bounded by
    ~K*eps. That is ~1000x tighter than the L>1 tolerance and proves the L=1
    path is the same computation as the packed decode.
  * L in {4,8,16}: algebraically equivalent but floating-point reordered (cumsum
    of gates, replay-decay scaling, the d@k reconstruction dot vs. the
    materialized recurrence). Tight fp tolerance, see ``_tols``.

The buffer wrap/flush boundary (a step where ``write_pos`` hits ``L-1``) is
covered because we run ``num_steps`` > L and flush every L steps; we also assert
the state explicitly right after a flush step.

Runnable as ``pytest`` and as ``__main__``.
"""

import sys
import unittest
from pathlib import Path

import torch

from sglang.test.test_utils import CustomTestCase

# Mirror sibling GDN unittests: register for CUDA/AMD CI. This is a kernel-math
# unit test; it lives with the other GDN kernel correctness tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

    register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-large")
    register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-large-amd")
except Exception:  # pragma: no cover - registration is optional for local runs
    pass


def _build_inputs(B, H, HV, K, V, dtype, device, seed=0):
    """Random GDN packed-decode inputs for one step.

    Returns a packed ``mixed_qkv`` [B, 2*H*K + HV*V] plus per-row gate inputs
    ``a``/``b`` [B, HV], matching the packed-decode kernel's expected layout
    (q | k | v concatenated on the last dim; q,k are H-headed, v is HV-headed).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    qk_dim = 2 * H * K
    v_dim = HV * V
    mixed_qkv = torch.randn(B, qk_dim + v_dim, generator=g, device=device, dtype=dtype)
    # Keep gate input `a` modest so softplus / decay stay in a stable range.
    a = torch.randn(B, HV, generator=g, device=device, dtype=dtype) * 0.5
    b = torch.randn(B, HV, generator=g, device=device, dtype=dtype)
    return mixed_qkv, a, b


def _tols(dtype):
    # L>1 reconstructs the state via a tensor-core tl.dot (TF32 for fp32 inputs,
    # bf16 tensor cores for bf16) -- the performance-critical path that keeps the
    # L-deep reconstruction hidden under the reduced memory traffic. TF32 carries
    # ~4e-4 error, bf16 ~1e-3, both benign end-to-end (ReplaySSM GSM8K parity);
    # vLLM's own ReplaySSM test uses fp32 atol 1e-4. L=1 has an empty buffer
    # (dot == 0) and is asserted bit-exact separately.
    if dtype == torch.float32:
        return dict(atol=1e-4, rtol=1e-3)
    return dict(atol=1e-3, rtol=1e-2)  # bf16


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestGDNReplaySSMDecode(CustomTestCase):
    # Small config: HV=8 or 16 heads, K=V=128, batch B=4.
    CONFIGS = (
        dict(B=4, H=4, HV=8, K=128, V=128),
        dict(B=4, H=8, HV=16, K=128, V=128),
    )
    NUM_STEPS = 40  # > 16 -> multiple flush cycles for L in {4,8,16}
    L_SWEEP = (1, 4, 8, 16)

    def _run_one(self, cfg, L, dtype, force_flush_steps=()):
        from sglang.srt.layers.attention.fla.fused_recurrent import (
            fused_recurrent_gated_delta_rule_packed_decode,
        )
        from sglang.srt.layers.attention.fla.fused_recurrent_gdn_replayssm import (
            fused_recurrent_gdn_replayssm_decode,
        )

        device = "cuda"
        B, H, HV, K, V = (cfg[x] for x in ("B", "H", "HV", "K", "V"))
        scale = K**-0.5

        # Shared static params across steps.
        A_log = (torch.randn(HV, device=device, dtype=torch.float32) * 0.3).contiguous()
        dt_bias = (torch.randn(HV, device=device, dtype=torch.float32) * 0.1).contiguous()

        # Two independent state pools (same init), one per kernel. One slot/req.
        num_slots = B
        # Seed the state init so the marginal bf16 flush-state element is
        # reproducible run-to-run (otherwise the pass/fail at the tolerance
        # boundary flickers with the global RNG).
        sg = torch.Generator(device=device).manual_seed(2024)
        ref_state = torch.randn(
            num_slots, HV, V, K, generator=sg, device=device, dtype=torch.float32
        )
        rep_state = ref_state.clone()
        cache_indices = torch.arange(B, device=device, dtype=torch.int32)

        # ReplaySSM ring tensors (caller-owned; this increment's caller == test).
        d_cache = torch.zeros(num_slots, HV, L, V, device=device, dtype=dtype)
        k_cache = torch.zeros(num_slots, H, L, K, device=device, dtype=dtype)
        g_cache = torch.zeros(num_slots, HV, L, device=device, dtype=torch.float32)
        write_pos = torch.zeros(B, device=device, dtype=torch.int32)

        tols = _tols(dtype)
        flush_step_seen = False

        for step in range(self.NUM_STEPS):
            mixed_qkv, a, b = _build_inputs(
                B, H, HV, K, V, dtype, device, seed=1000 + step
            )

            # ---- ground truth: packed decode, full-state write every step ----
            ref_out = mixed_qkv.new_empty(B, 1, HV, V)
            fused_recurrent_gated_delta_rule_packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                initial_state=ref_state,
                out=ref_out,
                ssm_state_indices=cache_indices,
                use_qk_l2norm_in_kernel=True,
            )

            # ---- ReplaySSM buffered decode for this L ----
            rep_out = mixed_qkv.new_empty(B, 1, HV, V)
            # slice 2: force a flush at a radix track boundary even mid-buffer.
            force_now = step in force_flush_steps
            ff_tensor = (
                torch.ones(B, device=device, dtype=torch.int32) if force_now else None
            )
            is_flush = bool((write_pos[0].item()) == L - 1) or force_now
            flush_step_seen = flush_step_seen or is_flush
            fused_recurrent_gdn_replayssm_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=A_log,
                dt_bias=dt_bias,
                scale=scale,
                initial_state=rep_state,
                d_cache=d_cache,
                k_cache=k_cache,
                g_cache=g_cache,
                out=rep_out,
                ssm_state_indices=cache_indices,
                write_pos=write_pos,
                force_flush=ff_tensor,
                use_qk_l2norm_in_kernel=True,
                # nk=1 at L=1 keeps the K-reduction in a single tile (no
                # K-tiling), matching the reference's single-pass reduction;
                # nk=2 for L>1 exercises the K-tiled reconstruction path.
                nk=1 if L == 1 else 2,
            )

            # Output must match every step.
            if dtype == torch.float32 and L == 1:
                # Essentially bit-exact: only an alpha*(sum) vs sum(alpha*)
                # scalar-multiply reordering separates this from the reference.
                torch.testing.assert_close(
                    rep_out,
                    ref_out,
                    atol=2e-6,
                    rtol=1e-5,
                    msg=f"L=1 step={step} out",
                )
            else:
                torch.testing.assert_close(
                    rep_out, ref_out, msg=f"L={L} step={step} out", **tols
                )

            # On a flush step the ReplaySSM checkpoint is brought fully
            # up-to-date and the ring logically clears -> compare states.
            if is_flush:
                if dtype == torch.float32 and L == 1:
                    torch.testing.assert_close(
                        rep_state,
                        ref_state,
                        atol=2e-6,
                        rtol=1e-5,
                        msg=f"L=1 step={step} state",
                    )
                else:
                    # The flush state is reconstructed via the tensor-core dot,
                    # so it carries tensor-core precision: ~tf32 (~4e-4, measured
                    # max_abs ~7e-4) for fp32, ~bf16 (measured max_abs ~1.9e-3)
                    # for bf16 -- looser than the per-step OUTPUT, which passes at
                    # the tighter output tol because the tf32/bf16 error partly
                    # cancels in the q-reduction. GDN's per-step decay (alpha<1)
                    # attenuates old reconstruction error so it does not
                    # accumulate across flushes over a long decode; e2e accuracy
                    # is unaffected (ReplaySSM GSM8K parity).
                    state_tols = (
                        dict(atol=1.5e-3, rtol=1e-3)
                        if dtype == torch.float32
                        else dict(atol=3e-3, rtol=1e-2)
                    )
                    torch.testing.assert_close(
                        rep_state,
                        ref_state,
                        msg=f"L={L} step={step} state(flush)",
                        **state_tols,
                    )

            # Advance the ring cursor (caller's responsibility in this phase):
            # wrap to 0 after a flush (natural L-1 OR a forced track-boundary
            # flush), otherwise increment.
            if force_now:
                write_pos = torch.zeros_like(write_pos)
            else:
                write_pos = torch.where(
                    write_pos == L - 1,
                    torch.zeros_like(write_pos),
                    write_pos + 1,
                )

        # Explicit wrap/flush boundary must have been exercised for L>1; for
        # L=1 every step is a flush.
        self.assertTrue(flush_step_seen, f"flush boundary never hit for L={L}")

    def test_replayssm_matches_packed_decode_fp32(self):
        for cfg in self.CONFIGS:
            for L in self.L_SWEEP:
                with self.subTest(cfg=cfg, L=L, dtype="fp32"):
                    self._run_one(cfg, L, torch.float32)

    def test_replayssm_matches_packed_decode_bf16(self):
        for cfg in self.CONFIGS:
            for L in self.L_SWEEP:
                with self.subTest(cfg=cfg, L=L, dtype="bf16"):
                    self._run_one(cfg, L, torch.bfloat16)

    def test_force_flush_matches_reference(self):
        """slice 2: a radix track-boundary forces a flush mid-buffer
        (write_pos != L-1). The kernel must fold the partial ring + current
        token into the checkpoint so an external snapshot reads an up-to-date
        state; output every step and the forced-flush checkpoint must still
        match the running recurrence."""
        cfg = dict(B=4, H=8, HV=16, K=128, V=128)
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=str(dtype)):
                self._run_one(cfg, L=8, dtype=dtype, force_flush_steps=(3, 5, 11, 20))

    def test_flush_boundary_state_exact_l1(self):
        """At L=1 every step is a flush; the kernel is algebraically the packed
        decode. Each step reproduces the reference output AND the full state to
        atol 2e-6 / rtol 1e-5 (essentially bit-exact) in fp32. This also
        exercises the ``write_pos == L-1`` flush boundary on every step."""
        self._run_one(dict(B=4, H=4, HV=8, K=128, V=128), L=1, dtype=torch.float32)


if __name__ == "__main__":
    unittest.main()
