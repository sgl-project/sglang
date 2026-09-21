"""Parity: KDA ReplaySSM fold-commit vs the recurrent verify kernel.

The fold kernel (`kda_replayssm_exact_fold_kernel`) replays a request's accepted
draft window from a checkpoint to reconstruct the committed SSM state, replacing
the per-step `intermediate_ssm` snapshots. This test pins the fold's committed
state to the recurrent verify kernel's state (derived-property parity).

Crucially it drives BOTH sides from raw (a, b, A_log, dt_bias):
  - baseline: the verify kernel (`fused_sigmoid_gating_delta_rule_update`, is_kda)
    forms the gate INTERNALLY and caches per-step states.
  - fold: the backend forms gk/beta in torch, writes them to the ring, and the
    fold kernel replays them.
So a gate-formula mismatch is caught here. This guards a real bug: the fold
originally formed the gate with plain `softplus` while K3's checkpoint uses the
safe gate (`lower_bound * sigmoid(exp(A_log) * x)`, gate_lower_bound=-5.0),
which silently committed the wrong state (gsm8k 0.955 -> 0.947). The earlier
same-gk fold test could not catch it because it fed both sides the same gk.
Running the safe-gate case on the pre-fix code turns it red.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

if not torch.cuda.is_available():
    import pytest

    pytest.skip(
        "KDA ReplaySSM fold parity needs CUDA (triton kernels).",
        allow_module_level=True,
    )

from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (  # noqa: E402
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.fla.kda_replayssm_spec_decode import (  # noqa: E402
    commit_kda_replayssm_spec,
)


class TestKDAReplaySSMFoldParity(CustomTestCase):
    B, T, HV, K, V = 3, 6, 4, 64, 64
    H = HV  # single k-head group
    L = 16

    def _parity(self, lower_bound):
        dev = "cuda"
        B, T, HV, K, V, H, L = self.B, self.T, self.HV, self.K, self.V, self.H, self.L
        scale = K**-0.5
        torch.manual_seed(0)

        q = torch.randn(B, T, H, K, device=dev, dtype=torch.float32)
        k = torch.randn(B, T, H, K, device=dev, dtype=torch.float32)
        v = torch.randn(B, T, HV, V, device=dev, dtype=torch.float32)
        a = torch.randn(B, T, HV, K, device=dev, dtype=torch.float32)
        b = torch.randn(B, T, HV, device=dev, dtype=torch.float32)
        A_log = torch.randn(HV, device=dev, dtype=torch.float32)
        dt_bias = torch.randn(HV, K, device=dev, dtype=torch.float32)
        h0 = torch.randn(B, HV, V, K, device=dev, dtype=torch.float32)

        slots = torch.arange(1, B + 1, device=dev, dtype=torch.int32)
        num_slots = B + 1

        # baseline: verify kernel forms the gate internally, caches per-step state
        h0_src = torch.zeros(num_slots, HV, V, K, device=dev, dtype=torch.float32)
        for j in range(B):
            h0_src[slots[j]] = h0[j]
        inter = torch.zeros(num_slots, T, HV, V, K, device=dev, dtype=torch.float32)
        fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=h0_src,
            initial_state_indices=slots,
            scale=scale,
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
            lower_bound=lower_bound,
            disable_state_update=True,
            intermediate_states_buffer=inter,
            intermediate_state_indices=slots,
            cache_steps=T,
        )
        accept = T  # commit the full window
        base = torch.stack([inter[slots[j], accept - 1] for j in range(B)], 0)

        # fold: torch forms gk/beta (matching the kernel's two branches), replays
        x = a + dt_bias.view(1, 1, HV, K)
        exp_a_log = torch.exp(A_log).view(1, 1, HV, 1)
        if lower_bound is not None:
            gk = lower_bound * torch.sigmoid(exp_a_log * x)
        else:
            gk = -exp_a_log * torch.nn.functional.softplus(x)
        beta = torch.sigmoid(b)

        rawv = torch.zeros(num_slots, HV, L, V, device=dev, dtype=torch.float32)
        rawk = torch.zeros(num_slots, H, L, K, device=dev, dtype=torch.float32)
        gkr = torch.zeros(num_slots, HV, L, K, device=dev, dtype=torch.float32)
        betar = torch.zeros(num_slots, HV, L, device=dev, dtype=torch.float32)
        ckpt = torch.zeros(num_slots, HV, V, K, device=dev, dtype=torch.float32)
        for j in range(B):
            s = slots[j].item()
            rawv[s, :, :T] = v[j].transpose(0, 1)
            rawk[s, :, :T] = k[j].transpose(0, 1)
            gkr[s, :, :T] = gk[j].transpose(0, 1)
            betar[s, :, :T] = beta[j].transpose(0, 1)
            ckpt[s] = h0[j]
        acc = torch.full((B,), accept, device=dev, dtype=torch.int32)
        commit_kda_replayssm_spec(
            ckpt,
            rawv,
            rawk,
            gkr,
            betar,
            slots,
            acc,
            max_cache_len=L,
            num_k_heads=H,
            use_qk_l2norm_in_kernel=True,
        )
        fold = torch.stack([ckpt[slots[j].item()] for j in range(B)], 0)

        rel = ((fold - base).abs().max() / base.abs().max().clamp_min(1e-6)).item()
        self.assertLess(rel, 1e-3, f"fold vs verify parity failed: rel={rel:.3e}")

    def test_safe_gate(self):
        # K3's gate: g = lower_bound * sigmoid(exp(A_log) * (a + dt_bias)).
        self._parity(lower_bound=-5.0)

    def test_softplus_gate(self):
        # Plain branch: g = -exp(A_log) * softplus(a + dt_bias).
        self._parity(lower_bound=None)


if __name__ == "__main__":
    unittest.main()
