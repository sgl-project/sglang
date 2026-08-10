"""Vendored CuTe DSL GDN MTP verify with the fused ReplaySSM ring-write.

The ring-write must be a pure side channel and the ring must feed the fold a
faithful raw window, so the anchor is:

  * the verify OUTPUT is bitwise unchanged by cache_ring (both the ilp4 and
    the wide_vec kernels);
  * rawv/rawk are bitwise copies of the kernel inputs; g matches the Triton
    gating and beta matches the fp32 sigmoid to fastmath tolerance;
  * folding the ring (Triton fold kernel) reproduces the CuTe DSL kernel's
    OWN committed state (disable_state_update=False run) to bf16-ulp
    tolerance -- the mixed-numerics bound replacing the triton-vs-triton
    bitwise anchor.
"""

import unittest

import torch

from sglang.kernels.ops.attention.cutedsl_gdn_mtp_ring import gated_delta_rule_mtp
from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_fold import (
    commit_gdn_replayssm_fold_all_layers,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=24, stage="base-b", runner_config="1-gpu-large")

T, H, HV, K, V, SLOTS = 4, 4, 16, 128, 128, 16
DEVICE = "cuda"


def _case(B):
    gen = torch.Generator(device=DEVICE).manual_seed(7)

    def rnd(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=gen)

    gating = {
        "A_log": (torch.randn(HV, device=DEVICE, generator=gen) * 0.1).float(),
        "dt_bias": (torch.randn(HV, device=DEVICE, generator=gen) * 0.1).float(),
    }
    inputs = {
        "q": rnd(B, T, H, K),
        "k": rnd(B, T, H, K),
        "v": rnd(B, T, HV, V),
        "a": rnd(B, T, HV),
        "b": rnd(B, T, HV),
    }
    state0 = rnd(SLOTS, HV, V, K)
    slots = torch.arange(B, device=DEVICE, dtype=torch.int32) + 2
    rings = {
        "rawv": torch.zeros(1, SLOTS, HV, T, V, device=DEVICE, dtype=torch.bfloat16),
        "rawk": torch.zeros(1, SLOTS, H, T, K, device=DEVICE, dtype=torch.bfloat16),
        "g": torch.zeros(1, SLOTS, HV, T, device=DEVICE, dtype=torch.float32),
        "beta": torch.zeros(1, SLOTS, HV, T, device=DEVICE, dtype=torch.float32),
    }
    return gating, inputs, state0, slots, rings


def _verify(gating, inputs, state, slots, rings=None, disable_state_update=True):
    kwargs = {}
    if rings is not None:
        kwargs.update(
            cache_ring=True,
            replayssm_rawv=rings["rawv"][0],
            replayssm_rawk=rings["rawk"][0],
            replayssm_g=rings["g"][0],
            replayssm_beta=rings["beta"][0],
        )
    return gated_delta_rule_mtp(
        gating["A_log"],
        inputs["a"],
        gating["dt_bias"],
        q=inputs["q"],
        k=inputs["k"],
        v=inputs["v"],
        b=inputs["b"],
        initial_state_source=state,
        initial_state_indices=slots,
        use_qk_l2norm_in_kernel=True,
        disable_state_update=disable_state_update,
        **kwargs,
    )


class TestGdnCuteDSLRingVerify(CustomTestCase):
    def _run(self, B):
        gating, inputs, state0, slots, rings = _case(B)

        out_ref = _verify(gating, inputs, state0.clone(), slots)
        out_ring = _verify(gating, inputs, state0.clone(), slots, rings=rings)
        self.assertTrue(torch.equal(out_ref, out_ring), f"{B=}")

        for i, s in enumerate(slots.tolist()):
            self.assertTrue(
                torch.equal(rings["rawv"][0, s], inputs["v"][i].transpose(0, 1))
            )
            self.assertTrue(
                torch.equal(rings["rawk"][0, s], inputs["k"][i].transpose(0, 1))
            )

        g_ref, _ = fused_gdn_gating(
            gating["A_log"],
            inputs["a"].view(B * T, HV),
            inputs["b"].view(B * T, HV),
            gating["dt_bias"],
        )
        g_ref = g_ref.view(B, T, HV).transpose(1, 2).float()
        beta_ref = torch.sigmoid(inputs["b"].float()).transpose(1, 2)
        self.assertLess((rings["g"][0, slots.long()] - g_ref).abs().max().item(), 5e-5)
        self.assertLess(
            (rings["beta"][0, slots.long()] - beta_ref).abs().max().item(), 5e-5
        )

        state_ref = state0.clone()
        _verify(gating, inputs, state_ref, slots, disable_state_update=False)
        fold_state = state0.clone().unsqueeze(0)
        commit_gdn_replayssm_fold_all_layers(
            checkpoint_state=fold_state,
            rawv_cache=rings["rawv"],
            rawk_cache=rings["rawk"],
            g_cache=rings["g"],
            beta_cache=rings["beta"],
            ssm_state_indices=slots,
            accept_lens=torch.full((B,), T, device=DEVICE, dtype=torch.int32),
            max_cache_len=T,
            num_k_heads=H,
        )
        touched = slots.long()
        err = (
            (fold_state[0, touched].float() - state_ref[touched].float())
            .abs()
            .max()
            .item()
        )
        self.assertLess(err, 3e-2, f"{B=} fold vs own update: {err}")
        untouched = [s for s in range(SLOTS) if s not in slots.tolist()]
        self.assertTrue(torch.equal(fold_state[0, untouched], state0[untouched]))

    def test_ilp4_small_batch(self):
        self._run(1)

    def test_wide_vec_batch(self):
        self._run(8)


if __name__ == "__main__":
    unittest.main()
