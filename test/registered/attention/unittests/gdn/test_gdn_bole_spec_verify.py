"""Bole spec-verify for GDN: factor verify + rebuild-on-commit.

Checks Bole against the recurrent verify with per-draft snapshots: the verify
output, the committed state of each request slot, the radix-track slot
rebuild, and that padded rows / untouched slots are left alone. Bole reorders
the floating-point math (TF32x3 dots over low-rank factors), so outputs are
compared by relative L2 error with the same bounds as SGLANG_BOLE_AUDIT.
"""

import unittest

import torch

from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.linear.bole_spec import BoleSpecVerifier
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")

LAYERS = 2
B, T = 4, 8
HK, HV = 4, 12
K = V = 128
NUM_SLOTS = 12
DEVICE = "cuda"
OUTPUT_TOL = 2e-3
STATE_TOL = {torch.float32: 2e-4, torch.bfloat16: 1e-2}


def _rel(actual, expected):
    actual, expected = actual.float(), expected.float()
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-15))


def _inputs(seed):
    gen = torch.Generator(device=DEVICE).manual_seed(seed)

    def rand(*shape, dtype=torch.bfloat16):
        return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=gen)

    return {
        "q": rand(1, B * T, HK, K),
        "k": rand(1, B * T, HK, K),
        "v": rand(1, B * T, HV, V),
        "a": rand(B * T, HV),
        "b": rand(B * T, HV),
        "A_log": rand(HV, dtype=torch.float32) * 0.5,
        "dt_bias": rand(HV, dtype=torch.float32) * 0.5,
    }


class TestGDNBoleSpecVerify(CustomTestCase):
    def _run(self, ssm_dtype, with_track):
        torch.manual_seed(0)
        temporal = (torch.randn(LAYERS, NUM_SLOTS, HV, V, K, device=DEVICE) * 0.05).to(
            ssm_dtype
        )
        before = temporal.clone()
        verifier = BoleSpecVerifier(temporal)
        # Row 3 is CUDA-graph padding (slot -1).
        slots = torch.tensor([5, 1, 9, -1], device=DEVICE, dtype=torch.int32)
        cu = torch.arange(0, (B + 1) * T, T, device=DEVICE, dtype=torch.int32)
        snapshots = []
        for layer in range(LAYERS):
            x = _inputs(seed=layer)
            snap = torch.zeros(B, T, HV, V, K, device=DEVICE, dtype=ssm_dtype)
            expected = fused_sigmoid_gating_delta_rule_update(
                A_log=x["A_log"],
                a=x["a"],
                dt_bias=x["dt_bias"],
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=x["q"],
                k=x["k"],
                v=x["v"],
                b=x["b"],
                initial_state_source=temporal[layer],
                initial_state_indices=slots,
                cu_seqlens=cu,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                intermediate_states_buffer=snap,
                intermediate_state_indices=torch.arange(
                    B, device=DEVICE, dtype=torch.int32
                ),
                cache_steps=T,
            )
            output = verifier.target_verify(
                layer,
                ssm_states=temporal[layer],
                cache_indices=slots,
                query_start_loc=cu,
                **x,
            )
            self.assertEqual(output.shape, expected.shape)
            valid = slice(0, 3 * T)  # skip the padded row
            self.assertLess(
                _rel(output[:, valid], expected[:, valid]), OUTPUT_TOL, layer
            )
            # Verify must not touch the state.
            torch.testing.assert_close(temporal, before, rtol=0, atol=0)
            snapshots.append(snap)

        steps = torch.tensor([2, 7, 0, -1], device=DEVICE, dtype=torch.int32)
        track_slots = track_steps = None
        if with_track:
            track_slots = torch.tensor(
                [7, -1, 11, -1], device=DEVICE, dtype=torch.int32
            )
            track_steps = torch.tensor([1, -1, 0, -1], device=DEVICE, dtype=torch.int32)
        verifier.commit(B, slots, steps, track_slots, track_steps)

        written = set()
        tol = STATE_TOL[ssm_dtype]
        for layer in range(LAYERS):
            for row in range(3):
                slot, step = int(slots[row]), int(steps[row])
                written.add(slot)
                err = _rel(temporal[layer, slot], snapshots[layer][row, step])
                self.assertLess(err, tol, (layer, slot, step))
            if with_track:
                for row in (0, 2):
                    slot, step = int(track_slots[row]), int(track_steps[row])
                    written.add(slot)
                    err = _rel(temporal[layer, slot], snapshots[layer][row, step])
                    self.assertLess(err, tol, ("track", layer, slot, step))
        untouched = [s for s in range(NUM_SLOTS) if s not in written]
        torch.testing.assert_close(
            temporal[:, untouched], before[:, untouched], rtol=0, atol=0
        )

    def test_fp32_state(self):
        self._run(torch.float32, with_track=False)

    def test_bf16_state(self):
        self._run(torch.bfloat16, with_track=False)

    def test_fp32_state_with_radix_track(self):
        self._run(torch.float32, with_track=True)


if __name__ == "__main__":
    unittest.main()
