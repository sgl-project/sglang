"""gfx950 correctness for the row-streaming KDA packed decode."""

import unittest

import torch

from sglang.kernels.ops.attention.fla import kda_packed_decode_rowstream as rowstream
from sglang.kernels.ops.attention.fla.fused_recurrent import (
    fused_recurrent_kda_packed_decode,
)
from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(
    est_time=120,
    stage="stage-b",
    runner_config="1-gpu-small-amd-mi35x",
)

K = V = 128
SLOTS = 70


def _inputs(batch, heads, seed, pad, state_dtype=torch.float32):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.float32, s=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * s).to(dtype)

    mixed = randn(batch, 2 * heads * K + heads * V, dtype=torch.bfloat16)
    a = randn(batch, heads * K, dtype=torch.bfloat16)
    b = randn(batch, heads, dtype=torch.bfloat16)
    A_log = randn(heads, s=0.1)
    dt_bias = randn(heads * K, s=0.1)
    slots = max(SLOTS, batch + 6)
    state = randn(slots, heads, V, K, dtype=state_dtype, s=0.1)
    idx = torch.randperm(slots, generator=g, device="cuda")[:batch].to(torch.int32)
    if pad:
        idx[batch // 2] = -1
    return mixed, a, b, A_log, dt_bias, state, idx


def _reference(mixed, a, b, A_log, dt_bias, state, idx, heads, lower_bound):
    n = mixed.shape[0]
    out = fused_sigmoid_gating_delta_rule_update(
        A_log=A_log,
        dt_bias=dt_bias,
        q=mixed[:, : heads * K].view(1, n, heads, K),
        k=mixed[:, heads * K : 2 * heads * K].view(1, n, heads, K),
        v=mixed[:, 2 * heads * K :].view(1, n, heads, V),
        a=a.view(1, n, heads, K),
        b=b.view(1, n, heads),
        initial_state_source=state,
        initial_state_indices=idx,
        cu_seqlens=torch.arange(n + 1, device="cuda", dtype=torch.int32),
        use_qk_l2norm_in_kernel=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        is_kda=True,
        lower_bound=lower_bound,
    )
    return out.view(n, heads, V)


@unittest.skipUnless(
    torch.cuda.is_available() and is_gfx95_supported(),
    "requires one gfx950 GPU",
)
class TestKdaPackedDecodeRowstream(CustomTestCase):
    def _check(
        self,
        batch,
        heads,
        lower_bound,
        pad,
        steps=1,
        seed=0,
        state_dtype=torch.float32,
    ):
        mixed, a, b, A_log, dt_bias, state0, idx = _inputs(
            batch, heads, seed, pad, state_dtype
        )
        ref_state, state = state0.clone(), state0.clone()
        out = torch.empty(batch, 1, heads, V, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(rowstream.covered(mixed, a, b, state, out, idx))
        for step in range(steps):
            if step:
                mixed, a, b = _inputs(batch, heads, seed + step, pad)[:3]
            ref_out = _reference(
                mixed, a, b, A_log, dt_bias, ref_state, idx, heads, lower_bound
            )
            fused_recurrent_kda_packed_decode(
                mixed,
                a,
                b,
                A_log,
                dt_bias,
                K**-0.5,
                state,
                out,
                idx,
                use_qk_l2norm_in_kernel=True,
                lower_bound=lower_bound,
            )
            live = idx >= 0
            torch.testing.assert_close(
                out.view(batch, heads, V)[live].float(),
                ref_out[live].float(),
                atol=2e-2,
                rtol=2e-2,
            )
            if pad:
                self.assertEqual(out.view(batch, heads, V)[~live].abs().max().item(), 0)
        self.assertTrue(torch.isfinite(state.float()).all())
        if state_dtype == torch.float32:
            torch.testing.assert_close(state, ref_state, atol=1e-5, rtol=1e-5)
        else:
            # Both kernels round the fp32 update to bf16; reduction order may
            # move a rounding by one ulp.
            torch.testing.assert_close(state, ref_state, atol=1e-3, rtol=1.6e-2)
        untouched = torch.ones(state.shape[0], dtype=torch.bool, device="cuda")
        untouched[idx[idx >= 0].long()] = False
        torch.testing.assert_close(state[untouched], state0[untouched], atol=0, rtol=0)

    def test_every_config_bucket(self):
        for batch in (1, 12, 16, 40, 56, 64):
            for heads in (16, 8):
                for lower_bound in (None, -5.0):
                    with self.subTest(
                        batch=batch, heads=heads, lower_bound=lower_bound
                    ):
                        self._check(batch, heads, lower_bound, pad=False)

    def test_padded_slot_and_untouched_pool(self):
        for batch in (2, 33, 64):
            with self.subTest(batch=batch):
                self._check(batch, 16, -5.0, pad=True)

    def test_state_matches_over_decode_steps(self):
        for batch in (4, 64):
            with self.subTest(batch=batch):
                self._check(batch, 16, -5.0, pad=False, steps=16)

    def test_bf16_state_every_config_bucket(self):
        buckets = {
            16: (1, 4, 5, 22, 32, 40, 48, 72, 80, 96),
            8: (1, 8, 12, 48, 56, 64, 80, 160, 192, 320, 512),
        }
        for heads, batches in buckets.items():
            for batch in batches:
                for seed in (0, 1):
                    with self.subTest(heads=heads, batch=batch, seed=seed):
                        self._check(
                            batch,
                            heads,
                            -5.0,
                            pad=False,
                            seed=seed,
                            state_dtype=torch.bfloat16,
                        )

    def test_bf16_state_padded_slot_and_decode_steps(self):
        for heads, batch in ((16, 64), (8, 128)):
            with self.subTest(heads=heads, batch=batch):
                self._check(batch, heads, -5.0, pad=True, state_dtype=torch.bfloat16)
                self._check(
                    batch,
                    heads,
                    -5.0,
                    pad=False,
                    steps=16,
                    state_dtype=torch.bfloat16,
                )

    def test_bf16_state_keeps_generic_kernel_outside_measured_cells(self):
        for heads, batch in ((16, 23), (16, 31), (4, 8), (32, 8)):
            with self.subTest(heads=heads, batch=batch):
                mixed, a, b, _, _, state, idx = _inputs(
                    batch, heads, 0, False, torch.bfloat16
                )
                out = torch.empty(
                    batch, 1, heads, V, device="cuda", dtype=torch.bfloat16
                )
                self.assertFalse(rowstream.covered(mixed, a, b, state, out, idx))
        mixed, a, b, _, _, state, idx = _inputs(26, 16, 0, False)
        out = torch.empty(26, 1, 16, V, device="cuda", dtype=torch.bfloat16)
        self.assertTrue(rowstream.covered(mixed, a, b, state, out, idx))

    def test_bf16_state_deterministic_and_graph_replay(self):
        for heads, batch in ((16, 64), (8, 64)):
            with self.subTest(heads=heads, batch=batch):
                mixed, a, b, A_log, dt_bias, state0, idx = _inputs(
                    batch, heads, 2, True, torch.bfloat16
                )
                states = [state0.clone() for _ in range(3)]
                outs = [
                    torch.empty(batch, 1, heads, V, device="cuda", dtype=torch.bfloat16)
                    for _ in range(3)
                ]

                def step(i):
                    fused_recurrent_kda_packed_decode(
                        mixed,
                        a,
                        b,
                        A_log,
                        dt_bias,
                        K**-0.5,
                        states[i],
                        outs[i],
                        idx,
                        use_qk_l2norm_in_kernel=True,
                        lower_bound=-5.0,
                    )

                step(0)
                step(1)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    step(2)
                states[2].copy_(state0)
                graph.replay()
                torch.cuda.synchronize()
                for i in (1, 2):
                    torch.testing.assert_close(states[i], states[0], atol=0, rtol=0)
                    torch.testing.assert_close(outs[i], outs[0], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
