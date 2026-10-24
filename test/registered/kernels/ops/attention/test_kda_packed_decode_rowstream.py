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


def _inputs(batch, heads, seed, pad):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, dtype=torch.float32, s=1.0):
        return (torch.randn(*shape, generator=g, device="cuda") * s).to(dtype)

    mixed = randn(batch, 2 * heads * K + heads * V, dtype=torch.bfloat16)
    a = randn(batch, heads * K, dtype=torch.bfloat16)
    b = randn(batch, heads, dtype=torch.bfloat16)
    A_log = randn(heads, s=0.1)
    dt_bias = randn(heads * K, s=0.1)
    state = randn(SLOTS, heads, V, K, s=0.1)
    idx = torch.randperm(SLOTS, generator=g, device="cuda")[:batch].to(torch.int32)
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
    def _check(self, batch, heads, lower_bound, pad, steps=1, seed=0):
        mixed, a, b, A_log, dt_bias, state0, idx = _inputs(batch, heads, seed, pad)
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
        torch.testing.assert_close(state, ref_state, atol=1e-5, rtol=1e-5)
        untouched = torch.ones(SLOTS, dtype=torch.bool, device="cuda")
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


if __name__ == "__main__":
    unittest.main()
