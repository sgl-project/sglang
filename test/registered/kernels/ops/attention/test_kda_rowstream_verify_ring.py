"""Parity for the row-streamed KDA ReplaySSM ring verify.

Checks ``kda_rowstream_verify_ring`` against the recurrent verify kernel
(``fused_sigmoid_gating_delta_rule_update`` with ``cache_ring=True``), which is
what the KDA backend dispatches today. The ring entries feed the commit-time
replay, so they must match bit for bit; the outputs come out of a reassociated
FP32 sum and are checked to chunked-vs-recurrent tolerance.
"""

import unittest

import torch

from sglang.kernels.ops.attention.linear.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.kernels.ops.attention.linear.kda_rowstream_verify import (
    kda_rowstream_verify_ring,
)
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")

_DTYPE = torch.bfloat16


def _rings(slots, H, HV, K, V, ring_len, device):
    return dict(
        replayssm_rawv=torch.zeros(
            slots, HV, ring_len, V, dtype=_DTYPE, device=device
        ),
        replayssm_rawk=torch.zeros(slots, H, ring_len, K, dtype=_DTYPE, device=device),
        replayssm_g=torch.zeros(
            slots, HV, ring_len, K, dtype=torch.float32, device=device
        ),
        replayssm_beta=torch.zeros(
            slots, HV, ring_len, dtype=torch.float32, device=device
        ),
    )


def _make_inputs(B, T, H, HV, K, V, slots, device, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    seq_len = B * T
    rnd = lambda *shape, dt=_DTYPE: torch.randn(*shape, generator=gen).to(device, dt)
    state_indices = torch.randperm(slots, generator=gen)[:B].to(torch.int32)
    # A padded request: zero state, and the ring row must stay untouched.
    state_indices[-1] = -1
    return dict(
        q=rnd(1, seq_len, H, K),
        k=rnd(1, seq_len, H, K),
        v=rnd(1, seq_len, HV, V),
        a=rnd(seq_len, HV * K) * 0.5,
        b=rnd(seq_len, HV),
        A_log=rnd(HV, dt=torch.float32) * 0.5,
        dt_bias=rnd(HV * K, dt=torch.float32) * 0.5,
        state=rnd(slots, HV, V, K, dt=torch.float32) * 0.2,
        state_indices=state_indices.to(device),
        cu_seqlens=torch.arange(0, seq_len + 1, T, dtype=torch.int32, device=device),
    )


@unittest.skipIf(not torch.cuda.is_available(), "needs a GPU")
class TestKDARowstreamVerifyRing(CustomTestCase):
    def _check(
        self,
        B,
        T,
        H,
        HV,
        K,
        V,
        lower_bound,
        ring_indices=False,
        block_v=None,
        batched_ab=False,
    ):
        device = "cuda"
        slots = B + 3
        ring_len = 2 * T
        inp = _make_inputs(B, T, H, HV, K, V, slots, device)
        ref_ring = _rings(slots, H, HV, K, V, ring_len, device)
        new_ring = _rings(slots, H, HV, K, V, ring_len, device)
        if ring_indices:
            rows = torch.arange(B, dtype=torch.int32, device=device)
            rows = torch.where(inp["state_indices"] < 0, -1, rows)
        else:
            rows = None

        if batched_ab:
            # The KDA backend hands the gating inputs over with a leading
            # batch axis of 1; both layouts must dispatch the same way.
            inp["a"] = inp["a"].unsqueeze(0)
            inp["b"] = inp["b"].unsqueeze(0)
        state_before = inp["state"].clone()
        ref = fused_sigmoid_gating_delta_rule_update(
            A_log=inp["A_log"],
            a=inp["a"],
            dt_bias=inp["dt_bias"],
            softplus_beta=1.0,
            softplus_threshold=20.0,
            q=inp["q"],
            k=inp["k"],
            v=inp["v"],
            b=inp["b"],
            initial_state_source=inp["state"],
            initial_state_indices=inp["state_indices"],
            cu_seqlens=inp["cu_seqlens"],
            use_qk_l2norm_in_kernel=True,
            is_kda=True,
            disable_state_update=True,
            intermediate_states_buffer=None,
            intermediate_state_indices=None,
            cache_steps=T,
            lower_bound=lower_bound,
            cache_ring=True,
            ring_indices=rows,
            **ref_ring,
        )
        out = kda_rowstream_verify_ring(
            inp["A_log"],
            inp["a"],
            inp["dt_bias"],
            inp["q"],
            inp["k"],
            inp["v"],
            inp["b"],
            initial_state_source=inp["state"],
            initial_state_indices=inp["state_indices"],
            cu_seqlens=inp["cu_seqlens"],
            draft_token_num=T,
            ring_indices=rows,
            lower_bound=lower_bound,
            block_v=block_v,
            **new_ring,
        )

        self.assertEqual(out.shape, ref.shape)
        self.assertGreater(ref.float().abs().max().item(), 1e-2)
        torch.testing.assert_close(out.float(), ref.float(), atol=1e-3, rtol=1.6e-2)
        # The commit fold replays these, so they must be identical.
        for name, value in new_ring.items():
            torch.testing.assert_close(value, ref_ring[name], atol=0, rtol=0, msg=name)
        # Verify never mutates the committed state pool.
        torch.testing.assert_close(inp["state"], state_before, atol=0, rtol=0)

    def test_safe_gate(self):
        self._check(B=3, T=6, H=2, HV=2, K=128, V=128, lower_bound=-5.0)

    def test_softplus_gate(self):
        self._check(B=2, T=6, H=2, HV=2, K=128, V=128, lower_bound=None, block_v=128)

    def test_gqa_small_head(self):
        self._check(B=2, T=8, H=1, HV=2, K=64, V=64, lower_bound=-5.0, block_v=32)

    def test_request_scoped_ring_indices(self):
        self._check(
            B=4, T=6, H=2, HV=2, K=128, V=128, lower_bound=-5.0, ring_indices=True
        )

    def test_batched_gating_inputs(self):
        self._check(B=3, T=6, H=2, HV=2, K=128, V=128, lower_bound=-5.0, batched_ab=True)

    def test_glm5_shape(self):
        self._check(B=20, T=6, H=64, HV=64, K=128, V=128, lower_bound=-5.0)


if __name__ == "__main__":
    unittest.main()
