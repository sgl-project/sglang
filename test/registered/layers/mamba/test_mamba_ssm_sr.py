"""GPU unit tests for Mamba1/Mamba2 stochastic rounding (selective_state_update).

Tests the hardware cvt.rs SR store in the selective_state_update kernel
(--mamba-ssm-enable-stochastic-rounding with --mamba-ssm-dtype float16/bfloat16).
cvt.rs requires SM100+; tests skip otherwise.
Run: ``python -m pytest test/registered/layers/mamba/test_mamba_ssm_sr.py``.
"""

import unittest

import torch

_HAS_SM100 = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
)


@unittest.skipUnless(_HAS_SM100, "SSM SR needs an SM100+ GPU (hardware cvt.rs)")
class TestSelectiveStateUpdateSR(unittest.TestCase):
    """SR for the Mamba1/Mamba2 decode kernel (selective_state_update)."""

    def _run(self, state_in, use_rs, seed_val, dtype):
        batch, nheads, dim, dstate = 4, 2, 32, 16
        torch.manual_seed(0)
        x = torch.randn(batch, nheads, dim, device="cuda", dtype=torch.float32)
        dt = torch.randn(batch, nheads, dim, device="cuda", dtype=torch.float32)
        A = torch.randn(nheads, dim, dstate, device="cuda", dtype=torch.float32)
        B = torch.randn(batch, 1, dstate, device="cuda", dtype=torch.float32)
        C = torch.randn(batch, 1, dstate, device="cuda", dtype=torch.float32)
        D = torch.randn(nheads, dim, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(nheads, dim, device="cuda", dtype=torch.float32)
        out = torch.empty_like(x)
        state = state_in.clone()
        seed = (
            torch.tensor([seed_val], device="cuda", dtype=torch.int64)
            if use_rs
            else None
        )
        from sglang.srt.layers.attention.mamba.ops.mamba_ssm import (
            selective_state_update,
        )

        selective_state_update(
            state,
            x,
            dt,
            A,
            B,
            C,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            out=out,
            rand_seed=seed,
            use_rs_rounding=use_rs,
            philox_rounds=10,
        )
        return state

    def test_sr_store_differs_from_rtn_and_varies(self):
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                state_in = torch.randn(
                    4, 2, 32, 16, device="cuda", dtype=dtype
                )
                rtn = self._run(state_in, use_rs=False, seed_val=0, dtype=dtype)
                sr1 = self._run(state_in, use_rs=True, seed_val=1, dtype=dtype)
                sr2 = self._run(state_in, use_rs=True, seed_val=2, dtype=dtype)
                self.assertTrue(torch.isfinite(sr1.float()).all().item())
                self.assertFalse(torch.equal(sr1, rtn))
                self.assertFalse(torch.equal(sr1, sr2))
                self.assertEqual(sr1.dtype, dtype)


if __name__ == "__main__":
    unittest.main()
