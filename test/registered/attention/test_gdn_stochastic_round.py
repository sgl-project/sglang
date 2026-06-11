"""GPU unit tests for GDN stochastic rounding (--mamba-ssm-enable-stochastic-rounding).

Covers the hardware cvt.rs SR helper (fp16/bf16: unbiased, on-grid, seed-controlled,
differs from RTN), the packed_decode kernel's in-place SR store, and the prefill
(chunk_delta_h) SR store. cvt.rs requires SM100+; tests skip otherwise.
Run: ``python -m pytest test/registered/attention/test_gdn_stochastic_round.py``.
"""

import unittest

import torch

_HAS_SM100 = (
    torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10
)

if _HAS_SM100:
    import triton
    import triton.language as tl

    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.srt.layers.attention.fla.fused_recurrent import (
        fused_recurrent_gated_delta_rule_packed_decode,
    )
    from sglang.srt.layers.attention.fla.stochastic_round import rs_round_state

    @triton.jit
    def _sr_test_kernel(
        x_ptr, seed_ptr, out_ptr, n, DTYPE: tl.constexpr, ROUNDS: tl.constexpr, BLOCK: tl.constexpr
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = rs_round_state(x, tl.load(seed_ptr), offs, DTYPE, ROUNDS)
        tl.store(out_ptr + offs, y, mask=mask)

    def _sr_round(x_f32, dtype, rounds=10, seed_val=12345):
        n = x_f32.numel()
        seed = torch.tensor([seed_val], device="cuda", dtype=torch.int64)
        out = torch.empty(n, device="cuda", dtype=dtype)
        tl_dtype = tl.float16 if dtype == torch.float16 else tl.bfloat16
        BLOCK = 1024
        _sr_test_kernel[(triton.cdiv(n, BLOCK),)](
            x_f32.contiguous().view(-1), seed, out, n,
            DTYPE=tl_dtype, ROUNDS=rounds, BLOCK=BLOCK,
        )
        return out.view(x_f32.shape)


@unittest.skipUnless(_HAS_SM100, "SSM SR needs an SM100+ GPU (hardware cvt.rs)")
class TestSSMStochasticRoundHelper(unittest.TestCase):
    def _unbiased_on_grid(self, dtype, spacing):
        lower, upper = 1.0, 1.0 + spacing
        x_val = 1.0 + 0.3 * spacing  # 30% toward `upper`
        x = torch.full((1_000_000,), x_val, device="cuda", dtype=torch.float32)
        y = _sr_round(x, dtype, seed_val=7).to(torch.float32)
        self.assertTrue(((y == lower) | (y == upper)).all().item())  # on grid
        self.assertAlmostEqual((y == upper).float().mean().item(), 0.3, delta=0.02)
        self.assertAlmostEqual(y.mean().item(), x_val, delta=1e-3)  # unbiased

    def test_fp16_unbiased_on_grid(self):
        self._unbiased_on_grid(torch.float16, 2.0 ** -10)

    def test_bf16_unbiased_on_grid(self):
        self._unbiased_on_grid(torch.bfloat16, 2.0 ** -7)

    def test_seed_controls_rounding(self):
        torch.manual_seed(0)
        x = torch.randn(8192, device="cuda", dtype=torch.float32)
        for dtype in (torch.float16, torch.bfloat16):
            a = _sr_round(x, dtype, seed_val=1)
            a2 = _sr_round(x, dtype, seed_val=1)
            b = _sr_round(x, dtype, seed_val=2)
            self.assertTrue(torch.equal(a, a2))  # same seed -> deterministic
            self.assertFalse(torch.equal(a, b))  # different seed -> different noise

    def test_differs_from_rtn_but_brackets_input(self):
        torch.manual_seed(0)
        x = torch.randn(8192, device="cuda", dtype=torch.float32)
        for dtype in (torch.float16, torch.bfloat16):
            sr = _sr_round(x, dtype, seed_val=3).to(torch.float32)
            rtn = x.to(dtype).to(torch.float32)
            self.assertFalse(torch.equal(sr, rtn))  # SR generally != RTN
            self.assertTrue(torch.isfinite(sr).all().item())
            self.assertLessEqual((sr - x).abs().max().item(), (rtn - x).abs().max().item() * 4 + 1e-3)


@unittest.skipUnless(_HAS_SM100, "SSM SR needs an SM100+ GPU (hardware cvt.rs)")
class TestPackedDecodeSR(unittest.TestCase):
    def _run(self, ssm_in, use_rs, seed_val, dtype):
        B, H, HV, K, V = 4, 2, 4, 8, 8
        torch.manual_seed(0)
        mixed_qkv = torch.randn(B, 2 * H * K + HV * V, device="cuda", dtype=torch.float32)
        a = torch.randn(B, HV, device="cuda", dtype=torch.float32)
        b = torch.randn(B, HV, device="cuda", dtype=torch.float32)
        A_log = torch.randn(HV, device="cuda", dtype=torch.float32)
        dt_bias = torch.randn(HV, device="cuda", dtype=torch.float32)
        out = torch.empty(B, 1, HV, V, device="cuda", dtype=torch.float32)
        idx = torch.arange(B, device="cuda", dtype=torch.int64)
        state = ssm_in.clone()
        seed = torch.tensor([seed_val], device="cuda", dtype=torch.int64) if use_rs else None
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv, a=a, b=b, A_log=A_log, dt_bias=dt_bias, scale=K ** -0.5,
            initial_state=state, out=out, ssm_state_indices=idx,
            use_qk_l2norm_in_kernel=True,
            rand_seed=seed, use_rs_rounding=use_rs, philox_rounds=10,
        )
        return state

    def test_sr_store_differs_from_rtn_and_varies(self):
        for dtype in (torch.float16, torch.bfloat16):
            ssm_in = torch.randn(4, 4, 8, 8, device="cuda", dtype=dtype)
            rtn = self._run(ssm_in, use_rs=False, seed_val=0, dtype=dtype)
            sr1 = self._run(ssm_in, use_rs=True, seed_val=1, dtype=dtype)
            sr2 = self._run(ssm_in, use_rs=True, seed_val=2, dtype=dtype)
            self.assertTrue(torch.isfinite(sr1.float()).all().item())
            self.assertFalse(torch.equal(sr1, rtn))
            self.assertFalse(torch.equal(sr1, sr2))
            self.assertEqual(sr1.dtype, dtype)


@unittest.skipUnless(_HAS_SM100, "SSM SR needs an SM100+ GPU (hardware cvt.rs)")
class TestChunkPrefillSR(unittest.TestCase):
    """Regression guard for the prefill (chunk_delta_h) SR store."""

    def _inputs(self, cache_dtype):
        N, H, K, V, Tseq = 2, 2, 128, 64, 64
        total_T = N * Tseq
        torch.manual_seed(0)
        act = torch.bfloat16
        q = torch.randn(1, total_T, H, K, device="cuda", dtype=act)
        k = torch.nn.functional.normalize(
            torch.randn(1, total_T, H, K, device="cuda", dtype=act), p=2, dim=-1
        )
        v = torch.randn(1, total_T, H, V, device="cuda", dtype=act)
        g = torch.nn.functional.logsigmoid(
            torch.rand(1, total_T, H, device="cuda", dtype=torch.float32)
        ).to(act)
        beta = torch.rand(1, total_T, H, device="cuda", dtype=act).sigmoid()
        cu = torch.tensor([0, Tseq, total_T], device="cuda", dtype=torch.int64)
        idx = torch.arange(N, device="cuda", dtype=torch.int64)
        ssm = torch.randn(N, H, V, K, device="cuda", dtype=cache_dtype)
        return q, k, v, g, beta, cu, idx, ssm

    def _run(self, inputs, use_rs, rng_seed):
        q, k, v, g, beta, cu, idx, ssm = inputs
        state = ssm.clone()
        torch.manual_seed(rng_seed)
        with envs.SGLANG_MAMBA_SSM_ENABLE_STOCHASTIC_ROUNDING.override(use_rs):
            chunk_gated_delta_rule(
                q, k, v, g, beta, initial_state=state, initial_state_indices=idx,
                cu_seqlens=cu, head_first=False, use_qk_l2norm_in_kernel=True,
            )
        return state

    def test_chunk_sr_compiles_and_rounds(self):
        for cache_dtype in (torch.float16, torch.bfloat16):
            inp = self._inputs(cache_dtype)
            rtn = self._run(inp, use_rs=False, rng_seed=0)
            sr1 = self._run(inp, use_rs=True, rng_seed=1)
            sr1b = self._run(inp, use_rs=True, rng_seed=1)
            sr2 = self._run(inp, use_rs=True, rng_seed=2)
            self.assertTrue(torch.isfinite(sr1.float()).all().item())
            self.assertFalse(torch.equal(sr1, rtn))
            self.assertTrue(torch.equal(sr1, sr1b))
            self.assertFalse(torch.equal(sr1, sr2))
            self.assertEqual(sr1.dtype, cache_dtype)


if __name__ == "__main__":
    unittest.main()
