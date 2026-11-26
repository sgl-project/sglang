"""Kimi-K3 MLA decode / target-verify fusions against the unfused chain they
replace: k3_mla_cat_cache (torch.cat + KV-cache write), k3_mla_absorb_q_cat_cache
(W_UK bmm + cat + write) and k3_mla_absorb_v_gate (W_UV bmm + sigmoid gate).
M = 65 is past the fused kernels' cap and must be rejected by the gates."""

import unittest

import torch

from sglang.kernels.ops.attention import k3_mla_absorb as ka
from sglang.kernels.ops.attention import k3_mla_cat_cache_hip as cc
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

H = 12  # K3 heads per rank (TP8)
DK, DN, DP, DV = 128, 512, 64, 128  # qk_nope, kv_lora, rope, v
ROWS = 4096
MS = (1, 17, 64)
CACHE_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
BF16 = torch.bfloat16
# fp32 accumulation like the bmm, only the summation order differs
TOL = dict(rtol=1.6e-2, atol=1e-2)


def _rand(*shape, scale=1.0):
    return (torch.randn(*shape, device="cuda") * scale).to(BF16)


def _decode_inputs(m, cache_dtype):
    q_nope = _rand(m, H, DK)
    q_pe = _rand(m, H, DP)
    k_nope = _rand(m, 1, DN)
    k_pe = _rand(m, 1, DP)
    kv = torch.zeros(ROWS, 1, DN + DP, device="cuda", dtype=cache_dtype)
    loc = torch.randperm(ROWS, device="cuda")[:m].to(torch.int64)
    return q_nope, q_pe, k_nope, k_pe, kv, loc


def _write_cache_ref(kv, loc, k_nope, k_pe):
    kv[loc] = torch.cat([k_nope, k_pe], dim=-1).to(kv.dtype)


class TestK3MlaCatCache(CustomTestCase):
    def test_matches_cat_and_set_kv_buffer(self):
        for m in MS:
            for dt in CACHE_DTYPES:
                with self.subTest(m=m, cache=dt):
                    _, q_pe, k_nope, k_pe, kv, loc = _decode_inputs(m, dt)
                    q_nope_out = _rand(m, H, DN)
                    kv_ref = kv.clone()
                    q_ref = torch.cat([q_nope_out, q_pe], dim=-1)
                    _write_cache_ref(kv_ref, loc, k_nope, k_pe)
                    self.assertTrue(cc.covered(q_nope_out, q_pe, k_nope, k_pe, kv, loc))
                    q = cc.k3_mla_cat_cache(q_nope_out, q_pe, k_nope, k_pe, kv, loc)
                    torch.testing.assert_close(q, q_ref, rtol=0, atol=0)
                    self.assertTrue(
                        torch.equal(kv.view(torch.uint8), kv_ref.view(torch.uint8))
                    )


class TestK3MlaAbsorbQCatCache(CustomTestCase):
    def test_matches_unfused_chain(self):
        w_kc = _rand(H, DK, DN, scale=0.05)
        for m in MS:
            for dt in CACHE_DTYPES:
                with self.subTest(m=m, cache=dt):
                    q_nope, q_pe, k_nope, k_pe, kv, loc = _decode_inputs(m, dt)
                    kv_ref = kv.clone()
                    q_ref = torch.cat(
                        [torch.bmm(q_nope.transpose(0, 1), w_kc).transpose(0, 1), q_pe],
                        dim=-1,
                    )
                    _write_cache_ref(kv_ref, loc, k_nope, k_pe)
                    self.assertTrue(
                        ka.covered_q(q_nope, q_pe, w_kc, k_nope, k_pe, kv, loc)
                    )
                    q = ka.k3_mla_absorb_q_cat_cache(
                        q_nope, q_pe, w_kc, k_nope, k_pe, kv, loc
                    )
                    torch.testing.assert_close(q, q_ref, **TOL)
                    self.assertTrue(
                        torch.equal(kv.view(torch.uint8), kv_ref.view(torch.uint8))
                    )

    def test_m65_falls_back(self):
        q_nope, q_pe, k_nope, k_pe, kv, loc = _decode_inputs(65, BF16)
        w_kc = _rand(H, DK, DN, scale=0.05)
        self.assertFalse(ka.covered_q(q_nope, q_pe, w_kc, k_nope, k_pe, kv, loc))


class TestK3MlaAbsorbVGate(CustomTestCase):
    def _ref(self, attn, w_vc, gate):
        m = attn.shape[0]
        y = torch.bmm(attn.transpose(0, 1), w_vc).transpose(0, 1).reshape(m, H * DV)
        return y if gate is None else y * torch.sigmoid(gate)

    def test_matches_unfused_chain(self):
        w_vc = _rand(H, DN, DV, scale=0.05)
        for m in MS:
            for with_gate in (False, True):
                with self.subTest(m=m, gate=with_gate):
                    # strided like the attention output view the model passes
                    attn = _rand(m, H, DN + DP)[..., :DN]
                    gate = _rand(m, H * DV) if with_gate else None
                    self.assertTrue(ka.covered_v(attn, w_vc, gate))
                    out = ka.k3_mla_absorb_v_gate(attn, w_vc, gate)
                    torch.testing.assert_close(out, self._ref(attn, w_vc, gate), **TOL)

    def test_m65_falls_back(self):
        w_vc = _rand(H, DN, DV, scale=0.05)
        self.assertFalse(ka.covered_v(_rand(65, H, DN), w_vc, None))

    def test_k_not_multiple_of_chunk(self):
        # loads along K are unmasked 512-wide chunks
        dk = 640
        w_vc = _rand(H, dk, DV, scale=0.05)
        self.assertFalse(ka.covered_v(_rand(4, H, dk), w_vc, None))


if __name__ == "__main__":
    unittest.main()
