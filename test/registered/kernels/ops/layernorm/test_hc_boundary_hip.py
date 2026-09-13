"""The ROCm fused mHC boundary, its gfx950 prefill regime and the norm launch hosting the reduce + sinkhorn must match the torch forms and the standalone launches bitwise."""

import unittest

import torch

from sglang.kernels.ops.layernorm.mhc import (
    _hc_split_sinkhorn_torch,
    hc_combine,
    hc_mix_stats_sinkhorn,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd-mi35x")


HC, H = 4, 5120
MIX = (2 + HC) * HC
ITERS, RMS_EPS, HC_EPS = 20, 1e-20, 1e-6
_IS_HIP = is_hip()


def _params(device):
    g = torch.Generator(device="cpu").manual_seed(0)
    hc_fn = (torch.randn(MIX, HC * H, generator=g) * 0.02).to(device)
    hc_scale = torch.tensor([0.7, 1.3, 0.9]).to(device)
    hc_base = (torch.randn(MIX, generator=g) * 0.3).to(device)
    return hc_fn, hc_scale, hc_base


def _ref_coefficients(residual, hc_fn, hc_scale, hc_base):
    """fp64 mixing statistics, then the in-tree torch sinkhorn."""
    x = residual.flatten(1).double()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + RMS_EPS)
    mixes = ((x @ hc_fn.double().T) * rsqrt).float().unsqueeze(1)
    pre, post, comb = _hc_split_sinkhorn_torch(
        mixes, hc_scale, hc_base, HC, ITERS, HC_EPS
    )
    return pre.squeeze(1), post.squeeze(1), comb.squeeze(1)


def _ref_post(x, residual, post, comb):
    out = post.unsqueeze(-1) * x.float().unsqueeze(1) + (
        comb.unsqueeze(-1) * residual.float().unsqueeze(2)
    ).sum(dim=1)
    return out.to(residual.dtype)


def _all_equal(a, b):
    return all((u is None and v is None) or torch.equal(u, v) for u, v in zip(a, b))


def _boundary_inputs(m, seed, hc_fn, hc_scale, hc_base):
    """A sublayer's x and residual with the coefficients the previous sinkhorn produced."""
    torch.manual_seed(seed)
    residual = torch.randn(m, HC, H, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(m, H, device="cuda", dtype=torch.bfloat16)
    pre_prev, post_in, comb_in = hc_mix_stats_sinkhorn(
        residual.flatten(1), hc_fn, hc_scale, hc_base, HC, ITERS, RMS_EPS, HC_EPS
    )
    return x, residual, post_in, comb_in, pre_prev


class TestHcMixStatsSinkhorn(CustomTestCase):
    def test_matches_reference_and_is_invariant(self):
        hc_fn, hc_scale, hc_base = _params("cuda")
        torch.manual_seed(1)
        m = 300
        residual = torch.randn(m, HC, H, device="cuda", dtype=torch.bfloat16)
        full = hc_mix_stats_sinkhorn(
            residual.flatten(1), hc_fn, hc_scale, hc_base, HC, ITERS, RMS_EPS, HC_EPS
        )
        ref = _ref_coefficients(residual, hc_fn, hc_scale, hc_base)
        for got, want in zip(full, ref):
            self.assertLess((got - want).abs().max().item(), 1e-4)
        for _ in range(3):
            again = hc_mix_stats_sinkhorn(
                residual.flatten(1),
                hc_fn,
                hc_scale,
                hc_base,
                HC,
                ITERS,
                RMS_EPS,
                HC_EPS,
            )
            self.assertTrue(_all_equal(again, full))
        for rows in ([0], list(range(0, 300, 7))):
            idx = torch.tensor(rows, device="cuda")
            sub = hc_mix_stats_sinkhorn(
                residual[idx].flatten(1).contiguous(),
                hc_fn,
                hc_scale,
                hc_base,
                HC,
                ITERS,
                RMS_EPS,
                HC_EPS,
            )
            self.assertTrue(_all_equal(sub, [t[idx] for t in full]), rows)


@unittest.skipUnless(_IS_HIP, "hc_boundary_fused is the ROCm path")
class TestHcBoundaryFused(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.layernorm.mhc_boundary_hip import hc_boundary_fused

        self.fused = hc_boundary_fused
        self.hc_fn, self.hc_scale, self.hc_base = _params("cuda")

    def _run(self, x, residual, post_in, comb_in, pre_prev):
        return self.fused(
            x,
            residual,
            post_in,
            comb_in,
            pre_prev,
            self.hc_fn,
            self.hc_scale,
            self.hc_base,
            HC,
            ITERS,
            RMS_EPS,
            HC_EPS,
        )

    def _inputs(self, m, seed):
        return _boundary_inputs(m, seed, self.hc_fn, self.hc_scale, self.hc_base)

    def test_matches_torch_forms(self):
        for m in (1, 33):
            x, residual, post_in, comb_in, pre_prev = self._inputs(m, m)
            res_out, y, pre, post, comb = self._run(
                x, residual, post_in, comb_in, pre_prev
            )
            self.assertEqual(res_out.shape, residual.shape)
            self.assertEqual(y.shape, (m, H))
            # a differently contracted fp32 chain rounded to bf16: one bf16 ulp, with an
            # absolute floor where the sum cancelled to near zero
            ref_res = _ref_post(x, residual, post_in, comb_in)
            self.assertEqual(res_out.dtype, ref_res.dtype)
            ulp = (
                res_out.view(torch.int16).int() - ref_res.view(torch.int16).int()
            ).abs()
            near_zero = (res_out.float() - ref_res.float()).abs() <= 2.0**-18
            self.assertTrue(
                bool(((ulp <= 1) | near_zero).all()),
                f"m={m}: {int(((ulp > 1) & ~near_zero).sum())} elements beyond one bf16 ulp",
            )
            # The collapse of the stored residual, as hc_combine computes it.
            y_ref = hc_combine(res_out.flatten(1), pre_prev, HC, torch.bfloat16)
            self.assertTrue(torch.equal(y, y_ref))
            # Coefficients of the stored residual against fp64.
            ref = _ref_coefficients(res_out, self.hc_fn, self.hc_scale, self.hc_base)
            for got, want in zip((pre, post, comb), ref):
                self.assertLess((got - want).abs().max().item(), 1e-4)

    def test_repeatable_and_batch_invariant(self):
        x, residual, post_in, comb_in, pre_prev = self._inputs(300, 7)
        full = self._run(x, residual, post_in, comb_in, pre_prev)
        for _ in range(3):
            self.assertTrue(
                _all_equal(self._run(x, residual, post_in, comb_in, pre_prev), full)
            )
        for rows in ([0], [299], list(range(0, 300, 7))):
            idx = torch.tensor(rows, device="cuda")
            sub = self._run(
                x[idx].contiguous(),
                residual[idx].contiguous(),
                post_in[idx].contiguous(),
                comb_in[idx].contiguous(),
                pre_prev[idx].contiguous(),
            )
            self.assertTrue(_all_equal(sub, [t[idx] for t in full]), rows)


@unittest.skipUnless(_IS_HIP, "rmsnorm_with_sinkhorn is the ROCm path")
class TestRmsnormWithSinkhorn(CustomTestCase):
    """The norm launch hosting the boundary's reduce + sinkhorn."""

    def setUp(self):
        from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
            hc_boundary_fused,
            hc_boundary_fused_deferred,
            rmsnorm_with_sinkhorn,
        )
        from sglang.kernels.ops.quantization.rmsnorm_fake_quant_amd_gfx95 import (
            rmsnorm_fake_quant_fp8,
        )

        self.fused = hc_boundary_fused
        self.deferred = hc_boundary_fused_deferred
        self.hosted = rmsnorm_with_sinkhorn
        self.norm = rmsnorm_fake_quant_fp8
        self.hc_fn, self.hc_scale, self.hc_base = _params("cuda")
        self.weight = (torch.rand(H, device="cuda") + 0.5).to(torch.bfloat16)

    def _boundary(self, fn, m, seed):
        x, residual, post_in, comb_in, pre_prev = _boundary_inputs(
            m, seed, self.hc_fn, self.hc_scale, self.hc_base
        )
        return fn(
            x,
            residual,
            post_in,
            comb_in,
            pre_prev,
            self.hc_fn,
            self.hc_scale,
            self.hc_base,
            HC,
            ITERS,
            RMS_EPS,
            HC_EPS,
        )

    def test_matches_standalone_launches(self):
        for m in (1, 1100):
            for emit_fp8 in (False, True):
                res, y, pre, post, comb = self._boundary(self.fused, m, m)
                quant, norm = self.norm(y, self.weight, 1e-6, emit_fp8=emit_fp8)
                res2, y2, coefficients = self._boundary(self.deferred, m, m)
                self.assertFalse(coefficients.materialized)
                quant2, norm2 = self.hosted(
                    y2, self.weight, 1e-6, coefficients, emit_fp8=emit_fp8
                )
                self.assertTrue(coefficients.materialized)
                self.assertTrue(_all_equal((res, y, norm), (res2, y2, norm2)))
                self.assertTrue(_all_equal((pre, post, comb), coefficients.tensors()))
                if emit_fp8:
                    self.assertTrue(torch.equal(quant.q, quant2.q))
                    self.assertTrue(torch.equal(quant.scale, quant2.scale))
                else:
                    self.assertTrue(torch.equal(quant.x, quant2.x))


def _prefill_available():
    if not is_hip() or not torch.cuda.is_available():
        return False
    from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
        _hc_boundary_prefill_available,
    )

    return _hc_boundary_prefill_available()


@unittest.skipUnless(_prefill_available(), "the prefill boundary kernel needs gfx950")
class TestHcBoundaryPrefill(CustomTestCase):
    def setUp(self):
        from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
            _hc_boundary_partials,
            hc_boundary_fused,
        )

        self.partials = _hc_boundary_partials
        self.fused = hc_boundary_fused
        self.hc_fn, self.hc_scale, self.hc_base = _params("cuda")

    def _inputs(self, m, seed):
        return _boundary_inputs(m, seed, self.hc_fn, self.hc_scale, self.hc_base)

    def _raw(self, x, residual, post_in, comb_in, pre_prev, prefill):
        """Both launches' raw outputs: (residual_out, y, part_mix, part_sq)."""
        has_post = x is not None
        residual_out = torch.empty_like(residual) if has_post else None
        y = (
            torch.empty((residual.shape[0], H), dtype=residual.dtype, device="cuda")
            if pre_prev is not None
            else None
        )
        part_mix, part_sq = self.partials(
            x,
            residual,
            post_in,
            comb_in,
            pre_prev,
            self.hc_fn,
            residual_out,
            y,
            hc_mult=HC,
            prefill=prefill,
        )
        return residual_out, y, part_mix, part_sq

    def test_matches_decode_kernel_bitwise(self):
        # Full blocks, partial last blocks and M below / above the switch.
        for m in (1, 4096):
            x, residual, post_in, comb_in, pre_prev = self._inputs(m, m)
            decode = self._raw(x, residual, post_in, comb_in, pre_prev, False)
            prefill = self._raw(x, residual, post_in, comb_in, pre_prev, True)
            self.assertTrue(_all_equal(prefill, decode), f"post+combine M={m}")
            decode = self._raw(None, residual, None, None, None, False)
            prefill = self._raw(None, residual, None, None, None, True)
            self.assertTrue(_all_equal(prefill, decode), f"stats-only M={m}")

    def test_row_alone_equals_row_in_prefill_batch(self):
        """A row alone and the same row inside a prefill batch must give bitwise equal coefficients and outputs."""
        from sglang.kernels.ops.layernorm.mhc_boundary_hip import (
            _HC_BOUNDARY_PREFILL_MIN_M,
        )

        m = 4 * _HC_BOUNDARY_PREFILL_MIN_M + 5
        x, residual, post_in, comb_in, pre_prev = self._inputs(m, 11)
        args = (self.hc_fn, self.hc_scale, self.hc_base, HC, ITERS, RMS_EPS, HC_EPS)
        full = self.fused(x, residual, post_in, comb_in, pre_prev, *args)
        for rows in ([0], list(range(0, m, 97))):
            idx = torch.tensor(rows, device="cuda")
            sub = self.fused(
                x[idx].contiguous(),
                residual[idx].contiguous(),
                post_in[idx].contiguous(),
                comb_in[idx].contiguous(),
                pre_prev[idx].contiguous(),
                *args,
            )
            self.assertTrue(_all_equal(sub, [t[idx] for t in full]), rows)
        # The stats-only form (a layer's first boundary) the same way.
        full = self.fused(None, residual, None, None, None, *args)
        idx = torch.tensor([0, 7, m - 1], device="cuda")
        sub = self.fused(None, residual[idx].contiguous(), None, None, None, *args)
        self.assertIsNone(sub[0])
        self.assertIsNone(sub[1])
        self.assertTrue(_all_equal(sub[2:], [t[idx] for t in full[2:]]))


if __name__ == "__main__":
    unittest.main()
