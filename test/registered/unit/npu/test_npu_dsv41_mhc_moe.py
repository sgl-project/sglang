"""CPU references for A3 DSV4.1 mHC and clipped W8A8 activations."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.npu.dsv4 import dsv4_mhc as npu_mhc
from sglang.srt.hardware_backend.npu.moe import activation
from sglang.srt.models import deepseek_v4_mhc as mhc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _hc_fixture(tokens=3):
    torch.manual_seed(12)
    cfg = mhc.HcConfig(4, 20, 1e-6, 1e-20, 8, True, False)
    hc = SimpleNamespace(
        cfg=cfg,
        fn=torch.randn(24, 32),
        scale=torch.tensor([0.2, 0.4, 0.7]),
        base=torch.randn(24),
        norm=torch.nn.Identity(),
    )
    return hc, torch.randn(tokens, 4, 8, dtype=torch.bfloat16)


class TestA3Mhc(unittest.TestCase):
    def test_triplet_matches_fused_sinkhorn_formula(self):
        hc, x = _hc_fixture()
        flat = x.flatten(1).float()
        mixes = (
            F.linear(flat, hc.fn)
            / (flat.square().mean(-1, keepdim=True) + hc.cfg.rms_eps).sqrt()
        )
        pre_ref = torch.sigmoid(mixes[:, :4] * hc.scale[0] + hc.base[:4]) + hc.cfg.eps
        post_ref = 2 * torch.sigmoid(mixes[:, 4:8] * hc.scale[1] + hc.base[4:8])
        mat = (mixes[:, 8:] * hc.scale[2] + hc.base[8:]).view(-1, 4, 4)
        mat = torch.exp(mat - mat.amax(-1, keepdim=True))
        mat = mat / mat.sum(-1, keepdim=True) + hc.cfg.eps
        mat = mat / (mat.sum(-2, keepdim=True) + hc.cfg.eps)
        for _ in range(19):
            mat = mat / (mat.sum(-1, keepdim=True) + hc.cfg.eps)
            mat = mat / (mat.sum(-2, keepdim=True) + hc.cfg.eps)
        actual = npu_mhc.mix_stats_torch(hc, x)
        for result, expected in zip(actual, (pre_ref, post_ref, mat)):
            torch.testing.assert_close(result, expected)
            self.assertEqual(result.dtype, torch.float32)

    def test_combine_keeps_previous_pre_and_caches_current_stats(self):
        hc, x = _hc_fixture()
        for previous in (None, torch.rand(3, 4)):
            with self.subTest(identity=previous is None):
                state = SimpleNamespace(residual=x, pre=previous, stats=None)
                with (
                    patch.object(mhc, "_is_npu", True),
                    patch.object(mhc, "is_npu_arch35", return_value=False),
                ):
                    actual = mhc._combine(hc, state, None)
                expected = (
                    x[:, 0]
                    if previous is None
                    else (x.float() * previous.unsqueeze(-1)).sum(1).bfloat16()
                )
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
                for result, reference in zip(
                    state.stats, npu_mhc.mix_stats_torch(hc, x)
                ):
                    torch.testing.assert_close(result, reference)

    def test_post_matrix_orientation_and_bf16_output(self):
        hc, residual = _hc_fixture()
        _, post_mix, comb = npu_mhc.mix_stats_torch(hc, residual)
        x = torch.randn(3, 8, dtype=torch.bfloat16)
        expected = (
            post_mix.unsqueeze(-1) * x.float().unsqueeze(1)
            + (comb.unsqueeze(-1) * residual.float().unsqueeze(2)).sum(1)
        ).bfloat16()
        with (
            patch.object(mhc, "_is_npu", True),
            patch.object(mhc, "is_npu_arch35", return_value=False),
        ):
            actual = mhc.post(hc.cfg, x, residual, post_mix, comb)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_idle_premix_has_no_kernel_call(self):
        hc, x = _hc_fixture(0)
        state = SimpleNamespace(residual=x, pre=None, stats=None)
        with patch.object(mhc, "is_npu_arch35", return_value=False):
            actual = mhc._npu_combine_premix(hc, state, x, None)
        self.assertEqual(tuple(actual.shape), (0, 8))
        self.assertEqual(
            [tuple(v.shape) for v in state.stats], [(0, 4), (0, 4), (0, 4, 4)]
        )

    def test_a5_fused_premix_and_post_calls_are_unchanged(self):
        hc, residual = _hc_fixture()
        previous = torch.rand(3, 4)
        pre, post_mix, comb = npu_mhc.mix_stats_torch(hc, residual)
        mixed = residual[:, 0].clone()
        state = SimpleNamespace(stats=None)
        with (
            patch.object(mhc, "is_npu_arch35", return_value=True),
            patch.object(
                torch.ops.custom,
                "npu_hc_pre_v2",
                create=True,
                return_value=(mixed, post_mix, comb, pre),
            ) as fused,
        ):
            actual = mhc._npu_combine_premix(hc, state, residual, previous)
        self.assertIs(actual, mixed)
        self.assertIs(fused.call_args.kwargs["pre_mix"], previous)
        self.assertIs(state.stats[0], pre)
        with (
            patch.object(mhc, "_is_npu", True),
            patch.object(mhc, "is_npu_arch35", return_value=True),
            patch.object(
                torch.ops.cann_ops_transformer,
                "mhc_post",
                create=True,
                return_value=residual,
            ) as fused,
        ):
            actual = mhc.post(hc.cfg, mixed, residual, post_mix, comb)
        self.assertIs(actual, residual)
        self.assertEqual(fused.call_count, 1)
        for a, b in zip(fused.call_args.args, (residual, comb, mixed, post_mix)):
            self.assertIs(a, b)


class TestA3W8A8Clipping(unittest.TestCase):
    def _check(self, x, groups=None, kind=1, valid=None):
        quant, scale = activation._clipped_swiglu_quant(x, 10.0, groups, kind)
        safe = x if valid is None else x.masked_fill(~valid.unsqueeze(-1), 0)
        gate, up = safe.float().chunk(2, -1)
        ref = F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)
        expected_scale = (ref.abs().amax(-1) / 127).clamp_min(1e-30)
        expected_q = (
            torch.floor(ref / expected_scale.unsqueeze(-1) + 0.5)
            .clamp(-128, 127)
            .to(torch.int8)
        )
        torch.testing.assert_close(quant, expected_q, atol=0, rtol=0)
        torch.testing.assert_close(scale, expected_scale, atol=0, rtol=0)
        self.assertTrue(torch.isfinite(scale).all())
        self.assertEqual(quant.dtype, torch.int8)
        self.assertEqual(scale.dtype, torch.float32)
        return quant, scale

    def test_clip_precedes_silu_and_not_oai(self):
        x = torch.tensor([[20, -20, 2, -3, 30, -30, 4, 5]], dtype=torch.bfloat16)
        q, scale = self._check(x)
        ref = F.silu(x[:, :4].float().clamp(max=10)) * x[:, 4:].float().clamp(-10, 10)
        error = (q.float() * scale.unsqueeze(-1) - ref).abs()
        self.assertTrue((error <= scale.unsqueeze(-1) * 0.501).all())

    def test_contiguous_groups_mask_uninitialized_rows_for_both_metadata_kinds(self):
        x = torch.randn(6, 8, dtype=torch.bfloat16)
        x[3:] = float("nan")
        valid = torch.arange(6) < 3
        for groups, kind in ((torch.tensor([1, 2]), 1), (torch.tensor([1, 3]), 0)):
            self._check(x, groups, kind, valid)

    def test_expert_major_low_latency_capacity(self):
        x = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        valid = torch.arange(4)[None, :] < torch.tensor([1, 3])[:, None]
        x.masked_fill_(~valid.unsqueeze(-1), float("nan"))
        self._check(x, torch.tensor([1, 3]), 1, valid)

    def test_zero_and_empty_rows_have_finite_scales(self):
        self._check(torch.zeros(3, 8, dtype=torch.bfloat16))
        q, scale = self._check(torch.empty(0, 8, dtype=torch.bfloat16))
        self.assertEqual(tuple(q.shape), (0, 4))
        self.assertEqual(tuple(scale.shape), (0,))

    def test_count_updates_are_not_host_specialized(self):
        x = torch.randn(6, 8, dtype=torch.bfloat16)
        traced = torch.jit.trace(
            lambda x, g: activation._clipped_swiglu_quant(x, 10, g, 1),
            (x, torch.tensor([1, 1])),
        )
        groups = torch.tensor([2, 3])
        actual = traced(x, groups)
        expected = activation._clipped_swiglu_quant(x, 10, groups, 1)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_existing_activation_wrappers_reuse_new_path(self):
        x = torch.randn(3, 8, dtype=torch.bfloat16)
        plain = activation.NPUSwigluQuant(swiglu_limit=10)
        grouped = activation.NPUSwigluDeepEPKernel(swiglu_limit=10)
        expected = activation._clipped_swiglu_quant(x, 10)
        for result in (
            plain._apply_activation(x),
            grouped._apply_activation(x, torch.tensor([1, 2]), 1),
        ):
            for a, b in zip(result, expected):
                torch.testing.assert_close(a, b, atol=0, rtol=0)

    def test_default_unclipped_activation_dispatch_is_unchanged(self):
        x = torch.randn(3, 8, dtype=torch.bfloat16)
        result = (torch.zeros(3, 4, dtype=torch.int8), torch.ones(3))
        with patch.object(
            torch.ops.npu, "npu_dequant_swiglu_quant", create=True, return_value=result
        ) as fused:
            actual = activation.NPUSwigluQuant()._apply_activation(x)
        self.assertIs(actual[0], result[0])
        self.assertEqual(
            fused.call_args.kwargs, {"quant_mode": 1, "activate_left": True}
        )
        grouped = activation.NPUSwigluDeepEPKernel.__new__(
            activation.NPUSwigluDeepEPKernel
        )
        grouped.swiglu_limit = None
        grouped._use_oai = False
        grouped.need_quant = True
        grouped._kernel = Mock(return_value=result)
        groups = torch.tensor([1, 2])
        actual = grouped._apply_activation(x, groups, 1)
        self.assertIs(actual[0], result[0])
        self.assertEqual(grouped._kernel.call_args.kwargs, {"need_quant": True})
        self.assertIs(grouped._kernel.call_args.args[1], groups)


if __name__ == "__main__":
    unittest.main()
