"""Unit tests for srt/layers/quantization/mxfp4_aiter_a16w4_moe.py (gfx942 A8W4 MoE)."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")

E, TOPK, HIDDEN, INTER = 32, 4, 1024, 256
FP4_DTYPE = torch.float4_e2m1fn_x2


def _is_gfx942():
    if not (is_hip() and torch.cuda.is_available()):
        return False
    return torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx942"


def _cos_diff(x, y):
    x, y = x.double(), y.double()
    return float(1 - 2 * (x * y).sum() / (x * x + y * y).sum())


class TestMxfp4AiterA16W4MoEMethod(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not _is_gfx942():
            raise unittest.SkipTest("A8W4 FlyDSL MoE kernels require gfx942 (MI300/MI325)")
        import aiter
        from aiter import QuantType, dtypes

        from sglang.srt.layers.quantization import mxfp4_aiter_a16w4_moe as mod

        cls.mod = mod
        torch.manual_seed(0)
        tq = aiter.get_torch_quant(QuantType.per_1x32)
        w1 = torch.randn((E, 2 * INTER, HIDDEN), dtype=torch.bfloat16, device="cuda")
        w2 = torch.randn((E, HIDDEN, INTER), dtype=torch.bfloat16, device="cuda")
        w1_qt, w1_scale = tq(w1, quant_dtype=dtypes.fp4x2)
        w2_qt, w2_scale = tq(w2, quant_dtype=dtypes.fp4x2)
        # Checkpoint layout as loaded by Fp8MoEMethod (store_dtype="mxfp4"):
        # raw uint8 bytes, separated [gate; up] w13 rows, e8m0 scale bytes.
        cls.w13 = w1_qt.view(torch.uint8).view(E, 2 * INTER, HIDDEN // 2)
        cls.w2 = w2_qt.view(torch.uint8).view(E, HIDDEN, INTER // 2)
        cls.w13_scale = w1_scale.view(torch.uint8).view(E, 2 * INTER, HIDDEN // 32)
        cls.w2_scale = w2_scale.view(torch.uint8).view(E, HIDDEN, INTER // 32)

    def _make_layer(self, fast_dec):
        layer = torch.nn.Module()
        for name, t in (
            ("w13_weight", self.w13),
            ("w2_weight", self.w2),
            ("w13_weight_scale_inv", self.w13_scale),
            ("w2_weight_scale_inv", self.w2_scale),
        ):
            layer.register_parameter(
                name, torch.nn.Parameter(t.clone(), requires_grad=False)
            )
        layer.dispatcher = MagicMock()
        method = self.mod.Mxfp4AiterA16W4MoEMethod(MagicMock(), prefix="experts")
        with patch.object(self.mod, "A8W4_FAST_DEC", fast_dec):
            method.process_weights_after_loading(layer)
        return method, layer

    def test_process_weights_after_loading(self):
        from aiter.ops.flydsl.kernels.moe_2stage_a16wmix.a8w4 import (
            channel_exponents,
        )
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        w1_std = shuffle_weight_a16w4(self.w13.view(FP4_DTYPE), 16, False)
        w2_std = shuffle_weight_a16w4(self.w2.view(FP4_DTYPE), 16, False)
        w1_std = w1_std.view(torch.uint8)
        w2_std = w2_std.view(torch.uint8)
        s1_std = shuffle_scale_a16w4(self.w13_scale.view(-1, HIDDEN // 32), E, False)
        s2_std = shuffle_scale_a16w4(self.w2_scale.view(-1, INTER // 32), E, False)
        s1_std = s1_std.view(torch.uint8).contiguous().view(-1)
        s2_std = s2_std.view(torch.uint8).contiguous().view(-1)
        # Per-output-channel max e8m0 code, from the unshuffled checkpoint scales.
        w1_exp_ref = self.w13_scale.amax(-1).int()  # [E, 2 * INTER]
        w2_exp_ref = self.w2_scale.amax(-1).int().view(-1)

        for fd in (0, 1, 2):
            with self.subTest(fast_dec=fd):
                _, layer = self._make_layer(fd)
                aw = layer.all_weights
                if fd:
                    w1_ref, s1_ref = self.mod.prep_w1_fast(w1_std, s1_std, w1_exp_ref, fd)
                    w2_ref, s2_ref = self.mod.prep_w2_fast(w2_std, s2_std, fd)
                else:
                    w1_ref, s1_ref, w2_ref, s2_ref = w1_std, s1_std, w2_std, s2_std

                self.assertEqual(aw["w1_fast_dec"], fd)
                self.assertEqual(aw["w2_fast_dec"], fd)
                torch.testing.assert_close(aw["w1_exp"], w1_exp_ref, rtol=0, atol=0)
                torch.testing.assert_close(
                    channel_exponents(w1_std, s1_std), w1_exp_ref, rtol=0, atol=0
                )
                for key, ref in (
                    ("w1", w1_ref),
                    ("w2", w2_ref),
                    ("w1_scale", s1_ref),
                    ("w2_scale", s2_ref),
                ):
                    self.assertEqual(aw[key].dtype, torch.uint8, key)
                    self.assertTrue(torch.equal(aw[key], ref), key)
                self.assertEqual(tuple(aw["w1"].shape), (E, 2 * INTER, HIDDEN // 2))
                self.assertEqual(tuple(aw["w2"].shape), (E, HIDDEN, INTER // 2))
                if fd == 2:  # W2 channel max codes appended after the scale bytes
                    tail = aw["w2_scale"][s2_std.numel() :].view(torch.int32)
                    torch.testing.assert_close(tail, w2_exp_ref, rtol=0, atol=0)
                else:
                    self.assertEqual(aw["w2_scale"].numel(), s2_std.numel())

                # The kernel buffers alias the layer storage (no second weight copy).
                for key, param in (
                    ("w1", layer.w13_weight),
                    ("w2", layer.w2_weight),
                    ("w1_scale", layer.w13_weight_scale_inv),
                    ("w2_scale", layer.w2_weight_scale_inv),
                ):
                    self.assertEqual(aw[key].data_ptr(), param.data_ptr(), key)
                for w in (layer.w13_weight, layer.w2_weight):
                    self.assertEqual(w.dtype, FP4_DTYPE)
                    self.assertTrue(w.is_shuffled)
                    self.assertFalse(w.gate_up_interleave)
                layer.dispatcher.set_quant_config.assert_called_once_with(
                    {"weight_dtype": FP4_DTYPE, "use_mxfp8": False}
                )

    def test_apply_matches_torch_reference(self):
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import fused_topk, torch_moe_stage1, torch_moe_stage2

        from sglang.srt.layers.moe.token_dispatcher.standard import (
            StandardCombineInput,
        )

        w1 = self.w13.view(dtypes.fp4x2)
        w2 = self.w2.view(dtypes.fp4x2)
        layers = {fd: self._make_layer(fd) for fd in (0, 2)}
        for tokens in (1, 37, 256):
            with self.subTest(tokens=tokens):
                x = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
                score = torch.randn((tokens, E), dtype=torch.bfloat16, device="cuda")
                topk_weights, topk_ids = fused_topk(x, score, TOPK, True)

                o1 = torch_moe_stage1(
                    x,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    dtype=torch.bfloat16,
                    activation=ActivationType.Silu,
                    quant_type=QuantType.per_1x32,
                    a1_scale=None,
                    w1_scale=self.w13_scale.view(dtypes.fp8_e8m0),
                    doweight=False,
                )
                ref = torch_moe_stage2(
                    o1.view(tokens, TOPK, INTER),
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    dtype=torch.bfloat16,
                    quant_type=QuantType.per_1x32,
                    w2_scale=self.w2_scale.view(dtypes.fp8_e8m0),
                    a2_scale=None,
                    doweight=True,
                )

                dispatch_output = SimpleNamespace(
                    hidden_states=x,
                    topk_output=SimpleNamespace(
                        topk_weights=topk_weights, topk_ids=topk_ids
                    ),
                )
                outs = {}
                for fd, (method, layer) in layers.items():
                    combine = method.apply(layer, dispatch_output)
                    self.assertIsInstance(combine, StandardCombineInput)
                    out = combine.hidden_states
                    self.assertEqual(tuple(out.shape), (tokens, HIDDEN))
                    self.assertEqual(out.dtype, torch.bfloat16)
                    self.assertFalse(out.isnan().any().item())
                    self.assertLess(_cos_diff(ref, out), 1e-2, f"fast_dec={fd}")
                    outs[fd] = out.clone()
                # Fast decode is bit-exact per GEMM; the e2e output differs only by
                # gemm2's bf16 atomic accumulation order.
                self.assertLess(_cos_diff(outs[0], outs[2]), 1e-4)

    def test_tuned_moe_kernel_combinations(self):
        """Every kernel kind tuned_moe can dispatch to (2-kernel base / nt4, fused
        single kernel with ksplit 1 / 2, persistent work-queue fused) on the same
        fd2 weights matches the 2-kernel default."""
        from aiter.fused_moe import fused_topk

        from sglang.srt.layers.quantization.flydsl_moe_tp_w4a8_gfx942 import (
            a8w4_moe_tuned as tm,
        )

        _, layer = self._make_layer(2)
        W = layer.all_weights
        cfgs = [
            dict(kind="2k", block_m=32, g1_bcm=2, g2="base"),
            dict(kind="2k", block_m=64, g1_bcm=1, g2="nt4"),
            dict(kind="1k", block_m=64, bcm=1, wpe=2, ksplit=1),
            dict(kind="1k", block_m=64, bcm=2, wpe=2, ksplit=2),
            dict(kind="1kq", block_m=64, bcm=1),
        ]
        for tokens in (1, 37, 512):
            x = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda")
            score = torch.randn((tokens, E), dtype=torch.bfloat16, device="cuda")
            tw, ti = fused_topk(x, score, TOPK, True)
            ref = tm.run_config(tm.DEFAULT_CFG, x, W, tw, ti).float().clone()
            self.assertEqual(
                tm.lookup_config(tokens, HIDDEN, INTER, E, TOPK), tm.DEFAULT_CFG
            )  # untuned shape -> default
            for cfg in cfgs:
                with self.subTest(tokens=tokens, cfg=tm.config_name(cfg)):
                    out = tm.run_config(cfg, x, W, tw, ti)
                    self.assertEqual(tuple(out.shape), (tokens, HIDDEN))
                    self.assertLess(_cos_diff(ref, out.float()), 1e-4)

    def test_tuned_lookup(self):
        """num_tokens rounds up to 2^N and clamps to the table range."""
        from sglang.srt.layers.quantization.flydsl_moe_tp_w4a8_gfx942 import (
            a8w4_moe_tuned as tm,
        )

        table = tm._load_table(tm.DEFAULT_CSV)
        key = (4096, 256, 256, 8, "uniform")
        self.assertIn(key, table)
        rows = dict(table[key])
        for tokens, t2 in ((1, 1), (3, 4), (1000, 1024), (3000, 4096), (10**6, 32768)):
            self.assertEqual(tm.lookup_config(tokens, *key), rows[t2])


if __name__ == "__main__":
    unittest.main()
