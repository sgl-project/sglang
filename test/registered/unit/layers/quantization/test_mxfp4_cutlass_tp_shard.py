import math
import sys
import types
import unittest
from unittest import mock

import torch

from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

MXFP4_BLOCK = 32
GPT_OSS_INTERMEDIATE = 2880
GPT_OSS_HIDDEN = 2880


def loader_rank_intermediate(intermediate_size, tp_size):
    # Mirrors per_rank_intermediate_size in
    # GptOssForCausalLM._load_mxfp4_experts_weights (models/gpt_oss.py).
    blocks = intermediate_size // MXFP4_BLOCK
    return math.ceil(blocks / tp_size) * MXFP4_BLOCK


class TestMxfp4CutlassTpShard(CustomTestCase):
    """Loading gpt-oss MXFP4 weights with tp=4 or tp=8 on the CUTLASS MoE path
    crashed with a size mismatch: the checkpoint is split into whole 32-element
    blocks per rank, which is wider than the buffer allocated for
    intermediate_size / tp.
    """

    def _create_weights(
        self, fi_kernel, tp_size, tp_rank=0, intermediate=GPT_OSS_INTERMEDIATE
    ):
        method = object.__new__(Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deep_gemm = False
        method.use_mega_moe = False
        method.use_aiter_moe = False
        method.use_flashinfer = True
        method._fi_kernel = fi_kernel
        self.method = method
        layer = torch.nn.Module()
        layer.num_local_experts = 2
        layer.moe_tp_size = tp_size
        layer.moe_tp_rank = tp_rank
        with override_platform(is_sm100=False):
            method.create_weights(
                layer,
                num_experts=2,
                hidden_size=GPT_OSS_HIDDEN,
                intermediate_size_per_partition=intermediate // tp_size,
                params_dtype=torch.bfloat16,
                with_bias=True,
            )
        return layer

    def test_buffers_match_the_loader_shard(self):
        for fi_kernel in ("cutlass_sm90", "cutlass_sm120"):
            for tp_size in (1, 2, 4, 8):
                with self.subTest(fi_kernel=fi_kernel, tp_size=tp_size):
                    layer = self._create_weights(fi_kernel, tp_size)
                    rows = loader_rank_intermediate(GPT_OSS_INTERMEDIATE, tp_size)
                    self.assertEqual(layer.w2_weight.shape[2], rows // 2)
                    self.assertEqual(
                        layer.w2_weight_scale.shape[2], rows // MXFP4_BLOCK
                    )
                    self.assertEqual(layer.w13_weight.shape[1], 2 * rows)
                    self.assertEqual(layer.w13_weight_bias.shape[1], 2 * rows)

    def test_humming_range_ignores_short_shard_tail(self):
        # The last rank loads fewer channels than its buffer holds; the rest
        # keeps scale 127 and must not reach Humming's per-expert range.
        # (3072, 2880): the config is wider than the checkpoint.
        tp_size, tp_rank = 4, 3
        for config_inter, expect_loaded in (
            (GPT_OSS_INTERMEDIATE, 672),
            (3072, 576),
        ):
            with self.subTest(config_intermediate=config_inter):
                # Mirrors the cutoff in _load_mxfp4_experts_weights (gpt_oss.py).
                per_rank = loader_rank_intermediate(config_inter, tp_size)
                loaded = (
                    min((tp_rank + 1) * per_rank, GPT_OSS_INTERMEDIATE)
                    - tp_rank * per_rank
                )
                self.assertEqual(loaded, expect_loaded)
                self._check_humming_inputs(config_inter, tp_size, tp_rank, loaded)

    def _check_humming_inputs(self, config_inter, tp_size, tp_rank, loaded):
        captured = []

        def fake_humming(weights, scales, **kwargs):
            captured.append(scales.clone())
            return weights, scales, torch.zeros(scales.shape[0], 2)

        fake_fused_moe = types.ModuleType("flashinfer.fused_moe")
        fake_fused_moe.preprocess_moe_weights_for_sm90_mixed_gemm_humming = (
            fake_humming
        )
        fake_flashinfer = types.ModuleType("flashinfer")
        fake_flashinfer.fused_moe = fake_fused_moe

        layer = self._create_weights("cutlass_sm90", tp_size, tp_rank, config_inter)
        layer.moe_runner_config = types.SimpleNamespace(gate_up_interleaved=True)
        layer.loaded_intermediate_size = loaded
        gen = torch.Generator().manual_seed(0)
        n_groups = -(-loaded // MXFP4_BLOCK)
        w13_real = layer.w13_weight_scale.data
        w2_real = layer.w2_weight_scale.data
        w13_real.fill_(127)
        w2_real.fill_(127)
        w13_real[:, : 2 * loaded, :] = torch.randint(
            100, 111, w13_real[:, : 2 * loaded, :].shape, generator=gen
        ).to(torch.uint8)
        w2_real[:, :, :n_groups] = torch.randint(
            100, 111, w2_real[:, :, :n_groups].shape, generator=gen
        ).to(torch.uint8)
        w13_src = w13_real.clone()
        w2_src = w2_real.clone()

        method = self.method
        method._use_sm90_humming = True
        with mock.patch.dict(
            sys.modules,
            {"flashinfer": fake_flashinfer, "flashinfer.fused_moe": fake_fused_moe},
        ):
            method._process_weights_for_sm90_cutlass(layer)

        w13_in, w2_in = captured
        n_pad = method._padded_intermediate
        k_groups = GPT_OSS_HIDDEN // MXFP4_BLOCK
        for name, got, src, expected_fill in (
            ("w13", w13_in, w13_src, w13_src[:, 0, 0]),
            ("w2", w2_in, w2_src, w2_src[:, 0, 0]),
        ):
            expected = expected_fill.view(-1, 1, 1).expand_as(got).clone()
            if name == "w13":
                expected[:, :loaded, :k_groups] = src[:, 1 : 2 * loaded : 2, :k_groups]
                expected[:, n_pad : n_pad + loaded, :k_groups] = src[
                    :, 0 : 2 * loaded : 2, :k_groups
                ]
            else:
                expected[:, :GPT_OSS_HIDDEN, :n_groups] = src[
                    :, :GPT_OSS_HIDDEN, :n_groups
                ]
            self.assertTrue(torch.equal(got, expected), name)
            self.assertFalse((got == 127).any(), name)


if __name__ == "__main__":
    unittest.main()
