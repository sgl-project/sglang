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

    def _create_weights(self, fi_kernel, tp_size, tp_rank=0):
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
                intermediate_size_per_partition=GPT_OSS_INTERMEDIATE // tp_size,
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
        # tp=4 rank 3 loads 672 of its 736 channels; the unwritten rows and
        # columns keep scale 127 and must not move Humming's per-expert range.
        tp_size, tp_rank = 4, 3
        loaded = GPT_OSS_INTERMEDIATE - tp_rank * loader_rank_intermediate(
            GPT_OSS_INTERMEDIATE, tp_size
        )
        self.assertEqual(loaded, 672)

        def fake_humming(weights, scales, **kwargs):
            exps = scales.flatten(1).to(torch.int32)
            return weights, scales, torch.stack(
                [exps.amax(dim=1), exps.amin(dim=1)], dim=1
            ).float()

        fake_fused_moe = types.ModuleType("flashinfer.fused_moe")
        fake_fused_moe.preprocess_moe_weights_for_sm90_mixed_gemm_humming = (
            fake_humming
        )
        fake_flashinfer = types.ModuleType("flashinfer")
        fake_flashinfer.fused_moe = fake_fused_moe

        residuals = []
        for tail_fill in (127, 105):
            layer = self._create_weights("cutlass_sm90", tp_size, tp_rank)
            layer.moe_runner_config = types.SimpleNamespace(gate_up_interleaved=True)
            gen = torch.Generator().manual_seed(0)
            for name, rows, cols in (
                ("w13_weight_scale", 2 * loaded, None),
                ("w2_weight_scale", None, loaded // MXFP4_BLOCK),
            ):
                scale = getattr(layer, name).data
                scale.fill_(tail_fill)
                region = scale[:, :rows, :cols]
                region.copy_(
                    torch.randint(100, 111, region.shape, generator=gen).to(
                        torch.uint8
                    )
                )
            method = self.method
            method._use_sm90_humming = True
            with mock.patch.dict(
                sys.modules,
                {"flashinfer": fake_flashinfer, "flashinfer.fused_moe": fake_fused_moe},
            ):
                method._process_weights_for_sm90_cutlass(layer)
            residuals.append(
                (
                    layer.w13_humming_residual_scale.clone(),
                    layer.w2_humming_residual_scale.clone(),
                )
            )

        self.assertTrue(torch.equal(residuals[0][0], residuals[1][0]))
        self.assertTrue(torch.equal(residuals[0][1], residuals[1][1]))


if __name__ == "__main__":
    unittest.main()
