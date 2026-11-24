import math
import unittest

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

    def _create_weights(self, fi_kernel, tp_size):
        method = object.__new__(Mxfp4MoEMethod)
        method.use_marlin = False
        method.use_deep_gemm = False
        method.use_mega_moe = False
        method.use_aiter_moe = False
        method.use_flashinfer = True
        method._fi_kernel = fi_kernel
        layer = torch.nn.Module()
        layer.num_local_experts = 2
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


if __name__ == "__main__":
    unittest.main()
