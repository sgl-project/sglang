"""SiluAndMul fused with the NVFP4 activation quant of the next linear (FlashInfer's kernel)
gives the same FP4 codes and block scales as silu_and_mul followed by fp4_quantize."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.activation import SiluAndMul, silu_and_mul
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW4A4Fp4,
)
from sglang.srt.layers.quantization.fp4_utils import fp4_quantize
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase, enter_scope, published_topology

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _sf_offsets(rows: int, groups: int) -> torch.Tensor:
    """Offsets of the [rows, groups] block scales in the 128x4 swizzled layout."""
    k_tiles = (groups + 3) // 4
    m = torch.arange(rows).unsqueeze(1)
    k = torch.arange(groups).unsqueeze(0)
    return (
        (m // 128) * k_tiles * 512
        + (k // 4) * 512
        + (m % 32) * 16
        + ((m % 128) // 32) * 4
        + (k % 4)
    ).cuda()


@unittest.skipIf(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10,
    "NVFP4 activation quantization requires SM100+",
)
class TestSiluAndMulNvfp4Quant(CustomTestCase):
    def setUp(self):
        enter_scope(self, published_topology())

    def _fused_act(self, global_scale):
        down_proj = SimpleNamespace(
            scheme=CompressedTensorsW4A4Fp4(), input_global_scale=global_scale
        )
        act = SiluAndMul()
        act.fuse_output_quant(down_proj)
        self.assertIs(act._quant_linear, down_proj)
        return act

    @torch.inference_mode()
    def test_matches_silu_and_mul_then_fp4_quantize(self):
        torch.manual_seed(0)
        for tokens in (1, 37, 300, 4100):
            for d in (1536, 4096, 14336):
                with self.subTest(tokens=tokens, d=d):
                    x = torch.randn(tokens, 2 * d, dtype=torch.bfloat16, device="cuda")
                    act_ref = torch.empty(tokens, d, dtype=x.dtype, device="cuda")
                    silu_and_mul(x, act_ref)
                    global_scale = (448.0 * 6.0 / act_ref.float().abs().amax()).reshape(
                        1
                    )
                    ref_fp4, ref_sf = fp4_quantize(act_ref, global_scale)

                    # The kernel itself, also below the fusion's token threshold.
                    fp4, sf, dtype, leading = self._fused_act(
                        global_scale
                    )._forward_nvfp4_quant(x)

                    self.assertEqual((dtype, leading), (torch.bfloat16, (tokens,)))
                    self.assertTrue(
                        torch.equal(fp4.view(torch.uint8), ref_fp4.view(torch.uint8))
                    )
                    offsets = _sf_offsets(tokens, d // 16)
                    self.assertTrue(
                        torch.equal(
                            sf.view(torch.uint8).flatten()[offsets],
                            ref_sf.view(torch.uint8).flatten()[offsets],
                        )
                    )

    @torch.inference_mode()
    def test_leading_dims_are_kept_for_the_linear(self):
        x = torch.randn(2, 128, 128, dtype=torch.bfloat16, device="cuda")
        fp4, _, _, leading = self._fused_act(
            torch.tensor([100.0], device="cuda")
        ).forward_cuda(x)
        self.assertEqual(leading, (2, 128))
        self.assertEqual(fp4.shape, (256, 32))

    @torch.inference_mode()
    def test_small_batches_return_the_activation(self):
        x = torch.randn(16, 128, dtype=torch.bfloat16, device="cuda")
        out = self._fused_act(torch.tensor([100.0], device="cuda")).forward_cuda(x)
        ref = torch.empty(16, 64, dtype=x.dtype, device="cuda")
        silu_and_mul(x, ref)
        self.assertTrue(torch.equal(out, ref))


if __name__ == "__main__":
    unittest.main()
