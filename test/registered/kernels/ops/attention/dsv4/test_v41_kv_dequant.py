"""The V4.1 paged dequant (bf16 prefill workspace) is bit-exact with the
pure-torch dequantizer of the fp8 (V41) and fp4 (V41_FP4) formats."""

import unittest

import torch

from sglang.kernels.ops.attention.dsv4.dequant_k_cache import dequantize_k_cache_paged
from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
from sglang.srt.layers.attention.dsv4 import torch_quant as tq
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

CASES = {
    KVLayout.V41: (tq.quantize_k_cache_v41, tq.dequantize_k_cache_v41),
    KVLayout.V41_FP4: (tq.quantize_k_cache_v41_fp4, tq.dequantize_k_cache_v41_fp4),
}


def bits(t: torch.Tensor) -> torch.Tensor:
    """bf16 as int16, so that -0.0 and NaN payloads compare exactly."""
    return t.contiguous().view(torch.int16)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestV41KVDequant(CustomTestCase):
    def _gather_ref(self, dequant, pages, page_size, ids):
        return dequant(pages, page_size).view(-1, 512)[ids.long()].unsqueeze(1)

    def test_quantized_pages(self):
        g = torch.Generator(device="cuda").manual_seed(0)
        for layout, (quant, dequant) in CASES.items():
            for page_size, num_pages in ((64, 9), (256, 3), (2, 50)):
                with self.subTest(layout=layout.name, page_size=page_size):
                    k = torch.randn(
                        num_pages,
                        page_size,
                        512,
                        generator=g,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    k = (
                        k
                        * torch.exp2(
                            torch.randint(
                                -12,
                                6,
                                (num_pages, page_size, 1),
                                generator=g,
                                device="cuda",
                            ).float()
                        )
                    ).to(torch.bfloat16)
                    k[0, 0, :32] = 0
                    k[0, 0, 32:48] = -0.0
                    pages = quant(k, page_bytes=layout.page_bytes(page_size))
                    ids = torch.randint(
                        0,
                        num_pages * page_size,
                        (777,),
                        generator=g,
                        device="cuda",
                        dtype=torch.int32,
                    )
                    got = dequantize_k_cache_paged(pages, ids, page_size, layout=layout)
                    self.assertEqual(got.shape, (777, 1, 512))
                    self.assertTrue(
                        torch.equal(
                            bits(got),
                            bits(self._gather_ref(dequant, pages, page_size, ids)),
                        )
                    )
                    # The fp4 cache dequantizes to the model's fake-quantized value
                    # (compared by value: the fake quant maps an exact -0.0 to +0.0).
                    if layout is KVLayout.V41_FP4:
                        expect = tq.fake_quant_compressed_kv(
                            k.view(-1, 512)[ids.long()]
                        ).unsqueeze(1)
                        self.assertTrue(torch.equal(got, expect))

    def test_random_bytes_and_workspace_slice(self):
        """Arbitrary payload bytes (scales in the quantizer's range) and an
        ``out`` that is a strided slice of a larger workspace."""
        g = torch.Generator(device="cuda").manual_seed(1)
        for layout, (_, dequant) in CASES.items():
            page_size, num_pages = 64, 7
            with self.subTest(layout=layout.name):
                pages = torch.randint(
                    0,
                    256,
                    (num_pages, layout.page_bytes(page_size)),
                    generator=g,
                    dtype=torch.uint8,
                    device="cuda",
                )
                if layout is KVLayout.V41:
                    lo = layout.scale_offset(page_size)
                    hi = lo + page_size * layout.scale_bytes
                    pages[:, lo:hi] = torch.randint(
                        100,
                        140,
                        (num_pages, hi - lo),
                        generator=g,
                        dtype=torch.uint8,
                        device="cuda",
                    )
                ids = torch.randint(
                    0,
                    num_pages * page_size,
                    (300,),
                    generator=g,
                    device="cuda",
                    dtype=torch.int64,
                )
                ref = self._gather_ref(dequant, pages, page_size, ids)
                workspace = torch.zeros(
                    305, 1, 512, dtype=torch.bfloat16, device="cuda"
                )
                out = dequantize_k_cache_paged(
                    pages, ids, page_size, out=workspace[5:], layout=layout
                )
                # NaN payloads (fp8 0x7F / e4m3 NaN scales) compare through their bits.
                self.assertTrue(torch.equal(bits(workspace[5:]), bits(ref)))
                self.assertEqual(int(workspace[:5].abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
