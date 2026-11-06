"""wo_a on FlashInfer's SM107 batched MXFP8 GEMM, fed by FlashMLA's FlashInfer-layout scales, gives the same
bytes as DeepGEMM's fp8_einsum on FlashMLA's default scales."""

import unittest

import torch

from sglang.kernels.ops.attention.flash_mla_interface import flash_mla_mega_prefill
from sglang.srt.layers.attention.deepseek_v4_mega_backend import quantize_wo_a
from sglang.srt.models import deepseek_v4_flashinfer_wo_a as fi_wo_a
from sglang.srt.models.deepseek_v4 import _fp8_wo_a_einsum
from sglang.srt.models.deepseek_v4_mhc import empty_deepgemm_mxfp8_pair
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

GROUPS, RANK = 8, 1024


@unittest.skipUnless(
    torch.cuda.is_available() and fi_wo_a._available(),
    "needs CC 10.7, FlashMLA's out_sf_flashinfer_layout and flashinfer.bmm_mxfp8(out_scale=...)",
)
class TestFlashInferWoA(CustomTestCase):
    def _attention(self, rows, flashinfer_layout):
        g = torch.Generator(device="cuda").manual_seed(rows)
        h_q, d_qk, s_kv = 64, 512, 4096
        q = torch.randn(
            rows, h_q, d_qk, device="cuda", dtype=torch.bfloat16, generator=g
        )
        kv = torch.randn(
            s_kv, 1, d_qk, device="cuda", dtype=torch.bfloat16, generator=g
        )
        indices = torch.randint(
            0, s_kv, (rows, 1, 512), device="cuda", dtype=torch.int32, generator=g
        )
        positions = torch.randint(
            0, 4096, (rows,), device="cuda", dtype=torch.int32, generator=g
        )
        cos_sin = torch.randn(4096, 64, device="cuda", generator=g)
        sink = torch.randn(h_q, device="cuda", generator=g)
        data, scale, _, _ = flash_mla_mega_prefill(
            q,
            kv,
            indices,
            positions,
            cos_sin,
            d_qk**-0.5,
            attn_sink=sink,
            use_legacy=False,
            out_sf_flashinfer_layout=flashinfer_layout,
        )
        return data, scale

    def test_matches_deepgemm(self):
        torch.manual_seed(0)
        weight = (
            torch.randn(GROUPS, RANK, 4096, device="cuda", dtype=torch.bfloat16) * 0.05
        )
        wq, ws = quantize_wo_a(weight)
        ws_fi = fi_wo_a.wo_a_scale_to_flashinfer(wq, ws)
        for rows in (128, 512, 4096):
            data, scale = self._attention(rows, False)
            data_fi, scale_fi = self._attention(rows, True)
            self.assertFalse(fi_wo_a.scales_in_flashinfer_layout(scale))
            self.assertTrue(fi_wo_a.scales_in_flashinfer_layout(scale_fi))

            y_ref, sf_ref = empty_deepgemm_mxfp8_pair(
                rows, GROUPS * RANK, "cuda", flashinfer=True
            )
            _fp8_wo_a_einsum(
                data,
                scale,
                wq,
                ws,
                recipe=(1, 1, 32),
                out=(y_ref.view(rows, GROUPS, -1), sf_ref),
            )
            y, sf = empty_deepgemm_mxfp8_pair(
                rows, GROUPS * RANK, "cuda", flashinfer=True
            )
            fi_wo_a.flashinfer_wo_a_einsum(data_fi, scale_fi, wq, ws_fi, (y, sf))
            self.assertTrue(
                torch.equal(y.view(torch.uint8), y_ref.view(torch.uint8)), rows
            )
            self.assertTrue(torch.equal(sf, sf_ref), rows)

            o_ref = _fp8_wo_a_einsum(data, scale, wq, ws, recipe=(1, 1, 32))
            o = torch.empty_like(o_ref)
            fi_wo_a.flashinfer_wo_a_einsum(data_fi, scale_fi, wq, ws_fi, o)
            torch.testing.assert_close(o, o_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
