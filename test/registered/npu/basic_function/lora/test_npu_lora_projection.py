"""Compare native Ascend fused LoRA projections with independent matmuls."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=30, suite="full-1-npu-a3", nightly=True)


@unittest.skipUnless(is_npu(), "Requires an Ascend NPU")
class TestAscendLoRAProjection(CustomTestCase):
    def test_qkv_and_gate_up_match_matmul(self):
        from sglang.srt.lora.backend.ascend_backend import AscendLoRABackend

        torch.set_num_threads(1)
        for dtype in (torch.float16, torch.bfloat16):
            for widths in ([128, 64, 64], [128, 128]):
                with self.subTest(dtype=dtype, widths=widths):
                    generator = torch.Generator().manual_seed(42)
                    ranks, max_rank, hidden = [8, 16], 16, 128
                    slices = len(widths)
                    x = torch.randn(3, hidden, generator=generator).to(dtype)
                    a = (
                        torch.randn(2, slices * max_rank, hidden, generator=generator)
                        * 0.02
                    ).to(dtype)
                    b = (
                        torch.randn(2, sum(widths), max_rank, generator=generator)
                        * 0.02
                    ).to(dtype)
                    base = (torch.randn(3, sum(widths), generator=generator) * 0.1).to(
                        dtype
                    )
                    offsets = [0]
                    for width in widths:
                        offsets.append(offsets[-1] + width)
                    expected = base.float().clone()
                    for token, slot in enumerate((0, 1, 1)):
                        rank = ranks[slot]
                        for part, (left, right) in enumerate(zip(offsets, offsets[1:])):
                            mid = (
                                x[token].float()
                                @ a[slot, part * rank : (part + 1) * rank].float().T
                            )
                            expected[token, left:right] += (
                                mid @ b[slot, left:right, :rank].float().T
                            )
                    backend = AscendLoRABackend(2, torch.device("npu"))
                    backend.batch_info = SimpleNamespace(
                        weight_indices=torch.tensor(
                            [0, 1], dtype=torch.int32, device="npu"
                        ),
                        seg_lens=torch.tensor([1, 2], dtype=torch.int32, device="npu"),
                        lora_ranks=torch.tensor(ranks, dtype=torch.int32, device="npu"),
                        scalings=torch.ones(2, dtype=torch.float16, device="npu"),
                    )
                    project = (
                        backend.run_qkv_lora
                        if slices == 3
                        else backend.run_gate_up_lora
                    )
                    actual = project(
                        x.npu(),
                        a.npu(),
                        b.npu(),
                        torch.tensor(offsets, dtype=torch.int32, device="npu"),
                        base_output=base.npu(),
                    )
                    self.assertTrue(torch.isfinite(actual).all())
                    torch.testing.assert_close(
                        actual.cpu(),
                        expected.to(dtype),
                        atol=1e-3,
                        rtol=1e-2 if dtype == torch.bfloat16 else 1e-3,
                    )


if __name__ == "__main__":
    unittest.main()
