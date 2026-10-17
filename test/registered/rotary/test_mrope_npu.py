"""M-RoPE handling in ``MRotaryEmbedding.forward_npu`` on Ascend NPU.

Both values this path derives for ``torch_npu.npu_mrope`` fail silently when
wrong -- the output looks plausible, accuracy just degrades. The section: a
literal ``[0, 0, 0]`` is the op's "M-RoPE disabled" sentinel, so passing it always
collapses multimodal prompts to 1D positions. The axis map: ``cache_mode`` must
follow ``mrope_interleaved``, or a model needing the round-robin mapping gets the
contiguous one instead.

Guarded by comparing against ``forward_triton``, the path ``forward_cuda`` takes
for sectioned M-RoPE.
"""

import unittest

import torch

from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.utils import is_npu
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=30, suite="full-1-npu-a3", nightly=True)


@unittest.skipUnless(is_npu(), "Ascend NPU not available")
class TestMRotaryEmbeddingNpuSection(CustomTestCase):
    def _assert_matches_cuda_path(self, mrope_interleaved):
        head_dim = 128
        rot = MRotaryEmbedding(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=8192,
            base=10000,
            is_neox_style=False,
            dtype=torch.bfloat16,
            mrope_section=[16, 24, 24],
            mrope_interleaved=mrope_interleaved,
        ).to("npu")

        torch.manual_seed(0)
        seq, heads = 8, 16
        positions = (
            torch.stack(
                [
                    torch.arange(seq),
                    torch.arange(seq) * 2 + 1,
                    torch.arange(seq) * 3 + 5,
                ]
            )
            .to(torch.int64)
            .npu()
        )
        q = torch.randn(seq, heads * head_dim, dtype=torch.bfloat16, device="npu")

        npu_q, _ = rot.forward_npu(positions, q.clone(), q.clone())
        triton_q, _ = rot.forward_triton(positions, q.clone(), q.clone())
        self.assertTrue(torch.equal(npu_q, triton_q))

    def test_contiguous_axis_map(self):
        # GLM-OCR: lanes split as [0]*16 [1]*24 [2]*24.
        self._assert_matches_cuda_path(mrope_interleaved=False)

    def test_interleaved_axis_map(self):
        # Qwen2-VL: axes assigned round-robin, which the op calls cache_mode
        # 'interleave'. Dropping it silently applies the contiguous map instead.
        self._assert_matches_cuda_path(mrope_interleaved=True)


if __name__ == "__main__":
    unittest.main()
