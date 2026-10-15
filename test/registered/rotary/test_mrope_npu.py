"""M-RoPE section handling in ``MRotaryEmbedding.forward_npu`` on Ascend NPU.

``forward_npu`` used to pass a literal ``mrope_section=[0, 0, 0]`` to
``torch_npu.npu_mrope``. The CANN op documents all-zero section as "M-RoPE not
enabled", so every axis got zero rotary lanes and multimodal prompts received
degenerate 1D positions -- on NPU only, and silently: the output looks
plausible, accuracy just degrades.

This guards the fix by comparing the NPU path against ``forward_triton``, the
path ``forward_cuda`` takes for sectioned M-RoPE.
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
    def test_forward_npu_matches_cuda_path(self):
        head_dim = 128
        rot = MRotaryEmbedding(
            head_size=head_dim,
            rotary_dim=head_dim,
            max_position_embeddings=8192,
            base=10000,
            is_neox_style=False,
            dtype=torch.bfloat16,
            mrope_section=[16, 24, 24],
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
        k = q.clone()

        npu_q, _ = rot.forward_npu(positions, q.clone(), k.clone())
        triton_q, _ = rot.forward_triton(positions, q.clone(), k.clone())

        # Before the fix these differed by ~1.8 (1D positions instead of 3-axis);
        # with the section passed through they are bit-identical.
        self.assertTrue(torch.equal(npu_q, triton_q))


if __name__ == "__main__":
    unittest.main()
