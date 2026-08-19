import unittest

import torch

from sglang.srt.layers.attention.vision import VisionSdpaAttention
from sglang.srt.utils import get_device
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

# Unit test for VisionSdpaAttention with flatten_batch=True. A caller that
# supplies its own attention_mask must be allowed to pass bsz > 1 (e.g. a video
# with multiple frames) -- previously this tripped `assert bsz == 1` (#28821).
register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


class TestVisionSdpaFlattenBatchBsz(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_default_device(get_device())

    def _run(self, bsz, seq_len=4, num_heads=2, head_dim=8):
        attn = VisionSdpaAttention(
            head_dim=head_dim,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            flatten_batch=True,
        )
        n = bsz * seq_len
        q = torch.randn(n, num_heads, head_dim)
        k = torch.randn(n, num_heads, head_dim)
        v = torch.randn(n, num_heads, head_dim)
        # Caller-supplied [bsz, 1, s, s] mask -> the flatten mask is not generated,
        # so bsz > 1 is handled by the normal batched path.
        mask = torch.ones(bsz, 1, seq_len, seq_len, dtype=torch.bool)
        out = attn.forward(q=q, k=k, v=v, bsz=bsz, attention_mask=mask)
        self.assertEqual(tuple(out.shape), (n, num_heads, head_dim))

    def test_bsz_one(self):
        self._run(bsz=1)

    def test_bsz_gt_one_with_mask(self):
        # bsz > 1 (e.g. multi-frame video through the SDPA mm attention backend)
        # must not trip the flatten_batch assert when a mask is provided (#28821).
        self._run(bsz=3)


if __name__ == "__main__":
    unittest.main()
