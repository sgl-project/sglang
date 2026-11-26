import unittest

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.beam_shared_context import shared_context_attention
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-large")


class TestBeamSharedContext(CustomTestCase):
    def test_ragged_suffix_matches_independent_attention(self):
        for head_dim, value_dim in ((256, 256), (576, 512)):
            with self.subTest(head_dim=head_dim, value_dim=value_dim):
                self._check_ragged_suffix(head_dim=head_dim, value_dim=value_dim)

    def _check_ragged_suffix(self, *, head_dim, value_dim):
        device = "cuda"
        torch.manual_seed(0)
        width, q_heads, kv_heads = 73, 4, 2
        q = torch.randn(
            2 * width, q_heads, head_dim, dtype=torch.bfloat16, device=device
        )
        k = torch.randn(503, kv_heads, head_dim, dtype=q.dtype, device=device)
        v = torch.randn_like(k) if value_dim == head_dim else k[:, :, :value_dim]
        rows = torch.randperm(2 * width, device=device).reshape(2, width)
        slots = torch.randperm(k.shape[0], device=device, dtype=torch.int32)
        prompts = (slots[:65], slots[65:211])
        suffix = slots[211:503].reshape(2, 2, width)
        out = shared_context_attention(
            q,
            k,
            v,
            rows,
            slots[:211],
            torch.tensor([0, 65, 211], dtype=torch.int32, device=device),
            suffix.reshape(2, -1),
            torch.tensor([1, 2], dtype=torch.int32, device=device),
            softmax_scale=head_dim**-0.5,
        )

        expected = q.new_empty((2 * width, q_heads, value_dim))
        for group, (prompt, decode_len) in enumerate(zip(prompts, (1, 2), strict=True)):
            beam_slots = torch.cat(
                [prompt.expand(width, -1), suffix[group, :decode_len].T], dim=1
            ).long()
            query = q[rows[group]].unsqueeze(2).float()
            keys, values = (
                buf[beam_slots].transpose(1, 2).repeat_interleave(2, dim=1).float()
                for buf in (k, v)
            )
            expected[rows[group]] = (
                F.scaled_dot_product_attention(query, keys, values)
                .squeeze(2)
                .to(q.dtype)
            )
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=5e-3)


if __name__ == "__main__":
    unittest.main()
