import unittest
import math

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F
from utils import precision

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-b-test-cpu")

HEAD_DIM = 128
FP8_DTYPE = torch.float8_e4m3fn
RAND_STD = 10.0
RAND_MEAN = 10.0


def fp8_mqa_logits_torch(
    q_fp8: torch.Tensor,
    k_fp8: torch.Tensor,
    k_scale: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Reference: out[i, j] = sum_h relu(q[i, h] . k[j]) * weight[i, h] * k_scale[j]."""
    q = q_fp8.to(torch.float32)
    k = k_fp8.to(torch.float32)
    score = torch.einsum("mhd,nd->mhn", q, k)
    score = F.relu(score) * weight.unsqueeze(-1)
    logits = score.sum(dim=1)
    return logits * k_scale.unsqueeze(0)


def _build_ragged_batch(
    request_lens,
    *,
    num_heads: int,
    gaps=None,
    q_dtype: torch.dtype = torch.bfloat16,
):
    """Build a ragged multi-request batch: request r contributes `request_lens[r]`
    query/key tokens, with causal self-attention scoped to its own local k range.
    `gaps[r]` (if given) inserts that many irrelevant filler k tokens BEFORE
    request r's own k range in the shared flat k buffer, so adjacent requests'
    k-ranges can be placed arbitrarily far apart (never read by any row's own
    [ks, ke), so must not affect correctness)."""
    torch.manual_seed(0)
    batch_size = len(request_lens)
    gaps = gaps or [0] * batch_size

    num_q = sum(request_lens)
    num_k = sum(request_lens) + sum(gaps)

    q = (torch.randn(num_q, num_heads, HEAD_DIM) * RAND_STD + RAND_MEAN).to(q_dtype)
    q_fp8 = q.to(FP8_DTYPE).contiguous()
    k = (torch.randn(num_k, HEAD_DIM) * RAND_STD + RAND_MEAN).to(q_dtype)
    k_fp8 = k.to(FP8_DTYPE).contiguous()
    k_scale = torch.rand(num_k, dtype=torch.float32) * 0.5 + 0.75
    weight = torch.randn(num_q, num_heads, dtype=torch.float32) * RAND_STD + RAND_MEAN

    ks = torch.empty(num_q, dtype=torch.int32)
    ke = torch.empty(num_q, dtype=torch.int32)
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32)

    q_pos = 0
    k_pos = 0
    for r, length in enumerate(request_lens):
        k_pos += gaps[r]
        k_start = k_pos
        for p in range(length):
            ks[q_pos + p] = k_start
            ke[q_pos + p] = k_start + p + 1
        q_pos += length
        k_pos += length
        cu_seqlens_q[r + 1] = q_pos

    return q_fp8, k_fp8, k_scale, weight, ks, ke, cu_seqlens_q


class TestFp8MqaLogitsCPU(CustomTestCase):
    def _assert_matches_reference(
        self, request_lens, *, num_heads=4, gaps=None, q_dtype=torch.bfloat16
    ):
        q_fp8, k_fp8, k_scale, weight, ks, ke, cu_seqlens_q = _build_ragged_batch(
            request_lens, num_heads=num_heads, gaps=gaps, q_dtype=q_dtype
        )

        actual = torch.ops.sgl_kernel.fp8_mqa_logits_cpu(
            q_fp8, k_fp8, k_scale, weight, ks, ke, cu_seqlens_q, False, 0
        )
        expected = fp8_mqa_logits_torch(q_fp8, k_fp8, k_scale, weight)

        self.assertEqual(actual.shape, (q_fp8.shape[0], k_fp8.shape[0]))
        self.assertEqual(actual.dtype, torch.float32)

        atol = rtol = precision[q_dtype]
        for i in range(q_fp8.shape[0]):
            s, e = int(ks[i].item()), int(ke[i].item())
            if s >= e:
                continue
            torch.testing.assert_close(
                actual[i, s:e], expected[i, s:e], atol=atol, rtol=rtol
            )

    def test_single_request(self):
        # batch_size == 1 - the only shape reachable in production today.
        self._assert_matches_reference([64])

    def test_multi_request_unaligned_lengths(self):
        # Lengths deliberately not multiples of BLOCK_M(32), including a request
        # boundary that lands mid-tile - exercises request-boundary-aligned tiling.
        self._assert_matches_reference([5, 17, 40, 100])

    def test_multi_request_far_apart_k_ranges(self):
        # Adjacent requests' k-ranges separated by a large gap of irrelevant
        # filler k tokens: guards against a tile straddling the boundary and
        # pulling the gap into its KV bounding box.
        self._assert_matches_reference([20, 20], gaps=[0, 5000])

    def test_many_small_requests(self):
        self._assert_matches_reference([3] * 20 + [7] * 10)


if __name__ == "__main__":
    unittest.main()
