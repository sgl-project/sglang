"""Deterministic fa2 prefill chunked on the 128-token grid must match the
unchunked prefill bit for bit; a chunk off the grid must not."""

import unittest

import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from sglang.srt.layers.attention.flashinfer_backend import PREFILL_Q_TILE_TOKENS
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

HEAD_DIM = 128
SPLIT_TILE = 4096
SEQ_LEN = 12 * PREFILL_Q_TILE_TOKENS
DTYPE = torch.bfloat16


def _prefill(wrapper, q, k, v, q_start, q_len, kv_len, num_qo_heads, num_kv_heads):
    device = q.device
    wrapper.plan(
        torch.tensor([0, q_len], dtype=torch.int32, device=device),
        torch.tensor([0, kv_len], dtype=torch.int32, device=device),
        torch.arange(kv_len, dtype=torch.int32, device=device),
        torch.ones(1, dtype=torch.int32, device=device),
        num_qo_heads,
        num_kv_heads,
        HEAD_DIM,
        1,
        causal=True,
        q_data_type=DTYPE,
        kv_data_type=DTYPE,
        fixed_split_size=SPLIT_TILE,
    )
    return wrapper.run(q[q_start : q_start + q_len], (k, v))


class TestFlashinferDeterministicChunkInvariance(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        workspace = torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        cls.wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )

    def _chunked_equals_full(self, *, num_qo_heads, num_kv_heads, bounds):
        gen = torch.Generator(device="cuda").manual_seed(0)
        q = torch.randn(
            SEQ_LEN, num_qo_heads, HEAD_DIM, dtype=DTYPE, device="cuda", generator=gen
        )
        k = torch.randn(
            SEQ_LEN, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda", generator=gen
        )
        v = torch.randn(
            SEQ_LEN, num_kv_heads, HEAD_DIM, dtype=DTYPE, device="cuda", generator=gen
        )
        args = (self.wrapper, q, k, v)
        full = _prefill(*args, 0, SEQ_LEN, SEQ_LEN, num_qo_heads, num_kv_heads)
        outs = []
        start = 0
        for end in list(bounds) + [SEQ_LEN]:
            outs.append(
                _prefill(*args, start, end - start, end, num_qo_heads, num_kv_heads)
            )
            start = end
        return torch.equal(torch.cat(outs), full)

    def test_chunks_on_tile_grid_match_unchunked(self):
        for num_qo_heads, num_kv_heads in ((8, 2), (8, 8)):
            for step in (1, 2, 3):
                bounds = range(
                    step * PREFILL_Q_TILE_TOKENS, SEQ_LEN, step * PREFILL_Q_TILE_TOKENS
                )
                with self.subTest(gqa=num_qo_heads // num_kv_heads, step=step):
                    self.assertTrue(
                        self._chunked_equals_full(
                            num_qo_heads=num_qo_heads,
                            num_kv_heads=num_kv_heads,
                            bounds=bounds,
                        )
                    )

    def test_chunk_off_tile_grid_differs(self):
        for num_qo_heads, num_kv_heads, bound in ((8, 2, 100), (8, 8, 64)):
            with self.subTest(gqa=num_qo_heads // num_kv_heads, bound=bound):
                self.assertFalse(
                    self._chunked_equals_full(
                        num_qo_heads=num_qo_heads,
                        num_kv_heads=num_kv_heads,
                        bounds=(bound,),
                    )
                )


if __name__ == "__main__":
    unittest.main()
