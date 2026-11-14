"""AITER paged decode must give a request the same bits whatever else is in the batch.

The AITER backend's non-MLA decode on gfx950 (``paged_attention_ragged``) splits
each request's KV into fixed 256-token partitions, so a request's output does not
depend on its batchmates. Deterministic inference needs exactly that from decode
attention, but nothing tests it: an AITER update that changed it, for example by
choosing the number of KV partitions from the batch, would go unnoticed. This
decodes a request alone and next to batchmates of other lengths and positions,
and compares its output bit for bit.
"""

import random
import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

try:
    from aiter import paged_attention_ragged

    from sglang.srt.layers.attention.aiter_backend import _AITER_PARTITION_SIZE_ROCM
except ImportError:
    paged_attention_ragged = None

NUM_HEADS, NUM_KV_HEADS, HEAD_DIM = 32, 8, 128
MAX_CONTEXT = 16384
# Target lengths cover one partial partition, partition edges and many partitions.
TARGET_LENS = (1, 255, 256, 257, 4000, 12000)
BATCH_SIZES = (2, 8, 33)


@unittest.skipIf(
    paged_attention_ragged is None or not is_gfx95_supported(),
    "needs AITER on gfx950",
)
class TestAiterDecodeBatchInvariance(CustomTestCase):
    def _decode(self, q, k_cache, v_cache, seq_lens, kv_cache_dtype, scale):
        """Decode one token per request; request i's KV is a contiguous slot range."""
        device = q.device
        bs = len(seq_lens)
        kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
        kv_indptr[1:] = torch.cumsum(
            torch.tensor(seq_lens, dtype=torch.int32, device=device), 0
        )
        kv_indices = torch.arange(sum(seq_lens), dtype=torch.int32, device=device)
        num_partitions = (MAX_CONTEXT + _AITER_PARTITION_SIZE_ROCM - 1) // (
            _AITER_PARTITION_SIZE_ROCM
        )
        # Sized as in AiterAttnBackend: fp32 partial outputs plus two fp32 stats.
        workspace = torch.empty(
            bs * NUM_HEADS * num_partitions * HEAD_DIM * 4
            + 2 * bs * NUM_HEADS * num_partitions * 4,
            dtype=torch.uint8,
            device=device,
        )
        out = torch.empty(bs, NUM_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
        one = torch.tensor([1.0], dtype=torch.float32, device=device)
        paged_attention_ragged(
            out,
            workspace,
            q,
            k_cache.view(-1, 1, NUM_KV_HEADS, HEAD_DIM),
            v_cache.view(-1, 1, NUM_KV_HEADS, HEAD_DIM),
            scale,
            kv_indptr,
            kv_indices,
            torch.ones(bs, dtype=torch.int32, device=device),
            1,
            num_partitions,
            None,
            kv_cache_dtype,
            "NHD",
            0.0,
            one,
            one,
            None,
            _AITER_PARTITION_SIZE_ROCM,
        )
        return out

    def _check(self, kv_cache_dtype, cache_dtype):
        device = "cuda"
        rng = random.Random(0)
        torch.manual_seed(0)
        scale = HEAD_DIM**-0.5

        def request(n):
            k = torch.randn(n, NUM_KV_HEADS, HEAD_DIM, device=device).to(cache_dtype)
            v = torch.randn(n, NUM_KV_HEADS, HEAD_DIM, device=device).to(cache_dtype)
            q = torch.randn(NUM_HEADS, HEAD_DIM, device=device, dtype=torch.bfloat16)
            return q, k, v

        for target_len in TARGET_LENS:
            target = request(target_len)
            alone = self._decode(
                target[0][None],
                target[1],
                target[2],
                [target_len],
                kv_cache_dtype,
                scale,
            )[0]
            for bs in BATCH_SIZES:
                mates = [request(rng.randrange(1, MAX_CONTEXT)) for _ in range(bs - 1)]
                pos = rng.randrange(bs)
                reqs = mates[:pos] + [target] + mates[pos:]
                out = self._decode(
                    torch.stack([r[0] for r in reqs]),
                    torch.cat([r[1] for r in reqs]),
                    torch.cat([r[2] for r in reqs]),
                    [r[1].shape[0] for r in reqs],
                    kv_cache_dtype,
                    scale,
                )
                self.assertTrue(
                    torch.equal(out[pos], alone),
                    f"{kv_cache_dtype}: a {target_len}-token request decoded at position "
                    f"{pos} of {bs} differs from the same request alone",
                )

    def test_bf16_kv_cache(self):
        self._check("auto", torch.bfloat16)

    def test_fp8_kv_cache(self):
        self._check("fp8_e4m3", torch.float8_e4m3fn)


if __name__ == "__main__":
    unittest.main()
