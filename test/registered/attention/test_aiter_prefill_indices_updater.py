"""kv_indices of the aiter prefill path are padded without a host read.

AiterIndicesUpdaterPrefill pads kv_indices past the last real index with the
first one. It must not read kv_indptr back for that: a host read stalls the CPU
until the GPU drains, once per prefill, and leaves the GPU idle while the next
step is prepared.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.aiter_backend import AiterIndicesUpdaterPrefill
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(torch.cuda.is_available(), "Test requires a GPU")
class TestAiterPrefillIndicesUpdater(unittest.TestCase):
    def _update(self, seq_lens, prefix_lens, seq_lens_sum):
        device = "cuda"
        bs = len(seq_lens)
        width = max(seq_lens) + 8
        g = torch.Generator().manual_seed(bs)
        # Row 0 is the padding row; requests use rows 1..bs of shuffled slots.
        req_to_token = (
            torch.randperm((bs + 1) * width, generator=g)
            .to(torch.int32)
            .reshape(bs + 1, width)
            .to(device)
        )
        updater = SimpleNamespace(
            kv_indptr=torch.zeros(bs + 2, dtype=torch.int32, device=device),
            qo_indptr=torch.zeros(bs + 2, dtype=torch.int32, device=device),
            req_to_token=req_to_token,
            kv_index_translator=None,
        )
        args = (
            torch.arange(1, bs + 1, dtype=torch.int64, device=device),
            torch.tensor(seq_lens, dtype=torch.int64, device=device),
            seq_lens_sum,
            torch.tensor(prefix_lens, dtype=torch.int64, device=device),
        )
        # Compile the index kernel first, then check that no call syncs.
        AiterIndicesUpdaterPrefill.update_single_wrapper(
            updater, *args, None, None, plan=None
        )
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            AiterIndicesUpdaterPrefill.update_single_wrapper(
                updater, *args, None, None, plan=None
            )
        finally:
            torch.cuda.set_sync_debug_mode(0)
        return updater, req_to_token.cpu()

    def _check(self, seq_lens, prefix_lens, seq_lens_sum):
        updater, req_to_token = self._update(seq_lens, prefix_lens, seq_lens_sum)
        kv_indices = updater.kv_indices.cpu()
        self.assertEqual(kv_indices.numel(), seq_lens_sum + 256)
        expected = torch.cat([req_to_token[i + 1, :n] for i, n in enumerate(seq_lens)])
        total = expected.numel()
        self.assertTrue(torch.equal(kv_indices[:total], expected))
        # Everything past the last real index repeats the first one.
        self.assertTrue(
            torch.equal(
                kv_indices[total:], expected[:1].expand(len(kv_indices) - total)
            )
        )
        extend = torch.tensor(seq_lens) - torch.tensor(prefix_lens)
        self.assertEqual(
            updater.qo_indptr[: len(seq_lens) + 1].tolist(),
            [0, *extend.cumsum(0).tolist()],
        )

    def test_single_sequence(self):
        self._check([16384], [0], 16384)
        self._check([40000], [23616], 40000)

    def test_several_sequences(self):
        self._check([300, 1, 4096], [0, 0, 1000], 4397)


if __name__ == "__main__":
    unittest.main()
