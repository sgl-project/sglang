"""Exercise AITER's token-level prefill index builder on fixed pools."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.aiter_backend import AiterIndicesUpdaterPrefill
from sglang.srt.mem_cache.kv.alloc.id_space import IdSpace, IdSpaceKind
from sglang.srt.mem_cache.kv.alloc.runner_ids import RunnerIds
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=10, suite="base-b-test-1-gpu-small-amd")


@unittest.skipUnless(
    torch.version.hip and torch.cuda.is_available(), "requires ROCm GPU"
)
class TestAiterFixedPoolPrefillIndices(unittest.TestCase):
    def test_page_sizes(self):
        for page_size in (1, 16):
            with self.subTest(page_size=page_size):
                self._check(page_size)

    def _check(self, page_size):
        rows = (2, 0, 1)
        lengths = (33, 5, 17)
        # Reverse page order so successive logical pages are not contiguous.
        table = (
            torch.arange(page_size, page_size + 3 * 48, dtype=torch.int32)
            .view(3, -1, page_size)
            .flip(1)
            .reshape(3, 48)
        )
        req_pool_indices = torch.tensor(rows, dtype=torch.int32, device="cuda")
        seq_lens_cpu = torch.tensor(lengths, dtype=torch.int32)
        seq_lens = seq_lens_cpu.cuda()

        # Index building needs only the fixed pool's identity ID space, not K/V buffers.
        full = IdSpace(key=(IdSpaceKind.FULL, None))
        runner_ids = RunnerIds(
            req_to_token=table.cuda(),
            token_to_kv_pool_allocator=None,
            token_to_kv_pool=SimpleNamespace(
                id_spaces=lambda: {IdSpaceKind.FULL: full}
            ),
            page_size=page_size,
            device="cuda",
        )
        plan = runner_ids.plan(
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            write_virtual=None,
        )
        # Set only the updater state used for metadata, without constructing a model.
        updater = AiterIndicesUpdaterPrefill.__new__(AiterIndicesUpdaterPrefill)
        updater.runner_ids = runner_ids
        updater.kv_indptr = torch.zeros(len(rows) + 1, dtype=torch.int32, device="cuda")
        updater.qo_indptr = torch.zeros_like(updater.kv_indptr)

        with patch.object(
            runner_ids, "pack_read_stream", wraps=runner_ids.pack_read_stream
        ) as pack:
            updater.update_single_wrapper(
                req_pool_indices,
                seq_lens,
                sum(lengths),
                prefix_lens=torch.zeros_like(seq_lens),
                encoder_lens=None,
                spec_info=None,
                plan=plan,
            )
        pack.assert_called_once()

        expected_indptr = torch.tensor((0, 33, 38, 55), dtype=torch.int32)
        expected_indices = torch.cat([table[row, :n] for row, n in zip(rows, lengths)])
        torch.testing.assert_close(updater.kv_indptr.cpu(), expected_indptr)
        # The additional 256 entries are padding for the attention consumer.
        torch.testing.assert_close(
            updater.kv_indices[: sum(lengths)].cpu(), expected_indices
        )


if __name__ == "__main__":
    unittest.main()
