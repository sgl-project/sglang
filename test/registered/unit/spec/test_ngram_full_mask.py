import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
maybe_stub_sgl_kernel()

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative import ngram_worker as ngram_worker_mod
from sglang.srt.speculative.ngram_worker import (
    NGRAMWorker,
    _prepare_full_verify_mask_buffer,
)


def _reference_full_verify_mask(
    seq_lens_cpu: list[int],
    mask: np.ndarray,
    draft_token_num: int,
) -> torch.Tensor:
    bs = len(seq_lens_cpu)
    if bs == 0:
        return torch.ones((0,), dtype=torch.bool)
    mask_3d = mask.reshape(bs, draft_token_num, draft_token_num)
    parts = []
    for i in range(bs):
        ones = torch.ones((draft_token_num, seq_lens_cpu[i]), dtype=torch.bool)
        req_mask = torch.cat((ones, torch.from_numpy(mask_3d[i])), dim=1)
        parts.append(req_mask.flatten())
    return torch.cat(parts, dim=0)


class TestNgramFullVerifyMask(CustomTestCase):
    def test_prepare_full_verify_mask_buffer_matches_reference(self):
        rng = np.random.default_rng(42)
        for dtype in (torch.int32, torch.int64):
            for draft_token_num in (1, 4, 8):
                for seq_lens_cpu in ([7], [1, 19, 3, 64], [0, 12, 5], [31] * 16):
                    with self.subTest(
                        dtype=dtype,
                        draft_token_num=draft_token_num,
                        seq_lens_cpu=seq_lens_cpu,
                    ):
                        bs = len(seq_lens_cpu)
                        seq_lens = torch.tensor(seq_lens_cpu, dtype=dtype)
                        mask = rng.integers(
                            0, 2, size=(bs * draft_token_num * draft_token_num,)
                        ).astype(np.bool_)
                        tree_mask = torch.from_numpy(mask.copy())

                        full_mask, dst_indices = _prepare_full_verify_mask_buffer(
                            seq_lens=seq_lens,
                            seq_lens_sum=sum(seq_lens_cpu),
                            draft_token_num=draft_token_num,
                        )
                        full_mask[dst_indices] = tree_mask

                        expected = _reference_full_verify_mask(
                            seq_lens_cpu, mask, draft_token_num
                        )
                        self.assertEqual(full_mask.dtype, torch.bool)
                        self.assertTrue(torch.equal(full_mask, expected))

    def test_prepare_for_speculative_decoding_full_mask_path(self):
        bs = 3
        draft_token_num = 4
        seq_lens_cpu = [5, 11, 2]
        seq_lens = torch.tensor(seq_lens_cpu, dtype=torch.int32)
        rng = np.random.default_rng(7)
        mask = rng.integers(0, 2, size=(bs * draft_token_num * draft_token_num,)).astype(
            np.bool_
        )
        req_drafts = np.arange(bs * draft_token_num, dtype=np.int64)

        worker = object.__new__(NGRAMWorker)
        worker.device = "cpu"
        worker.draft_token_num = draft_token_num
        worker.retrieve_indexes_batch = {
            bs: torch.zeros((bs, draft_token_num), dtype=torch.int64)
        }
        worker.retrieve_next_token_batch = {
            bs: torch.zeros((bs, draft_token_num), dtype=torch.int64)
        }
        worker.retrieve_next_sibling_batch = {
            bs: torch.zeros((bs, draft_token_num), dtype=torch.int64)
        }
        worker.positions_batch = {
            bs: torch.zeros((bs * draft_token_num,), dtype=torch.int64)
        }
        preserved_tree_mask = torch.zeros(
            (bs * draft_token_num * draft_token_num,), dtype=torch.bool
        )
        worker.tree_mask_batch = {bs: preserved_tree_mask}
        worker.draft_tokens_batch = {
            bs: torch.zeros((bs * draft_token_num,), dtype=torch.int64)
        }
        worker._prepare_draft_tokens = lambda batch: (req_drafts, mask)

        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            reqs=[object() for _ in range(bs)],
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=None,
            has_grammar=False,
            req_pool_indices=torch.arange(bs, dtype=torch.int32),
            req_to_token_pool=SimpleNamespace(
                req_to_token=torch.zeros((bs, 32), dtype=torch.int32)
            ),
            mamba_track_indices=None,
        )

        with (
            patch.object(ngram_worker_mod, "_is_cpu", False),
            patch.object(ngram_worker_mod, "USE_FULL_MASK", True),
            patch.object(
                ngram_worker_mod,
                "reconstruct_indices_from_tree_mask_func",
                 lambda *args, **kwargs: None,
            ),
            patch.object(
                ngram_worker_mod,
                "assign_extend_cache_locs_func",
                lambda **kwargs: torch.zeros((bs * draft_token_num,), dtype=torch.int64),
            ),
            patch.object(
                ngram_worker_mod,
                "prepare_mamba_track_for_verify",
                lambda _batch: None,
            ),
        ):
            worker._prepare_for_speculative_decoding(batch)

        expected = _reference_full_verify_mask(seq_lens_cpu, mask, draft_token_num)
        self.assertTrue(torch.equal(batch.spec_info.custom_mask, expected))
        self.assertTrue(
            torch.equal(worker.tree_mask_batch[bs], torch.from_numpy(mask))
        )


if __name__ == "__main__":
    unittest.main()
