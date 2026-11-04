"""The post-verify KV move takes its targets from the verify window, not from
req_to_token: the scheduler may rewrite a finished request's row once verify's
read-done event fires, and a reread would move KV into the new owner's slots."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.spec_utils import move_accept_tokens_to_target_kvcache
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


class _RecordingKVCache:
    def move_kv_cache(self, tgt_loc, src_loc):
        self.tgt_loc, self.src_loc = tgt_loc.clone(), src_loc.clone()


@unittest.skipUnless(torch.cuda.is_available(), "the move kernels are Triton")
class TestMoveAcceptTokensTargets(CustomTestCase):
    def _run(self, window, accept_rows):
        device = "cuda"
        rows = torch.tensor([5, 2, 7], device=device)
        seq_lens = torch.tensor([10, 20, 30], device=device)
        req_to_token = torch.arange(512, dtype=torch.int32, device=device).view(8, 64)
        out_cache_loc = torch.cat(
            [req_to_token[r, s : s + window] for r, s in zip(rows, seq_lens)]
        ).to(torch.int64)
        accept_index = torch.tensor(accept_rows, dtype=torch.int32, device=device)
        num_accepted = (accept_index != -1).sum(dim=1)

        expected_tgt = torch.zeros(accept_index.numel(), dtype=torch.int64)
        packed = torch.cat(
            [
                req_to_token[r, s : s + n]
                for r, s, n in zip(rows, seq_lens, num_accepted)
            ]
        )
        expected_tgt[: packed.numel()] = packed.cpu()
        valid = accept_index[accept_index != -1].long()
        expected_src = torch.zeros(accept_index.numel(), dtype=torch.int64)
        expected_src[: valid.numel()] = out_cache_loc[valid].cpu()

        # The finished requests' rows go to new requests after read-done.
        req_to_token[rows] = -7

        kv_cache = _RecordingKVCache()
        move_accept_tokens_to_target_kvcache(
            SimpleNamespace(
                seq_lens=seq_lens,
                req_pool_indices=rows,
                req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
                out_cache_loc=out_cache_loc,
            ),
            accept_index,
            num_accepted - 1,
            SimpleNamespace(get_kvcache=lambda: kv_cache),
        )
        self.assertTrue(torch.equal(kv_cache.tgt_loc.cpu(), expected_tgt))
        self.assertTrue(torch.equal(kv_cache.src_loc.cpu(), expected_src))

    def test_chain_targets_ignore_rewritten_rows(self):
        # NGRAM: accept_index spans the whole window.
        self._run(
            window=6,
            accept_rows=[
                [0, 1, 2, -1, -1, -1],
                [6, -1, -1, -1, -1, -1],
                [12, 13, 14, 15, 16, 17],
            ],
        )

    def test_tree_targets_ignore_rewritten_rows(self):
        # topk > 1: the accepted path is shorter than the window and not a prefix of it.
        self._run(
            window=8,
            accept_rows=[
                [0, 2, 5, -1],
                [8, -1, -1, -1],
                [16, 19, 20, 23],
            ],
        )


if __name__ == "__main__":
    unittest.main()
