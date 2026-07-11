"""Regression test: the final Mamba snapshot must land on a matchable position.

Prefix matching looks up at most page_aligned(input_len - 1). When the final
prefill chunk length is an exact multiple of mamba_cache_chunk_size, a snapshot
at the end position is therefore one chunk past what any later request can
match, and the whole prefix silently degrades to the last chunk-boundary
snapshot (or zero). The final track must move one chunk earlier in that case,
and only in that case.
"""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MCS = 64


class TestMambaFinalTrackAlignment(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        server_args = ServerArgs(model_path="dummy", page_size=_MCS)
        server_args._mamba_cache_chunk_size = _MCS
        set_global_server_args_for_scheduler(server_args)
        cls.batch = SimpleNamespace(req_to_token_pool=MagicMock())
        cls.batch.req_to_token_pool.get_mamba_ping_pong_other_idx.return_value = 1

    def _prepare(self, prefix_len, extend_len, fill_len):
        req = Req(
            rid="r",
            origin_input_text="",
            origin_input_ids=array("q"),
            sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
        )
        req.prefix_indices = torch.zeros(prefix_len, dtype=torch.int64)
        req.full_untruncated_fill_ids = array("q", range(fill_len))
        req.set_extend_range(prefix_len, prefix_len + extend_len)
        req.mamba_ping_pong_track_buffer = torch.tensor([3, 7], dtype=torch.int64)
        req.mamba_next_track_idx = 0
        req.mamba_branching_seqlen = None
        entry = ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend(
            self.batch, req
        )
        return entry, req

    def test_final_chunk_aligned_moves_one_chunk_earlier(self):
        entry, req = self._prepare(prefix_len=0, extend_len=5120, fill_len=5120)
        self.assertTrue(entry.track_mask)
        self.assertEqual(req.mamba_last_track_seqlen, 5056)
        self.assertEqual(entry.track_seqlen, 5057)

    def test_final_chunk_aligned_with_prefix(self):
        entry, req = self._prepare(prefix_len=8192, extend_len=8192, fill_len=16384)
        self.assertEqual(req.mamba_last_track_seqlen, 16320)
        self.assertEqual(entry.track_seqlen, 16321)

    def test_final_chunk_unaligned_unchanged(self):
        entry, req = self._prepare(prefix_len=0, extend_len=5121, fill_len=5121)
        self.assertEqual(req.mamba_last_track_seqlen, 5120)
        self.assertEqual(entry.track_seqlen, 5121)

    def test_final_chunk_single_chunk_guard(self):
        entry, req = self._prepare(prefix_len=8192, extend_len=_MCS, fill_len=8256)
        self.assertEqual(req.mamba_last_track_seqlen, 8256)
        self.assertEqual(entry.track_seqlen, 8256)

    def test_intermediate_chunk_unchanged(self):
        entry, req = self._prepare(prefix_len=0, extend_len=8192, fill_len=16385)
        self.assertEqual(req.mamba_last_track_seqlen, 8192)
        self.assertEqual(entry.track_seqlen, 8192)

    def test_below_chunk_not_tracked(self):
        entry, req = self._prepare(prefix_len=8192, extend_len=63, fill_len=8255)
        self.assertFalse(entry.track_mask)


if __name__ == "__main__":
    unittest.main()
