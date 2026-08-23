import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.test.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMambaTrackChunkSize(unittest.TestCase):
    def _prepare_track_entry(self, extend_length: int):
        batch = object.__new__(ScheduleBatch)
        batch.req_to_token_pool = SimpleNamespace(
            get_mamba_ping_pong_other_idx=lambda _: 1
        )
        batch.tree_cache = SimpleNamespace(page_size=128)
        req = SimpleNamespace(
            extend_range=SimpleNamespace(length=extend_length),
            prefix_indices=range(16384),
            mamba_ping_pong_track_buffer=torch.tensor([7, 8]),
            mamba_next_track_idx=0,
            mamba_branching_seqlen=None,
        )
        server_args = SimpleNamespace(
            mamba_cache_chunk_size=128,
            mamba_state_chunk_size=64,
        )

        with patch(
            "sglang.srt.managers.schedule_batch.get_server_args",
            return_value=server_args,
        ), patch(
            "sglang.srt.managers.schedule_batch.mamba_cache_chunk_size",
            return_value=128,
        ), patch(
            "sglang.srt.managers.schedule_batch.mamba_checkpoint_grid",
            return_value=128,
        ), patch(
            "sglang.srt.managers.schedule_batch.mamba_extra_buffer_lazy_enabled",
            return_value=False,
        ):
            entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)

        return entry, req

    def test_scheduler_tracks_cache_boundary_with_finer_state_chunks(self):
        entry, req = self._prepare_track_entry(12739)
        self.assertTrue(entry.track_mask)
        self.assertEqual(entry.track_index, 7)
        self.assertEqual(entry.track_seqlen, 29057)
        self.assertEqual(req.mamba_last_track_seqlen, 29056)

    def test_ssm_index_uses_kernel_state_chunk_size(self):
        backend = object.__new__(MambaAttnBackendBase)
        backend.device = torch.device("cpu")
        forward_batch = SimpleNamespace(
            mamba_track_mask=torch.tensor([True]),
            extend_seq_lens=torch.tensor([12739]),
            mamba_track_indices=torch.tensor([7]),
            mamba_track_seqlens=torch.tensor([29057]),
            extend_prefix_lens=torch.tensor([16384]),
        )
        server_args = SimpleNamespace(mamba_state_chunk_size=64)

        with patch(
            "sglang.srt.layers.attention.hybrid_linear_attn_backend.get_server_args",
            return_value=server_args,
        ):
            h_src, h_dst, final_src, final_dst = backend._init_track_ssm_indices(
                torch.tensor([3]), forward_batch
            )

        self.assertEqual(h_src.tolist(), [198])
        self.assertEqual(h_dst.tolist(), [7])
        self.assertEqual(final_src.tolist(), [])
        self.assertEqual(final_dst.tolist(), [])


if __name__ == "__main__":
    unittest.main()
