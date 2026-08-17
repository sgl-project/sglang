import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    MambaAttnBackendBase,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import Range
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestMambaCheckpointChunkSizes(unittest.TestCase):
    def test_npu_state_chunk_can_be_smaller_than_cache_chunk(self):
        server_args = object.__new__(ServerArgs)

        with (
            patch.object(
                ServerArgs,
                "get_model_config",
                return_value=SimpleNamespace(hf_config=SimpleNamespace()),
            ),
            patch(
                "sglang.srt.server_args.resolved_view",
                return_value=SimpleNamespace(page_size=128),
            ),
        ):
            self.assertEqual(server_args.npu_mamba_state_chunk_size, 64)
            self.assertEqual(server_args.mamba_cache_chunk_size, 128)

    def test_default_backend_keeps_cache_chunk_semantics(self):
        backend = object.__new__(MambaAttnBackendBase)
        with patch(
            "sglang.srt.layers.attention.hybrid_linear_attn_backend."
            "mamba_cache_chunk_size",
            return_value=128,
        ):
            self.assertEqual(backend._mamba_state_chunk_size(), 128)

    def test_scheduler_marks_interior_cache_boundary_for_h(self):
        batch = object.__new__(ScheduleBatch)
        batch.tree_cache = SimpleNamespace(page_size=128)
        batch.req_to_token_pool = SimpleNamespace(
            get_mamba_ping_pong_other_idx=lambda index: 1 - index
        )
        req = SimpleNamespace(
            extend_range=Range(0, 3008),
            prefix_indices=[],
            mamba_ping_pong_track_buffer=torch.tensor([10, 11]),
            mamba_next_track_idx=0,
            mamba_last_track_idx=None,
            mamba_last_track_seqlen=None,
            mamba_branching_seqlen=None,
        )

        with (
            patch(
                "sglang.srt.managers.schedule_batch.mamba_cache_chunk_size",
                return_value=128,
            ),
            patch(
                "sglang.srt.managers.schedule_batch.npu_mamba_state_chunk_size",
                return_value=64,
            ),
            patch("sglang.srt.managers.schedule_batch._is_npu", True),
            patch(
                "sglang.srt.managers.schedule_batch.mamba_checkpoint_grid",
                return_value=128,
            ),
            patch(
                "sglang.srt.managers.schedule_batch.mamba_extra_buffer_lazy_enabled",
                return_value=False,
            ),
        ):
            entry = batch._mamba_radix_cache_v2_req_prepare_for_extend(req)

        self.assertTrue(entry.track_mask)
        self.assertEqual(entry.track_seqlen, 2945)
        self.assertEqual(req.mamba_last_track_seqlen, 2944)

    def test_gdn_h_indices_use_state_chunk_not_cache_chunk(self):
        backend = object.__new__(MambaAttnBackendBase)
        backend.device = torch.device("cpu")
        forward_batch = SimpleNamespace(
            mamba_track_mask=torch.tensor([True, True]),
            extend_seq_lens=torch.tensor([3000, 3008]),
            mamba_track_indices=torch.tensor([20, 21]),
            mamba_track_seqlens=torch.tensor([3000, 2945]),
            extend_prefix_lens=torch.tensor([0, 0]),
        )

        with patch.object(backend, "_mamba_state_chunk_size", return_value=64):
            h_src, h_dst, final_src, final_dst = backend._init_track_ssm_indices(
                torch.tensor([10, 11]), forward_batch
            )

        torch.testing.assert_close(h_src, torch.tensor([46, 93]))
        torch.testing.assert_close(h_dst, torch.tensor([20, 21]))
        self.assertEqual(final_src.numel(), 0)
        self.assertEqual(final_dst.numel(), 0)


if __name__ == "__main__":
    unittest.main()
