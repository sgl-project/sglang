"""Focused tests for persistent DCP a2a exchange buffers."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-large")


class TestDCPA2APersistentBuffers(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required")
        cls.device = "cuda"

    def setUp(self):
        super().setUp()
        from sglang.srt.layers.dcp import comm as dcp_comm

        dcp_comm._A2A_PERSISTENT_BUFFERS.clear()

    def tearDown(self):
        from sglang.srt.layers.dcp import comm as dcp_comm

        dcp_comm._A2A_PERSISTENT_BUFFERS.clear()
        super().tearDown()

    def _make_mock_group(self, world_size):
        group = MagicMock()
        group.world_size = world_size

        def identity_a2a(output, input_):
            output.copy_(input_)

        group.all_to_all_single = MagicMock(side_effect=identity_a2a)
        return group

    def _make_inputs(self, N=2, B=4, H_per_rank=8, D=128):
        H = N * H_per_rank
        attn_out = torch.randn(B, H, D, device=self.device, dtype=torch.bfloat16)
        attn_lse = torch.randn(B, H, device=self.device, dtype=torch.float32)
        return attn_out, attn_lse

    def test_same_shape_reuses_persistent_buffers(self):
        from sglang.srt.layers.dcp import comm as dcp_comm

        N = 2
        group = self._make_mock_group(N)
        attn_out, attn_lse = self._make_inputs(N=N)

        dcp_comm.dcp_a2a_lse_reduce(attn_out, attn_lse, group, is_lse_base_on_e=True)
        self.assertEqual(len(dcp_comm._A2A_PERSISTENT_BUFFERS), 1)

        first = next(iter(dcp_comm._A2A_PERSISTENT_BUFFERS.values()))
        send_ptr = first["send_combined"].data_ptr()
        recv_ptr = first["recv_combined"].data_ptr()

        dcp_comm.dcp_a2a_lse_reduce(attn_out, attn_lse, group, is_lse_base_on_e=True)

        self.assertEqual(len(dcp_comm._A2A_PERSISTENT_BUFFERS), 1)
        second = next(iter(dcp_comm._A2A_PERSISTENT_BUFFERS.values()))
        self.assertEqual(second["send_combined"].data_ptr(), send_ptr)
        self.assertEqual(second["recv_combined"].data_ptr(), recv_ptr)

    def test_first_seen_during_capture_is_not_cached(self):
        from sglang.srt.layers.dcp import comm as dcp_comm

        N = 2
        group = self._make_mock_group(N)
        attn_out, attn_lse = self._make_inputs(N=N)

        with patch("torch.cuda.is_current_stream_capturing", return_value=True):
            dcp_comm.dcp_a2a_lse_reduce(
                attn_out, attn_lse, group, is_lse_base_on_e=True
            )

        self.assertEqual(dcp_comm._A2A_PERSISTENT_BUFFERS, {})

    def test_explicit_cuda_graph_buffers_bypass_cache(self):
        from sglang.srt.layers.dcp import comm as dcp_comm

        N, B, H_per_rank, D = 2, 4, 8, 128
        group = self._make_mock_group(N)
        attn_out, attn_lse = self._make_inputs(N=N, B=B, H_per_rank=H_per_rank, D=D)

        graph_buffers = {
            "send_combined": torch.empty(
                N,
                B,
                H_per_rank,
                D + 2,
                device=self.device,
                dtype=torch.bfloat16,
            ),
            "recv_combined": torch.empty(
                N,
                B,
                H_per_rank,
                D + 2,
                device=self.device,
                dtype=torch.bfloat16,
            ),
        }

        dcp_comm.dcp_a2a_lse_reduce(
            attn_out,
            attn_lse,
            group,
            is_lse_base_on_e=True,
            cuda_graph_buffers=graph_buffers,
        )

        self.assertEqual(dcp_comm._A2A_PERSISTENT_BUFFERS, {})


if __name__ == "__main__":
    unittest.main()
