import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers.attention.cutedsl_mla_backend import CuteDslMLABackend
from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCuteDslMLADCPMetadata(CustomTestCase):
    @patch("sglang.srt.layers.attention.cutedsl_mla_backend.get_parallel")
    def test_local_max_seq_len_uses_cyclic_partition(self, get_parallel):
        backend = object.__new__(CuteDslMLABackend)
        get_parallel.return_value = SimpleNamespace(
            dcp_enabled=True, dcp_size=4, dcp_rank=1
        )

        self.assertEqual(backend._get_dcp_local_max_seq_len(10), 3)

    @patch("sglang.srt.layers.attention.cutedsl_mla_backend.get_parallel")
    def test_local_max_seq_len_depends_on_rank(self, get_parallel):
        backend = object.__new__(CuteDslMLABackend)
        get_parallel.return_value = SimpleNamespace(
            dcp_enabled=True, dcp_size=4, dcp_rank=3
        )

        self.assertEqual(backend._get_dcp_local_max_seq_len(10), 2)

    @patch("sglang.srt.layers.attention.cutedsl_mla_backend.get_parallel")
    def test_local_max_seq_len_stays_positive(self, get_parallel):
        backend = object.__new__(CuteDslMLABackend)
        get_parallel.return_value = SimpleNamespace(
            dcp_enabled=True, dcp_size=4, dcp_rank=3
        )

        self.assertEqual(backend._get_dcp_local_max_seq_len(1), 1)

    @patch("sglang.srt.layers.attention.cutedsl_mla_backend.get_parallel")
    @patch.object(TRTLLMMLABackend, "_init_cuda_graph_metadata")
    def test_target_verify_graph_metadata_keeps_global_lengths(
        self, parent_init, get_parallel
    ):
        backend = object.__new__(CuteDslMLABackend)
        backend.max_context_len = 128
        backend.num_draft_tokens = 4
        backend.forward_decode_metadata = SimpleNamespace(
            seq_lens_k=torch.zeros(2, dtype=torch.int32),
            max_seq_len_k=128,
        )
        get_parallel.return_value = SimpleNamespace(
            dcp_enabled=True, dcp_size=2, dcp_rank=0
        )
        forward_mode = MagicMock()
        forward_mode.is_target_verify.return_value = True

        backend._init_cuda_graph_metadata(
            bs=2,
            num_tokens=8,
            forward_mode=forward_mode,
            seq_lens=torch.ones(2, dtype=torch.int32),
            device=torch.device("cpu"),
        )

        parent_init.assert_called_once()
        self.assertEqual(backend.forward_decode_metadata.global_seq_lens_k.shape, (2,))
        self.assertEqual(backend.forward_decode_metadata.max_seq_len_k, (128 + 4) // 2)


if __name__ == "__main__":
    unittest.main()
