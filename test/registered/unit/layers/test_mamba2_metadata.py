import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.mamba.mamba2_metadata import (
    ForwardMetadata,
    Mamba2Metadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMamba2Metadata(unittest.TestCase):
    def test_dp_idle_fabricated_prefill_has_no_decode_requests(self):
        forward_metadata = ForwardMetadata(
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            mamba_cache_indices=torch.tensor([-1], dtype=torch.int32),
        )
        forward_batch = SimpleNamespace(
            batch_size=1,
            seq_lens=torch.tensor([1], dtype=torch.int32),
            extend_num_tokens=1,
            extend_seq_lens=torch.tensor([1], dtype=torch.int32),
            extend_seq_lens_cpu=[1],
            extend_prefix_lens=torch.tensor([0], dtype=torch.int32),
            _original_batch_size=0,
            _original_forward_mode=ForwardMode.IDLE,
            forward_mode=ForwardMode.EXTEND,
            spec_info=None,
        )

        metadata = Mamba2Metadata.prepare_mixed(
            forward_metadata,
            chunk_size=256,
            forward_batch=forward_batch,
        )

        self.assertEqual(metadata.num_prefills, 1)
        self.assertEqual(metadata.num_prefill_tokens, 1)
        self.assertEqual(metadata.num_decodes, 0)


if __name__ == "__main__":
    unittest.main()
