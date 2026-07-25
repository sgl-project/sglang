import unittest
from unittest import mock

import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


class _DeviceSeqLensGuard:
    def cpu(self):
        raise AssertionError("device seq_lens must not be copied to the host")

    def item(self):
        raise AssertionError("device seq_lens must not be scalarized on the host")

    def tolist(self):
        raise AssertionError("device seq_lens must not be materialized on the host")


@unittest.skipUnless(torch.cuda.is_available(), "GPU-backed DSV4 import is required")
class TestDSV4DSparkDeviceLengths(unittest.TestCase):
    def test_device_only_draft_block_never_materializes_seq_lens_on_host(self):
        from sglang.srt.layers.attention.deepseek_v4_backend import (
            DSV4Metadata,
            DeepseekV4AttnBackend,
        )

        backend = object.__new__(DeepseekV4AttnBackend)
        backend.req_to_token = object()
        seq_lens = _DeviceSeqLensGuard()
        seq_lens_casual = torch.tensor([11, 12, 13, 21, 22, 23])
        core_attn_metadata = object()
        backend._dspark_seq_lens_casual = mock.Mock(return_value=seq_lens_casual)
        backend.make_core_attn_metadata = mock.Mock(return_value=core_attn_metadata)

        metadata = backend.init_forward_metadata_dspark_draft_block(
            max_seq_len=4096,
            req_pool_indices=torch.tensor([4, 7]),
            seq_lens=seq_lens,
            seq_lens_cpu=None,
            out_cache_loc=torch.arange(6),
            block_size=3,
        )

        self.assertIsInstance(metadata, DSV4Metadata)
        self.assertIs(metadata.core_attn_metadata, core_attn_metadata)
        self.assertIsNone(metadata.indexer_metadata)
        backend._dspark_seq_lens_casual.assert_called_once_with(
            seq_lens=seq_lens, block_size=3
        )
        kwargs = backend.make_core_attn_metadata.call_args.kwargs
        torch.testing.assert_close(
            kwargs["req_pool_indices_repeated"], torch.tensor([4, 4, 4, 7, 7, 7])
        )
        self.assertIs(kwargs["seq_lens_casual"], seq_lens_casual)
        self.assertFalse(kwargs["need_compress"])
        self.assertTrue(kwargs["is_prefill"])
        self.assertEqual(kwargs["dspark_block_size"], 3)


if __name__ == "__main__":
    unittest.main()
