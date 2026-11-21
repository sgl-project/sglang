"""Single-die smoke test for NPU PLE offload with both host-table backends."""

import tempfile
import unittest
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.models.qwen4_exp import Qwen4ExpPinnedHostEmbedding


class TestNpuPleOffload(unittest.TestCase):
    def test_pinned_and_file_gather(self):
        if not torch.npu.is_available():
            self.skipTest("NPU not available")
        device = torch.device("npu:0")
        torch.npu.set_device(device)

        vocab_size, width = 128, 160
        generator = torch.Generator().manual_seed(42)
        weights = torch.randn(vocab_size, width, generator=generator).bfloat16()
        ids = torch.arange(65536) % (vocab_size + 2) - 1
        valid = (ids >= 0) & (ids < vocab_size)
        expected = weights[ids.masked_fill(~valid, 0)].masked_fill(~valid[:, None], 0)
        device_ids = ids.to(device)

        for backend in ("pinned", "file"):
            with (
                self.subTest(backend=backend),
                tempfile.TemporaryDirectory() as table_dir,
            ):
                with torch.device("meta"):
                    source = VocabParallelEmbedding(
                        vocab_size, width, params_dtype=torch.bfloat16, enable_tp=False
                    )
                source.register_buffer("weight_scale", torch.ones(1, device=device))
                embedding = Qwen4ExpPinnedHostEmbedding(
                    source, backend=backend, table_dir=table_dir
                )
                try:
                    embedding.weight_loader(embedding.weight, weights)
                    if backend == "file":
                        self.assertEqual(
                            Path(embedding.weight.untyped_storage().filename).parent,
                            Path(table_dir),
                        )
                    actual = embedding.gather(device_ids).cpu()
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                finally:
                    if embedding._file_rss_trimmer is not None:
                        embedding._file_rss_trimmer.close()
                    embedding._npu_uva_table.close()


if __name__ == "__main__":
    unittest.main()
