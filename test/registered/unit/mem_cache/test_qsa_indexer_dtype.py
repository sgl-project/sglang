"""CPU unit tests for ``--qsa-indexer-dtype``: CLI value resolution and the
per-token cell it charges the KV budget for."""

import unittest

import torch

from sglang.srt.mem_cache.qsa_kv_pool import (
    QSA_INDEXER_DTYPE_CHOICES,
    QSATokenToKVPool,
    resolve_qsa_indexer_dtype,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestQSAIndexerDtype(CustomTestCase):
    def test_choices_resolve_to_storage_dtypes(self):
        self.assertEqual(resolve_qsa_indexer_dtype("auto"), torch.bfloat16)
        self.assertEqual(resolve_qsa_indexer_dtype("bfloat16"), torch.bfloat16)
        self.assertEqual(resolve_qsa_indexer_dtype("fp8_e4m3"), torch.float8_e4m3fn)
        # Every CLI choice resolves.
        for name in QSA_INDEXER_DTYPE_CHOICES:
            resolve_qsa_indexer_dtype(name)

    def test_unknown_choice_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported --qsa-indexer-dtype"):
            resolve_qsa_indexer_dtype("fp8_e5m2")

    def test_fp8_halves_the_compressed_cell(self):
        # Qwen3.8-Flash-Next shape; the cell is the compressed key only.
        shape = dict(kv_heads=1, head_dim=128, compress_ratio=4, num_layers=12)
        bf16 = QSATokenToKVPool.qsa_bytes_per_token(**shape)
        fp8 = QSATokenToKVPool.qsa_bytes_per_token(
            **shape, compressed_dtype=torch.float8_e4m3fn
        )
        self.assertEqual(bf16, 128 * 2 // 4 * 12)
        self.assertEqual(fp8, bf16 // 2)
        self.assertEqual(QSATokenToKVPool.index_state_dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
