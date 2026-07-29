import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.trtllm_mha_backend import (
    TRTLLMHAAttnBackend,
    TRTLLMMHAMetadata,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class TestTRTLLMMHAZigzagPageTables(CustomTestCase):
    def _backend(self, swa_pool=None):
        backend = object.__new__(TRTLLMHAAttnBackend)
        backend._swa_kv_pool = swa_pool
        return backend

    def _metadata(self):
        return TRTLLMMHAMetadata(
            page_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
            swa_page_table=torch.tensor([[30, 31], [40, 41]], dtype=torch.int32),
        )

    def test_builds_prev_then_next_page_tables_for_cp_v2(self):
        backend = self._backend()
        metadata = self._metadata()

        with patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
            return_value=True,
        ):
            backend._build_zigzag_page_tables(metadata, SimpleNamespace())

        self.assertEqual(
            metadata.zigzag_page_table.tolist(),
            [[10, 11], [20, 21], [10, 11], [20, 21]],
        )
        self.assertEqual(
            metadata.zigzag_swa_page_table.tolist(),
            [[30, 31], [40, 41], [30, 31], [40, 41]],
        )

    def test_leaves_zigzag_page_tables_unset_without_cp_v2(self):
        backend = self._backend()
        metadata = self._metadata()

        with patch(
            "sglang.srt.layers.attention.trtllm_mha_backend.is_cp_v2_active",
            return_value=False,
        ):
            backend._build_zigzag_page_tables(metadata, SimpleNamespace())

        self.assertIsNone(metadata.zigzag_page_table)
        self.assertIsNone(metadata.zigzag_swa_page_table)

    def test_selects_full_or_swa_zigzag_page_table_per_layer(self):
        swa_pool = SimpleNamespace(
            layers_mapping={
                0: (None, False),
                1: (None, True),
            }
        )
        backend = self._backend(swa_pool=swa_pool)
        metadata = self._metadata()
        metadata.zigzag_page_table = torch.tensor([[1], [2]], dtype=torch.int32)
        metadata.zigzag_swa_page_table = torch.tensor([[3], [4]], dtype=torch.int32)
        backend.forward_metadata = metadata

        full_table = backend._get_zigzag_layer_page_table(SimpleNamespace(layer_id=0))
        swa_table = backend._get_zigzag_layer_page_table(SimpleNamespace(layer_id=1))

        self.assertIs(full_table, metadata.zigzag_page_table)
        self.assertIs(swa_table, metadata.zigzag_swa_page_table)


if __name__ == "__main__":
    unittest.main()
