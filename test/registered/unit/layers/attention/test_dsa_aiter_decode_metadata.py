import unittest
from unittest.mock import Mock

import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAiterDSADecodeMetadata(CustomTestCase):
    def test_metadata_is_reused_only_within_one_decode(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.kv_indptr = torch.zeros(3, dtype=torch.int32)
        backend.aiter_dsa_decode_metadata_owner = None
        backend.aiter_dsa_decode_kv_last_page_lens = None
        backend.aiter_dsa_decode_persistent_kwargs = None

        prepared_results = [
            {
                "kv_last_page_lens": object(),
                "work_meta_data": object(),
            },
            {
                "kv_last_page_lens": object(),
                "work_meta_data": object(),
            },
        ]
        backend._prepare_aiter_dsa_decode_metadata = Mock(side_effect=prepared_results)

        def get_metadata(owner, page_table):
            return backend._get_aiter_dsa_decode_metadata(
                metadata_owner=owner,
                page_table_1=page_table,
                qo_indptr=object(),
                bs=2,
                max_seqlen_q=1,
                q_dtype=torch.bfloat16,
                kv_dtype=torch.float8_e4m3fn,
            )

        first_owner = object()
        first = get_metadata(
            first_owner,
            torch.tensor([[4, 5, -1], [7, -1, -1]], dtype=torch.int32),
        )
        reused = get_metadata(
            first_owner,
            torch.tensor([[8, 9, -1], [10, -1, -1]], dtype=torch.int32),
        )

        self.assertEqual(
            backend._prepare_aiter_dsa_decode_metadata.call_count,
            1,
        )
        self.assertIs(first[0], reused[0])
        self.assertIs(first[1], reused[1])
        torch.testing.assert_close(
            backend.kv_indptr,
            torch.tensor([0, 2, 3], dtype=torch.int32),
        )

        second = get_metadata(
            object(),
            torch.tensor([[11, -1, -1], [12, 13, -1]], dtype=torch.int32),
        )

        self.assertEqual(
            backend._prepare_aiter_dsa_decode_metadata.call_count,
            2,
        )
        self.assertIsNot(first[0], second[0])
        self.assertIsNot(first[1], second[1])
        torch.testing.assert_close(
            backend.kv_indptr,
            torch.tensor([0, 1, 3], dtype=torch.int32),
        )

    def test_cuda_graph_warmup_invalidates_decode_owner(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.aiter_dsa_decode_metadata_owner = object()

        backend.on_after_cuda_graph_warmup()

        self.assertIsNone(backend.aiter_dsa_decode_metadata_owner)


if __name__ == "__main__":
    unittest.main()
