import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAiterDSADecodeMetadata(CustomTestCase):
    @patch("sglang.srt.layers.attention.dsa_backend.get_mla_metadata_v1")
    def test_metadata_uses_only_active_kv_indptr_range(self, get_metadata):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend._ensure_aiter_dsa_decode_metadata_buffer = Mock()
        backend.aiter_dsa_kv_last_page_lens = torch.empty(8, dtype=torch.int32)
        backend.num_head_padded = 16
        backend.dsa_index_topk = 2048
        backend.aiter_dsa_max_split_per_batch = 64
        backend.aiter_dsa_work_metadata = object()
        backend.aiter_dsa_work_info_set = object()
        backend.aiter_dsa_work_indptr = object()
        backend.aiter_dsa_reduce_indptr = object()
        backend.aiter_dsa_reduce_final_map = object()
        backend.aiter_dsa_reduce_partial_map = object()

        qo_indptr = torch.tensor([0, 1, 2], dtype=torch.int32)
        kv_indptr = torch.tensor([0, 5, 11, 99, 99], dtype=torch.int32)

        backend._prepare_aiter_dsa_decode_metadata(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            bs=2,
            max_seqlen_q=1,
            q_dtype=torch.bfloat16,
            kv_dtype=torch.float8_e4m3fn,
        )

        passed_kv_indptr = get_metadata.call_args.args[1]
        torch.testing.assert_close(passed_kv_indptr, kv_indptr[:3])
        self.assertEqual(passed_kv_indptr.shape, (3,))


if __name__ == "__main__":
    unittest.main()
