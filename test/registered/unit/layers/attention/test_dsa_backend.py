import unittest
from unittest.mock import Mock

from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestAiterDSAMetadataReuse(CustomTestCase):
    def test_metadata_is_reused_only_within_one_forward(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.aiter_dsa_extend_metadata_owner = None
        backend.aiter_dsa_extend_kv_last_page_lens = None
        backend.aiter_dsa_extend_persistent_kwargs = None

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

        def get_metadata(owner):
            return backend._get_aiter_dsa_extend_metadata(
                owner,
                object(),
                object(),
                1,
                1,
                None,
                None,
            )

        first_forward = object()
        first = get_metadata(first_forward)
        reused = get_metadata(first_forward)

        self.assertEqual(backend._prepare_aiter_dsa_decode_metadata.call_count, 1)
        self.assertIs(first[0], reused[0])
        self.assertIs(first[1], reused[1])

        second = get_metadata(object())

        self.assertEqual(backend._prepare_aiter_dsa_decode_metadata.call_count, 2)
        self.assertIsNot(first[0], second[0])
        self.assertIsNot(first[1], second[1])


if __name__ == "__main__":
    unittest.main()
