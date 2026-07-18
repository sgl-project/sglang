import unittest

from sglang.srt.layers.attention.dsa.graph_metadata import (
    get_dsa_cuda_graph_metadata_key,
    load_dsa_cuda_graph_metadata,
    store_dsa_cuda_graph_metadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDsaGraphMetadataKey(unittest.TestCase):
    def test_draft_extend_does_not_alias_target_verify(self):
        target_key = get_dsa_cuda_graph_metadata_key(8, ForwardMode.TARGET_VERIFY)
        draft_key = get_dsa_cuda_graph_metadata_key(8, ForwardMode.DRAFT_EXTEND_V2)

        self.assertEqual(target_key, 8)
        self.assertEqual(draft_key, (8, ForwardMode.DRAFT_EXTEND_V2))
        self.assertNotEqual(target_key, draft_key)

    def test_cache_keeps_target_and_draft_metadata_for_same_batch(self):
        cache = {}
        target_metadata = object()
        draft_metadata = object()

        store_dsa_cuda_graph_metadata(
            cache, 8, ForwardMode.TARGET_VERIFY, target_metadata
        )
        store_dsa_cuda_graph_metadata(
            cache, 8, ForwardMode.DRAFT_EXTEND_V2, draft_metadata
        )

        self.assertEqual(len(cache), 2)
        self.assertIs(
            load_dsa_cuda_graph_metadata(cache, 8, ForwardMode.TARGET_VERIFY),
            target_metadata,
        )
        self.assertIs(
            load_dsa_cuda_graph_metadata(cache, 8, ForwardMode.DRAFT_EXTEND_V2),
            draft_metadata,
        )


if __name__ == "__main__":
    unittest.main()
