"""CPU checks for the opt-in metadata-fusion envelope and reuse guards."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sglang.srt.layers.attention.dsa.dsa_backend_kpool import (
    _is_kpool_metadata_fusion_supported,
)
from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestFusionContract(unittest.TestCase):
    def test_only_supported_pool_page_topk_geometry_is_enabled(self):
        for pool, page, topk, expected in (
            (1, 64, 2048, False),
            (2, 64, 2048, True),
            (4, 64, 2048, True),
            (3, 64, 2048, False),
            (4, 128, 2048, False),
            (4, 64, 2049, False),
        ):
            with self.subTest(pool=pool, page=page, topk=topk):
                self.assertEqual(
                    _is_kpool_metadata_fusion_supported(pool, page, topk), expected
                )

    def test_optional_derived_buffer_mismatch_rejects_reuse(self):
        fields = dict(
            paged_mqa_schedule_metadata=None,
            topk_v2_plan=None,
            pooled_cache_seqlens_int32=None,
            pooled_real_page_table=None,
            pooled_paged_mqa_schedule_metadata=None,
            kpool_write_plan=None,
        )
        dst, src = SimpleNamespace(**fields), SimpleNamespace(**fields)
        self.assertTrue(
            DeepseekSparseAttnBackend._sibling_replay_metadata_compatible(dst, src)
        )
        src.topk_v2_plan = object()
        self.assertFalse(
            DeepseekSparseAttnBackend._sibling_replay_metadata_compatible(dst, src)
        )

    def test_non_cuda_recomputes_instead_of_partial_copy(self):
        dst = SimpleNamespace(
            decode_cuda_graph_metadata={},
            init_forward_metadata_replay_cuda_graph_from_precomputed=MagicMock(),
        )
        src = SimpleNamespace(decode_cuda_graph_metadata={})
        precomputed = object()
        with patch(
            "sglang.srt.layers.attention.dsa_backend.is_cuda", return_value=False
        ):
            DeepseekSparseAttnBackend._copy_replay_metadata_from_sibling(
                dst, src, 2, precomputed, ForwardMode.DECODE
            )
        dst.init_forward_metadata_replay_cuda_graph_from_precomputed.assert_called_once_with(
            bs=2, precomputed=precomputed, forward_mode=ForwardMode.DECODE
        )


if __name__ == "__main__":
    unittest.main()
