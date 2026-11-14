import unittest

from sglang.multimodal_gen.runtime.layers.attention.turbo_layer import (
    _resolve_turbo_wan_sparse_backend,
)
from sglang.multimodal_gen.runtime.platforms.interface import AttentionBackendEnum


class TestTurboWanBackendSelection(unittest.TestCase):
    def test_non_sparse_requested_backend_falls_back_to_attention_type(self):
        selected, warning = _resolve_turbo_wan_sparse_backend(
            attention_type="sla",
            requested_attention_backend="fa",
        )

        self.assertEqual(selected, AttentionBackendEnum.SLA_ATTN)
        self.assertIsNotNone(warning)
        self.assertIn("TurboWan only supports", warning)
        self.assertIn("attention_backend='fa'", warning)

    def test_sagesla_attention_type_prefers_sage_sparse_backend(self):
        selected, warning = _resolve_turbo_wan_sparse_backend(
            attention_type="sagesla",
            requested_attention_backend="torch_sdpa",
        )

        self.assertEqual(selected, AttentionBackendEnum.SAGE_SLA_ATTN)
        self.assertIsNotNone(warning)

    def test_requested_sparse_backend_is_honored(self):
        selected, warning = _resolve_turbo_wan_sparse_backend(
            attention_type="sla",
            requested_attention_backend="sage_sla_attn",
        )

        self.assertEqual(selected, AttentionBackendEnum.SAGE_SLA_ATTN)
        self.assertIsNone(warning)

    def test_supported_backend_filter_is_respected(self):
        selected, warning = _resolve_turbo_wan_sparse_backend(
            attention_type="sla",
            requested_attention_backend=None,
            supported_attention_backends={AttentionBackendEnum.SAGE_SLA_ATTN},
        )

        self.assertEqual(selected, AttentionBackendEnum.SAGE_SLA_ATTN)
        self.assertIsNone(warning)

    def test_empty_supported_backend_intersection_keeps_turbowan_choices(self):
        selected, warning = _resolve_turbo_wan_sparse_backend(
            attention_type="sla",
            requested_attention_backend=None,
            supported_attention_backends={AttentionBackendEnum.FA},
        )

        self.assertEqual(selected, AttentionBackendEnum.SLA_ATTN)
        self.assertIsNone(warning)


if __name__ == "__main__":
    unittest.main()
