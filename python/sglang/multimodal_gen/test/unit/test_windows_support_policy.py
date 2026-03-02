"""Unit tests for Windows native-support policy and backend fallback semantics."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from sglang.multimodal_gen.registry import get_model_info


class TestBackendFallbackPolicy(unittest.TestCase):
    def tearDown(self) -> None:
        get_model_info.cache_clear()

    def test_auto_backend_falls_back_to_diffusers_on_config_read_failure(self) -> None:
        get_model_info.cache_clear()
        sentinel = object()

        with patch(
            "sglang.multimodal_gen.registry._discover_and_register_pipelines"
        ), patch(
            "sglang.multimodal_gen.registry.os.path.exists", return_value=False
        ), patch(
            "sglang.multimodal_gen.registry.maybe_download_model_index",
            side_effect=RuntimeError("config read failed"),
        ), patch(
            "sglang.multimodal_gen.registry._get_diffusers_model_info",
            return_value=sentinel,
        ):
            model_info = get_model_info("dummy-model", backend="auto")

        self.assertIs(model_info, sentinel)

    def test_sglang_backend_does_not_fallback_to_diffusers_on_config_read_failure(
        self,
    ) -> None:
        get_model_info.cache_clear()

        with patch(
            "sglang.multimodal_gen.registry._discover_and_register_pipelines"
        ), patch(
            "sglang.multimodal_gen.registry.os.path.exists", return_value=False
        ), patch(
            "sglang.multimodal_gen.registry.maybe_download_model_index",
            side_effect=RuntimeError("config read failed"),
        ), patch(
            "sglang.multimodal_gen.registry._get_diffusers_model_info"
        ) as mocked_diffusers:
            model_info = get_model_info("dummy-model", backend="sglang")

        mocked_diffusers.assert_not_called()
        self.assertIsNone(model_info)


if __name__ == "__main__":
    unittest.main()
