# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from huggingface_hub.errors import (
    GatedRepoError,
    RemoteEntryNotFoundError,
    RepositoryNotFoundError,
)

from sglang.cli.utils import get_is_diffusion_model
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _hub_error(error_type, status_code):
    request = httpx.Request("GET", "https://huggingface.co/test/model")
    response = httpx.Response(status_code, request=request)
    return error_type("test error", response=response)


class TestDiffusionModelDetection(CustomTestCase):
    def setUp(self):
        super().setUp()
        for target in (
            "sglang.cli.utils._is_diffusion_model_from_registry",
            "sglang.cli.utils._is_overlay_diffusion_model",
            "sglang.cli.utils.os.path.isdir",
        ):
            patcher = patch(target, return_value=False)
            patcher.start()
            self.addCleanup(patcher.stop)

        modelscope_patcher = patch(
            "sglang.cli.utils.envs.SGLANG_USE_MODELSCOPE.get", return_value=False
        )
        modelscope_patcher.start()
        self.addCleanup(modelscope_patcher.stop)

    @patch("sglang.cli.utils.HfApi")
    @patch("huggingface_hub.hf_hub_download")
    def test_gated_diffusion_repo_uses_metadata_fallback(
        self, mock_download, mock_hf_api
    ):
        mock_download.side_effect = _hub_error(GatedRepoError, 401)
        mock_hf_api.return_value.model_info.return_value = SimpleNamespace(
            library_name="diffusers"
        )

        self.assertTrue(get_is_diffusion_model("test/gated-diffusion-model"))
        mock_hf_api.return_value.model_info.assert_called_once_with(
            "test/gated-diffusion-model"
        )

    @patch("sglang.cli.utils.HfApi")
    @patch("huggingface_hub.hf_hub_download")
    def test_gated_non_diffusion_repo_is_not_detected(self, mock_download, mock_hf_api):
        mock_download.side_effect = _hub_error(GatedRepoError, 401)
        mock_hf_api.return_value.model_info.return_value = SimpleNamespace(
            library_name="transformers"
        )

        self.assertFalse(get_is_diffusion_model("test/gated-llm-model"))
        mock_hf_api.return_value.model_info.assert_called_once_with(
            "test/gated-llm-model"
        )

    @patch("sglang.cli.utils.HfApi")
    @patch("huggingface_hub.hf_hub_download")
    def test_non_gated_hub_errors_do_not_use_metadata_fallback(
        self, mock_download, mock_hf_api
    ):
        for error_type in (RemoteEntryNotFoundError, RepositoryNotFoundError):
            with self.subTest(error_type=error_type.__name__):
                mock_download.side_effect = _hub_error(error_type, 404)
                self.assertFalse(get_is_diffusion_model("test/llm-model"))
        mock_hf_api.assert_not_called()


if __name__ == "__main__":
    unittest.main()
