# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from sglang.multimodal_gen.configs.pipeline_configs.llada_image import (
    LLaDAImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.image_api import (
    _build_sampling_params_or_400,
    _early_validate_edit_bounds,
    edits,
    generations,
)
from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
    ImageGenerationsRequest,
)


class TestLLaDAImageAPI(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.server_args = SimpleNamespace(
            pipeline_config=LLaDAImagePipelineConfig(),
            sp_degree=2,
            input_save_path=None,
            output_path=None,
            llada_image_max_pixel_area=None,
            llada_image_max_text_tokens=None,
            llada_image_max_total_pixel_area=None,
        )
        self.raw_request = SimpleNamespace(headers={})

    async def test_generation_sp_rejects_multiple_outputs_before_scheduling(self):
        request = ImageGenerationsRequest(prompt="a red car", n=2)
        fallback = HTTPException(status_code=418, detail="request was not validated")

        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.get_global_server_args",
                return_value=self.server_args,
            ),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.build_sampling_params",
                side_effect=fallback,
            ),
            self.assertRaises(HTTPException) as context,
        ):
            await generations(request, self.raw_request)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("n=1", context.exception.detail)

    async def test_edit_sp_rejects_multiple_outputs_before_saving_input(self):
        fallback = HTTPException(status_code=418, detail="request was not validated")

        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.get_global_server_args",
                return_value=self.server_args,
            ),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.save_image_to_path",
                new=AsyncMock(side_effect=fallback),
            ),
            self.assertRaises(HTTPException) as context,
        ):
            await edits(
                raw_request=self.raw_request,
                image=None,
                image_array=None,
                url=["https://example.com/source.png"],
                url_array=None,
                prompt="make the car blue",
                n=2,
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("n=1", context.exception.detail)

    async def test_edit_rejects_oversized_area_before_source_processing(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.get_global_server_args",
                return_value=self.server_args,
            ),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.save_image_to_path",
                new=AsyncMock(side_effect=AssertionError("source was processed")),
            ),
            self.assertRaises(HTTPException) as context,
        ):
            await edits(
                raw_request=self.raw_request,
                image=None,
                image_array=None,
                url=["https://example.com/source.png"],
                url_array=None,
                prompt="make the car blue",
                size="4096x4096",
                n=1,
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("exceeds the supported maximum", context.exception.detail)

    async def test_edit_rejects_multiple_sources_before_source_processing(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.get_global_server_args",
                return_value=self.server_args,
            ),
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.save_image_to_path",
                new=AsyncMock(side_effect=AssertionError("source was processed")),
            ),
            self.assertRaises(HTTPException) as context,
        ):
            await edits(
                raw_request=self.raw_request,
                image=None,
                image_array=None,
                url=[
                    "https://example.com/source-1.png",
                    "https://example.com/source-2.png",
                ],
                url_array=None,
                prompt="make the car blue",
                n=1,
                size=None,
            )

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("exactly one source image", context.exception.detail)

    def test_build_sampling_params_maps_value_error_to_400(self):
        with (
            patch(
                "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.build_sampling_params",
                side_effect=ValueError("bad request field"),
            ),
            self.assertRaises(HTTPException) as context,
        ):
            _build_sampling_params_or_400("rid", prompt="x")

        self.assertEqual(context.exception.status_code, 400)
        self.assertEqual(context.exception.detail, "bad request field")

    def test_build_sampling_params_passthrough_on_success(self):
        marker = object()
        with patch(
            "sglang.multimodal_gen.runtime.entrypoints.openai.image_api.build_sampling_params",
            return_value=marker,
        ):
            self.assertIs(_build_sampling_params_or_400("rid", prompt="x"), marker)

    def test_early_validate_edit_bounds_skips_absent_or_unparseable_size(self):
        _early_validate_edit_bounds(self.server_args, None)
        _early_validate_edit_bounds(self.server_args, "banana")
        with self.assertRaises(HTTPException) as context:
            _early_validate_edit_bounds(self.server_args, "4096x4096")
        self.assertEqual(context.exception.status_code, 400)


if __name__ == "__main__":
    unittest.main()
