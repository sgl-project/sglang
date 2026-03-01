"""
Unit tests for IMAGE_NUM_LIMITATION in BaseMultimodalProcessor.

Covers:
    - Default image limit enforcement
    - Subclass override of IMAGE_NUM_LIMITATION
    - CLI override via server_args.limit_mm_data_per_request
    - No error when image_data is None or within limit
"""

from unittest.mock import Mock

import pytest

from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class ConcreteProcessor(BaseMultimodalProcessor):
    """Minimal concrete subclass for testing (ABC can't be instantiated directly)."""

    async def process_mm_data_async(self, **kwargs):
        return {}


class HighLimitProcessor(BaseMultimodalProcessor):
    """Processor with a higher image limit (like InternVL)."""

    IMAGE_NUM_LIMITATION = 12

    async def process_mm_data_async(self, **kwargs):
        return {}


def make_processor(cls=ConcreteProcessor, limit_mm_data_per_request=None):
    """Helper to create a processor with mocked dependencies."""
    server_args = Mock()
    server_args.limit_mm_data_per_request = limit_mm_data_per_request
    return cls(
        hf_config=Mock(),
        server_args=server_args,
        _processor=Mock(),
        transport_mode="ipc",
        skip_mm_pool=True,
    )


class TestGetImageLimit:
    """Tests for _get_image_limit() priority logic."""

    def test_default_limit(self):
        proc = make_processor()
        assert proc._get_image_limit() == 5

    def test_subclass_override(self):
        proc = make_processor(cls=HighLimitProcessor)
        assert proc._get_image_limit() == 12

    def test_cli_override_takes_priority(self):
        proc = make_processor(limit_mm_data_per_request={"image": 20})
        assert proc._get_image_limit() == 20

    def test_cli_override_on_subclass(self):
        proc = make_processor(
            cls=HighLimitProcessor,
            limit_mm_data_per_request={"image": 3},
        )

        assert proc._get_image_limit() == 3

    def test_cli_with_no_image_key_falls_back_to_default(self):
        proc = make_processor(limit_mm_data_per_request={"video": 2})
        assert proc._get_image_limit() == 5


class TestImageNumLimitation:
    """Tests for image count validation in load_mm_data()."""

    def test_within_limit_no_error(self):
        proc = make_processor()
        # 3 images, limit is 5 - should not raise
        image_data = ["img1.jpg", "img2.jpg", "img3.jpg"]
        multimodal_tokens = Mock(spec=MultimodalSpecialTokens)
        multimodal_tokens.get_combined_regex.return_value = Mock(
            match=Mock(return_value=None)
        )
        # load_mm_data will fail later (no real processor), but should NOT
        # fail at the image count check. We catch the later error.

        try:
            proc.load_mm_data(
                prompt="describe these",
                multimodal_tokens=multimodal_tokens,
                image_data=image_data,
            )
        except ValueError as e:
            if "exceeds" in str(e):
                pytest.fail(f"Should not raise limit error for 3 images: {e}")
        except Exception:
            pass  # Other errors from mocked dependencies are expected

    def test_exceeds_limit_raises_error(self):
        proc = make_processor()
        image_data = [f"image{i}.jpg" for i in range(10)]
        multimodal_tokens = Mock(spec=MultimodalSpecialTokens)
        with pytest.raises(ValueError, match="exceeds"):
            proc.load_mm_data(
                prompt="describe these",
                multimodal_tokens=multimodal_tokens,
                image_data=image_data,
            )

    def test_no_images_no_error(self):
        proc = make_processor()
        multimodal_tokens = Mock(spec=MultimodalSpecialTokens)
        multimodal_tokens.get_combined_regex.return_value = Mock(
            match=Mock(return_value=None)
        )
        try:
            proc.load_mm_data(
                prompt="hello",
                multimodal_tokens=multimodal_tokens,
                image_data=None,
            )
        except ValueError as e:
            if "exceeds" in str(e):
                pytest.fail(f"Should not raise limit error for None: {e}")
        except Exception:
            pass

    def test_high_limit_processor_allows_more(self):
        proc = make_processor(cls=HighLimitProcessor)
        image_data = [f"img{i}.jpg" for i in range(10)]
        multimodal_tokens = Mock(spec=MultimodalSpecialTokens)
        multimodal_tokens.get_combined_regex.return_value = Mock(
            match=Mock(return_value=None)
        )
        try:
            proc.load_mm_data(
                prompt="describe these",
                multimodal_tokens=multimodal_tokens,
                image_data=image_data,
            )
        except ValueError as e:
            if "exceeds" in str(e):
                pytest.fail(f"Should not raise for 10 images with limit 12: {e}")
        except Exception:
            pass

    def test_cli_override_allows_more(self):
        proc = make_processor(limit_mm_data_per_request={"image": 20})
        image_data = [f"img{i}.jpg" for i in range(15)]
        multimodal_tokens = Mock(spec=MultimodalSpecialTokens)
        multimodal_tokens.get_combined_regex.return_value = Mock(
            match=Mock(return_value=None)
        )

        try:
            proc.load_mm_data(
                prompt="describe these",
                multimodal_tokens=multimodal_tokens,
                image_data=image_data,
            )
        except ValueError as e:
            if "exceeds" in str(e):
                pytest.fail(f"Should not raise for 15 images with CLI limit 20: {e}")
        except Exception:
            pass
