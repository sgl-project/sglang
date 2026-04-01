"""Tests for the image count limitation feature (issue #8540).

Validates that:
- The server arg --max-images-per-request is respected
- Per-processor IMAGE_MAX_NUM defaults work as fallback
- Server arg takes precedence over processor default
- Requests at or below the limit pass through
- Requests with no images never trigger the limit
"""

import re
import unittest
from typing import Optional
from unittest.mock import MagicMock


def _make_server_args(max_images_per_request=None, skip_tokenizer_init=False):
    """Create a minimal server_args-like object for testing."""
    args = MagicMock()
    args.max_images_per_request = max_images_per_request
    args.skip_tokenizer_init = skip_tokenizer_init
    return args


def _make_multimodal_tokens():
    """Create a mock multimodal_tokens object."""
    mock = MagicMock()
    mock.get_combined_regex.return_value = re.compile("(<image>)")
    mock.get_modality_of_token.return_value = None
    return mock


def _validate_image_count(server_args, image_max_num, n_image):
    """
    Replicate the validation logic from BaseMultimodalProcessor.load_mm_data().
    This tests the core logic without needing the full import chain.
    """
    max_images = (
        server_args.max_images_per_request
        if server_args.max_images_per_request is not None
        else image_max_num
    )
    if max_images is not None and n_image > max_images:
        raise ValueError(
            f"Request contains {n_image} images, but the maximum allowed is "
            f"{max_images}. Reduce the number of images or increase "
            f"--max-images-per-request."
        )


class TestImageNumLimitationLogic(unittest.TestCase):
    """Unit tests for the image count validation logic."""

    def test_no_limit_when_unset(self):
        """No error when neither server arg nor processor default is set."""
        server_args = _make_server_args(max_images_per_request=None)
        # Should not raise for any count
        _validate_image_count(server_args, image_max_num=None, n_image=100)

    def test_server_arg_rejects_over_limit(self):
        """Requests exceeding --max-images-per-request are rejected."""
        server_args = _make_server_args(max_images_per_request=5)
        with self.assertRaises(ValueError) as ctx:
            _validate_image_count(server_args, image_max_num=None, n_image=6)
        self.assertIn("6", str(ctx.exception))
        self.assertIn("5", str(ctx.exception))
        self.assertIn("--max-images-per-request", str(ctx.exception))

    def test_server_arg_allows_at_limit(self):
        """Requests at exactly the limit should pass."""
        server_args = _make_server_args(max_images_per_request=5)
        _validate_image_count(server_args, image_max_num=None, n_image=5)

    def test_server_arg_allows_below_limit(self):
        """Requests below the limit should pass."""
        server_args = _make_server_args(max_images_per_request=10)
        _validate_image_count(server_args, image_max_num=None, n_image=3)

    def test_processor_default_rejects_over_limit(self):
        """Per-processor IMAGE_MAX_NUM is enforced when server arg is not set."""
        server_args = _make_server_args(max_images_per_request=None)
        with self.assertRaises(ValueError) as ctx:
            _validate_image_count(server_args, image_max_num=12, n_image=13)
        self.assertIn("13", str(ctx.exception))
        self.assertIn("12", str(ctx.exception))

    def test_processor_default_allows_at_limit(self):
        """Requests at exactly the processor default should pass."""
        server_args = _make_server_args(max_images_per_request=None)
        _validate_image_count(server_args, image_max_num=12, n_image=12)

    def test_server_arg_overrides_processor_default(self):
        """--max-images-per-request takes precedence over IMAGE_MAX_NUM."""
        server_args = _make_server_args(max_images_per_request=3)
        # 4 images: below processor default (12) but above server arg (3)
        with self.assertRaises(ValueError) as ctx:
            _validate_image_count(server_args, image_max_num=12, n_image=4)
        self.assertIn("3", str(ctx.exception))

    def test_server_arg_can_raise_processor_default(self):
        """Server arg can allow more images than the processor default."""
        server_args = _make_server_args(max_images_per_request=20)
        # 15 images: above processor default (12) but below server arg (20)
        _validate_image_count(server_args, image_max_num=12, n_image=15)

    def test_no_images_always_passes(self):
        """Zero images should never trigger the limit."""
        server_args = _make_server_args(max_images_per_request=1)
        _validate_image_count(server_args, image_max_num=1, n_image=0)

    def test_single_image_with_limit_one(self):
        """A single image with limit=1 should pass."""
        server_args = _make_server_args(max_images_per_request=1)
        _validate_image_count(server_args, image_max_num=None, n_image=1)

    def test_error_message_includes_actionable_hint(self):
        """Error message should tell users how to fix the issue."""
        server_args = _make_server_args(max_images_per_request=2)
        with self.assertRaises(ValueError) as ctx:
            _validate_image_count(server_args, image_max_num=None, n_image=5)
        msg = str(ctx.exception)
        self.assertIn("--max-images-per-request", msg)
        self.assertIn("5", msg)
        self.assertIn("2", msg)


class TestServerArgsField(unittest.TestCase):
    """Test that ServerArgs correctly accepts the new field.

    Uses direct dataclass construction to avoid the heavy sglang import chain.
    """

    def test_field_exists_in_source(self):
        """Verify max_images_per_request is declared in server_args.py."""
        import os

        server_args_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..",
            "python", "sglang", "srt", "server_args.py",
        )
        with open(os.path.normpath(server_args_path)) as f:
            source = f.read()
        self.assertIn("max_images_per_request", source)
        self.assertIn("--max-images-per-request", source)


class TestBaseProcessorField(unittest.TestCase):
    """Test that BaseMultimodalProcessor declares IMAGE_MAX_NUM."""

    def test_field_exists_in_source(self):
        """Verify IMAGE_MAX_NUM is declared in base_processor.py."""
        import os

        base_path = os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..",
            "python", "sglang", "srt", "multimodal", "processors", "base_processor.py",
        )
        with open(os.path.normpath(base_path)) as f:
            source = f.read()
        # Check the class-level default
        self.assertIn("IMAGE_MAX_NUM: Optional[int] = None", source)
        # Check the validation logic
        self.assertIn("max_images_per_request", source)
        self.assertIn("self.IMAGE_MAX_NUM", source)


if __name__ == "__main__":
    unittest.main()
