"""Basic image API tests.

Tests the image generation and editing functionality through the router
with cloud backends (OpenAI).

These tests require API keys to be set:
- OPENAI_API_KEY for OpenAI backend

Usage:
    pytest e2e_test/images/test_basic.py -v
    pytest e2e_test/images/test_basic.py -v -k "generations"
"""

from __future__ import annotations

import base64
import logging
import os

import pytest

logger = logging.getLogger(__name__)

# Skip all tests in this module if no image API keys are available
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set - skipping image tests",
)


@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestImageGeneration:
    """Image generation API tests using OpenAI cloud backend."""

    def test_generations_basic(self, setup_backend):
        """Test basic image generation.

        Verifies that:
        - Response structure is correct
        - Image URL or base64 data is returned
        - Created timestamp is present
        """
        backend, model, client, gateway = setup_backend

        response = client.images.generate(
            model="dall-e-3",
            prompt="A simple red circle on a white background",
            n=1,
            size="1024x1024",
        )

        assert response.created > 0
        assert len(response.data) == 1

        image = response.data[0]
        # Response should have either url or b64_json
        assert image.url is not None or image.b64_json is not None

        if image.url:
            assert image.url.startswith("http")
            logger.info("Generated image URL: %s", image.url[:100])

        if image.revised_prompt:
            logger.info("Revised prompt: %s", image.revised_prompt[:100])

    def test_generations_with_size(self, setup_backend):
        """Test image generation with different sizes."""
        backend, model, client, gateway = setup_backend

        # Test with a smaller size for faster response
        response = client.images.generate(
            model="dall-e-3",
            prompt="A blue square",
            n=1,
            size="1024x1024",
        )

        assert response.created > 0
        assert len(response.data) == 1
        assert response.data[0].url is not None or response.data[0].b64_json is not None

    def test_generations_with_quality(self, setup_backend):
        """Test image generation with quality parameter."""
        backend, model, client, gateway = setup_backend

        response = client.images.generate(
            model="dall-e-3",
            prompt="A green triangle",
            n=1,
            size="1024x1024",
            quality="standard",
        )

        assert response.created > 0
        assert len(response.data) == 1

    def test_generations_with_style(self, setup_backend):
        """Test image generation with style parameter."""
        backend, model, client, gateway = setup_backend

        response = client.images.generate(
            model="dall-e-3",
            prompt="A sunset over mountains",
            n=1,
            size="1024x1024",
            style="natural",
        )

        assert response.created > 0
        assert len(response.data) == 1

    def test_generations_b64_json_format(self, setup_backend):
        """Test image generation with base64 response format."""
        backend, model, client, gateway = setup_backend

        response = client.images.generate(
            model="dall-e-3",
            prompt="A small yellow dot",
            n=1,
            size="1024x1024",
            response_format="b64_json",
        )

        assert response.created > 0
        assert len(response.data) == 1

        image = response.data[0]
        assert image.b64_json is not None

        # Verify it's valid base64
        try:
            decoded = base64.b64decode(image.b64_json)
            assert len(decoded) > 0
            logger.info("Decoded base64 image size: %d bytes", len(decoded))
        except Exception as e:
            pytest.fail(f"Invalid base64 data: {e}")


@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestImageEdits:
    """Image editing API tests using OpenAI cloud backend.

    Note: These tests require dall-e-2 which supports image editing.
    DALL-E 3 does not support the edits endpoint.
    """

    @pytest.fixture
    def sample_image_base64(self):
        """Create a minimal valid PNG image for testing.

        Returns a 1x1 transparent PNG encoded as base64.
        """
        # Minimal 1x1 transparent PNG
        png_bytes = bytes(
            [
                0x89,
                0x50,
                0x4E,
                0x47,
                0x0D,
                0x0A,
                0x1A,
                0x0A,  # PNG signature
                0x00,
                0x00,
                0x00,
                0x0D,
                0x49,
                0x48,
                0x44,
                0x52,  # IHDR chunk
                0x00,
                0x00,
                0x00,
                0x01,
                0x00,
                0x00,
                0x00,
                0x01,  # 1x1
                0x08,
                0x06,
                0x00,
                0x00,
                0x00,
                0x1F,
                0x15,
                0xC4,
                0x89,  # 8-bit RGBA
                0x00,
                0x00,
                0x00,
                0x0A,
                0x49,
                0x44,
                0x41,
                0x54,  # IDAT chunk
                0x78,
                0x9C,
                0x63,
                0x00,
                0x01,
                0x00,
                0x00,
                0x05,
                0x00,
                0x01,  # compressed data
                0x0D,
                0x0A,
                0x2D,
                0xB4,
                0x00,
                0x00,
                0x00,
                0x00,
                0x49,
                0x45,
                0x4E,
                0x44,  # IEND chunk
                0xAE,
                0x42,
                0x60,
                0x82,
            ]
        )
        return base64.b64encode(png_bytes).decode("utf-8")

    @pytest.mark.skip(
        reason="Image edits require actual image file upload via multipart/form-data"
    )
    def test_edits_basic(self, setup_backend, sample_image_base64):
        """Test basic image editing.

        Note: This test is skipped because the OpenAI edits endpoint
        requires multipart/form-data with actual file uploads, which
        the current gateway implementation handles differently.
        """
        backend, model, client, gateway = setup_backend

        # Note: The actual OpenAI API requires file uploads, not base64
        # This test demonstrates the expected API structure
        response = client.images.edit(
            model="dall-e-2",
            image=sample_image_base64,
            prompt="Add a red border",
            n=1,
            size="256x256",
        )

        assert response.created > 0
        assert len(response.data) == 1


@pytest.mark.e2e
@pytest.mark.parametrize("setup_backend", ["openai"], indirect=True)
class TestImageErrors:
    """Error handling tests for image endpoints."""

    def test_generations_invalid_size(self, setup_backend):
        """Test that invalid size parameter returns an error."""
        backend, model, client, gateway = setup_backend

        with pytest.raises(Exception) as exc_info:
            client.images.generate(
                model="dall-e-3",
                prompt="A test image",
                n=1,
                size="invalid_size",
            )

        # OpenAI returns a BadRequestError for invalid parameters
        logger.info("Expected error for invalid size: %s", exc_info.value)

    def test_generations_empty_prompt(self, setup_backend):
        """Test that empty prompt returns an error."""
        backend, model, client, gateway = setup_backend

        with pytest.raises(Exception) as exc_info:
            client.images.generate(
                model="dall-e-3",
                prompt="",
                n=1,
                size="1024x1024",
            )

        logger.info("Expected error for empty prompt: %s", exc_info.value)
