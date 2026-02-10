"""
Tests for update_weights_from_disk API in SGLang-D (diffusion engine).

This tests the ability to dynamically update model weights without restarting the server,
which is critical for RL workflows and iterative fine-tuning scenarios.
"""

from __future__ import annotations

import os

import pytest
import requests

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.test_server_utils import (
    ServerContext,
    ServerManager,
)
from sglang.multimodal_gen.test.test_utils import get_dynamic_server_port

logger = init_logger(__name__)

# Default model for testing - use a small/fast model, need to be an image diffusion model
DEFAULT_DIFFUSION_MODEL = os.environ.get(
    "SGLANG_TEST_DIFFUSION_MODEL", "black-forest-labs/FLUX.2-klein-4B"
)


@pytest.fixture(scope="class")
def diffusion_server_for_weight_update():
    """Start a diffusion server for weight update tests."""
    port = get_dynamic_server_port()
    wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))

    manager = ServerManager(
        model=DEFAULT_DIFFUSION_MODEL,
        port=port,
        wait_deadline=wait_deadline,
        extra_args="--num-gpus 1",
    )

    ctx = manager.start()

    try:
        yield ctx
    finally:
        ctx.cleanup()


class TestUpdateWeightsFromDisk:
    """Test suite for update_weights_from_disk API."""

    def _get_base_url(self, ctx: ServerContext) -> str:
        return f"http://localhost:{ctx.port}"

    def _update_weights(
        self,
        base_url: str,
        model_path: str,
        flush_cache: bool = True,
        target_modules: list[str] | None = None,
        timeout: int = 300,
    ) -> dict:
        """Call update_weights_from_disk API."""
        payload = {
            "model_path": model_path,
            "flush_cache": flush_cache,
        }
        if target_modules is not None:
            payload["target_modules"] = target_modules

        response = requests.post(
            f"{base_url}/update_weights_from_disk",
            json=payload,
            timeout=timeout,
        )
        return response.json(), response.status_code

    def test_update_weights_same_model(
        self, diffusion_server_for_weight_update: ServerContext
    ):
        """Test updating weights with the same model (should succeed)."""
        base_url = self._get_base_url(diffusion_server_for_weight_update)

        result, status_code = self._update_weights(base_url, DEFAULT_DIFFUSION_MODEL)
        logger.info(f"Update result: {result}")

        assert status_code == 200, f"Expected 200, got {status_code}"
        assert result.get("success", False), f"Update failed: {result.get('message')}"

    def test_update_weights_with_flush_cache(
        self, diffusion_server_for_weight_update: ServerContext
    ):
        """Test updating weights with flush_cache=True."""
        base_url = self._get_base_url(diffusion_server_for_weight_update)

        result, status_code = self._update_weights(
            base_url,
            DEFAULT_DIFFUSION_MODEL,
            flush_cache=True,
        )

        assert status_code == 200
        assert result.get("success", False), f"Update failed: {result.get('message')}"

    def test_update_weights_without_flush_cache(
        self, diffusion_server_for_weight_update: ServerContext
    ):
        """Test updating weights with flush_cache=False."""
        base_url = self._get_base_url(diffusion_server_for_weight_update)

        result, status_code = self._update_weights(
            base_url,
            DEFAULT_DIFFUSION_MODEL,
            flush_cache=False,
        )

        assert status_code == 200
        assert result.get("success", False), f"Update failed: {result.get('message')}"

    def test_update_weights_nonexistent_model(
        self, diffusion_server_for_weight_update: ServerContext
    ):
        """Test that updating with non-existent model fails gracefully."""
        base_url = self._get_base_url(diffusion_server_for_weight_update)

        result, status_code = self._update_weights(
            base_url,
            "/nonexistent/path/to/model",
            timeout=60,
        )
        logger.info(f"Update result for nonexistent model: {result}")

        # Should fail gracefully
        assert not result.get("success", True), "Should fail for nonexistent model"

    def test_update_weights_missing_model_path(
        self, diffusion_server_for_weight_update: ServerContext
    ):
        """Test that request without model_path returns 400."""
        base_url = self._get_base_url(diffusion_server_for_weight_update)

        response = requests.post(
            f"{base_url}/update_weights_from_disk",
            json={},
            timeout=30,
        )

        # Should return 400 Bad Request
        assert response.status_code == 400, f"Expected 400, got {response.status_code}"

    def test_update_weights_specific_modules(
        self, diffusion_server_for_weight_update: ServerContext
    ):
        """Test updating only specific modules (e.g., transformer only)."""
        base_url = self._get_base_url(diffusion_server_for_weight_update)

        # Try to update only transformer module
        result, status_code = self._update_weights(
            base_url,
            DEFAULT_DIFFUSION_MODEL,
            target_modules=["transformer"],
        )
        logger.info(f"Update specific modules result: {result}")

        # This might fail if the model doesn't have a transformer module
        # or if weights for only transformer aren't available
        # The test verifies the API handles target_modules parameter
        assert status_code == 200

    def test_update_weights_nonexistent_module(
        self, diffusion_server_for_weight_update: ServerContext
    ):
        """Test that requesting a non-existent module name fails with a clear error."""
        base_url = self._get_base_url(diffusion_server_for_weight_update)

        result, status_code = self._update_weights(
            base_url,
            DEFAULT_DIFFUSION_MODEL,
            target_modules=["nonexistent_module"],
            timeout=60,
        )
        logger.info(f"Update nonexistent module result: {result}")

        assert status_code == 400, f"Expected 400, got {status_code}"
        assert not result.get("success", True), "Should fail for nonexistent module"
        assert "not found in pipeline" in result.get("message", "")


class TestUpdateWeightsFromDiskWithOffload:
    """Test update_weights_from_disk with layerwise offload enabled."""

    @pytest.fixture(scope="class")
    def diffusion_server_with_offload(self):
        """Start a diffusion server with layerwise offload enabled."""
        port = get_dynamic_server_port()
        wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))

        manager = ServerManager(
            model=DEFAULT_DIFFUSION_MODEL,
            port=port,
            wait_deadline=wait_deadline,
            extra_args="--num-gpus 1 --dit-layerwise-offload true",
        )

        ctx = manager.start()

        try:
            yield ctx
        finally:
            ctx.cleanup()

    def _get_base_url(self, ctx: ServerContext) -> str:
        return f"http://localhost:{ctx.port}"

    def _update_weights(
        self, base_url: str, model_path: str, **kwargs
    ) -> tuple[dict, int]:
        payload = {"model_path": model_path, **kwargs}
        response = requests.post(
            f"{base_url}/update_weights_from_disk",
            json=payload,
            timeout=kwargs.get("timeout", 300),
        )
        return response.json(), response.status_code

    def test_update_weights_with_offload_enabled(
        self, diffusion_server_with_offload: ServerContext
    ):
        """Test that weight update works correctly when layerwise offload is enabled."""
        base_url = self._get_base_url(diffusion_server_with_offload)

        logger.info("Testing weight update with offload enabled")

        result, status_code = self._update_weights(base_url, DEFAULT_DIFFUSION_MODEL)
        logger.info(f"Update result: {result}")

        assert status_code == 200, f"Expected 200, got {status_code}"
        assert result.get("success", False), f"Update failed: {result.get('message')}"

        # Verify no shape mismatch warnings in the message
        message = result.get("message", "")
        assert "Shape mismatch" not in message, f"Shape mismatch detected: {message}"


class TestUpdateWeightsEndToEnd:
    """End-to-end tests: verify generation works after weight update."""

    @pytest.fixture(scope="class")
    def diffusion_server_e2e(self):
        """Start a diffusion server for E2E tests."""
        port = get_dynamic_server_port()
        wait_deadline = float(os.environ.get("SGLANG_TEST_WAIT_SECS", "600"))

        manager = ServerManager(
            model=DEFAULT_DIFFUSION_MODEL,
            port=port,
            wait_deadline=wait_deadline,
            extra_args="--num-gpus 1",
        )

        ctx = manager.start()

        try:
            yield ctx
        finally:
            ctx.cleanup()

    def _get_base_url(self, ctx: ServerContext) -> str:
        return f"http://localhost:{ctx.port}"

    def _generate_image(self, base_url: str, prompt: str = "a cat") -> dict:
        """Generate an image using the OpenAI-compatible API."""
        from openai import OpenAI

        client = OpenAI(
            api_key="sglang-test",
            base_url=f"{base_url}/v1",
        )

        response = client.images.generate(
            model="default",
            prompt=prompt,
            n=1,
            size="512x512",
            response_format="b64_json",  # Avoid needing cloud storage
        )

        return response

    def test_generation_after_weight_update(self, diffusion_server_e2e: ServerContext):
        """Test that generation still works after updating weights."""
        base_url = self._get_base_url(diffusion_server_e2e)

        # Generate before update
        logger.info("Generating image before weight update...")
        response_before = self._generate_image(base_url, "a beautiful sunset")
        assert response_before.data, "Generation before update failed"
        logger.info("Generation before update succeeded")

        # Update weights
        update_response = requests.post(
            f"{base_url}/update_weights_from_disk",
            json={"model_path": DEFAULT_DIFFUSION_MODEL, "flush_cache": True},
            timeout=300,
        )
        assert update_response.json().get("success"), "Weight update failed"
        logger.info("Weight update succeeded")

        # Generate after update
        logger.info("Generating image after weight update...")
        response_after = self._generate_image(base_url, "a beautiful sunrise")
        assert response_after.data, "Generation after update failed"
        logger.info("Generation after update succeeded")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
