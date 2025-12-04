# SPDX-License-Identifier: Apache-2.0
"""
cache-dit acceleration tests for diffusion models.

This test suite validates that cache-dit works correctly with SGLang's diffusion pipeline.
It runs a subset of the standard diffusion tests with cache-dit enabled.

Note: cache-dit only supports single-GPU inference (world_size=1).
In distributed environments, cache-dit is automatically disabled.

Usage:
    # Run all cache-dit tests
    pytest python/sglang/multimodal_gen/test/server/test_server_cache_dit.py

    # Run specific test
    pytest python/sglang/multimodal_gen/test/server/test_server_cache_dit.py -k qwen_image

    # Generate baseline for new cache-dit test
    SGLANG_GEN_BASELINE=1 pytest -s python/sglang/multimodal_gen/test/server/test_server_cache_dit.py -k qwen_image

    # Run with custom cache-dit parameters
    SGLANG_CACHE_DIT_FN=2 SGLANG_CACHE_DIT_RDT=0.4 pytest -s test_server_cache_dit.py
"""

from __future__ import annotations

import pytest

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.test_server_common import (  # noqa: F401
    DiffusionServerBase,
    diffusion_server,
)
from sglang.multimodal_gen.test.server.testcase_configs import (
    DiffusionSamplingParams,
    DiffusionServerArgs,
    DiffusionTestCase,
)

logger = init_logger(__name__)


# Default cache-dit configuration for CI tests (can still be overridden at runtime)
CACHE_DIT_DEFAULT_CONFIG = {
    "SGLANG_CACHE_DIT_ENABLED": "true",
    # DBCache parameters
    "SGLANG_CACHE_DIT_FN": "1",  # First N blocks to always compute
    "SGLANG_CACHE_DIT_BN": "0",  # Last N blocks to always compute
    "SGLANG_CACHE_DIT_WARMUP": "8",  # Warmup steps before caching
    "SGLANG_CACHE_DIT_RDT": "0.35",  # Residual difference threshold
    "SGLANG_CACHE_DIT_MC": "3",  # Max continuous cached steps
    # TaylorSeer parameters
    "SGLANG_CACHE_DIT_TAYLORSEER": "true",  # Enable TaylorSeer calibrator
    "SGLANG_CACHE_DIT_TS_ORDER": "1",  # Taylor expansion order
}

# SCM configuration for additional speedup tests
CACHE_DIT_SCM_CONFIG = {
    **CACHE_DIT_DEFAULT_CONFIG,
    "SGLANG_CACHE_DIT_SCM_PRESET": "medium",  # SCM preset
    "SGLANG_CACHE_DIT_SCM_POLICY": "dynamic",  # SCM policy
}

# Dual-transformer (Wan2.2) configuration adds secondary overrides
CACHE_DIT_DUAL_TRANSFORMER_CONFIG = {
    **CACHE_DIT_DEFAULT_CONFIG,
    "SGLANG_CACHE_DIT_SECONDARY_FN": "2",  # More first blocks computed
    "SGLANG_CACHE_DIT_SECONDARY_BN": "1",  # More last blocks computed
    "SGLANG_CACHE_DIT_SECONDARY_RDT": "0.25",  # Lower threshold (more conservative)
}


def _apply_cache_dit_env(monkeypatch: pytest.MonkeyPatch, config: dict[str, str]) -> None:
    """Apply cache-dit env vars for a test case."""
    logger.info("Setting cache-dit environment variables:")
    for key, value in config.items():
        # Skip keys not provided explicitly (especially in overrides)
        if value is None:
            continue
        logger.info("  %s=%s", key, value)
        monkeypatch.setenv(key, value)


# cache-dit test cases - subset of ONE_GPU_CASES_A suitable for cache-dit testing
# Note: Using smaller output sizes and fewer warmup steps for faster CI
CACHE_DIT_CASES: list[DiffusionTestCase] = [
    # Qwen-Image with cache-dit (DBCache + TaylorSeer)
    DiffusionTestCase(
        "cache_dit_qwen_image_t2i",
        DiffusionServerArgs(
            model_path="Qwen/Qwen-Image",
            modality="image",
            warmup_text=1,
            warmup_edit=0,
        ),
        DiffusionSamplingParams(
            prompt="A futuristic cityscape at sunset with flying cars",
            output_size="512x512",  # Smaller for faster CI
        ),
    ),
    # FLUX with cache-dit
    DiffusionTestCase(
        "cache_dit_flux_image_t2i",
        DiffusionServerArgs(
            model_path="black-forest-labs/FLUX.1-dev",
            modality="image",
            warmup_text=1,
            warmup_edit=0,
        ),
        DiffusionSamplingParams(
            prompt="A curious raccoon in a forest",
            output_size="512x512",  # Smaller for faster CI
        ),
    ),
]


# Additional test cases for SCM (Step Computation Masking)
# These are skipped by default as they require longer runtime
CACHE_DIT_SCM_CASES: list[DiffusionTestCase] = [
    # Qwen-Image with cache-dit + SCM medium preset
    DiffusionTestCase(
        "cache_dit_scm_qwen_image_t2i",
        DiffusionServerArgs(
            model_path="Qwen/Qwen-Image",
            modality="image",
            warmup_text=1,
            warmup_edit=0,
        ),
        DiffusionSamplingParams(
            prompt="A futuristic cityscape at sunset with flying cars",
            output_size="512x512",
        ),
    ),
]


# Dual-transformer test cases (Wan2.2 with high-noise + low-noise experts)
# Note: These tests require Wan2.2 model which has two transformer blocks
# Skip by default as Wan2.2 is a large model requiring significant resources
CACHE_DIT_DUAL_TRANSFORMER_CASES: list[DiffusionTestCase] = [
    # Wan2.2 T2V with cache-dit on both transformers
    DiffusionTestCase(
        "cache_dit_wan2.2_dual_t2v",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.2-T2V-14B",
            modality="video",
            warmup_text=1,
            warmup_edit=0,
        ),
        DiffusionSamplingParams(
            prompt="A serene mountain landscape with flowing water",
            output_size="512x288",  # Smaller for faster CI
            num_frames=17,  # Fewer frames for faster CI
        ),
    ),
    # Wan2.2 I2V with cache-dit on both transformers
    DiffusionTestCase(
        "cache_dit_wan2.2_dual_i2v",
        DiffusionServerArgs(
            model_path="Wan-AI/Wan2.2-I2V-14B",
            modality="video",
            warmup_text=1,
            warmup_edit=0,
        ),
        DiffusionSamplingParams(
            prompt="A gentle breeze moves through the scene",
            output_size="512x288",
            num_frames=17,
        ),
    ),
]


class TestCacheDitOneGpu(DiffusionServerBase):
    """Validates cache-dit on 1-GPU diffusion models."""

    @pytest.fixture(params=CACHE_DIT_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each cache-dit test."""
        return request.param

    @pytest.fixture(autouse=True)
    def setup_cache_dit_env(self, monkeypatch: pytest.MonkeyPatch):
        """Set up cache-dit environment variables for tests."""
        _apply_cache_dit_env(monkeypatch, CACHE_DIT_DEFAULT_CONFIG)


@pytest.mark.skip(reason="SCM tests require longer runtime, run manually")
class TestCacheDitScmOneGpu(DiffusionServerBase):
    """cache-dit with SCM presets (run manually due to runtime)."""

    @pytest.fixture(params=CACHE_DIT_SCM_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each SCM test."""
        return request.param

    @pytest.fixture(autouse=True)
    def setup_scm_env(self, monkeypatch: pytest.MonkeyPatch):
        """Set up cache-dit + SCM environment variables for tests."""
        _apply_cache_dit_env(monkeypatch, CACHE_DIT_SCM_CONFIG)


@pytest.mark.skip(reason="Wan2.2 tests require large GPU memory, run manually")
class TestCacheDitDualTransformer(DiffusionServerBase):
    """cache-dit regression for dual-transformer models such as Wan2.2."""

    @pytest.fixture(params=CACHE_DIT_DUAL_TRANSFORMER_CASES, ids=lambda c: c.id)
    def case(self, request) -> DiffusionTestCase:
        """Provide a DiffusionTestCase for each dual-transformer test."""
        return request.param

    @pytest.fixture(autouse=True)
    def setup_dual_transformer_env(self, monkeypatch: pytest.MonkeyPatch):
        """Set up cache-dit environment for dual-transformer models."""
        _apply_cache_dit_env(monkeypatch, CACHE_DIT_DUAL_TRANSFORMER_CONFIG)
