"""Checkpoint-backed fixtures for remote LFM3 preprocessing tests."""

import os

import pytest
from transformers import AutoConfig, AutoFeatureExtractor


@pytest.fixture(scope="session")
def lfm3_audio_reference():
    model = os.environ.get("SGLANG_TEST_LFM3_AUDIO_MODEL")
    if model is None:
        pytest.skip("Set SGLANG_TEST_LFM3_AUDIO_MODEL to test checkpoint-provided code")
    return (
        AutoConfig.from_pretrained(model, trust_remote_code=True),
        AutoFeatureExtractor.from_pretrained(model, trust_remote_code=True),
    )


@pytest.fixture(scope="session")
def lfm3_omni_reference():
    model = os.environ.get("SGLANG_TEST_LFM3_OMNI_MODEL")
    if model is None:
        pytest.skip("Set SGLANG_TEST_LFM3_OMNI_MODEL to test checkpoint-provided code")
    return AutoConfig.from_pretrained(model, trust_remote_code=True)
