import logging
import sys
from types import SimpleNamespace

import pytest

from sglang.srt.configs.model_config import ModelConfig


def _build_model_config_stub(
    *,
    model_path: str,
    model_type: str,
    architectures: list[str] | None = None,
    vision_model_type: str | None = None,
) -> ModelConfig:
    config = object.__new__(ModelConfig)
    config.model_path = model_path

    hf_config = SimpleNamespace(
        model_type=model_type,
        architectures=architectures or [],
    )
    if vision_model_type is not None:
        hf_config.vision_config = SimpleNamespace(model_type=vision_model_type)

    config.hf_config = hf_config
    return config


def _mock_transformers_version(monkeypatch, version: str) -> None:
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__=version))


def test_verify_transformers_version_glm4_moe_lite_no_downgrade_warning(
    monkeypatch, caplog
):
    _mock_transformers_version(monkeypatch, "5.3.0.dev0")
    model_config = _build_model_config_stub(
        model_path="/ssd512g/models/GLM-4.7-Flash",
        model_type="glm4_moe_lite",
        architectures=["Glm4MoeLiteForCausalLM"],
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        model_config._verify_transformers_version()

    assert not any(
        "downgrading to transformers==4.57.1" in message
        for message in caplog.messages
    )


def test_verify_transformers_version_glm4_moe_lite_requires_tf5(monkeypatch):
    _mock_transformers_version(monkeypatch, "4.57.1")
    model_config = _build_model_config_stub(
        model_path="/ssd512g/models/GLM-4.7-Flash",
        model_type="glm4_moe_lite",
        architectures=["Glm4MoeLiteForCausalLM"],
    )

    with pytest.raises(ValueError, match="Please upgrade transformers to >= 5.0.0"):
        model_config._verify_transformers_version()


def test_verify_transformers_version_non_glm4_moe_lite_warns_on_tf5(
    monkeypatch, caplog
):
    _mock_transformers_version(monkeypatch, "5.3.0.dev0")
    model_config = _build_model_config_stub(
        model_path="/tmp/llama",
        model_type="llama",
        architectures=["LlamaForCausalLM"],
    )

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        model_config._verify_transformers_version()

    assert any(
        "downgrading to transformers==4.57.1" in message
        for message in caplog.messages
    )
