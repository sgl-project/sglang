# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.utils.torch_compile import (
    BLACKWELL_TORCH_COMPILE_MODE,
    DEFAULT_TORCH_COMPILE_MODE,
    TORCH_COMPILE_MODE_ENV,
    resolve_torch_compile_mode,
)


def test_resolve_torch_compile_mode_uses_env_override(monkeypatch):
    monkeypatch.setenv(TORCH_COMPILE_MODE_ENV, "max-autotune")

    with patch(
        "sglang.multimodal_gen.runtime.utils.torch_compile.current_platform.is_blackwell",
        return_value=True,
    ):
        assert resolve_torch_compile_mode(None) == "max-autotune"


def test_resolve_torch_compile_mode_keeps_default_off_blackwell(monkeypatch):
    monkeypatch.delenv(TORCH_COMPILE_MODE_ENV, raising=False)

    with patch(
        "sglang.multimodal_gen.runtime.utils.torch_compile.current_platform.is_blackwell",
        return_value=False,
    ):
        assert resolve_torch_compile_mode(None) == DEFAULT_TORCH_COMPILE_MODE


def test_resolve_torch_compile_mode_uses_default_on_blackwell(monkeypatch):
    monkeypatch.delenv(TORCH_COMPILE_MODE_ENV, raising=False)

    with patch(
        "sglang.multimodal_gen.runtime.utils.torch_compile.current_platform.is_blackwell",
        return_value=True,
    ):
        assert resolve_torch_compile_mode(None) == BLACKWELL_TORCH_COMPILE_MODE


def test_resolve_torch_compile_mode_keeps_explicit_model_mode(monkeypatch):
    monkeypatch.delenv(TORCH_COMPILE_MODE_ENV, raising=False)
    config = SimpleNamespace(torch_compile_mode="reduce-overhead")

    with patch(
        "sglang.multimodal_gen.runtime.utils.torch_compile.current_platform.is_blackwell",
        return_value=True,
    ):
        assert resolve_torch_compile_mode(config) == "reduce-overhead"
