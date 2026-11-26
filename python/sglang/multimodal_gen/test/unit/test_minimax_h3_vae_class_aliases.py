# SPDX-License-Identifier: Apache-2.0
"""Comfy/FastVideo checkpoint class names must resolve to the native H3 VAEs."""

import pytest

from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.models.vaes.minimax_h3 import (
    MiniMaxH3AudioVAE,
    MiniMaxH3VideoVAE,
)


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("AutoencoderKLMiniMaxH3", MiniMaxH3VideoVAE),
        ("AutoencoderKLMiniMaxH3Audio", MiniMaxH3AudioVAE),
    ],
)
def test_comfy_vae_class_name_resolves_to_the_native_vae(alias, expected):
    vae_cls, arch = ModelRegistry.resolve_model_cls(alias)
    assert vae_cls is expected
    assert arch == expected.__name__
