# SPDX-License-Identifier: Apache-2.0
"""Per-construction caches must neither consume nor publish global RoPE state."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.runtime.models.encoders.qwen_vl_rope import (
    build_qwen_vl_text_rope,
    isolated_qwen_vl_rope_cache,
)
from sglang.srt.layers.rotary_embedding import factory


def build():
    return build_qwen_vl_text_rope(
        SimpleNamespace(
            head_dim=128,
            max_position_embeddings=16,
            rope_theta=5000000,
            rope_scaling={"rope_type": "default", "mrope_section": [24, 20, 20]},
        ),
        mrope_interleaved=True,
    )


def test_scoped_meta_rope_preserves_aliases_without_mutating_real_model():
    with torch.device("cpu"):
        ordinary = build()
    old = dict(factory._ROPE_DICT)
    ptr = ordinary.cos_sin_cache.data_ptr()
    with isolated_qwen_vl_rope_cache(), torch.device("meta"):
        meta = build()
        assert meta is build()
        assert meta is not ordinary
        assert all(buffer.is_meta for buffer in meta.buffers())
    assert factory._ROPE_DICT == old
    assert ordinary.cos_sin_cache.data_ptr() == ptr
    with isolated_qwen_vl_rope_cache(), torch.device("cpu"):
        fresh = build()
        assert fresh is build()
        assert fresh is not meta and fresh is not ordinary
        torch.testing.assert_close(fresh.cos_sin_cache, ordinary.cos_sin_cache)
    assert factory._ROPE_DICT == old


def test_nested_rope_scope_and_error_restore_previous_cache():
    with isolated_qwen_vl_rope_cache(), torch.device("cpu"):
        outer = build()
        with pytest.raises(RuntimeError, match="failed init"):
            with isolated_qwen_vl_rope_cache():
                assert build() is not outer
                raise RuntimeError("failed init")
        assert build() is outer
