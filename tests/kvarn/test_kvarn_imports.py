# SPDX-License-Identifier: Apache-2.0
"""Import smoke tests for KVarN sglang integration modules.

These tests verify that the KVarN modules can be imported without errors.
They use the conftest.py shim to avoid pulling in the full sglang package.

Run:
    python -m pytest tests/kvarn/test_kvarn_imports.py -v
"""

import pytest

# These imports rely on the conftest.py shim for the parent packages.
# The kvarn sub-packages themselves have no heavy deps.


def test_import_config():
    from sglang.srt.layers.quantization.kvarn.config import KVARN_PRESETS, KVarNConfig

    assert "kvarn_k4v4_g128" in KVARN_PRESETS
    cfg = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=128)
    assert cfg.key_bits == 4


def test_import_sinkhorn():
    from sglang.srt.layers.quantization.kvarn.sinkhorn import variance_normalize

    assert callable(variance_normalize)


def test_import_store():
    from sglang.srt.layers.quantization.kvarn.store import kvarn_store_tile_k

    assert callable(kvarn_store_tile_k)


def test_import_dequant():
    from sglang.srt.layers.quantization.kvarn.dequant import kvarn_dequant_tile_k

    assert callable(kvarn_dequant_tile_k)


def test_import_hadamard():
    from sglang.srt.layers.quantization.kvarn.hadamard import build_hadamard

    assert callable(build_hadamard)


def test_import_triton_sinkhorn():
    from sglang.srt.layers.attention.kvarn_ops.triton_sinkhorn import (
        kvarn_sinkhorn_triton,
    )

    assert callable(kvarn_sinkhorn_triton)


def test_import_triton_decode():
    from sglang.srt.layers.attention.kvarn_ops.triton_decode import (
        _kvarn_build_packed_kv_kernel,
        _kvarn_fused_decode_kernel,
        _kvarn_fused_verify_stage1,
        _kvarn_scatter_store_kernel,
        kvarn_decode_attention,
        kvarn_scatter_store,
        kvarn_verify_attention,
    )

    # These are Triton JIT kernels — just verify they're decorated.
    assert _kvarn_scatter_store_kernel is not None
    assert _kvarn_build_packed_kv_kernel is not None
    assert _kvarn_fused_decode_kernel is not None
    assert _kvarn_fused_verify_stage1 is not None
    assert callable(kvarn_decode_attention)
    assert callable(kvarn_verify_attention)
    assert callable(kvarn_scatter_store)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
