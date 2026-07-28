"""Phase 13: Protocol and failure tests.

Tests bounded failure handling for protocol violations.
"""
from __future__ import annotations

import os
import sys
import torch

sys.path.insert(0, "/home/liang/sglang/python")

from sglang.srt.distributed.pp_packed_transport import (
    PPSchemaCache, validate_pp_packed_header, validate_pp_buffer_capacity,
    PPSchemaKey, calculate_pp_buffer_layout, pack_pp_proxy_tensors,
    unpack_pp_proxy_tensors, PROTOCOL_VERSION,
    _dtype_to_id, _id_to_dtype,
)
from sglang.srt.speculative.glm52_eagle3_pp import (
    validate_pp_proxy_keys, validate_capture_layers,
    REQUIRED_PP_PROXY_KEYS, GLM52_EAGLE3_AUX_PP_KEY,
)


def test_unknown_schema_id():
    """Unknown schema_id should raise diagnostic error."""
    cache = PPSchemaCache()
    try:
        validate_pp_packed_header(
            schema_id=999, active_rows=4, presence_mask=0xFF,
            expected_max_rows=64, recv_schema_cache=cache,
        )
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "unknown schema_id" in str(e).lower() or "schema" in str(e).lower()
    print("  Unknown schema_id: PASSED")


def test_negative_active_rows():
    """Negative active_rows should raise."""
    cache = PPSchemaCache()
    key = PPSchemaKey(0xFF, 64, 3, 8, 2, 64)
    entry = cache.register(key, 1000, 100, {}, {})
    try:
        validate_pp_packed_header(
            schema_id=entry.schema_id, active_rows=-1, presence_mask=0xFF,
            expected_max_rows=64, recv_schema_cache=cache,
        )
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "negative" in str(e).lower()
    print("  Negative active rows: PASSED")


def test_capacity_overflow():
    """active_rows exceeding max should raise."""
    cache = PPSchemaCache()
    key = PPSchemaKey(0xFF, 64, 3, 8, 2, 64)
    entry = cache.register(key, 1000, 100, {}, {})
    try:
        validate_pp_packed_header(
            schema_id=entry.schema_id, active_rows=128, presence_mask=0xFF,
            expected_max_rows=64, recv_schema_cache=cache,
        )
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "exceeds" in str(e).lower()
    print("  Capacity overflow: PASSED")


def test_presence_bitmask_mismatch():
    """Presence bitmask mismatch should raise."""
    cache = PPSchemaCache()
    key = PPSchemaKey(0xFF, 64, 3, 8, 2, 64)
    entry = cache.register(key, 1000, 100, {}, {})
    try:
        validate_pp_packed_header(
            schema_id=entry.schema_id, active_rows=4, presence_mask=0x0F,
            expected_max_rows=64, recv_schema_cache=cache,
        )
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "mismatch" in str(e).lower()
    print("  Presence bitmask mismatch: PASSED")


def test_missing_required_key():
    """Missing required key should raise."""
    try:
        validate_pp_proxy_keys(
            available_keys=["hidden_states"],
            pp_rank=1, tp_rank=0, forward_mode="DECODE",
            active_token_rows=4, remote_capture_layers_exist=False,
        )
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "residual" in str(e)
    print("  Missing required key: PASSED")


def test_missing_aux_on_pp1():
    """Missing aux key on PP1 with remote capture layers should raise."""
    try:
        validate_pp_proxy_keys(
            available_keys=["hidden_states", "residual"],
            pp_rank=1, tp_rank=0, forward_mode="DECODE",
            active_token_rows=4, remote_capture_layers_exist=True,
            slot_ownership={2: 0, 5: 1, 8: 1},
        )
        assert False, "Should have raised for missing aux"
    except RuntimeError as e:
        assert GLM52_EAGLE3_AUX_PP_KEY in str(e)
    print("  Missing aux on PP1: PASSED")


def test_buffer_capacity_insufficient():
    """Insufficient buffer capacity should raise."""
    cache = PPSchemaCache()
    key = PPSchemaKey(0xFF, 64, 3, 8, 2, 64)
    entry = cache.register(key, 10000, 1000, {}, {})
    
    small_db = torch.zeros(100, dtype=torch.bfloat16)
    small_cb = torch.zeros(10, dtype=torch.int32)
    
    try:
        validate_pp_buffer_capacity(small_db, small_cb, entry)
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "capacity" in str(e).lower()
    print("  Buffer capacity insufficient: PASSED")


def test_dtype_mapping():
    """Test dtype to ID and back mapping."""
    for dtype in [torch.float32, torch.float16, torch.bfloat16, torch.int32, torch.int64]:
        id_ = _dtype_to_id(dtype)
        assert id_ >= 0, f"Unknown dtype: {dtype}"
        recovered = _id_to_dtype(id_)
        assert recovered == dtype, f"Round-trip failed: {dtype} -> {id_} -> {recovered}"
    print("  Dtype mapping: PASSED")


def test_protocol_version():
    """Verify protocol version is defined."""
    assert PROTOCOL_VERSION == 1
    print("  Protocol version: PASSED")


def test_schema_cache_eviction():
    """Test schema cache evicts old entries."""
    cache = PPSchemaCache(max_entries=4)
    for i in range(10):
        key = PPSchemaKey(i, 64, 3, 8, 2, 64)
        cache.register(key, 1000, 100, {}, {})
    
    assert cache.size <= 4
    assert cache.evictions >= 6
    print(f"  Schema cache eviction: size={cache.size}, evictions={cache.evictions} PASSED")


if __name__ == "__main__":
    print("=== Phase 13: Protocol and Failure Tests ===")
    
    tests = [
        test_unknown_schema_id,
        test_negative_active_rows,
        test_capacity_overflow,
        test_presence_bitmask_mismatch,
        test_missing_required_key,
        test_missing_aux_on_pp1,
        test_buffer_capacity_insufficient,
        test_dtype_mapping,
        test_protocol_version,
        test_schema_cache_eviction,
    ]
    
    for test in tests:
        test()
    
    print(f"\n=== All {len(tests)} Phase 13 tests PASSED ===")
