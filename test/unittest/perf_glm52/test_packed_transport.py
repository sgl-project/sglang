"""CPU-only correctness tests for packed PP transport.

Verifies that pack/unpack produces exactly equivalent logical tensors
to the original tensor-dict path.
"""
import math
import struct
import threading

import pytest
import torch

from sglang.srt.distributed.pp_packed_transport import (
    BIT_HIDDEN, BIT_RESIDUAL, BIT_AUX, BIT_TOPK,
    BIT_NEXT_TOKEN_IDS, BIT_ACCEPT_LENS, BIT_NEW_SEQ_LENS, BIT_BONUS_TOKENS,
    PROTOCOL_VERSION, SCHEMA_CACHE_MAX,
    PPSchemaKey, PPSchemaCache, PPSchemaEntry, PPStaticBufferRegistry,
    calculate_pp_buffer_layout,
    pack_pp_proxy_tensors,
    unpack_pp_proxy_tensors,
    validate_pp_packed_header,
    validate_pp_buffer_capacity,
    _dtype_to_id, _id_to_dtype,
)


HIDDEN_SIZE = 5120
CAPTURE_LAYERS = 3
TOPK_SIZE = 64


def _make_tensor_dict(rows, include_aux=True, include_topk=True,
                      include_relay=False, dtype=torch.bfloat16):
    d = {
        "hidden_states": torch.randn(rows, HIDDEN_SIZE, dtype=dtype),
        "residual": torch.randn(rows, HIDDEN_SIZE, dtype=dtype),
    }
    if include_aux:
        d["glm52_eagle3_aux_hidden_states"] = torch.randn(
            rows, CAPTURE_LAYERS, HIDDEN_SIZE, dtype=dtype
        )
    if include_topk:
        d["topk_indices"] = torch.randint(0, TOPK_SIZE, (rows, TOPK_SIZE), dtype=torch.int32)
    if include_relay:
        d["next_token_ids"] = torch.randint(0, 151552, (rows,), dtype=torch.int64)
        d["accept_lens"] = torch.randint(1, 8, (rows,), dtype=torch.int32)
        d["new_seq_lens"] = torch.randint(1, 200000, (rows,), dtype=torch.int32)
        d["bonus_tokens"] = torch.randint(0, 151552, (rows,), dtype=torch.int64)
    return d


class TestLayoutCalculation:
    def test_hidden_only_layout(self):
        d = {"hidden_states": torch.randn(4, HIDDEN_SIZE, dtype=torch.bfloat16)}
        key, data_nelem, ctrl_nelem, d_off, c_off = calculate_pp_buffer_layout(
            d, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE, max_rows_bucket=16
        )
        assert key.presence_mask == BIT_HIDDEN
        assert data_nelem == 16 * HIDDEN_SIZE
        assert ctrl_nelem == 0
        assert "hidden_states" in d_off

    def test_full_layout(self):
        d = _make_tensor_dict(4)
        key, data_nelem, ctrl_nelem, d_off, c_off = calculate_pp_buffer_layout(
            d, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE, max_rows_bucket=16
        )
        assert key.presence_mask == (BIT_HIDDEN | BIT_RESIDUAL | BIT_AUX | BIT_TOPK)
        assert data_nelem == 16 * HIDDEN_SIZE * (1 + 1 + CAPTURE_LAYERS)
        assert ctrl_nelem == 16 * TOPK_SIZE
        assert len(d_off) == 3
        assert len(c_off) == 1

    def test_relay_layout(self):
        d = _make_tensor_dict(4, include_aux=False, include_topk=False, include_relay=True)
        key, data_nelem, ctrl_nelem, d_off, c_off = calculate_pp_buffer_layout(
            d, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE, max_rows_bucket=16
        )
        assert key.presence_mask == (BIT_HIDDEN | BIT_RESIDUAL | BIT_NEXT_TOKEN_IDS |
                                      BIT_ACCEPT_LENS | BIT_NEW_SEQ_LENS | BIT_BONUS_TOKENS)
        assert data_nelem == 16 * HIDDEN_SIZE * 2
        # 4 control tensors, each 16 elements
        assert ctrl_nelem == 16 * 4


class TestPackUnpack:
    def _round_trip(self, rows, bucket, **kwargs):
        dtype = kwargs.get("dtype", torch.bfloat16)
        d = _make_tensor_dict(rows, **kwargs)
        key, data_nelem, ctrl_nelem, d_off, c_off = calculate_pp_buffer_layout(
            d, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE, max_rows_bucket=bucket
        )
        data_buf = torch.zeros(max(data_nelem, 1), dtype=dtype)
        ctrl_buf = torch.zeros(max(ctrl_nelem, 1), dtype=torch.int32)

        pack_pp_proxy_tensors(d, data_buf, ctrl_buf, d_off, c_off, rows)

        # Build schema entry for unpack
        entry = PPSchemaEntry(
            schema_id=0, key=key,
            data_num_elements=data_nelem, control_num_elements=ctrl_nelem,
            data_offsets=d_off, control_offsets=c_off,
        )
        result = unpack_pp_proxy_tensors(
            data_buf, ctrl_buf, entry, rows, torch.device("cpu"),
            dtype, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE
        )
        return d, result

    def test_bf16_hidden_residual_aux_topk(self):
        d, result = self._round_trip(4, 16, include_aux=True, include_topk=True)
        for key in d:
            assert key in result, f"Missing key: {key}"
            assert torch.equal(d[key], result[key]), f"Mismatch for {key}"

    def test_bf16_hidden_only(self):
        d, result = self._round_trip(4, 16, include_aux=False, include_topk=False)
        for key in d:
            assert torch.equal(d[key], result[key]), f"Mismatch for {key}"

    def test_fp16_dtype(self):
        d, result = self._round_trip(8, 16, dtype=torch.float16)
        for key in d:
            assert torch.equal(d[key], result[key]), f"Mismatch for {key}"

    def test_relay_tensors(self):
        d, result = self._round_trip(4, 16, include_aux=False, include_topk=False, include_relay=True)
        for key in d:
            assert torch.equal(d[key], result[key]), f"Mismatch for {key}"

    def test_single_row(self):
        d, result = self._round_trip(1, 16)
        for key in d:
            assert torch.equal(d[key], result[key]), f"Mismatch for {key}"

    def test_large_rows(self):
        d, result = self._round_trip(256, 256)
        for key in d:
            assert torch.equal(d[key], result[key]), f"Mismatch for {key}"

    def test_non_contiguous_input(self):
        d = _make_tensor_dict(4)
        # Make hidden_states non-contiguous via slicing
        big = torch.randn(8, HIDDEN_SIZE, dtype=torch.bfloat16)
        d["hidden_states"] = big[::2]  # Strided view -> non-contiguous
        assert not d["hidden_states"].is_contiguous()

        key, data_nelem, ctrl_nelem, d_off, c_off = calculate_pp_buffer_layout(
            d, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE, max_rows_bucket=16
        )
        data_buf = torch.zeros(max(data_nelem, 1), dtype=torch.bfloat16)
        ctrl_buf = torch.zeros(max(ctrl_nelem, 1), dtype=torch.int32)
        pack_pp_proxy_tensors(d, data_buf, ctrl_buf, d_off, c_off, 4)

        entry = PPSchemaEntry(
            schema_id=0, key=key,
            data_num_elements=data_nelem, control_num_elements=ctrl_nelem,
            data_offsets=d_off, control_offsets=c_off,
        )
        result = unpack_pp_proxy_tensors(
            data_buf, ctrl_buf, entry, 4, torch.device("cpu"),
            torch.bfloat16, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE
        )
        # hidden_states should be contiguous in result
        assert result["hidden_states"].is_contiguous()
        assert torch.equal(d["hidden_states"].contiguous(), result["hidden_states"])


class TestBatchTransitions:
    def test_large_to_small(self):
        """16 → 1: packed buffer sized for 16, only 1 row active."""
        d_large = _make_tensor_dict(16)
        d_small = _make_tensor_dict(1)

        key, data_nelem, ctrl_nelem, d_off, c_off = calculate_pp_buffer_layout(
            d_large, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE, max_rows_bucket=16
        )
        data_buf = torch.zeros(max(data_nelem, 1), dtype=torch.bfloat16)
        ctrl_buf = torch.zeros(max(ctrl_nelem, 1), dtype=torch.int32)

        # Pack large
        pack_pp_proxy_tensors(d_large, data_buf, ctrl_buf, d_off, c_off, 16)
        # Pack small into same buffer
        pack_pp_proxy_tensors(d_small, data_buf, ctrl_buf, d_off, c_off, 1)

        entry = PPSchemaEntry(
            schema_id=0, key=key,
            data_num_elements=data_nelem, control_num_elements=ctrl_nelem,
            data_offsets=d_off, control_offsets=c_off,
        )
        result = unpack_pp_proxy_tensors(
            data_buf, ctrl_buf, entry, 1, torch.device("cpu"),
            torch.bfloat16, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE
        )
        for key in d_small:
            assert torch.equal(d_small[key], result[key])
            # Verify no stale data from large batch
            assert result[key].shape[0] == 1

    def test_transition_sequence(self):
        """1 → 16 → 1: verify no stale rows."""
        bucket = 16
        d1 = _make_tensor_dict(1)
        d16 = _make_tensor_dict(16)
        d1b = _make_tensor_dict(1)

        key, data_nelem, ctrl_nelem, d_off, c_off = calculate_pp_buffer_layout(
            d16, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE, max_rows_bucket=bucket
        )
        data_buf = torch.zeros(max(data_nelem, 1), dtype=torch.bfloat16)
        ctrl_buf = torch.zeros(max(ctrl_nelem, 1), dtype=torch.int32)
        entry = PPSchemaEntry(
            schema_id=0, key=key,
            data_num_elements=data_nelem, control_num_elements=ctrl_nelem,
            data_offsets=d_off, control_offsets=c_off,
        )

        for d, rows in [(d1, 1), (d16, 16), (d1b, 1)]:
            pack_pp_proxy_tensors(d, data_buf, ctrl_buf, d_off, c_off, rows)
            result = unpack_pp_proxy_tensors(
                data_buf, ctrl_buf, entry, rows, torch.device("cpu"),
                torch.bfloat16, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE
            )
            for k in d:
                assert torch.equal(d[k], result[k]), f"Mismatch at rows={rows}, key={k}"
                assert result[k].shape[0] == rows

    def test_64_to_4_to_64(self):
        bucket = 64
        d64a = _make_tensor_dict(64)
        d4 = _make_tensor_dict(4)
        d64b = _make_tensor_dict(64)

        key, data_nelem, ctrl_nelem, d_off, c_off = calculate_pp_buffer_layout(
            d64a, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE, max_rows_bucket=bucket
        )
        data_buf = torch.zeros(max(data_nelem, 1), dtype=torch.bfloat16)
        ctrl_buf = torch.zeros(max(ctrl_nelem, 1), dtype=torch.int32)
        entry = PPSchemaEntry(
            schema_id=0, key=key,
            data_num_elements=data_nelem, control_num_elements=ctrl_nelem,
            data_offsets=d_off, control_offsets=c_off,
        )

        for d, rows in [(d64a, 64), (d4, 4), (d64b, 64)]:
            pack_pp_proxy_tensors(d, data_buf, ctrl_buf, d_off, c_off, rows)
            result = unpack_pp_proxy_tensors(
                data_buf, ctrl_buf, entry, rows, torch.device("cpu"),
                torch.bfloat16, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE
            )
            for k in d:
                assert torch.equal(d[k], result[k]), f"Mismatch at rows={rows}, key={k}"


class TestSchemaCache:
    def test_cache_hit_miss(self):
        cache = PPSchemaCache()
        key = PPSchemaKey(
            presence_mask=0xF, hidden_size=5120, capture_layers=3,
            topk_size=64, dtype_id=2, max_rows_bucket=16,
        )
        assert cache.lookup(key) is None
        assert cache.misses == 1
        assert cache.hits == 0

        cache.register(key, 100, 200, {}, {})
        entry = cache.lookup(key)
        assert entry is not None
        assert cache.hits == 1

    def test_cache_eviction(self):
        cache = PPSchemaCache(max_entries=3)
        for i in range(5):
            key = PPSchemaKey(
                presence_mask=i, hidden_size=5120, capture_layers=3,
                topk_size=64, dtype_id=2, max_rows_bucket=16,
            )
            cache.register(key, 100, 200, {}, {})
        assert cache.size <= 3
        assert cache.evictions >= 2

    def test_get_by_id(self):
        cache = PPSchemaCache()
        key = PPSchemaKey(
            presence_mask=0xF, hidden_size=5120, capture_layers=3,
            topk_size=64, dtype_id=2, max_rows_bucket=16,
        )
        entry = cache.register(key, 100, 200, {}, {})
        assert cache.get_by_id(entry.schema_id) is not None
        assert cache.get_by_id(999) is None


class TestValidation:
    def test_unknown_schema_id(self):
        cache = PPSchemaCache()
        with pytest.raises(RuntimeError, match="unknown schema_id"):
            validate_pp_packed_header(999, 4, 0xF, 16, cache)

    def test_negative_active_rows(self):
        cache = PPSchemaCache()
        key = PPSchemaKey(0xF, 5120, 3, 64, 2, 16)
        entry = cache.register(key, 100, 200, {}, {})
        with pytest.raises(RuntimeError, match="negative active_rows"):
            validate_pp_packed_header(entry.schema_id, -1, 0xF, 16, cache)

    def test_active_rows_overflow(self):
        cache = PPSchemaCache()
        key = PPSchemaKey(0xF, 5120, 3, 64, 2, 16)
        entry = cache.register(key, 100, 200, {}, {})
        with pytest.raises(RuntimeError, match="exceeds"):
            validate_pp_packed_header(entry.schema_id, 32, 0xF, 16, cache)

    def test_presence_mismatch(self):
        cache = PPSchemaCache()
        key = PPSchemaKey(0xF, 5120, 3, 64, 2, 16)
        entry = cache.register(key, 100, 200, {}, {})
        with pytest.raises(RuntimeError, match="bitmask mismatch"):
            validate_pp_packed_header(entry.schema_id, 4, 0x3, 16, cache)

    def test_buffer_capacity_overflow(self):
        key = PPSchemaKey(0xF, 5120, 3, 64, 2, 16)
        entry = PPSchemaEntry(
            schema_id=0, key=key, data_num_elements=1000,
            control_num_elements=200, data_offsets={}, control_offsets={},
        )
        small_buf = torch.zeros(100, dtype=torch.bfloat16)
        ctrl_buf = torch.zeros(200, dtype=torch.int32)
        with pytest.raises(RuntimeError, match="data buffer capacity"):
            validate_pp_buffer_capacity(small_buf, ctrl_buf, entry)


class TestStaticBufferRegistry:
    def test_allocation_and_reuse(self):
        reg = PPStaticBufferRegistry(torch.device("cpu"))
        d1, c1 = reg.get_or_allocate(16, 1000, 200, torch.bfloat16)
        assert reg.allocation_count == 1
        d2, c2 = reg.get_or_allocate(16, 1000, 200, torch.bfloat16)
        assert reg.allocation_count == 1  # No new allocation
        assert d1.data_ptr() == d2.data_ptr()  # Same buffer
        assert c1.data_ptr() == c2.data_ptr()

    def test_multiple_buckets(self):
        reg = PPStaticBufferRegistry(torch.device("cpu"))
        reg.get_or_allocate(16, 100, 20, torch.bfloat16)
        reg.get_or_allocate(64, 400, 80, torch.bfloat16)
        assert reg.size == 2

    def test_reset(self):
        reg = PPStaticBufferRegistry(torch.device("cpu"))
        reg.get_or_allocate(16, 100, 20, torch.bfloat16)
        reg.reset()
        assert reg.size == 0


class TestDtypeMapping:
    def test_roundtrip(self):
        for dtype in [torch.float32, torch.float16, torch.bfloat16,
                      torch.int32, torch.int64]:
            assert _id_to_dtype(_dtype_to_id(dtype)) == dtype


class TestMissingKeys:
    def test_missing_required_hidden(self):
        d = {"residual": torch.randn(4, HIDDEN_SIZE, dtype=torch.bfloat16)}
        key, data_nelem, ctrl_nelem, d_off, c_off = calculate_pp_buffer_layout(
            d, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE, max_rows_bucket=16
        )
        # Should not have hidden in offsets
        assert "hidden_states" not in d_off
        assert not (key.presence_mask & BIT_HIDDEN)  # Bit is 0

    def test_missing_optional_topk(self):
        d = _make_tensor_dict(4, include_topk=False)
        key, _, _, d_off, c_off = calculate_pp_buffer_layout(
            d, HIDDEN_SIZE, CAPTURE_LAYERS, TOPK_SIZE, max_rows_bucket=16
        )
        assert not (key.presence_mask & BIT_TOPK)
        assert "topk_indices" not in c_off
