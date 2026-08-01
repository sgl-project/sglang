"""Tests for MooncakeStore logical anchor marker key methods."""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List


class TestMooncakeStoreLogicalAnchor:
    """Test logical anchor marker key methods in MooncakeStore."""

    @pytest.fixture
    def mooncake_store(self):
        """Create a MooncakeStore instance with mocked store."""
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

        # Mock the underlying store
        mock_store = Mock()
        store = MooncakeStore.__new__(MooncakeStore)
        store.store = mock_store
        return store

    def test_logical_anchor_key_generation(self, mooncake_store):
        """Test that marker keys are generated with _logical_kv suffix."""
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

        key = "test_prefix_123"
        marker_key = MooncakeStore._logical_anchor_key(key)
        assert marker_key == "test_prefix_123_logical_kv"

    def test_batch_set_logical_anchor_new_keys(self, mooncake_store):
        """Test setting marker keys for new logical anchor pages."""
        keys = ["key1", "key2", "key3"]

        # Mock _batch_exist to return all 0 (keys don't exist)
        mooncake_store.store.batch_is_exist.return_value = [0, 0, 0]
        mooncake_store.store.put.return_value = 0  # success

        results = mooncake_store._batch_set_logical_anchor(keys)

        # All should succeed
        assert results == [True, True, True]

        # Verify put was called with marker keys and 1-byte marker
        assert mooncake_store.store.put.call_count == 3
        calls = mooncake_store.store.put.call_args_list
        for i, call in enumerate(calls):
            marker_key = f"{keys[i]}_logical_kv"
            assert call[0][0] == marker_key
            assert call[0][1] == b"1"  # 1-byte marker

    def test_batch_set_logical_anchor_existing_keys(self, mooncake_store):
        """Test that existing marker keys are skipped."""
        keys = ["key1", "key2", "key3"]

        # Mock _batch_exist: key1 exists, key2 and key3 don't
        mooncake_store.store.batch_is_exist.return_value = [1, 0, 0]
        mooncake_store.store.put.return_value = 0

        results = mooncake_store._batch_set_logical_anchor(keys)

        # key1 should be True (already existed), key2 and key3 should succeed
        assert results == [True, True, True]

        # Verify put was only called for key2 and key3 (not key1)
        assert mooncake_store.store.put.call_count == 2
        calls = mooncake_store.store.put.call_args_list
        assert calls[0][0][0] == "key2_logical_kv"
        assert calls[1][0][0] == "key3_logical_kv"

    def test_batch_set_logical_anchor_partial_failure(self, mooncake_store):
        """Test handling of partial write failures."""
        keys = ["key1", "key2"]

        # Mock _batch_exist to return all 0 (keys don't exist)
        mooncake_store.store.batch_is_exist.return_value = [0, 0]
        # Mock put: key1 succeeds, key2 fails
        mooncake_store.store.put.side_effect = [0, -1]

        results = mooncake_store._batch_set_logical_anchor(keys)

        # key1 should succeed, key2 should fail
        assert results == [True, False]

    def test_batch_get_logical_anchor_all_exist(self, mooncake_store):
        """Test checking marker keys when all exist."""
        keys = ["key1", "key2", "key3"]

        # Mock _batch_exist to return all 1 (keys exist)
        mooncake_store.store.batch_is_exist.return_value = [1, 1, 1]

        results = mooncake_store._batch_get_logical_anchor(keys)

        assert results == [True, True, True]

        # Verify batch_is_exist was called with marker keys
        mooncake_store.store.batch_is_exist.assert_called_once_with(
            ["key1_logical_kv", "key2_logical_kv", "key3_logical_kv"]
        )

    def test_batch_get_logical_anchor_partial_exist(self, mooncake_store):
        """Test checking marker keys when some exist."""
        keys = ["key1", "key2", "key3"]

        # Mock _batch_exist: key1 and key3 exist, key2 doesn't
        mooncake_store.store.batch_is_exist.return_value = [1, 0, 1]

        results = mooncake_store._batch_get_logical_anchor(keys)

        assert results == [True, False, True]

    def test_batch_get_logical_anchor_none_exist(self, mooncake_store):
        """Test checking marker keys when none exist."""
        keys = ["key1", "key2"]

        # Mock _batch_exist to return all 0 (keys don't exist)
        mooncake_store.store.batch_is_exist.return_value = [0, 0]

        results = mooncake_store._batch_get_logical_anchor(keys)

        assert results == [False, False]

    def test_batch_set_empty_list(self, mooncake_store):
        """Test setting empty list of keys."""
        keys = []

        # Mock _batch_exist to return empty list
        mooncake_store.store.batch_is_exist.return_value = []

        results = mooncake_store._batch_set_logical_anchor(keys)

        assert results == []
        mooncake_store.store.put.assert_not_called()

    def test_batch_get_empty_list(self, mooncake_store):
        """Test getting empty list of keys."""
        keys = []

        # Mock _batch_exist to return empty list
        mooncake_store.store.batch_is_exist.return_value = []

        results = mooncake_store._batch_get_logical_anchor(keys)

        assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
