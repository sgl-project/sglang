"""Tests for draft pool graceful degradation with storage backend."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import logging


class TestDraftPoolGracefulDegradation:
    """Test draft pool registration graceful degradation."""

    @pytest.fixture
    def cache_controller(self):
        """Create a minimal HiCacheController instance for testing."""
        from sglang.srt.managers.cache_controller import HiCacheController

        ctrl = HiCacheController.__new__(HiCacheController)
        ctrl.enable_storage = True
        ctrl.has_draft = False
        ctrl.storage_backend_type = "mooncake"
        ctrl.storage_config = Mock()
        ctrl.storage_config.should_split_heads = False
        ctrl.draft_page_get_func = None
        ctrl.draft_page_set_func = None
        return ctrl

    def test_maybe_register_draft_no_storage(self, cache_controller):
        """Test that registration is skipped when storage is disabled."""
        cache_controller.enable_storage = False
        cache_controller.has_draft = True

        cache_controller._maybe_register_draft_with_storage()
        assert cache_controller.draft_page_get_func is None
        assert cache_controller.draft_page_set_func is None

    def test_maybe_register_draft_no_draft(self, cache_controller):
        """Test that registration is skipped when draft is not enabled."""
        cache_controller.enable_storage = True
        cache_controller.has_draft = False

        cache_controller._maybe_register_draft_with_storage()
        assert cache_controller.draft_page_get_func is None
        assert cache_controller.draft_page_set_func is None

    def test_maybe_register_draft_mooncake_success(self, cache_controller):
        """Test successful Mooncake draft pool registration wires up v2 funcs."""
        cache_controller.enable_storage = True
        cache_controller.has_draft = True
        cache_controller.storage_backend_type = "mooncake"

        mock_draft_pool = Mock()
        mock_draft_pool.size = 1024
        mock_draft_pool.page_size = 64
        cache_controller.mem_pool_host_draft = mock_draft_pool

        mock_backend = Mock()
        mock_backend.register_mem_host_pool_v2 = Mock()
        cache_controller.storage_backend = mock_backend

        cache_controller._maybe_register_draft_with_storage()

        mock_backend.register_mem_host_pool_v2.assert_called_once()
        assert cache_controller.draft_page_get_func == cache_controller._draft_page_get_v2
        assert cache_controller.draft_page_set_func == cache_controller._draft_page_set_v2

    def test_maybe_register_draft_mooncake_registration_failure(self, cache_controller, caplog):
        """Test graceful degradation when Mooncake registration fails (EAGLE + mmap)."""
        cache_controller.enable_storage = True
        cache_controller.has_draft = True
        cache_controller.storage_backend_type = "mooncake"

        mock_draft_pool = Mock()
        mock_draft_pool.size = 1024
        mock_draft_pool.page_size = 64
        cache_controller.mem_pool_host_draft = mock_draft_pool

        mock_backend = Mock()
        mock_backend.register_mem_host_pool_v2 = Mock(
            side_effect=RuntimeError("Buffer outside shared memory segment")
        )
        cache_controller.storage_backend = mock_backend

        with caplog.at_level(logging.WARNING):
            cache_controller._maybe_register_draft_with_storage()

        assert cache_controller.draft_page_get_func is None
        assert cache_controller.draft_page_set_func is None
        assert any(
            "Draft L3 (storage backend) operations will be skipped" in record.message
            for record in caplog.records
        )

    def test_maybe_register_draft_split_heads_disabled(self, cache_controller, caplog):
        """Test that split_heads config disables draft L3 on Mooncake."""
        cache_controller.enable_storage = True
        cache_controller.has_draft = True
        cache_controller.storage_backend_type = "mooncake"
        cache_controller.storage_config.should_split_heads = True

        mock_draft_pool = Mock()
        cache_controller.mem_pool_host_draft = mock_draft_pool
        cache_controller.storage_backend = Mock()

        with caplog.at_level(logging.WARNING):
            cache_controller._maybe_register_draft_with_storage()

        assert cache_controller.draft_page_get_func is None
        assert cache_controller.draft_page_set_func is None
        assert any("should_split_heads" in record.message for record in caplog.records)

    def test_maybe_register_draft_generic_backend(self, cache_controller):
        """Test generic backends wire up generic funcs."""
        cache_controller.enable_storage = True
        cache_controller.has_draft = True
        cache_controller.storage_backend_type = "hf3fs"

        mock_draft_pool = Mock()
        cache_controller.mem_pool_host_draft = mock_draft_pool
        cache_controller.storage_backend = Mock()

        with pytest.MonkeyPatch.context() as m:
            # hf3fs is in the unsupported list, should warn and leave funcs None
            cache_controller._maybe_register_draft_with_storage()
            assert cache_controller.draft_page_get_func is None
            assert cache_controller.draft_page_set_func is None

    def test_maybe_register_draft_generic_fallback(self, cache_controller):
        """Test unknown backends fall through to generic funcs."""
        cache_controller.enable_storage = True
        cache_controller.has_draft = True
        cache_controller.storage_backend_type = "some_new_backend"

        mock_draft_pool = Mock()
        cache_controller.mem_pool_host_draft = mock_draft_pool
        cache_controller.storage_backend = Mock()

        cache_controller._maybe_register_draft_with_storage()

        assert cache_controller.draft_page_get_func == cache_controller._draft_page_get_generic
        assert cache_controller.draft_page_set_func == cache_controller._draft_page_set_generic

    def test_draft_page_set_no_func_is_noop(self, cache_controller):
        """Test _draft_page_set is a no-op when draft_page_set_func is None."""
        cache_controller.draft_page_set_func = None
        # Should not raise
        cache_controller._draft_page_set(["hash1"], [0, 1, 2])

    def test_draft_page_get_no_func_is_noop(self, cache_controller):
        """Test _draft_page_get is a no-op when draft_page_get_func is None."""
        cache_controller.draft_page_get_func = None
        # Should not raise
        cache_controller._draft_page_get(["hash1"], [0, 1, 2])

    def test_draft_page_set_exception_swallowed(self, cache_controller):
        """Test _draft_page_set swallows exceptions (best-effort)."""
        def failing_func(hashes, indices):
            raise RuntimeError("storage backend error")

        cache_controller.draft_page_set_func = failing_func
        # Should not raise
        cache_controller._draft_page_set(["hash1"], [0, 1, 2])

    def test_draft_page_get_exception_swallowed(self, cache_controller):
        """Test _draft_page_get swallows exceptions (best-effort)."""
        def failing_func(hashes, indices):
            raise RuntimeError("storage backend error")

        cache_controller.draft_page_get_func = failing_func
        # Should not raise
        cache_controller._draft_page_get(["hash1"], [0, 1, 2])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
