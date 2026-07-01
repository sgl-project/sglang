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

        # Create a minimal controller without full initialization
        ctrl = HiCacheController.__new__(HiCacheController)
        ctrl.enable_storage = True
        ctrl.has_draft = False
        ctrl.storage_backend_type = "mooncake"
        return ctrl

    def test_maybe_register_draft_no_storage(self, cache_controller):
        """Test that registration is skipped when storage is disabled."""
        cache_controller.enable_storage = False
        cache_controller.has_draft = True

        # Should not raise and should not attempt registration
        cache_controller._maybe_register_draft_with_storage()
        assert not hasattr(cache_controller, 'draft_registered_in_storage')

    def test_maybe_register_draft_no_draft(self, cache_controller):
        """Test that registration is skipped when draft is not enabled."""
        cache_controller.enable_storage = True
        cache_controller.has_draft = False

        # Should not raise and should not attempt registration
        cache_controller._maybe_register_draft_with_storage()
        assert not hasattr(cache_controller, 'draft_registered_in_storage')

    def test_maybe_register_draft_no_v2_support(self, cache_controller):
        """Test graceful handling when backend doesn't support v2 registration."""
        cache_controller.enable_storage = True
        cache_controller.has_draft = True

        # Mock storage backend without v2 support
        mock_backend = Mock()
        del mock_backend.register_mem_host_pool_v2
        cache_controller.storage_backend = mock_backend

        # Should not raise
        cache_controller._maybe_register_draft_with_storage()
        assert cache_controller.draft_registered_in_storage == False

    def test_maybe_register_draft_success(self, cache_controller):
        """Test successful draft pool registration."""
        cache_controller.enable_storage = True
        cache_controller.has_draft = True

        # Mock draft pool
        mock_draft_pool = Mock()
        mock_draft_pool.size = 1024
        mock_draft_pool.page_size = 64
        cache_controller.mem_pool_host_draft = mock_draft_pool

        # Mock storage backend with v2 support
        mock_backend = Mock()
        mock_backend.register_mem_host_pool_v2 = Mock()
        cache_controller.storage_backend = mock_backend

        # Should succeed and set flag
        cache_controller._maybe_register_draft_with_storage()

        # Verify registration was called
        mock_backend.register_mem_host_pool_v2.assert_called_once()
        assert cache_controller.draft_registered_in_storage == True

    def test_maybe_register_draft_registration_failure(self, cache_controller, caplog):
        """Test graceful handling of registration failure."""
        cache_controller.enable_storage = True
        cache_controller.has_draft = True

        # Mock draft pool
        mock_draft_pool = Mock()
        mock_draft_pool.size = 1024
        mock_draft_pool.page_size = 64
        cache_controller.mem_pool_host_draft = mock_draft_pool

        # Mock storage backend that raises an error
        mock_backend = Mock()
        mock_backend.register_mem_host_pool_v2 = Mock(
            side_effect=RuntimeError("Buffer outside shared memory segment")
        )
        cache_controller.storage_backend = mock_backend

        # Should not raise, should log warning
        with caplog.at_level(logging.WARNING):
            cache_controller._maybe_register_draft_with_storage()

        # Verify warning was logged
        assert any("Failed to register draft host pool" in record.message
                   for record in caplog.records)
        assert cache_controller.draft_registered_in_storage == False

    def test_maybe_register_draft_mmap_allocator(self, cache_controller, caplog):
        """Test the specific EAGLE + Mooncake standalone case."""
        cache_controller.enable_storage = True
        cache_controller.has_draft = True

        # Mock draft pool with mmap allocator (not MooncakeHostTensorAllocator)
        mock_draft_pool = Mock()
        mock_draft_pool.size = 1024
        mock_draft_pool.page_size = 64
        cache_controller.mem_pool_host_draft = mock_draft_pool

        # Mock storage backend that fails due to allocator mismatch
        mock_backend = Mock()
        mock_backend.register_mem_host_pool_v2 = Mock(
            side_effect=ValueError(
                "Buffer not in Mooncake shared memory segment: "
                "allocator is HostTensorAllocator, expected MooncakeHostTensorAllocator"
            )
        )
        cache_controller.storage_backend = mock_backend

        # Should handle gracefully
        with caplog.at_level(logging.WARNING):
            cache_controller._maybe_register_draft_with_storage()

        # Verify graceful degradation
        assert cache_controller.draft_registered_in_storage == False
        assert any("L3 (storage backend) operations will be skipped" in record.message
                   for record in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
