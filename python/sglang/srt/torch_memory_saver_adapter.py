import logging
import threading
import time
from abc import ABC
from contextlib import contextmanager

try:
    import torch_memory_saver

    # Use the global singleton instance
    _memory_saver = torch_memory_saver.memory_saver
    import_error = None
except ImportError as e:
    import_error = e
    _memory_saver = None

logger = logging.getLogger(__name__)


@contextmanager
def configure_subprocess(enable: bool):
    if not enable:
        logger.debug("torch memory saver is disabled, skipping configure_subprocess")
        yield
        return

    if import_error is not None:
        logger.warning(
            "torch-memory-saver is not installed. configure_subprocess will do nothing."
        )
        yield
        return

    with torch_memory_saver.configure_subprocess():
        yield


@contextmanager
def _noop_context():
    """Noop context manager"""
    yield


class TorchMemorySaverAdapter:
    """Adapter for TorchMemorySaver with tag-based control"""

    def __init__(self, enabled: bool = True):
        """
        Initialize adapter with enable/disable control

        Args:
            enabled: Whether to enable memory saving functionality
        """
        self._user_enabled = enabled

        if import_error is not None:
            logger.warning(
                "torch-memory-saver is not installed. Please install it "
                "via `pip3 install torch-memory-saver`. "
            )

        logger.info(f"TorchMemorySaver adapter initialized with enabled={enabled}")

    def region(self, tag: str):
        """Context manager for memory region with specified tag"""
        if self.enabled:
            # Use the real torch_memory_saver context manager
            logger.info(f"enter tms region for tag: {tag}")
            return _memory_saver.region(tag=tag)
        else:
            # Use noop context manager when disabled
            logger.debug(f"memory saver disabled, using noop context for tag: {tag}")
            return _noop_context()

    def pause(self, tag: str):
        """Pause memory for specific tag"""
        if self.enabled:
            logger.info(f"enter tms pause for tag: {tag}")
            _memory_saver.pause(tag=tag)
        else:
            logger.debug(f"memory saver disabled, noop pause for tag: {tag}")

    def resume(self, tag: str):
        """Resume memory for specific tag"""
        if self.enabled:
            logger.info(f"enter tms resume for tag: {tag}")
            _memory_saver.resume(tag=tag)
        else:
            logger.debug(f"memory saver disabled, noop resume for tag: {tag}")

    def pause_weights(self):
        """Convenience method to pause weights memory"""
        self.pause("weights")

    def resume_weights(self):
        """Convenience method to resume weights memory"""
        self.resume("weights")

    def pause_kv_cache(self):
        """Convenience method to pause kv_cache memory"""
        self.pause("kv_cache")

    def resume_kv_cache(self):
        """Convenience method to resume kv_cache memory"""
        self.resume("kv_cache")

    @property
    def enabled(self):
        """Check if memory saver is enabled (both user setting and library availability)"""
        return (
            self._user_enabled and _memory_saver is not None and _memory_saver.enabled
        )


# Factory function for backward compatibility
def torch_memory_saver_adapter(enabled: bool = True):
    """Create a TorchMemorySaverAdapter instance with specified enabled state"""
    return TorchMemorySaverAdapter(enabled=enabled)
