import logging
import threading
import time
from abc import ABC
from contextlib import contextmanager

try:
    import torch_memory_saver

    # Global lock to coordinate memory operations between instances
    _memory_operation_lock = threading.Lock()

    # Create memory savers with staggered timing to avoid address conflicts
    def _create_memory_saver_with_delay(instance_id):
        """Create memory saver with delay to avoid address space conflicts."""
        # Small delay to let CUDA settle between instance creations
        time.sleep(0.01 * instance_id)
        return torch_memory_saver.TorchMemorySaver()

    # Memory savers organized by purpose
    _memory_savers = {
        "weights": _create_memory_saver_with_delay(0),
        "kv_cache": _create_memory_saver_with_delay(1),
    }

    # Pre-created adapter instances for reuse
    _adapters = {}
    _noop_adapter = None

    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


@contextmanager
def configure_subprocess(enable: bool):
    """Configure subprocess for torch memory saver. Call this once per process."""
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

    # Use the real torch_memory_saver configure_subprocess context manager
    with torch_memory_saver.configure_subprocess():
        yield


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool, tag: str = "weights"):
        """Create or retrieve adapter for specific memory type.

        This method implements a singleton pattern where each tag corresponds to a unique
        memory saver instance. Multiple calls with the same tag will return the same adapter
        instance, ensuring consistent memory management within a single SGLang process.

        Memory saver instances are created with staggered timing to prevent CUDA virtual
        address space conflicts that can occur when multiple instances are initialized
        simultaneously.

        Args:
            enable: Whether to enable memory saver functionality
            tag: Memory type identifier, must be either "weights" or "kv_cache"

        Returns:
            TorchMemorySaverAdapter: Adapter instance for the specified memory type

        Raises:
            ValueError: If tag is not "weights" or "kv_cache"
            ImportError: If enable=True but torch-memory-saver is not installed
        """
        global _adapters, _noop_adapter

        if tag not in ["weights", "kv_cache"]:
            raise ValueError(f"Invalid tag '{tag}'. Must be 'weights' or 'kv_cache'")

        if enable and import_error is not None:
            logger.warning(
                "enable_memory_saver is enabled, but "
                "torch-memory-saver is not installed. Please install it "
                "via `pip3 install torch-memory-saver`. "
            )
            raise import_error

        if enable:
            if tag not in _adapters:
                _adapters[tag] = _TorchMemorySaverAdapterReal(tag=tag)
            return _adapters[tag]
        else:
            if _noop_adapter is None:
                _noop_adapter = _TorchMemorySaverAdapterNoop()
            return _noop_adapter

    def check_validity(self, caller_name):
        if not self.enabled:
            logger.warning(
                f"`{caller_name}` will not save memory because torch_memory_saver is not enabled. "
                f"Potential causes: `enable_memory_saver` is false, or torch_memory_saver has installation issues."
            )

    def region(self):
        raise NotImplementedError

    def pause(self):
        raise NotImplementedError

    def resume(self):
        raise NotImplementedError

    @property
    def enabled(self):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    def __init__(self, tag: str):
        self.tag = tag

    def _get_memory_saver(self):
        """Get the appropriate memory saver instance based on tag."""
        return _memory_savers[self.tag]

    def region(self):
        return self._get_memory_saver().region()

    def pause(self):
        print(f"pause {self.tag} memory saver")

        with _memory_operation_lock:
            return self._get_memory_saver().pause()

    def resume(self):
        print(f"resume {self.tag} memory saver")

        # Serialize resume operations and add retry with backoff
        with _memory_operation_lock:
            max_retries = 5
            base_delay = 0.01

            for attempt in range(max_retries):
                try:
                    # Add small delay between different instances
                    if self.tag == "kv_cache":
                        time.sleep(0.05)  # kv_cache waits a bit longer

                    result = self._get_memory_saver().resume()
                    print(f"resume {self.tag} memory saver succeeded")
                    return result

                except Exception as e:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    print(
                        f"resume {self.tag} memory saver attempt {attempt + 1}/{max_retries} failed: {e}"
                    )

                    if attempt == max_retries - 1:
                        print(
                            f"resume {self.tag} memory saver failed after {max_retries} attempts"
                        )
                        raise e

                    print(f"retrying in {delay:.3f} seconds...")
                    time.sleep(delay)

    @property
    def enabled(self):
        return self._get_memory_saver().enabled


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    def __init__(self):
        pass

    @contextmanager
    def region(self):
        yield

    def pause(self):
        pass

    def resume(self):
        pass

    @property
    def enabled(self):
        return False
