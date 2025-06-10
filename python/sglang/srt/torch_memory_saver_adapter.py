import logging
import threading
import time
from abc import ABC
from contextlib import contextmanager

try:
    import torch_memory_saver

    def _create_memory_saver_with_delay(instance_id):
        # Small delay to let CUDA settle between instance creations
        time.sleep(1 * instance_id)
        return torch_memory_saver.TorchMemorySaver()

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


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool, tag: str = "weights"):
        global _adapters, _noop_adapter

        if tag not in ["weights", "kv_cache"]:
            raise ValueError(f"Invalid tag '{tag}'. Must be 'weights' or 'kv_cache'")

        if enable and import_error is not None:
            logger.warning(
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
        return self._get_memory_saver().pause()

    def resume(self):
        return self._get_memory_saver().resume()

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
