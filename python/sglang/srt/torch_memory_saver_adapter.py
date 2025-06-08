import logging
from abc import ABC
from contextlib import contextmanager

try:
    import torch_memory_saver

    # Primary Memory Saver for Model Weight
    _primary_memory_saver = torch_memory_saver.TorchMemorySaver()

    # Secondary Memory Saver for KV Cache
    _secondary_memory_saver = torch_memory_saver.TorchMemorySaver()

    # Pre-created adapter instances for reuse
    _primary_adapter = None
    _secondary_adapter = None
    _noop_adapter = None

    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


@contextmanager
def configure_subprocess(enable: bool):
    """Configure subprocess for torch memory saver. Call this once per process."""
    print("configure_subprocess 222!")

    if not enable:
        print("configure_subprocess 333!")
        logger.debug("torch memory saver is disabled, skipping configure_subprocess")
        yield
        return

    if import_error is not None:
        print("configure_subprocess 444!")
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
    def create(enable: bool, is_primary: bool = True):
        global _primary_adapter, _secondary_adapter, _noop_adapter

        if enable and import_error is not None:
            logger.warning(
                "enable_memory_saver is enabled, but "
                "torch-memory-saver is not installed. Please install it "
                "via `pip3 install torch-memory-saver`. "
            )
            raise import_error

        if enable:
            if is_primary:
                if _primary_adapter is None:
                    _primary_adapter = _TorchMemorySaverAdapterReal(is_primary=True)
                return _primary_adapter
            else:
                if _secondary_adapter is None:
                    _secondary_adapter = _TorchMemorySaverAdapterReal(is_primary=False)
                return _secondary_adapter
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
    def __init__(self, is_primary: bool = True):
        self.is_primary = is_primary

    def _get_memory_saver(self):
        """Get the appropriate memory saver instance based on is_primary flag."""
        return _primary_memory_saver if self.is_primary else _secondary_memory_saver

    def region(self):
        return self._get_memory_saver().region()

    def pause(self):
        if self.is_primary:
            print("pause primary memory saver")
        else:
            print("pause secondary memory saver")
        return self._get_memory_saver().pause()

    def resume(self):
        return self._get_memory_saver().resume()

    @property
    def enabled(self):
        return self._get_memory_saver().enabled


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    def __init__(self, is_primary: bool = True):
        self.is_primary = is_primary

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
