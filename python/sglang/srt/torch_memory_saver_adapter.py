import logging
from abc import ABC
from contextlib import contextmanager

try:
    import torch_memory_saver

    _primary_memory_saver = torch_memory_saver.TorchMemorySaver()
    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool):
        if enable and import_error is not None:
            logger.warning(
                "enable_memory_saver is enabled, but "
                "torch-memory-saver is not installed. Please install it "
                "via `pip3 install torch-memory-saver`. "
            )
            raise import_error
        return (
            _TorchMemorySaverAdapterReal() if enable else _TorchMemorySaverAdapterNoop()
        )

    def check_validity(self, caller_name):
        if not self.enabled:
            logger.warning(
                f"`{caller_name}` will not save memory because torch_memory_saver is not enabled. "
                f"Potential causes: `enable_memory_saver` is false, or torch_memory_saver has installation issues."
            )

    def configure_subprocess(self):
        raise NotImplementedError

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
    def configure_subprocess(self):
        return torch_memory_saver.configure_subprocess()

    def region(self):
        return _primary_memory_saver.region()

    def pause(self):
        return _primary_memory_saver.pause()

    def resume(self):
        return _primary_memory_saver.resume()

    @property
    def enabled(self):
        return _primary_memory_saver.enabled


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

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
