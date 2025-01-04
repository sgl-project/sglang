from abc import ABC
from contextlib import contextmanager

try:
    import torch_memory_saver

    _primary_memory_saver = torch_memory_saver.TorchMemorySaver()
except ImportError:
    pass


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool):
        return (
            _TorchMemorySaverAdapterReal() if enable else _TorchMemorySaverAdapterNoop()
        )

    def configure_subprocess(self):
        raise NotImplementedError

    def region(self):
        raise NotImplementedError

    def pause(self):
        raise NotImplementedError

    def resume(self):
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
