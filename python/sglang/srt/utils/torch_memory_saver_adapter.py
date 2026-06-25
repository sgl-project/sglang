import logging
from abc import ABC
from contextlib import contextmanager

from sglang.srt.utils.common import is_xpu

try:
    import torch_memory_saver

    # Intel XPU requires hook_mode="torch" (in-process pluggable allocator);
    # the LD_PRELOAD-based preload mode is CUDA/HIP-only. Set it before the
    # singleton is initialized on first use.
    if is_xpu():
        torch_memory_saver.torch_memory_saver.hook_mode = "torch"

    _memory_saver = torch_memory_saver.torch_memory_saver
    import_error = None
except ImportError as e:
    import_error = e
    pass

logger = logging.getLogger(__name__)


class TorchMemorySaverAdapter(ABC):
    @staticmethod
    def create(enable: bool):
        if enable and import_error is not None:
            if is_xpu():
                # XPU ships no prebuilt wheel; it is built from source against the
                # local oneAPI + torch-XPU runtime. TMS_PLATFORM=xpu forces the XPU
                # backend; --no-build-isolation lets the build see torch and match
                # the libsycl ABI to it.
                logger.warning(
                    "enable_memory_saver is enabled, but torch-memory-saver is "
                    "not installed. On Intel XPU, build it from source with Intel "
                    "oneAPI on PATH: `TMS_PLATFORM=xpu pip3 install "
                    "--no-build-isolation "
                    "git+https://github.com/fzyzcjy/torch_memory_saver.git`."
                )
            else:
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

    def region(self, tag: str, enable_cpu_backup: bool = False):
        raise NotImplementedError

    def cuda_graph(self, **kwargs):
        raise NotImplementedError

    def disable(self):
        raise NotImplementedError

    def pause(self, tag: str):
        raise NotImplementedError

    def resume(self, tag: str):
        raise NotImplementedError

    @property
    def enabled(self):
        raise NotImplementedError


class _TorchMemorySaverAdapterReal(TorchMemorySaverAdapter):
    """Adapter for TorchMemorySaver with tag-based control.

    Backed by the upstream torch_memory_saver package (CUDA VMM, and Intel XPU
    via Level Zero). On XPU the package uses an in-process pluggable allocator
    (hook_mode="torch") rather than the CUDA LD_PRELOAD path, so
    configure_subprocess() (nothing to preload) and cuda_graph() (no pauseable
    graph-capture path) are no-ops there.
    """

    def configure_subprocess(self):
        if is_xpu():
            # XPU uses an in-process pluggable allocator; nothing to preload.
            return self._noop_context()
        return torch_memory_saver.configure_subprocess()

    def region(self, tag: str, enable_cpu_backup: bool = False):
        return _memory_saver.region(tag=tag, enable_cpu_backup=enable_cpu_backup)

    def cuda_graph(self, **kwargs):
        if is_xpu():
            # XPU does not support the memory-saver pauseable graph-capture path.
            return self._noop_context()
        return _memory_saver.cuda_graph(**kwargs)

    @contextmanager
    def _noop_context(self, **kwargs):
        yield

    def disable(self):
        return _memory_saver.disable()

    def pause(self, tag: str):
        return _memory_saver.pause(tag=tag)

    def resume(self, tag: str):
        return _memory_saver.resume(tag=tag)

    @property
    def enabled(self):
        return _memory_saver is not None and _memory_saver.enabled


class _TorchMemorySaverAdapterNoop(TorchMemorySaverAdapter):
    @contextmanager
    def configure_subprocess(self):
        yield

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):
        yield

    @contextmanager
    def cuda_graph(self, **kwargs):
        yield

    @contextmanager
    def disable(self):
        yield

    def pause(self, tag: str):
        pass

    def resume(self, tag: str):
        pass

    @property
    def enabled(self):
        return False
