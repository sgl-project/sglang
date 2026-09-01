"""Install compatibility stubs needed by SGLang on Apple Silicon MPS.

``torch.mps`` lacks several APIs that ``torch.cuda`` provides (``Stream``,
``set_device``, ``get_device_properties``, …).  Rather than scattering
``hasattr`` / ``getattr`` guards throughout the codebase, we monkey-patch
``torch.mps`` once at startup so that generic device-agnostic code paths
just work. Triton is also stubbed because it is unavailable on macOS.
"""

from __future__ import annotations

import functools
import importlib
import platform
import sys
import types
from dataclasses import dataclass, field
from typing import Any


class _StubBase:
    """A base class that any mock attribute can safely be subclassed from.

    Used when external code does ``class Foo(triton.runtime.KernelInterface):``.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)


class _MockModule(types.ModuleType):
    """A module whose every attribute is itself a ``_MockModule``.

    When called (e.g. ``@triton.jit``), it acts as a pass-through decorator so
    that kernel *definitions* are syntactically valid even though they will never
    be compiled.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.__path__: list[str] = []  # make it look like a package
        self.__package__ = name
        self.__file__ = __file__
        self._children: dict[str, object] = {}
        # Set __spec__ so that importlib.util.find_spec() works on cached modules
        self.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)

    def __getattr__(self, name: str):
        """Handle attribute access by creating and returning a child _MockModule."""
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        full = f"{self.__name__}.{name}"
        if full in sys.modules:
            return sys.modules[full]
        # If the name looks like a class (CamelCase / uppercase), return a
        # stub class that can be used as a base class for inheritance.
        if name[0:1].isupper():
            stub_cls = type(name, (_StubBase,), {"__module__": self.__name__})
            self._children[name] = stub_cls
            return stub_cls
        child = _MockModule(full)
        sys.modules[full] = child
        self._children[name] = child
        return child

    def __call__(self, *args, **kwargs):
        # Direct decorator usage:  @triton.jit  (receives the function)
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        # Parameterised decorator: @triton.jit(...)  → returns a decorator
        def _decorator(fn):
            return fn

        return _decorator

    def __instancecheck__(self, instance):
        """Return False for all instance checks against the mock."""
        return False

    def __contains__(self, item):
        """Return False for all membership checks."""
        return False

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

    def __bool__(self):
        return False

    def __repr__(self):
        return f"<triton-stub {self.__name__!r}>"


def _cdiv(a: int, b: int) -> int:
    """Ceiling division – mirrors ``triton.cdiv``."""
    return -(a // -b)


def _next_power_of_2(n: int) -> int:
    """Mirrors ``triton.next_power_of_2``."""
    return 1 << (n - 1).bit_length() if n > 0 else 1


class _Config:
    """Minimal stand-in for ``triton.Config`` used in ``@triton.autotune``."""

    def __init__(self, kwargs=None, num_warps=4, num_stages=2, **extra):
        self.kwargs = kwargs or {}
        self.num_warps = num_warps
        self.num_stages = num_stages


class _TritonFinder:
    """A meta-path finder that intercepts all ``import triton.*`` statements.

    When Python encounters ``import triton.backends.compiler``, it walks the
    dotted path and tries to import each component.  Our mock module's
    ``__getattr__`` handles *attribute* access, but the import machinery uses
    ``importlib`` finders, not attribute access, for sub-module resolution.
    This finder bridges that gap by creating ``_MockModule`` instances for any
    ``triton.*`` sub-module that isn't already in ``sys.modules``.
    """

    def find_spec(self, fullname, path=None, target=None):
        """PEP 451 meta-path finder for ``triton.*`` sub-modules."""
        if fullname == "triton" or fullname.startswith("triton."):
            if fullname in sys.modules:
                return getattr(sys.modules[fullname], "__spec__", None)
            # Create and register the mock so the import machinery finds it
            mod = _MockModule(fullname)
            sys.modules[fullname] = mod
            parts = fullname.rsplit(".", 1)
            if len(parts) == 2:
                parent_name, child_name = parts
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    setattr(parent, child_name, mod)
            return mod.__spec__
        return None


def _make_mock(name: str) -> _MockModule:
    """Create a ``_MockModule`` and register it in ``sys.modules``."""
    mod = _MockModule(name)
    sys.modules[name] = mod
    return mod


class Stream:
    """Minimal stand-in for ``torch.cuda.Stream``.

    MPS does not expose user-visible streams.  Every method is a no-op so
    that code written for CUDA's multi-stream model still runs.
    """

    def __init__(self, device: Any = None, priority: int = 0) -> None:
        pass

    def synchronize(self) -> None:
        pass

    def wait_stream(self, stream: Any) -> None:
        pass

    def wait_event(self, event: Any) -> None:
        pass

    def record_event(self, event: Any = None) -> Any:
        return None

    def query(self) -> bool:
        return True

    # context-manager protocol (``with stream:``)
    def __enter__(self) -> Stream:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class Event:
    """Minimal stand-in for ``torch.cuda.Event``."""

    def __init__(self, enable_timing: bool = False) -> None:
        pass

    def record(self, stream: Any = None) -> None:
        pass

    def wait(self, stream: Any = None) -> None:
        pass

    def query(self) -> bool:
        return True

    def synchronize(self) -> None:
        pass

    def elapsed_time(self, end_event: Any) -> float:
        return 0.0


class StreamContext:
    """Minimal stand-in for ``torch.cuda.StreamContext``."""

    def __init__(self, stream: Any = None) -> None:
        pass

    def __enter__(self) -> StreamContext:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


_default_stream = Stream()


def current_stream(device: Any = None) -> Stream:
    """Return the default (and only) MPS stream."""
    return _default_stream


def stream(s: Any) -> Stream:
    """Return a context manager that is a no-op on MPS."""
    return s if s is not None else _default_stream


def set_device(device: Any) -> None:  # noqa: ARG001
    """Set the current device. This is a no-op for MPS as it has exactly one device."""
    pass


def current_device() -> int:
    """Return the index of the current MPS device (always 0)."""
    return 0


def device_count() -> int:
    """Return the number of available MPS devices (always 1)."""
    return 1


@dataclass
class _MPSDeviceProperties:
    """Mimics the object returned by ``torch.cuda.get_device_properties``."""

    name: str = "Apple MPS"
    total_memory: int = 0  # populated at install time
    multi_processor_count: int = 0
    warp_size: int = 32
    is_integrated: bool = True
    major: int = 0
    minor: int = 0
    # Extra attrs some callers inspect
    _extra: dict = field(default_factory=dict)

    def __getattr__(self, name: str) -> Any:
        # Return a safe default for any attribute we didn't anticipate
        try:
            return self._extra[name]
        except KeyError:
            return None


_cached_props: _MPSDeviceProperties | None = None


def get_device_properties(device: Any = 0) -> _MPSDeviceProperties:  # noqa: ARG001
    """Return the properties of the MPS device. Results are cached after first call."""
    global _cached_props
    if _cached_props is None:
        import psutil

        _cached_props = _MPSDeviceProperties(
            total_memory=psutil.virtual_memory().total,
        )
    return _cached_props


class _MPSMemoryTracker:
    """Tracks peak memory values on top of ``torch.mps`` current-value APIs.

    * ``memory_allocated`` → ``torch.mps.current_allocated_memory()``
    * ``memory_reserved``  → ``torch.mps.driver_allocated_memory()``
    * ``max_memory_*``     → high-water marks of the above
    """

    def __init__(self) -> None:
        self._peak_allocated: int = 0
        self._peak_reserved: int = 0

    def memory_allocated(self, device: Any = None) -> int:  # noqa: ARG002
        import torch

        val = torch.mps.current_allocated_memory()
        if val > self._peak_allocated:
            self._peak_allocated = val
        return val

    def memory_reserved(self, device: Any = None) -> int:  # noqa: ARG002
        import torch

        val = torch.mps.driver_allocated_memory()
        if val > self._peak_reserved:
            self._peak_reserved = val
        return val

    def max_memory_allocated(self, device: Any = None) -> int:  # noqa: ARG002
        self.memory_allocated()
        return self._peak_allocated

    def max_memory_reserved(self, device: Any = None) -> int:  # noqa: ARG002
        self.memory_reserved()
        return self._peak_reserved

    def reset_peak_memory_stats(self, device: Any = None) -> None:  # noqa: ARG002
        import torch

        self._peak_allocated = torch.mps.current_allocated_memory()
        self._peak_reserved = torch.mps.driver_allocated_memory()


_memory_tracker = _MPSMemoryTracker()


def _patch_non_blocking() -> None:
    """Force ``non_blocking=False`` for copies targeting the MPS device.

    Unlike CUDA, MPS does not guarantee that a subsequent kernel on the same
    "stream" will wait for an async host-to-device transfer to finish.  Reading
    the tensor before the transfer completes yields uninitialised (garbage)
    data.  Patching ``Tensor.to`` and ``Tensor.copy_`` centrally avoids having
    to sprinkle ``non_blocking=not is_mps()`` at every call-site.
    """
    import torch

    _original_to = torch.Tensor.to

    @functools.wraps(_original_to)
    def _patched_to(self, *args, **kwargs):
        if kwargs.get("non_blocking"):
            # Detect target device from positional or keyword args
            device = None
            if args and isinstance(args[0], (str, torch.device)):
                device = torch.device(args[0]) if isinstance(args[0], str) else args[0]
            elif "device" in kwargs:
                d = kwargs["device"]
                device = torch.device(d) if isinstance(d, str) else d
            if device is not None and device.type == "mps":
                kwargs = {**kwargs, "non_blocking": False}
        return _original_to(self, *args, **kwargs)

    torch.Tensor.to = _patched_to

    _original_copy_ = torch.Tensor.copy_

    @functools.wraps(_original_copy_)
    def _patched_copy_(self, src, non_blocking=False):
        if non_blocking and self.device.type == "mps":
            non_blocking = False
        return _original_copy_(self, src, non_blocking=non_blocking)

    torch.Tensor.copy_ = _patched_copy_


_platform_stubs_installed = False


def install_platform_stubs() -> None:
    """Install the Triton and MPS compatibility stubs when they are needed."""
    global _platform_stubs_installed
    if _platform_stubs_installed:
        return

    if sys.platform != "darwin" or platform.machine() != "arm64":
        return

    try:
        import torch
    except ImportError:
        return

    if not torch.backends.mps.is_available():
        return

    if "triton" not in sys.modules and importlib.util.find_spec("triton") is None:
        # Register the meta-path finder first so later ``import triton.X``
        # statements are handled by the stub.
        sys.meta_path.insert(0, _TritonFinder())

        triton = _make_mock("triton")
        triton.__version__ = "3.0.0"
        triton.cdiv = _cdiv
        triton.next_power_of_2 = _next_power_of_2
        triton.Config = _Config

        # triton.language (commonly imported as ``tl``)
        tl = _make_mock("triton.language")

        class _constexpr:
            """Stand-in for ``tl.constexpr`` as an annotation and value wrapper."""

            def __init__(self, value=None):
                self.value = value

            def __repr__(self):
                return f"constexpr({self.value!r})"

        tl.constexpr = _constexpr
        triton.language = tl

        # triton.language.extra.libdevice
        extra = _make_mock("triton.language.extra")
        tl.extra = extra
        libdevice = _make_mock("triton.language.extra.libdevice")
        extra.libdevice = libdevice

        # triton.runtime.jit (JITFunction is used in isinstance checks)
        runtime = _make_mock("triton.runtime")
        triton.runtime = runtime
        jit_mod = _make_mock("triton.runtime.jit")

        class _JITFunction:
            """Dummy type for ``triton.runtime.jit.JITFunction`` checks."""

            pass

        class _KernelInterface(_StubBase):
            pass

        jit_mod.JITFunction = _JITFunction
        jit_mod.KernelInterface = _KernelInterface
        runtime.jit = jit_mod

        # Torch 2.13 imports these as classes while initializing Inductor, even on
        # MPS where no Triton kernel is compiled. Define them explicitly so the
        # catch-all meta-path finder does not materialize class names as modules.
        autotuner = _make_mock("triton.runtime.autotuner")

        class _OutOfResources(Exception):
            pass

        class _PTXASError(Exception):
            pass

        autotuner.OutOfResources = _OutOfResources
        autotuner.PTXASError = _PTXASError
        runtime.autotuner = autotuner

        compiler_root = _make_mock("triton.compiler")

        class _CompiledKernel(_StubBase):
            pass

        compiler_root.CompiledKernel = _CompiledKernel
        compiler_impl = _make_mock("triton.compiler.compiler")

        class _ASTSource(_StubBase):
            pass

        compiler_impl.ASTSource = _ASTSource
        compiler_impl.triton_key = lambda: "triton-stub"
        compiler_root.compiler = compiler_impl
        triton.compiler = compiler_root

        # triton.runtime.driver
        driver = _make_mock("triton.runtime.driver")
        runtime.driver = driver

        # triton.testing
        testing = _make_mock("triton.testing")
        triton.testing = testing

        # triton.tools / triton.tools.tensor_descriptor
        tools = _make_mock("triton.tools")
        triton.tools = tools
        td = _make_mock("triton.tools.tensor_descriptor")
        tools.tensor_descriptor = td

        # triton.backends / triton.backends.compiler
        backends = _make_mock("triton.backends")
        triton.backends = backends
        compiler = _make_mock("triton.backends.compiler")

        class _GPUTarget(_StubBase):
            pass

        compiler.GPUTarget = _GPUTarget
        backends.compiler = compiler

    mps = torch.mps
    # Only patch attributes that are actually missing
    for name, obj in [
        ("Stream", Stream),
        ("StreamContext", StreamContext),
        ("Event", Event),
        ("current_stream", current_stream),
        ("stream", stream),
        ("set_device", set_device),
        ("current_device", current_device),
        ("device_count", device_count),
        ("get_device_properties", get_device_properties),
        ("reset_peak_memory_stats", _memory_tracker.reset_peak_memory_stats),
        ("memory_allocated", _memory_tracker.memory_allocated),
        ("memory_reserved", _memory_tracker.memory_reserved),
        ("max_memory_allocated", _memory_tracker.max_memory_allocated),
        ("max_memory_reserved", _memory_tracker.max_memory_reserved),
    ]:
        if not hasattr(mps, name):
            setattr(mps, name, obj)

    _patch_non_blocking()

    _platform_stubs_installed = True
