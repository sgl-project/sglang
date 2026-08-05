"""
Mock triton module for platforms where triton is not available (e.g., macOS/MPS).

This module provides stub implementations of triton APIs so that modules which
import triton at the top level can be loaded without error.  The actual triton
kernels are never executed on these platforms – alternative backends (e.g. SDPA
for MPS) are used instead.

Usage – call ``install()`` **before** any ``import triton`` in the process:

    from sglang._triton_stub import install
    install()
"""

import sys
import types


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
        import importlib

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


def install() -> None:
    """Register a mock ``triton`` package in *sys.modules*.

    This is a no-op if a real ``triton`` is already importable.
    """
    if "triton" in sys.modules:
        return
    # Check whether a real triton exists before installing the stub.
    import importlib.util

    if importlib.util.find_spec("triton") is not None:
        return

    # Register the meta-path finder FIRST so that any ``import triton.X``
    # during the rest of install() (or later) is handled.
    sys.meta_path.insert(0, _TritonFinder())

    triton = _make_mock("triton")
    triton.__version__ = "3.0.0"
    triton.cdiv = _cdiv
    triton.next_power_of_2 = _next_power_of_2
    triton.Config = _Config

    # triton.language  (commonly imported as ``tl``)
    tl = _make_mock("triton.language")

    class _constexpr:
        """Stand-in for ``tl.constexpr`` – works as both annotation and value wrapper."""

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

    # triton.runtime.jit  (JITFunction used in isinstance checks)
    runtime = _make_mock("triton.runtime")
    triton.runtime = runtime
    jit_mod = _make_mock("triton.runtime.jit")

    class _JITFunction:
        """Dummy so ``isinstance(fn, triton.runtime.jit.JITFunction)`` works."""

        pass

    jit_mod.JITFunction = _JITFunction
    runtime.jit = jit_mod

    # triton.runtime.driver  (used by fla/utils.py)
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

    # triton.backends / triton.backends.compiler  (used by torch._inductor)
    backends = _make_mock("triton.backends")
    triton.backends = backends
    compiler = _make_mock("triton.backends.compiler")
    backends.compiler = compiler
