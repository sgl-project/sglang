# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""
Platform Interface - The Foundation of Multi-Vendor Support
============================================================

This module defines the Platform base class, the core abstraction that enables
SGLang to run on multiple GPU vendors (NVIDIA, AMD, Intel, Huawei, Moore Threads).

This is the foundation for the RFC "Unifying Multi-Vendor Support via a Platform
Interface in sglang" (https://github.com/sgl-project/sglang/issues/15299).

KEY CONCEPTS:
-------------

1. **Platform Class** (Base class for all vendors)
   - Each GPU vendor has a Platform subclass (CudaPlatform, RocmPlatform, etc.)
   - Provides: device info, op registry, server arg post-processing, modules

2. **Op Registry** (Maps op names to platform-specific kernels)
   - Supports lazy imports via OpSpec (module path + attribute name)
   - Supports out-of-class registration via register_op()
   - Ops are looked up via get_op_by_name()

3. **PlatformModules** (Lazy access to vendor-specific modules)
   - Allows accessing vendor modules (torch_npu, sgl_kernel) without if-else
   - Example: current_platform.modules.npu_swiglu(x)

4. **Server Arg Post-processing** (Platform-specific defaults)
   - Each platform can override postprocess_server_args() to set defaults

LAZY IMPORT SUPPORT:
--------------------
Ops can be registered with lazy imports to reduce startup time:

    # Instead of importing at registration time:
    from sgl_kernel import silu_and_mul
    register_op(PlatformEnum.CUDA, "silu_and_mul", silu_and_mul)

    # Use OpSpec for lazy import (imported only when op is first called):
    register_op(PlatformEnum.CUDA, "silu_and_mul",
                OpSpec("sgl_kernel", "silu_and_mul"))

OUT-OF-CLASS REGISTRATION:
--------------------------
Ops can be registered from anywhere, not just in Platform subclasses:

    # In a separate file (e.g., my_custom_ops.py):
    from sglang.srt.platforms.interface import register_op, OpSpec, PlatformEnum

    # Register a custom op for CUDA
    register_op(PlatformEnum.CUDA, "my_custom_op",
                OpSpec("my_kernel_lib", "my_custom_op"))

    # Or register with a callable directly
    register_op(PlatformEnum.CUDA, "my_custom_op", my_custom_op_fn)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from sglang.srt.ops.base import OpProxy
    from sglang.srt.server_args import ServerArgs


# =============================================================================
# Op Specification for Lazy Imports
# =============================================================================


@dataclass(frozen=True, slots=True)
class OpSpec:
    """Specification for a lazily-imported operation.

    Instead of importing a kernel at registration time, OpSpec stores the
    import path and resolves it only when the op is first accessed.

    This provides:
    - Faster startup (no kernel imports until needed)
    - No import errors if the kernel library isn't installed
    - Ability to register ops before their dependencies are available

    Example:
        # Lazy import: sgl_kernel.silu_and_mul will be imported on first use
        spec = OpSpec("sgl_kernel", "silu_and_mul")

        # Nested attribute access (e.g., torch_npu.npu_swiglu)
        spec = OpSpec("torch_npu", "npu_swiglu")

    Attributes:
        module: The module path to import (e.g., "sgl_kernel", "torch_npu")
        attr: The attribute name to get from the module (e.g., "silu_and_mul")
    """

    module: str
    attr: str

    def resolve(self) -> Callable:
        """Import the module and return the callable.

        Returns:
            The resolved callable from the module.

        Raises:
            ImportError: If the module cannot be imported.
            AttributeError: If the attribute doesn't exist in the module.
        """
        mod = import_module(self.module)
        return getattr(mod, self.attr)


# =============================================================================
# Global Op Registry
# =============================================================================
# Maps (PlatformEnum, op_name) -> OpSpec | Callable
# This allows registering ops from anywhere, not just in Platform subclasses.

_global_op_registry: dict[tuple["PlatformEnum", str], OpSpec | Callable] = {}

# Cache for resolved ops (OpSpec -> Callable)
_resolved_ops_cache: dict[tuple["PlatformEnum", str], Callable] = {}


def register_op(
    platform: "PlatformEnum",
    op_name: str,
    impl: OpSpec | Callable,
) -> None:
    """Register an operation for a specific platform.

    This function allows registering ops from anywhere in the codebase,
    not just inside Platform subclass files. This is similar to PyTorch's
    dispatch mechanism where ops can be registered externally.

    Args:
        platform: The platform to register the op for (e.g., PlatformEnum.CUDA)
        op_name: The canonical name of the operation (e.g., "silu_and_mul")
        impl: Either an OpSpec for lazy import, or a callable directly

    Example:
        # Register with lazy import (recommended for startup performance)
        register_op(PlatformEnum.CUDA, "silu_and_mul",
                    OpSpec("sgl_kernel", "silu_and_mul"))

        # Register with a callable directly
        register_op(PlatformEnum.CUDA, "my_op", my_op_function)

        # Register from a separate file (out-of-class registration)
        # In my_custom_kernels.py:
        from sglang.srt.platforms.interface import register_op, OpSpec, PlatformEnum
        register_op(PlatformEnum.CUDA, "custom_attention",
                    OpSpec("my_attention_lib", "flash_attention"))
    """
    key = (platform, op_name)
    _global_op_registry[key] = impl
    # Clear cache if re-registering
    if key in _resolved_ops_cache:
        del _resolved_ops_cache[key]


def get_registered_op(platform: "PlatformEnum", op_name: str) -> Callable | None:
    """Get a registered operation, resolving lazy imports if needed.

    This function handles the lazy import resolution: if an op was registered
    with an OpSpec, it will be imported on first access and cached.

    Args:
        platform: The platform to look up the op for
        op_name: The canonical name of the operation

    Returns:
        The callable implementation, or None if not registered.
    """
    key = (platform, op_name)

    # Check resolved cache first
    if key in _resolved_ops_cache:
        return _resolved_ops_cache[key]

    # Check global registry
    if key not in _global_op_registry:
        return None

    impl = _global_op_registry[key]

    # Resolve OpSpec if needed
    if isinstance(impl, OpSpec):
        try:
            resolved = impl.resolve()
            _resolved_ops_cache[key] = resolved
            return resolved
        except (ImportError, AttributeError):
            # Import failed - return None (will fall back to native impl)
            return None
    else:
        # Already a callable
        _resolved_ops_cache[key] = impl
        return impl


def list_registered_ops(platform: "PlatformEnum" | None = None) -> list[str]:
    """List all registered ops, optionally filtered by platform.

    Args:
        platform: If provided, only list ops for this platform.
                  If None, list all ops across all platforms.

    Returns:
        List of op names.
    """
    if platform is None:
        return list(set(op_name for _, op_name in _global_op_registry.keys()))
    return [op_name for (p, op_name) in _global_op_registry.keys() if p == platform]


# =============================================================================
# Platform Enum
# =============================================================================


class PlatformEnum(Enum):
    """Enumeration of supported platforms.

    Each platform represents a distinct GPU/accelerator vendor or fallback.
    The enum is used for platform property checks (is_cuda, is_rocm, etc.).
    """

    CUDA = auto()  # NVIDIA CUDA GPUs
    ROCM = auto()  # AMD ROCm/HIP GPUs
    MUSA = auto()  # Moore Threads MUSA GPUs
    NPU = auto()  # Huawei Ascend NPUs
    XPU = auto()  # Intel XPU accelerators
    HPU = auto()  # Intel Habana Gaudi HPUs
    CPU = auto()  # CPU fallback (no accelerator)


class PlatformModules:
    """
    Lazy-loading container for platform-specific modules and functions.

    PROBLEM:
    --------
    Different platforms have vendor-specific modules (torch_npu, sgl_kernel, etc.)
    that provide specialized functions. Without this class, you'd need:

        # OLD - if-else at every call site
        if current_platform.is_npu:
            import torch_npu
            result = torch_npu.npu_swiglu(x)
        elif current_platform.is_hip:
            from sgl_kernel import gelu_quick
            gelu_quick(x, out)
        # ... more platforms ...

    SOLUTION:
    ---------
    PlatformModules provides a unified interface:

        # NEW - no if-else needed, works on any platform
        result = current_platform.modules.npu_swiglu(x)  # Only on NPU
        current_platform.modules.gelu_quick(x, out)       # Only on HIP

    If you call a function on the wrong platform, you get a clear error:
        "Platform 'cuda' does not provide 'npu_swiglu'. This operation may not
         be supported on this platform."

    HOW IT WORKS:
    -------------
    1. Each Platform subclass implements _get_module_attr(name)
    2. PlatformModules.__getattr__ calls _get_module_attr on first access
    3. Results are cached for performance
    """

    def __init__(self, platform: "Platform"):
        self._platform = platform
        self._cache: dict[str, Any] = {}

    def __getattr__(self, name: str) -> Any:
        """Lazy load and cache module attributes.

        This uses Python's __getattr__ magic: only called when the attribute
        is not found via normal lookup. This allows us to dynamically resolve
        platform-specific functions.
        """
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        if name not in self._cache:
            value = self._platform._get_module_attr(name)
            if value is None:
                raise AttributeError(
                    f"Platform '{self._platform.device_name}' does not provide '{name}'. "
                    f"This operation may not be supported on this platform."
                )
            self._cache[name] = value

        return self._cache[name]

    def has(self, name: str) -> bool:
        """Check if a module attribute is available without raising.

        Use this before calling a platform-specific function to avoid errors:
            if current_platform.modules.has("gelu_quick"):
                current_platform.modules.gelu_quick(x, out)
        """
        try:
            self._platform._get_module_attr(name)
            return True
        except (ImportError, AttributeError):
            return False


# =============================================================================
# Platform Base Class
# =============================================================================


class Platform(ABC):
    """
    Base class for all platform implementations.

    Each GPU vendor has a Platform subclass that provides:
    - Device information (device_name, device_type)
    - Op registry (_ops, get_op, get_op_by_name)
    - Module access (_get_module_attr for PlatformModules)
    - Server arg defaults (postprocess_server_args)

    IMPLEMENTING A NEW PLATFORM:
    ----------------------------
    See platforms/cuda.py for a complete example. Key steps:

    1. Create a new file: platforms/myplatform.py
    2. Subclass Platform and set class attributes:
       - _enum = PlatformEnum.MYPLATFORM
       - device_name = "mydevice"
       - device_type = "mydevice"

    3. Implement _init_ops() to register kernels:
       @classmethod
       def _init_ops(cls) -> dict[str, Callable]:
           from my_kernel_lib import my_silu_and_mul
           return {"silu_and_mul": my_silu_and_mul}

    4. Implement get_op() for op lookup:
       def get_op(self, op: "OpProxy") -> Callable | None:
           if MyPlatform._ops is None:
               MyPlatform._ops = self._init_ops()
           return MyPlatform._ops.get(op.name)

    5. (Optional) Implement _get_module_attr() for PlatformModules
    6. (Optional) Implement postprocess_server_args() for defaults
    7. Add detection logic in platforms/__init__.py _detect_platform()
    """

    _enum: PlatformEnum
    device_name: str  # Human-readable name, e.g., "cuda", "hip", "npu"
    device_type: str  # torch device type, e.g., "cuda" for both CUDA and ROCm

    # Op registry - maps op names to implementations (lazy-loaded)
    _ops: dict[str, Callable] | None = None

    # Modules container (lazy-initialized)
    _modules: "PlatformModules | None" = None

    # === Platform Modules ===

    @property
    def modules(self) -> "PlatformModules":
        """
        Access platform-specific modules and functions.

        This provides a unified way to access platform-specific functionality
        without if-else checks at the call site.

        Example:
            # Access torch_npu functions (only works on NPU platform)
            result = current_platform.modules.npu_swiglu(x)

            # Access sgl_kernel functions
            current_platform.modules.gelu_quick(x, out)
        """
        if self._modules is None:
            self._modules = PlatformModules(self)
        return self._modules

    def _get_module_attr(self, name: str) -> Any | None:
        """
        Get a platform-specific module attribute.

        Override this in subclasses to provide platform-specific modules.

        Args:
            name: The attribute name to look up.

        Returns:
            The attribute value, or None if not available.
        """
        return None

    # === Platform Properties ===

    @property
    def is_cuda(self) -> bool:
        return self._enum == PlatformEnum.CUDA

    @property
    def is_rocm(self) -> bool:
        return self._enum == PlatformEnum.ROCM

    @property
    def is_hip(self) -> bool:
        """Alias for is_rocm for compatibility."""
        return self._enum == PlatformEnum.ROCM

    @property
    def is_musa(self) -> bool:
        return self._enum == PlatformEnum.MUSA

    @property
    def is_npu(self) -> bool:
        return self._enum == PlatformEnum.NPU

    @property
    def is_xpu(self) -> bool:
        return self._enum == PlatformEnum.XPU

    @property
    def is_hpu(self) -> bool:
        return self._enum == PlatformEnum.HPU

    @property
    def is_cpu(self) -> bool:
        return self._enum == PlatformEnum.CPU

    # === Platform Capabilities ===

    @property
    def has_amx(self) -> bool:
        """Check if this platform has AMX (Advanced Matrix Extensions) support.

        Only applicable for CPU platforms. Other platforms return False.

        Returns:
            True if AMX is supported, False otherwise.
        """
        return False

    # === Op Registry ===
    #
    # The op registry supports two sources:
    # 1. Class-level _ops dict (populated by _init_ops() in subclasses)
    # 2. Global registry (populated by register_op() from anywhere)
    #
    # The global registry is checked first, allowing external registration
    # to override class-level ops if needed.

    @abstractmethod
    def get_op(self, op: "OpProxy") -> Callable | None:
        """Get the platform-specific implementation of an operation.

        Args:
            op: The OpProxy object representing the operation.

        Returns:
            The callable implementation, or None if not available.
        """
        ...

    def get_op_by_name(self, name: str) -> Callable | None:
        """Get the platform-specific implementation of an operation by name.

        This method checks two sources in order:
        1. Global registry (register_op() - supports lazy imports)
        2. Class-level _ops dict (from _init_ops())

        The global registry takes precedence, allowing external registration
        to override class-level ops.

        Args:
            name: The canonical name of the operation.

        Returns:
            The callable implementation, or None if not available.
        """
        # First, check global registry (supports lazy imports and external registration)
        global_impl = get_registered_op(self._enum, name)
        if global_impl is not None:
            return global_impl

        # Fall back to class-level _ops dict
        if self._ops is not None:
            return self._ops.get(name)

        return None

    def has_op(self, op: "OpProxy") -> bool:
        """Check if an operation is available on this platform."""
        return self.get_op(op) is not None

    def has_op_by_name(self, name: str) -> bool:
        """Check if an operation is available by name."""
        return self.get_op_by_name(name) is not None

    def list_available_ops(self) -> list[str]:
        """List all available ops on this platform.

        Combines ops from both the global registry and class-level _ops.
        """
        ops = set()
        # Add ops from global registry
        ops.update(list_registered_ops(self._enum))
        # Add ops from class-level _ops
        if self._ops is not None:
            ops.update(self._ops.keys())
        return list(ops)

    # === Server Argument Post-processing ===

    def postprocess_server_args(self, args: "ServerArgs") -> None:
        """Post-process server arguments for this platform.

        This method is called during ServerArgs initialization to apply
        platform-specific defaults and adjustments. Override this in
        platform subclasses to set platform-specific backend defaults,
        enable/disable features, or apply other configuration changes.

        Args:
            args: The ServerArgs instance being initialized. The method
                should modify this object in-place.

        Example:
            class HpuPlatform(Platform):
                def postprocess_server_args(self, args: ServerArgs) -> None:
                    args.attention_backend = "torch_native"
                    args.sampling_backend = "pytorch"
        """
        # Default implementation does nothing.
        # Platform subclasses should override this to apply platform-specific settings.
        pass
