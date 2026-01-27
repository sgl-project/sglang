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
OpProxy Base Class - Alternative to Function-Based Ops
=======================================================

This module provides OpProxy, an alternative approach to defining platform-agnostic
operations. This is part of **RFC Feature #3: Platform-specific Op / Kernel Resolution**.

TWO APPROACHES TO DEFINING OPS:
-------------------------------

1. **Function-Based (Recommended)** - See ops/activation.py
   - Define regular Python functions with docstrings and type hints
   - Use _get_impl() for platform dispatch
   - Better IDE support (docstrings appear directly in hover)
   - Simpler to understand and maintain

2. **OpProxy-Based (Advanced)** - This module
   - Define ops as OpProxy instances
   - More structured, but docstrings don't appear in IDE hover
   - Useful when you need advanced features (is_available, etc.)

WHEN TO USE WHICH:
------------------

Use Function-Based (activation.py style) when:
- You want the best IDE experience (docstrings in hover)
- You're defining simple ops with straightforward signatures
- You prefer explicit, readable code

Use OpProxy when:
- You need to check if an op is available before calling
- You're building a dynamic dispatch system
- You need the OpProxy.is_available property

EXAMPLE - Function-Based (Recommended):
---------------------------------------

    def silu_and_mul(x: torch.Tensor, out: torch.Tensor) -> None:
        '''Fused SiLU activation and multiply.

        Full docstring visible in IDE hover!
        '''
        impl = _get_impl("silu_and_mul", _silu_and_mul_native)
        impl(x, out)

EXAMPLE - OpProxy-Based (Advanced):
-----------------------------------

    silu_and_mul: OpProxy[[torch.Tensor, torch.Tensor], None] = OpProxy(
        "silu_and_mul",
        doc="Fused SiLU and multiply",  # Docstring via property, less visible
        fallback=_silu_and_mul_native,
    )

    # Check availability before use
    if silu_and_mul.is_available:
        silu_and_mul(x, out)
    else:
        # Handle unsupported platform
        ...

For this demo, we use the function-based approach in activation.py because
it provides the best IDE experience, which is one of the main goals of the RFC.
"""

from __future__ import annotations

from typing import Callable, Generic, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


class OpProxy(Generic[P, R]):
    """
    A type-safe proxy for platform-specific operations.

    OpProxy wraps an operation name and provides:
    - Runtime platform dispatch via __call__
    - Availability checking via is_available property
    - Cached implementation lookup for performance

    This class is provided for advanced use cases. For most ops,
    the simpler function-based approach in activation.py is preferred.

    Example Usage:
        # Create an OpProxy (typically done at module level)
        my_op: OpProxy[[torch.Tensor], torch.Tensor] = OpProxy(
            "my_op",
            fallback=lambda x: x.clone(),  # Native fallback
        )

        # Use like a regular function
        result = my_op(tensor)

        # Check availability
        if my_op.is_available:
            print("Optimized kernel available!")

    Type Parameters:
        P: Parameter specification (argument types)
        R: Return type
    """

    __slots__ = ("_name", "_doc", "_fallback", "_impl_cache")

    def __init__(
        self,
        name: str,
        doc: str = "",
        fallback: Callable[P, R] | None = None,
    ):
        """
        Initialize an OpProxy.

        Args:
            name: The canonical name of this operation (used for registry lookup)
            doc: Documentation string for the operation
            fallback: Optional native Python/PyTorch fallback implementation
        """
        self._name = name
        self._doc = doc
        self._fallback = fallback
        self._impl_cache: Callable[P, R] | None = None

    @property
    def name(self) -> str:
        """The canonical name of this operation."""
        return self._name

    def _get_impl(self) -> Callable[P, R]:
        """Get the implementation, with caching."""
        if self._impl_cache is not None:
            return self._impl_cache

        from sglang.srt.platforms import current_platform

        impl = current_platform.get_op(self)

        if impl is not None:
            self._impl_cache = impl
            return impl

        if self._fallback is not None:
            self._impl_cache = self._fallback
            return self._fallback

        raise RuntimeError(
            f"Operation '{self._name}' is not available on platform "
            f"'{current_platform.device_name}' and no fallback is provided.\n"
            f"Available ops: {current_platform.list_available_ops()}"
        )

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Dispatch to the platform-specific implementation."""
        return self._get_impl()(*args, **kwargs)

    @property
    def is_available(self) -> bool:
        """Check if this op is available on the current platform."""
        from sglang.srt.platforms import current_platform

        return current_platform.has_op(self) or self._fallback is not None

    def __repr__(self) -> str:
        return f"OpProxy({self._name!r})"

    @property
    def __doc__(self) -> str | None:
        return self._doc
