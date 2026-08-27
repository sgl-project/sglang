import os
from typing import Any, Callable, TypeVar, cast, overload

F = TypeVar("F", bound=Callable[..., Any])


def _wrap_debug_kernel(func: F, op_name: str | None = None) -> F:
    try:
        if int(os.environ.get("SGLANG_KERNEL_API_LOGLEVEL", "0")) == 0:
            return func
    except Exception:
        return func

    try:
        from sglang.kernel_api_logging import debug_kernel_api
    except Exception:
        return func

    if getattr(func, "_debug_kernel_wrapped", False):
        return func

    wrapped = debug_kernel_api(func, op_name=op_name)
    setattr(wrapped, "_debug_kernel_wrapped", True)
    return cast(F, wrapped)


@overload
def maybe_wrap_debug_kernel(func: F) -> F: ...


@overload
def maybe_wrap_debug_kernel(func: F, op_name: str) -> F: ...


@overload
def maybe_wrap_debug_kernel(*, op_name: str | None = None) -> Callable[[F], F]: ...


def maybe_wrap_debug_kernel(
    func: F | None = None, op_name: str | None = None
) -> F | Callable[[F], F]:
    if func is None:
        return lambda wrapped_func: _wrap_debug_kernel(wrapped_func, op_name)

    return _wrap_debug_kernel(func, op_name)
