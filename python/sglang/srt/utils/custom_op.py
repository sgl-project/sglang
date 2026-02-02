from __future__ import annotations

import inspect
from typing import Any, Callable, List, Optional, TypeVar, Union, overload

import torch

F = TypeVar("F", bound=Callable)


@overload
def register_custom_op(
    fn: F,
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    out_shape: Optional[Union[int, str]] = None,
    eager: bool = True,
) -> F: ...


@overload
def register_custom_op(
    fn: F,
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    fake_impl: Optional[Callable],
    eager: bool = True,
) -> F: ...


@overload
def register_custom_op(
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    out_shape: Optional[Union[int, str]] = None,
    eager: bool = True,
) -> Callable[[F], F]: ...


@overload
def register_custom_op(
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    fake_impl: Optional[Callable],
    eager: bool = True,
) -> Callable[[F], F]: ...


# Real implementation
def register_custom_op(
    fn: Optional[Callable] = None,
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    eager: bool = True,
    **extra_kwargs,
) -> Any:
    """
    A decorator to register a custom operator.

    Example usage:
    ```python
    # inplace operator, out_shape is None by default
    @register_custom_op(mutates_args=["x"])
    def add_1_(x: torch.Tensor) -> None:
        x.add_(1)

    # operator with output, out_shape indicates the position of output
    @register_custom_op(mutates_args=["x"], out_shape=0)
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x.add_(y)
    ```

    :param fn: The function to be registered as a custom operator.
               If None, return a decorator.
    :type fn: Callable
    :param op_name: The name of the operator. If None, use the function name
    :type op_name: Optional[str]
    :param mutates_args: A list of argument names that are mutated in-place.
    :type mutates_args: List[str]
    :param out_shape: The position (int for positional, str for keyword) of the output-shape tensor.
                      It is used to generate a fake implementation for torch.compile compatibility.
                      If the operator is inplace and has no output, set to None.
    :type out_shape: Optional[List[Union[int, str]]]
    :param fake_impl: A fake implementation for the operator.
                      Only one of `out_shape` or `fake_impl` should be provided.
    :type fake_impl: Optional[Callable]
    :param eager: Whether to register the operator eagerly.
                  If False, the registration will be deferred until the first call.
                  If you met any issue with torch.compile, try to set eager=True.
                  Currently, to avoid misuse, we set eager=True by default.
    :type eager: bool
    :return: The registered JIT custom operator, or a decorator.
             NOTE: the real register will occur at the first call of the function.
    :rtype: Callable
    """
    extra_kwarg_keys = set(extra_kwargs.keys())
    expected_kwarg_keys = set({"out_shape", "fake_impl"})
    assert (
        expected_kwarg_keys >= extra_kwarg_keys
    ), f"Unexpected extra kwargs: {extra_kwarg_keys - expected_kwarg_keys}"

    has_out_shape = "out_shape" in extra_kwargs
    has_fake_impl = "fake_impl" in extra_kwargs
    assert not (
        has_out_shape and has_fake_impl
    ), "Only one of `out_shape` or `fake_impl` should be provided."
    # Assume inplace if neither out_shape nor fake_impl is provided
    if not (has_out_shape or has_fake_impl):
        extra_kwargs["out_shape"] = None

    def decorator(op_func: Callable) -> Callable:
        wrapper = CustomOpWrapper(
            op_name=op_name or op_func.__name__,
            op_func=op_func,
            mutates_args=mutates_args or [],
            **extra_kwargs,
        )
        return wrapper.real_impl if eager else wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


class CustomOpWrapper:
    def __init__(
        self,
        op_name: str,
        op_func: Callable,
        mutates_args: List[str],
        **extra_kwargs,
    ):
        self.op_name = op_name
        self.op_func = op_func
        self.mutates_args = mutates_args
        self.extra_kwargs = extra_kwargs
        self._impl: Optional[Callable] = None

    def __call__(self, *args, **kwargs):
        return self.real_impl(*args, **kwargs)

    @property
    def real_impl(self) -> Callable:
        if self._impl is None:
            if not hasattr(torch.ops.sglang, self.op_name):
                from sglang.srt.utils.common import direct_register_custom_op

                # NOTE(dark): if torch compile fail here, mark the decorator as eager
                # lazy registration does not work with torch compile
                direct_register_custom_op(
                    op_name=self.op_name,
                    op_func=self.op_func,
                    mutates_args=self.mutates_args,
                    fake_impl=self.fake_impl,
                )
            self._impl = getattr(torch.ops.sglang, self.op_name)
            assert self._impl is not None
        return self._impl

    @property
    def fake_impl(self) -> Callable:
        if "fake_impl" in self.extra_kwargs:
            return self.extra_kwargs["fake_impl"]
        assert "out_shape" in self.extra_kwargs
        signature = inspect.signature(self.op_func)
        out_shape = self.extra_kwargs["out_shape"]
        # check out_shape in signature

        def fake_impl(*args, **kwargs):
            if out_shape is None:
                return None
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            try:
                return torch.empty_like(
                    bound.args[out_shape]
                    if isinstance(out_shape, int)
                    else bound.arguments[out_shape]
                )
            except (IndexError, KeyError):
                raise RuntimeError(
                    f"Cannot find output argument at position `{out_shape}` for "
                    f"custom operator `{self.op_name}` with signature `{signature}`."
                )

        return fake_impl
