from __future__ import annotations

import inspect
from typing import Any, Callable, List, Optional, TypeVar, Union, overload

import torch
import torch.library

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


def register_custom_op_from_extern(
    fn: Callable,
    *,
    op_name: Optional[str] = None,
    mutates_args: Optional[List[str]] = None,
    out_shape: Optional[Union[int, str]] = None,
    out_dtype: Optional[torch.dtype] = None,
    fake_impl: Optional[Callable] = None,
    computed_args: Optional[dict] = None,
) -> Callable:
    """Wrap an external library function as a custom op for torch.compile compatibility.

    Use this to wrap functions from external libraries (e.g. flashinfer kernels) that
    perform operations incompatible with torch.compile/dynamo tracing, such as JIT
    compilation, file I/O, or dynamic module loading.

    The wrapped function becomes an opaque node in the compiled graph. Dynamo will
    not trace inside it, avoiding tracing failures. A fake implementation is used
    for shape/dtype propagation during compilation.

    The external function must have type annotations compatible with
    ``torch.library.infer_schema`` (``torch.Tensor``, ``int``, ``float``, ``bool``,
    ``Optional[torch.Tensor]``, etc.).

    This function is idempotent: calling it multiple times with the same ``op_name``
    (or ``fn.__name__``) safely skips re-registration.

    Example usage::

        from flashinfer.fused_moe import trtllm_fp8_block_scale_moe

        trtllm_fp8_block_scale_moe = register_custom_op_from_extern(
            trtllm_fp8_block_scale_moe,
            out_shape="hidden_states",
            out_dtype=torch.bfloat16,
            computed_args={
                "tune_max_num_tokens": lambda hidden_states, **kw: next_power_of_2(
                    hidden_states.shape[0]
                ),
            },
        )

    :param fn: The external function to wrap.
    :param op_name: The name of the custom operator.
                    Defaults to ``fn.__name__``.
    :param mutates_args: A list of argument names that are mutated in-place.
                         Defaults to ``[]``.
    :param out_shape: The position (int) or name (str) of the argument whose shape
                      matches the output tensor. Used to auto-generate a fake
                      implementation. Set to ``None`` for inplace-only operators.
    :param out_dtype: Override the output dtype in the fake implementation.
                      If ``None``, ``torch.empty_like`` is used (same dtype as the
                      reference tensor). Useful when the output dtype differs from
                      the input (e.g. fp8 input -> bf16 output).
    :param fake_impl: A custom fake implementation for shape/dtype propagation.
                      Only one of ``out_shape`` or ``fake_impl`` should be provided.
    :param computed_args: A dict mapping argument names to callables. These arguments
                          are excluded from the custom op schema and computed inside
                          the op body at runtime. Each callable receives the other
                          arguments as keyword args and returns the computed value.
                          Use this for arguments that vary dynamically (e.g.
                          ``tune_max_num_tokens``) to avoid torch.compile recompilation.
    :return: The registered custom op callable (``torch.ops.sglang.<op_name>``).
    """
    name = op_name or fn.__name__
    computed_args = computed_args or {}

    assert not (
        out_shape is not None and fake_impl is not None
    ), "Only one of `out_shape` or `fake_impl` should be provided."

    # If computed_args specified, create a wrapper with a reduced signature
    # that computes the excluded args inside the op body.
    if computed_args:
        original_fn = fn
        original_sig = inspect.signature(fn)

        # Build new signature excluding computed args
        new_params = [
            p
            for param_name, p in original_sig.parameters.items()
            if param_name not in computed_args
        ]
        new_sig = original_sig.replace(parameters=new_params)

        def wrapper(*args, **kwargs):
            bound = new_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            # Compute excluded args from the bound arguments
            for arg_name, compute_fn in computed_args.items():
                bound.arguments[arg_name] = compute_fn(**bound.arguments)
            return original_fn(**bound.arguments)

        wrapper.__name__ = fn.__name__
        wrapper.__qualname__ = fn.__qualname__
        wrapper.__module__ = fn.__module__
        wrapper.__signature__ = new_sig
        # Build annotations without computed args, preserving return type
        wrapper.__annotations__ = {
            k: v
            for k, v in getattr(fn, "__annotations__", {}).items()
            if k not in computed_args
        }
        fn = wrapper

    # Generate fake_impl from out_shape if needed
    fake_sig = inspect.signature(fn)
    if fake_impl is None and out_shape is not None:

        def _fake_impl(*args, **kwargs):
            bound = fake_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            try:
                ref = (
                    bound.args[out_shape]
                    if isinstance(out_shape, int)
                    else bound.arguments[out_shape]
                )
            except (IndexError, KeyError):
                raise RuntimeError(
                    f"Cannot find output argument at position `{out_shape}` for "
                    f"external function `{name}` with signature `{fake_sig}`."
                )
            if out_dtype is not None:
                return torch.empty(ref.shape, dtype=out_dtype, device=ref.device)
            return torch.empty_like(ref)

        fake_impl = _fake_impl
    elif fake_impl is None:
        fake_impl = lambda *args, **kwargs: None

    from sglang.srt.utils.common import direct_register_custom_op

    direct_register_custom_op(
        op_name=name,
        op_func=fn,
        mutates_args=mutates_args or [],
        fake_impl=fake_impl,
    )

    return getattr(torch.ops.sglang, name)
