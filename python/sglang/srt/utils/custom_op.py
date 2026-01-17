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


