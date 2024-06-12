"""Public APIs of the language."""

import os
import re
from typing import Callable, List, Optional, Union

from sglang.backend.base_backend import BaseBackend
from sglang.global_config import global_config
from sglang.lang.ir import (
    SglExpr,
    SglExprList,
    SglFunction,
    SglGen,
    SglImage,
    SglRoleBegin,
    SglRoleEnd,
    SglSelect,
    SglVideo,
)


def function(
    func: Optional[Callable] = None, num_api_spec_tokens: Optional[int] = None
):
    if func:
        return SglFunction(func, num_api_spec_tokens=num_api_spec_tokens)

    def decorator(func):
        return SglFunction(func, num_api_spec_tokens=num_api_spec_tokens)

    return decorator


def Runtime(*args, **kwargs):
    # Avoid importing unnecessary dependency
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from sglang.srt.server import Runtime

    return Runtime(*args, **kwargs)


def set_default_backend(backend: BaseBackend):
    global_config.default_backend = backend


def flush_cache(backend: Optional[BaseBackend] = None):
    backend = backend or global_config.default_backend
    if backend is None:
        return False
    return backend.flush_cache()


def get_server_args(backend: Optional[BaseBackend] = None):
    backend = backend or global_config.default_backend
    if backend is None:
        return None
    return backend.get_server_args()


def gen(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    ignore_eos: Optional[bool] = None,
    dtype: Optional[type] = None,
    choices: Optional[List[str]] = None,
    regex: Optional[str] = None,
):
    if choices:
        return SglSelect(name, choices, 0.0 if temperature is None else temperature)

    # check regex is valid
    if regex is not None:
        try:
            re.compile(regex)
        except re.error as e:
            raise e

    return SglGen(
        name,
        max_tokens,
        stop,
        temperature,
        top_p,
        top_k,
        frequency_penalty,
        presence_penalty,
        ignore_eos,
        dtype,
        regex,
    )


def gen_int(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    ignore_eos: Optional[bool] = None,
):
    return SglGen(
        name,
        max_tokens,
        stop,
        temperature,
        top_p,
        top_k,
        frequency_penalty,
        presence_penalty,
        ignore_eos,
        int,
        None,
    )


def gen_string(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    ignore_eos: Optional[bool] = None,
):
    return SglGen(
        name,
        max_tokens,
        stop,
        temperature,
        top_p,
        top_k,
        frequency_penalty,
        presence_penalty,
        ignore_eos,
        str,
        None,
    )


def image(expr: SglExpr):
    return SglImage(expr)


def video(path: str, num_frames: int):
    return SglVideo(path, num_frames)


def select(
    name: Optional[str] = None,
    choices: Optional[List[str]] = None,
    temperature: float = 0.0,
):
    assert choices is not None
    return SglSelect(name, choices, temperature)


def _role_common(name: str, expr: Optional[SglExpr] = None):
    if expr is None:
        return SglExprList([SglRoleBegin(name), SglRoleEnd(name)])
    else:
        return SglExprList([SglRoleBegin(name), expr, SglRoleEnd(name)])


def system(expr: Optional[SglExpr] = None):
    return _role_common("system", expr)


def user(expr: Optional[SglExpr] = None):
    return _role_common("user", expr)


def assistant(expr: Optional[SglExpr] = None):
    return _role_common("assistant", expr)


def user_begin():
    return SglRoleBegin("user")


def user_end():
    return SglRoleEnd("user")


def assistant_begin():
    return SglRoleBegin("assistant")


def assistant_end():
    return SglRoleEnd("assistant")
