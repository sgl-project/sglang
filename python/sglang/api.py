"""Public APIs of the language."""

import re
from typing import Callable, List, Optional, Union

from sglang.global_config import global_config
from sglang.lang.backend.base_backend import BaseBackend
from sglang.lang.choices import ChoicesSamplingMethod, token_length_normalized
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
    from sglang.lang.backend.runtime_endpoint import Runtime

    return Runtime(*args, **kwargs)


def Engine(*args, **kwargs):
    # Avoid importing unnecessary dependency
    from sglang.srt.entrypoints.engine import Engine

    return Engine(*args, **kwargs)


def set_default_backend(backend: BaseBackend):
    global_config.default_backend = backend


def flush_cache(backend: Optional[BaseBackend] = None):
    backend = backend or global_config.default_backend
    if backend is None:
        return False

    # If backend is Runtime
    if hasattr(backend, "endpoint"):
        backend = backend.endpoint
    return backend.flush_cache()


def get_server_info(backend: Optional[BaseBackend] = None):
    backend = backend or global_config.default_backend
    if backend is None:
        return None

    # If backend is Runtime
    if hasattr(backend, "endpoint"):
        backend = backend.endpoint
    return backend.get_server_info()


def gen(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    min_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    ignore_eos: Optional[bool] = None,
    return_logprob: Optional[bool] = None,
    logprob_start_len: Optional[int] = None,
    top_logprobs_num: Optional[int] = None,
    return_text_in_logprobs: Optional[bool] = None,
    dtype: Optional[Union[type, str]] = None,
    choices: Optional[List[str]] = None,
    choices_method: Optional[ChoicesSamplingMethod] = None,
    regex: Optional[str] = None,
    json_schema: Optional[str] = None,
):
    """Call the model to generate. See the meaning of the arguments in docs/sampling_params.md"""

    if choices:
        return SglSelect(
            name,
            choices,
            0.0 if temperature is None else temperature,
            token_length_normalized if choices_method is None else choices_method,
        )

    # check regex is valid
    if regex is not None:
        try:
            re.compile(regex)
        except re.error as e:
            raise e

    return SglGen(
        name,
        max_tokens,
        min_tokens,
        stop,
        stop_token_ids,
        temperature,
        top_p,
        top_k,
        min_p,
        frequency_penalty,
        presence_penalty,
        ignore_eos,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        return_text_in_logprobs,
        dtype,
        regex,
        json_schema,
    )


def gen_int(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    ignore_eos: Optional[bool] = None,
    return_logprob: Optional[bool] = None,
    logprob_start_len: Optional[int] = None,
    top_logprobs_num: Optional[int] = None,
    return_text_in_logprobs: Optional[bool] = None,
):
    return SglGen(
        name,
        max_tokens,
        None,
        stop,
        stop_token_ids,
        temperature,
        top_p,
        top_k,
        min_p,
        frequency_penalty,
        presence_penalty,
        ignore_eos,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        return_text_in_logprobs,
        int,
        None,
    )


def gen_string(
    name: Optional[str] = None,
    max_tokens: Optional[int] = None,
    stop: Optional[Union[str, List[str]]] = None,
    stop_token_ids: Optional[List[int]] = None,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    min_p: Optional[float] = None,
    frequency_penalty: Optional[float] = None,
    presence_penalty: Optional[float] = None,
    ignore_eos: Optional[bool] = None,
    return_logprob: Optional[bool] = None,
    logprob_start_len: Optional[int] = None,
    top_logprobs_num: Optional[int] = None,
    return_text_in_logprobs: Optional[bool] = None,
):
    return SglGen(
        name,
        max_tokens,
        None,
        stop,
        stop_token_ids,
        temperature,
        top_p,
        top_k,
        min_p,
        frequency_penalty,
        presence_penalty,
        ignore_eos,
        return_logprob,
        logprob_start_len,
        top_logprobs_num,
        return_text_in_logprobs,
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
    choices_method: ChoicesSamplingMethod = token_length_normalized,
):
    assert choices is not None
    return SglSelect(name, choices, temperature, choices_method)


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


def system_begin():
    return SglRoleBegin("system")


def system_end():
    return SglRoleEnd("system")


def user_begin():
    return SglRoleBegin("user")


def user_end():
    return SglRoleEnd("user")


def assistant_begin():
    return SglRoleBegin("assistant")


def assistant_end():
    return SglRoleEnd("assistant")
