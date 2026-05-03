# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/attention/selector.py

import os
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from typing import cast

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import STR_BACKEND_ENV_VAR, resolve_obj_by_qualname

logger = init_logger(__name__)


def backend_name_to_enum(backend_name: str) -> AttentionBackendEnum | None:
    """
    Convert a string backend name to a _Backend enum value.

    Returns:
    * _Backend: enum value if backend_name is a valid in-tree type
    * None: otherwise it's an invalid in-tree type or an out-of-tree platform is
            loaded.
    """
    assert backend_name is not None
    return (
        AttentionBackendEnum[backend_name]
        if backend_name in AttentionBackendEnum.__members__
        else None
    )


def get_env_variable_attn_backend() -> AttentionBackendEnum | None:
    """
    Get the backend override specified by the sglang-diffusion attention
    backend environment variable, if one is specified.

    Returns:

    * _Backend enum value if an override is specified
    * None otherwise
    """
    backend_name = os.environ.get(STR_BACKEND_ENV_VAR)
    return None if backend_name is None else backend_name_to_enum(backend_name)


# Global state allows a particular choice of backend
# to be forced, overriding the logic which auto-selects
# a backend based on system & workload configuration
# (default behavior if this variable is None)
#
# THIS SELECTION TAKES PRECEDENCE OVER THE
# FASTVIDEO ATTENTION BACKEND ENVIRONMENT VARIABLE
forced_attn_backend: AttentionBackendEnum | None = None
component_forced_attn_backend: ContextVar[AttentionBackendEnum | None] = ContextVar(
    "component_forced_attn_backend", default=None
)
component_attn_backend_name: ContextVar[str | None] = ContextVar(
    "component_attn_backend_name", default=None
)


def global_force_attn_backend(attn_backend: AttentionBackendEnum | None) -> None:
    """
    Force all attention operations to use a specified backend.

    Passing `None` for the argument re-enables automatic
    backend selection.,

    Arguments:

    * attn_backend: backend selection (None to revert to auto)
    """
    global forced_attn_backend
    forced_attn_backend = attn_backend


def get_global_forced_attn_backend() -> AttentionBackendEnum | None:
    """
    Get the currently-forced choice of attention backend,
    or None if auto-selection is currently enabled.
    """
    return forced_attn_backend


def get_component_forced_attn_backend() -> AttentionBackendEnum | None:
    return component_forced_attn_backend.get()


def get_component_attn_backend_name() -> str | None:
    return component_attn_backend_name.get()


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    supported_attention_backends: set[AttentionBackendEnum] | None = None,
    selected_attention_backend: AttentionBackendEnum | None = None,
) -> type[AttentionBackend]:
    if supported_attention_backends is None:
        be_tuple = tuple()
    else:
        # Sort the backend names to ensure consistent cache key
        be_tuple = tuple(
            sorted(list(supported_attention_backends), key=lambda b: b.name)
        )

    selected_backend = (
        selected_attention_backend
        or get_component_forced_attn_backend()
        or get_global_forced_attn_backend()
    )
    if selected_backend is None:
        server_args = get_global_server_args()
        if server_args.attention_backend is not None:
            try:
                selected_backend = AttentionBackendEnum[
                    server_args.attention_backend.upper()
                ]
            except KeyError:
                raise ValueError(
                    f"Invalid attention backend '{server_args.attention_backend}' specified via command line. "
                    f"Available options are: {[e.name.lower() for e in AttentionBackendEnum]}"
                )

    return _cached_get_attn_backend(
        head_size,
        dtype,
        be_tuple,
        selected_backend,
        get_component_attn_backend_name(),
    )


@cache
def _cached_get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    supported_attention_backends: tuple[AttentionBackendEnum],
    selected_backend: AttentionBackendEnum | None,
    component_name: str | None,
) -> type[AttentionBackend]:
    from sglang.multimodal_gen.runtime.platforms import current_platform

    supported_attention_backends = set(supported_attention_backends)
    component_suffix = f" for {component_name}" if component_name else ""
    backend_not_specified = selected_backend is None

    # get device-specific attn_backend
    if len(supported_attention_backends) == 0:
        # all attention backends are allowed
        pass
    elif selected_backend is None and len(supported_attention_backends) == 1:
        selected_backend = next(iter(supported_attention_backends))
    elif selected_backend is None:
        logger.debug("Attention backend not specified%s", component_suffix)
    elif selected_backend not in supported_attention_backends:
        supported_attention_backends_str = [
            supported_attention_backend.__str__()
            for supported_attention_backend in supported_attention_backends
        ]
        logger.debug(
            "Selected attention backend: '%s' not in supported attention backends: %s%s",
            selected_backend,
            supported_attention_backends_str,
            component_suffix,
        )
        selected_backend = None

    attention_cls = current_platform.get_attn_backend_cls_str(
        selected_backend, head_size, dtype
    )
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}"
        )
    attention_backend_cls = cast(
        type[AttentionBackend], resolve_obj_by_qualname(attention_cls)
    )
    if component_name:
        backend_name = attention_backend_cls.get_enum().name.lower()
        if backend_not_specified:
            logger.info(
                "Attention backend not specified for %s, using %s backend for %s",
                component_name,
                backend_name,
                component_name,
            )
        else:
            logger.info("Using %s backend for %s", backend_name, component_name)
    return attention_backend_cls


@contextmanager
def component_force_attn_backend_context_manager(
    attn_backend: AttentionBackendEnum | None,
    component_name: str | None = None,
) -> Generator[None, None, None]:
    if attn_backend is None and component_name is None:
        yield
        return

    backend_token = (
        component_forced_attn_backend.set(attn_backend)
        if attn_backend is not None
        else None
    )
    component_token = (
        component_attn_backend_name.set(component_name)
        if component_name is not None
        else None
    )
    try:
        yield
    finally:
        if component_token is not None:
            component_attn_backend_name.reset(component_token)
        if backend_token is not None:
            component_forced_attn_backend.reset(backend_token)


@contextmanager
def global_force_attn_backend_context_manager(
    attn_backend: AttentionBackendEnum,
) -> Generator[None, None, None]:
    """
    Globally force a sglang-diffusion attention backend override within a
    context manager, reverting the global attention backend
    override to its prior state upon exiting the context
    manager.

    Arguments:
    * attn_backend: attention backend to force

    Returns:

    * Generator
    """

    # Save the current state of the global backend override (if any)
    original_value = get_global_forced_attn_backend()

    # Globally force the new backend override
    global_force_attn_backend(attn_backend)

    # Yield control back to the enclosed code block
    try:
        yield
    finally:
        # Revert the original global backend override, if any
        global_force_attn_backend(original_value)
