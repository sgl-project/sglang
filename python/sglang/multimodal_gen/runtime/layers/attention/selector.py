# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/attention/selector.py

import os
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from typing import NamedTuple, cast

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
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


class ComponentAttnBackendContext(NamedTuple):
    backend: AttentionBackendEnum | None
    component_name: str | None
    selected_backends: dict[str, str | None]
    allow_global_backend_fallback: bool = False


component_attn_backend_context: ContextVar[ComponentAttnBackendContext | None] = (
    ContextVar("component_attn_backend_context", default=None)
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


def get_component_attn_backend_context() -> ComponentAttnBackendContext | None:
    return component_attn_backend_context.get()


def get_component_forced_attn_backend() -> AttentionBackendEnum | None:
    context = get_component_attn_backend_context()
    return context.backend if context is not None else None


def get_component_attn_backend_name() -> str | None:
    context = get_component_attn_backend_context()
    return context.component_name if context is not None else None


def _component_allows_global_backend_fallback() -> bool:
    context = get_component_attn_backend_context()
    return context is not None and context.allow_global_backend_fallback


def _record_component_attn_backend(backend_name: str, reason: str | None) -> bool:
    context = get_component_attn_backend_context()
    if context is None or context.component_name is None:
        return False

    existing_reason = context.selected_backends.get(backend_name)
    if backend_name not in context.selected_backends or existing_reason is None:
        context.selected_backends[backend_name] = reason
    return True


def _log_component_attn_backend_summary(
    context: ComponentAttnBackendContext | None,
) -> None:
    if (
        context is None
        or context.component_name is None
        or not context.selected_backends
    ):
        return

    backend_parts = []
    for backend_name, reason in context.selected_backends.items():
        if reason:
            backend_parts.append(f"{backend_name} ({reason})")
        else:
            backend_parts.append(backend_name)

    logger.info_once(
        f"Attention backends for {context.component_name}: "
        f"{', '.join(backend_parts)}"
    )


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    supported_attention_backends: set[AttentionBackendEnum] | None = None,
    selected_attention_backend: AttentionBackendEnum | None = None,
    attention_requirements: AttentionRequirements | None = None,
    default_attention_backend: AttentionBackendEnum | None = None,
    is_cross_attention: bool = False,
) -> type[AttentionBackend]:
    requirements = attention_requirements or AttentionRequirements()
    if supported_attention_backends is None:
        be_tuple = tuple()
    else:
        # Sort the backend names to ensure consistent cache key
        be_tuple = tuple(
            sorted(list(supported_attention_backends), key=lambda b: b.name)
        )

    selected_backend = selected_attention_backend
    selected_from_global_cli = False
    selection_is_explicit = selected_backend is not None
    if selected_backend is None:
        selected_backend = get_global_forced_attn_backend()
        selection_is_explicit = selected_backend is not None
    if selected_backend is None:
        selected_backend = get_component_forced_attn_backend()
        selection_is_explicit = selected_backend is not None
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
            selection_is_explicit = isinstance(
                server_args, ServerArgs
            ) and server_args.is_arg_explicitly_set("attention_backend")
            selected_from_global_cli = selection_is_explicit

    if selected_backend is None:
        selected_backend = default_attention_backend

    allowed_fallback_reason = None
    if selected_backend is None:
        allowed_fallback_reason = "platform default fallback"
    elif is_cross_attention and selected_backend.is_sparse:
        allowed_fallback_reason = "dense cross-attention fallback"
    elif selected_from_global_cli and (
        default_attention_backend is not None
        or _component_allows_global_backend_fallback()
    ):
        # The global CLI backend is strict for DiT components. Auxiliary
        # components may instead use a declared default or platform-compatible
        # backend. A component-specific CLI override otherwise remains strict.
        allowed_fallback_reason = "global backend fallback"
    elif not selection_is_explicit:
        allowed_fallback_reason = "platform default fallback"

    constraint_backend = None
    if selected_backend is None and len(be_tuple) == 1:
        constraint_backend = be_tuple[0].name.lower()

    candidate_backends = [selected_backend]
    if allowed_fallback_reason is not None:
        for candidate in (default_attention_backend, None, *be_tuple):
            if candidate not in candidate_backends:
                candidate_backends.append(candidate)

    supported_backends = set(be_tuple)
    attention_backend_cls = None
    fallback_reason = None
    selection_error = None
    unsupported_backend_name = None
    unsupported_requirements = ()
    for candidate_index, candidate in enumerate(candidate_backends):
        try:
            candidate_cls = _cached_get_attn_backend(
                head_size,
                dtype,
                be_tuple,
                candidate,
            )
        except ValueError as error:
            if selection_error is None:
                selection_error = error
            continue

        candidate_backend = candidate_cls.get_enum()
        candidate_name = candidate_backend.name.lower()
        if is_cross_attention and candidate_backend.is_sparse:
            if selection_error is None:
                selection_error = ValueError(
                    f"Sparse attention backend '{candidate_name}' cannot serve "
                    "cross-attention"
                )
            continue
        if supported_backends and not _is_backend_supported(
            candidate_backend, supported_backends
        ):
            if selection_error is None:
                selection_error = ValueError(
                    f"Attention backend '{candidate_name}' is not supported by this "
                    f"attention layer; supported backends: "
                    f"{[str(backend) for backend in be_tuple]}"
                )
            continue

        missing_requirements = candidate_cls.unsupported_requirements(requirements)
        if missing_requirements:
            if not unsupported_requirements:
                unsupported_backend_name = candidate_name
                unsupported_requirements = missing_requirements
            continue

        attention_backend_cls = candidate_cls
        if candidate_index > 0:
            fallback_reason = allowed_fallback_reason
        break

    if attention_backend_cls is None:
        component_name = get_component_attn_backend_name()
        component_suffix = (
            f" for component '{component_name}'" if component_name is not None else ""
        )
        if unsupported_requirements:
            raise ValueError(
                f"Attention backend '{unsupported_backend_name}' does not implement "
                f"{', '.join(unsupported_requirements)}{component_suffix}"
            )
        if selection_error is not None:
            raise ValueError(
                f"{selection_error}{component_suffix}"
            ) from selection_error
        raise ValueError(
            f"No compatible attention backend is available{component_suffix}"
        )

    backend_name = attention_backend_cls.get_enum().name.lower()
    reason = fallback_reason
    if reason is None and backend_name == constraint_backend:
        reason = "component constraint"
    if not _record_component_attn_backend(backend_name, reason):
        reason_suffix = f" ({reason})" if reason else ""
        logger.info_once(f"Using {backend_name} attention backend{reason_suffix}")
    return attention_backend_cls


@cache
def _cached_get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    supported_attention_backends: tuple[AttentionBackendEnum],
    selected_backend: AttentionBackendEnum | None,
) -> type[AttentionBackend]:
    from sglang.multimodal_gen.runtime.platforms import current_platform

    supported_attention_backends = set(supported_attention_backends)

    # get device-specific attn_backend
    if len(supported_attention_backends) == 0:
        # all attention backends are allowed
        pass
    elif selected_backend is None and len(supported_attention_backends) == 1:
        selected_backend = next(iter(supported_attention_backends))
    elif selected_backend is not None and not _is_backend_supported(
        selected_backend, supported_attention_backends
    ):
        supported_attention_backends_str = [
            supported_attention_backend.__str__()
            for supported_attention_backend in supported_attention_backends
        ]
        raise ValueError(
            f"Attention backend '{selected_backend}' is not supported by this "
            f"attention layer; supported backends: {supported_attention_backends_str}"
        )

    attention_cls = current_platform.get_attn_backend_cls_str(
        selected_backend, head_size, dtype
    )
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}"
        )
    return cast(type[AttentionBackend], resolve_obj_by_qualname(attention_cls))


def _is_backend_supported(
    selected_backend: AttentionBackendEnum,
    supported_attention_backends: set[AttentionBackendEnum],
) -> bool:
    if selected_backend in supported_attention_backends:
        return True
    if selected_backend == AttentionBackendEnum.TORCH_CUDNN_SDPA:
        return AttentionBackendEnum.TORCH_SDPA in supported_attention_backends
    if selected_backend == AttentionBackendEnum.DYNAMIC_CUDNN_SDPA:
        return (
            AttentionBackendEnum.FA in supported_attention_backends
            and AttentionBackendEnum.TORCH_SDPA in supported_attention_backends
        )
    return False


@contextmanager
def component_attn_backend_context_manager(
    attn_backend: AttentionBackendEnum | None,
    component_name: str | None = None,
    allow_global_backend_fallback: bool = False,
) -> Generator[None, None, None]:
    if attn_backend is None and component_name is None:
        yield
        return

    token = component_attn_backend_context.set(
        ComponentAttnBackendContext(
            attn_backend,
            component_name,
            {},
            allow_global_backend_fallback,
        )
    )
    try:
        yield
    finally:
        context = component_attn_backend_context.get()
        _log_component_attn_backend_summary(context)
        component_attn_backend_context.reset(token)


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
