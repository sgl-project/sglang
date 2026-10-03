# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/attention/selector.py

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from pkgutil import resolve_name
from typing import NamedTuple, cast

import torch

from sglang.multimodal_gen.runtime.layers.attention.backends.attention_backend import (
    AttentionBackend,
    AttentionRequirements,
)
from sglang.multimodal_gen.runtime.layers.attention.roles import (
    AttentionRole,
    make_component_role_key,
)
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum
from sglang.multimodal_gen.runtime.server_args import ServerArgs, get_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Global state allows a particular choice of backend
# to be forced, overriding the logic which auto-selects
# a backend based on system & workload configuration
# (default behavior if this variable is None)
forced_attn_backend: AttentionBackendEnum | None = None


class ComponentAttnBackendContext(NamedTuple):
    backend: AttentionBackendEnum | None
    component_name: str | None
    selected_backends: dict[str, str | None]
    backend_by_role: dict[AttentionRole, AttentionBackendEnum]
    # Same records as ``selected_backends``, but attributed to the role of the
    # layer that made them. ``None`` collects selections made outside a layer,
    # which belong to no single role.
    selected_backends_by_role: dict[AttentionRole | None, dict[str, str | None]]
    allow_global_backend_fallback: bool = False
    require_backend_selection: bool = False


class ComponentAttentionBackendNotAppliedError(ValueError):
    """An explicit component backend did not control its attention layers."""


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


def claim_deferred_component_attn_backend() -> AttentionBackendEnum | None:
    """Capture an override whose compatible backend is resolved on first use."""
    context = get_component_attn_backend_context()
    if context is None or context.backend is None:
        return None
    _record_component_attn_backend(
        context.backend.name.lower(), "deferred first-use selection"
    )
    return context.backend


def get_component_forced_attn_backend_for_role(
    role: AttentionRole,
) -> AttentionBackendEnum | None:
    """Role-specific component override (e.g. ``transformer.cross``).

    Takes precedence over the component-wide backend when both are configured.
    """
    context = get_component_attn_backend_context()
    return context.backend_by_role.get(role) if context is not None else None


def get_component_attn_backend_name() -> str | None:
    context = get_component_attn_backend_context()
    return context.component_name if context is not None else None


def _component_allows_global_backend_fallback() -> bool:
    context = get_component_attn_backend_context()
    return context is not None and context.allow_global_backend_fallback


def _merge_selection(
    selections: dict[str, str | None], backend_name: str, reason: str | None
) -> None:
    if backend_name not in selections:
        selections[backend_name] = reason
    elif reason is None:
        # unrestricted selection must not be hidden by a later valid fallback
        selections[backend_name] = None


def _record_component_attn_backend(
    backend_name: str, reason: str | None, role: AttentionRole | None = None
) -> bool:
    context = get_component_attn_backend_context()
    if context is None or context.component_name is None:
        return False

    _merge_selection(context.selected_backends, backend_name, reason)
    _merge_selection(
        context.selected_backends_by_role.setdefault(role, {}), backend_name, reason
    )
    return True


def _format_component_attn_selections(selections: dict[str, str | None]) -> str:
    return ", ".join(
        f"{backend_name} ({reason})" if reason else backend_name
        for backend_name, reason in selections.items()
    )


def _log_component_attn_backend_summary(
    context: ComponentAttnBackendContext | None,
) -> None:
    if (
        context is None
        or context.component_name is None
        or not context.selected_backends
    ):
        return

    by_role = context.selected_backends_by_role
    summary = ""
    # Break the summary out per role only when a role override is configured, so
    # it can be confirmed from the log without re-deriving which layers it
    # reached. Components with no override keep the flat single-line format.
    if context.backend_by_role:
        parts = [
            f"{role.value}={_format_component_attn_selections(by_role[role])}"
            for role in AttentionRole
            if by_role.get(role)
        ]
        if by_role.get(None):
            parts.append(_format_component_attn_selections(by_role[None]))
        summary = "; ".join(parts)
    if not summary:
        summary = _format_component_attn_selections(context.selected_backends)

    logger.info_once(f"Attention backends for {context.component_name}: {summary}")


def _validate_selection_against_request(
    selections: dict[str, str | None], requested_name: str, target: str
) -> None:
    """Check that ``target``'s layers honored ``requested_name``.

    A divergence is tolerated only when every diverging selection carries a
    reason, i.e. it went through an allowed fallback such as the dense
    replacement for a sparse backend in cross-attention.
    """
    unexplained = sorted(
        backend_name
        for backend_name, reason in selections.items()
        if backend_name != requested_name and reason is None
    )
    if unexplained:
        detail = (
            f"also selected {', '.join(unexplained)} without an allowed fallback"
            if requested_name in selections
            else f"selected {', '.join(unexplained)} without an allowed fallback"
        )
        raise ComponentAttentionBackendNotAppliedError(
            f"Attention backend '{requested_name}' was requested for {target}, "
            f"but it {detail}"
        )


def _validate_component_attn_backend_selection(
    context: ComponentAttnBackendContext,
) -> None:
    component_name = context.component_name or "component"

    # A role override is honored when the layers of that role selected it, or
    # fell back for a recorded reason. Having no layers of that role at all
    # means the override silently did nothing, which is the same failure the
    # component-wide check below reports.
    for role, backend in context.backend_by_role.items():
        target = f"component '{make_component_role_key(component_name, role)}'"
        selections = context.selected_backends_by_role.get(role)
        requested_name = backend.name.lower()
        if not selections:
            raise ComponentAttentionBackendNotAppliedError(
                f"Attention backend '{requested_name}' was requested for {target}, "
                f"but the component constructed no {role.value}-attention layers"
            )
        _validate_selection_against_request(selections, requested_name, target)

    if not context.require_backend_selection or context.backend is None:
        return

    # Roles carrying their own override are out of the component-wide backend's
    # scope, so judge it only on what is left.
    residual: dict[str, str | None] = {}
    for role, selections in context.selected_backends_by_role.items():
        if role is not None and role in context.backend_by_role:
            continue
        for backend_name, reason in selections.items():
            _merge_selection(residual, backend_name, reason)

    requested_name = context.backend.name.lower()
    target = f"component '{component_name}'"
    if not residual:
        if context.selected_backends_by_role:
            # Every layer this component built was diverted by a role override,
            # so the component-wide backend had nothing left to apply to.
            return
        raise ComponentAttentionBackendNotAppliedError(
            f"Attention backend '{requested_name}' was requested for {target}, "
            "but it did not construct any SGLang-selectable attention layers"
        )
    if requested_name not in residual:
        raise ComponentAttentionBackendNotAppliedError(
            f"Attention backend '{requested_name}' was requested for {target}, "
            f"but it selected {', '.join(sorted(residual))} instead"
        )
    _validate_selection_against_request(residual, requested_name, target)


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    supported_attention_backends: set[AttentionBackendEnum] | None = None,
    selected_attention_backend: AttentionBackendEnum | None = None,
    attention_requirements: AttentionRequirements | None = None,
    default_attention_backend: AttentionBackendEnum | None = None,
    is_cross_attention: bool = False,
) -> type[AttentionBackend]:
    """Resolve an attention backend for one layer.

    ``supported_attention_backends`` constrains automatic selection only. An
    explicitly requested backend may be newer than a model's preference set;
    it is admitted when the platform resolves it and the backend satisfies the
    layer's semantic requirements.
    """
    attention_role = AttentionRole.CROSS if is_cross_attention else AttentionRole.SELF
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
    selected_from_role_override = False
    if selected_backend is None:
        selected_backend = get_component_forced_attn_backend_for_role(attention_role)
        selection_is_explicit = selected_backend is not None
        selected_from_role_override = selection_is_explicit
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

    automatic_backends = set(be_tuple)
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
        explicit_candidate = selection_is_explicit and candidate_index == 0
        if (
            automatic_backends
            and not explicit_candidate
            and not _is_backend_supported(candidate_backend, automatic_backends)
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
    if selected_from_role_override and fallback_reason is not None:
        # An explicit role request that could not be used is allowed to fall
        # back, but silently doing so would look like the override took effect.
        component_name = get_component_attn_backend_name() or "component"
        logger.warning_once(
            f"Attention backend '{selected_backend.name.lower()}' was requested "
            f"for '{make_component_role_key(component_name, attention_role)}' but "
            f"could not be used ({fallback_reason}); using {backend_name} instead"
        )
    if reason is None and selected_from_role_override:
        # Diverging from the component-wide backend is the point of a role
        # override, so record it as an explained selection.
        reason = f"{attention_role.value}-attention override"
    if reason is None and backend_name == constraint_backend:
        reason = "component constraint"
    if not _record_component_attn_backend(backend_name, reason, attention_role):
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

    attention_cls = current_platform.get_attn_backend_cls_str(
        selected_backend, head_size, dtype
    )
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}"
        )
    return cast(type[AttentionBackend], resolve_name(attention_cls))


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
    backend_by_role: dict[AttentionRole, AttentionBackendEnum] | None = None,
    allow_global_backend_fallback: bool = False,
    require_backend_selection: bool | None = None,
    require_component_backend_selection: bool | None = None,
) -> Generator[None, None, None]:
    if attn_backend is None and component_name is None and not backend_by_role:
        yield
        return

    if require_backend_selection is None:
        require_backend_selection = (
            require_component_backend_selection
            if require_component_backend_selection is not None
            else attn_backend is not None
        )
    elif require_component_backend_selection is not None:
        raise ValueError("Specify only one component backend selection requirement")

    if backend_by_role:
        # Announce the configured overrides up front; the summary logged on exit
        # reports which layers each one actually reached.
        role_parts = ", ".join(
            f"{role.value}={backend.name.lower()}"
            for role, backend in sorted(
                backend_by_role.items(), key=lambda item: item[0].value
            )
        )
        logger.info(
            f"Per-role attention backend overrides for "
            f"{component_name or 'component'}: {role_parts}"
        )

    token = component_attn_backend_context.set(
        ComponentAttnBackendContext(
            attn_backend,
            component_name,
            {},
            dict(backend_by_role or {}),
            {},
            allow_global_backend_fallback,
            require_backend_selection,
        )
    )
    try:
        yield
        context = component_attn_backend_context.get()
        _validate_component_attn_backend_selection(context)
        _log_component_attn_backend_summary(context)
    finally:
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
