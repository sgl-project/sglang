# SPDX-License-Identifier: Apache-2.0
"""Launcher and child-process selection of out-of-tree megakernel providers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import msgspec
import torch.distributed as dist

from sglang.kernels.plugin_kernels.megakernel.spec import (
    MEGAKERNEL_API_VERSION,
    BuildContext,
    BuiltMegakernel,
    MegakernelSpec,
    find_spec,
    find_target_registration,
)
from sglang.srt.arg_groups.model_override_base import model_config_of, resolving_view
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.plugins import load_plugins
from sglang.srt.runtime_context import get_device, get_parallel

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class MegakernelSelection(msgspec.Struct, frozen=True):
    """Provider identity carried by the resolved launch to child processes."""

    architecture: str
    target: str
    path: str
    api_version: int


def get_oot_megakernel_selection(server_args: ServerArgs) -> MegakernelSelection | None:
    return getattr(server_args, "_oot_megakernel_selection", None)


def select_oot_megakernel(server_args: ServerArgs) -> None:
    """Resolve the exact registration from metadata before other settings use it."""
    from sglang.srt.platforms import current_platform

    if not current_platform.is_out_of_tree():
        return
    cfg = resolving_view(server_args)
    requested = cfg.enable_nvidia_megakernel
    capabilities = current_platform.capabilities
    if requested and not (
        capabilities.supports_megakernel and capabilities.accepts_legacy_megakernel_flag
    ):
        raise ValueError(
            f"--enable-nvidia-megakernel is unsupported on {current_platform.device_name}"
        )
    if not capabilities.supports_megakernel:
        return

    target = current_platform.get_megakernel_target()
    if not isinstance(target, str) or not target:
        raise ValueError(
            f"{current_platform.device_name} advertises megakernel support but "
            "did not provide a device target"
        )
    load_plugins()
    hf_config = model_config_of(server_args).hf_config
    registration = find_target_registration(hf_config.architectures, target=target)
    if registration is None:
        if requested or current_platform.requires_megakernel(cfg, hf_config):
            raise ValueError(
                f"a megakernel is required but no provider is registered for "
                f"{hf_config.architectures} on {target}"
            )
        return

    architecture, path = registration
    if cfg.enable_amd_monokernel:
        raise ValueError(
            "an out-of-tree megakernel cannot be used with --enable-amd-monokernel"
        )
    server_args._oot_megakernel_selection = MegakernelSelection(
        architecture, target, path, MEGAKERNEL_API_VERSION
    )
    if requested:
        logger.warning(
            "--enable-nvidia-megakernel is deprecated on %s; the provider is selected automatically",
            current_platform.device_name,
        )
        # Keep the raw input for diagnostics and prevent entry into stock CUDA dispatch.
        declare_resolution(
            server_args, "select_oot_megakernel", enable_nvidia_megakernel=False
        )


def load_selected_oot_megakernel(server_args: ServerArgs) -> MegakernelSpec | None:
    """Load the selected provider after checking its identity in this process."""
    selection = get_oot_megakernel_selection(server_args)
    if selection is None:
        return None

    from sglang.srt.platforms import current_platform

    if not (
        current_platform.is_out_of_tree()
        and current_platform.capabilities.supports_megakernel
        and current_platform.get_megakernel_target() == selection.target
    ):
        raise ValueError(
            f"this process cannot serve the selected megakernel {selection}"
        )

    load_plugins()
    registration = find_target_registration(
        [selection.architecture], target=selection.target
    )
    if registration != (selection.architecture, selection.path):
        raise ValueError(
            f"this process has megakernel registration {registration!r}; "
            f"the launcher selected {selection.path!r} for "
            f"{selection.architecture} on {selection.target}"
        )
    spec, reason = find_spec(
        architectures=[selection.architecture],
        isa=selection.target,
        allow_default=False,
    )
    if spec is None:
        raise ValueError(f"cannot load the selected megakernel: {reason}")
    if spec.api_version != selection.api_version:
        raise ValueError(
            f"megakernel {selection.path!r} uses API {spec.api_version}; "
            f"the launcher selected API {selection.api_version}"
        )
    return spec


def validate_oot_megakernel(server_args: ServerArgs) -> None:
    """Validate the selected spec against the final resolved launch settings."""
    spec = load_selected_oot_megakernel(server_args)
    if spec is None:
        return

    from sglang.srt.kernel_plugins.nvidia.binding import (
        megakernel_config_disable_reasons,
    )

    cfg = resolving_view(server_args)
    model_config = model_config_of(server_args)
    selection = get_oot_megakernel_selection(server_args)
    if selection.architecture not in (model_config.hf_config.architectures or ()):
        raise ValueError(f"model architecture changed after selecting {selection}")
    reasons = megakernel_config_disable_reasons(
        spec=spec, cfg=cfg, hf_config=model_config.hf_config, model_config=model_config
    )
    if reasons:
        raise ValueError(
            f"megakernel {spec.name} cannot serve this setup:\n  - "
            + "\n  - ".join(reasons)
        )


@dataclass
class _InstalledMegakernel:
    built: BuiltMegakernel
    dispatch: Callable
    pool_layout: Any
    prepared: bool = False


def _installed(model) -> _InstalledMegakernel | None:
    return getattr(model, "_oot_megakernel", None)


def oot_built_megakernel(model) -> BuiltMegakernel | None:
    installed = _installed(model)
    return None if installed is None else installed.built


def oot_megakernel_dispatch(model) -> Callable | None:
    installed = _installed(model)
    return None if installed is None else installed.dispatch


def oot_megakernel_pool_layout(model) -> Any:
    installed = _installed(model)
    return None if installed is None else installed.pool_layout


def _gather_tp(value: Any) -> list[Any]:
    group = get_parallel().tp_group
    if group.world_size == 1:
        return [value]
    values = [None] * group.world_size
    dist.all_gather_object(values, value, group=group.cpu_group)
    return values


_T = TypeVar("_T")


def _local_phase(name: str, run: Callable[[], _T]) -> _T:
    """Report rank-local failures before peers enter another bootstrap phase."""
    try:
        result = run()
    except Exception as error:
        local_error = f"{type(error).__name__}: {error}"
    else:
        local_error = None
    errors = [
        f"rank {rank}: {error}"
        for rank, error in enumerate(_gather_tp(local_error))
        if error is not None
    ]
    if errors:
        raise ValueError(f"out-of-tree megakernel {name} failed: " + "; ".join(errors))
    return result


def install_oot_megakernel(model_runner: ModelRunner) -> None:
    """Build the selected provider after weights are loaded, before KV sizing."""
    selection = get_oot_megakernel_selection(model_runner.server_args)
    if selection is None or model_runner.is_draft_worker:
        return

    from sglang.srt.kernel_plugins.nvidia.binding import deadline_s
    from sglang.srt.platforms import current_platform

    model = model_runner.model
    hf_config = model_runner.model_config.hf_config

    def preflight():
        spec = load_selected_oot_megakernel(model_runner.server_args)
        if selection.architecture not in (hf_config.architectures or ()):
            raise ValueError(f"loaded model does not match the selected {selection}")
        if (
            spec.binding
            or spec.tensor_verify_build
            or spec.prepare
            or spec.prefill_graph
        ):
            raise ValueError("the selected provider uses CUDA-specific binding entries")
        if spec.requires.resident_blocks:
            raise ValueError("the selected provider requires a CUDA SM residency check")
        if spec.model_disable_reasons is not None:
            reasons = spec.model_disable_reasons(model, hf_config)
            if reasons:
                raise ValueError("; ".join(reasons))
        return spec

    spec = _local_phase("model validation", preflight)

    def build():
        tokens = model_runner.decode_num_tokens_per_req()
        context = BuildContext(
            widths=tuple(range(tokens, spec.max_rows + 1, tokens)),
            tokens_per_request=tokens,
            deadline_s=deadline_s(get_device().watchdog_timeout),
        )
        built = spec.build(model, hf_config, context)
        if spec.max_rows > 0 and not built.widths:
            raise ValueError(f"{spec.name} built no widths from {context.widths}")
        layout = getattr(built, "pool_layout", None)
        if layout is not None:
            if model_runner.spec_algorithm.is_speculative():
                raise ValueError(
                    "a custom megakernel pool layout cannot share pools with a speculative draft"
                )
            current_platform.activate_megakernel_pool_layout(layout)
        dispatch = current_platform.install_megakernel_dispatch(
            model=model, built=built, binding=context.binding
        )
        if not callable(dispatch):
            raise TypeError("the platform must return a whole-step megakernel callable")
        return _InstalledMegakernel(built, dispatch, layout)

    installed = _local_phase("build", build)
    layout = installed.pool_layout
    signature = (
        selection,
        tuple(installed.built.widths),
        type(layout).__module__,
        type(layout).__qualname__,
    )
    if any(other != signature for other in _gather_tp(signature)):
        raise ValueError(
            f"ranks disagree on the selected megakernel or its build: {signature}"
        )
    model.__dict__["_oot_megakernel"] = installed
    model._derived_weight_cache_error = (
        f"Online weight updates are not supported with {spec.name}"
    )


def prepare_oot_megakernel(model_runner: ModelRunner) -> None:
    """Prepare the installed delegate once the pools and backends exist."""
    installed = _installed(model_runner.model)
    if installed is None or installed.prepared:
        return
    prepare = getattr(installed.dispatch, "prepare", None)
    if prepare is not None:
        prepare(model_runner)
    installed.prepared = True
