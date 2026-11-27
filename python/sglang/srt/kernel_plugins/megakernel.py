# SPDX-License-Identifier: Apache-2.0
"""Launcher and child-process selection of out-of-tree megakernel providers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import msgspec

from sglang.kernels.plugin_kernels.megakernel.spec import (
    MEGAKERNEL_API_VERSION,
    MegakernelSpec,
    find_spec,
    find_target_registration,
)
from sglang.srt.arg_groups.model_override_base import model_config_of, resolving_view
from sglang.srt.arg_groups.overrides import declare_resolution
from sglang.srt.plugins import load_plugins

if TYPE_CHECKING:
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


def check_oot_megakernel_worker(server_args: ServerArgs) -> None:
    """Keep a selected provider from reaching the stock model worker."""
    if load_selected_oot_megakernel(server_args) is not None:
        raise ValueError(
            "this SGLang build can select the out-of-tree megakernel but "
            "does not yet support installing it in a model worker"
        )
