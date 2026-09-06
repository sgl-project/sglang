# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/platforms/__init__.py

import os
import traceback
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from importlib.metadata import EntryPoint, entry_points

from sglang.multimodal_gen import envs

# imported by other files, do not remove
from sglang.multimodal_gen.runtime.platforms.interface import (  # noqa: F401
    AttentionBackendEnum,
    Platform,
    PlatformEnum,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import resolve_obj_by_qualname

logger = init_logger(__name__)

PLATFORM_PLUGINS_GROUP = "sglang.multimodal_gen.platforms"
_BUILTIN_PLATFORM_QUALNAMES = {
    "cpu": "sglang.multimodal_gen.runtime.platforms.cpu.CpuPlatform",
    "cuda": "sglang.multimodal_gen.runtime.platforms.cuda.CudaPlatform",
    "rocm": "sglang.multimodal_gen.runtime.platforms.rocm.RocmPlatform",
    "xpu": "sglang.multimodal_gen.runtime.platforms.xpu.XpuPlatform",
    "mps": "sglang.multimodal_gen.runtime.platforms.mps.MpsPlatform",
    "npu": "sglang.multimodal_gen.runtime.platforms.npu.NPUPlatformBase",
    "musa": "sglang.multimodal_gen.runtime.platforms.musa.MusaPlatform",
}
BUILTIN_PLATFORM_NAMES = frozenset(_BUILTIN_PLATFORM_QUALNAMES)
# XPU has historically been auto-detected but was not accepted by the override.
_BUILTIN_PLATFORM_OVERRIDE_NAMES = BUILTIN_PLATFORM_NAMES - {"xpu"}


def cuda_platform_plugin() -> str | None:
    is_cuda = False

    try:
        from sglang.multimodal_gen.utils import import_pynvml

        pynvml = import_pynvml()  # type: ignore[no-untyped-call]
        pynvml.nvmlInit()
        try:
            # NOTE: Edge case: sgl_diffusion cpu build on a GPU machine.
            # Third-party pynvml can be imported in cpu build,
            # we need to check if sgl_diffusion is built with cpu too.
            # Otherwise, sgl_diffusion will always activate cuda plugin
            # on a GPU machine, even if in a cpu build.
            is_cuda = pynvml.nvmlDeviceGetCount() > 0
        finally:
            pynvml.nvmlShutdown()
    except Exception as e:
        if "nvml" not in e.__class__.__name__.lower():
            # If the error is not related to NVML, re-raise it.
            raise e

        # CUDA is supported on Jetson, but NVML may not be.
        import os

        def cuda_is_jetson() -> bool:
            return os.path.isfile("/etc/nv_tegra_release") or os.path.exists(
                "/sys/class/tegra-firmware"
            )

        if cuda_is_jetson():
            is_cuda = True
        else:
            # NVML is NVIDIA-specific. CUDA-compatible stacks (e.g. Iluvatar
            # CoreX) expose devices through torch's CUDA API without shipping
            # libnvidia-ml. Only fall back when NVML itself is unavailable;
            # a successful NVML init that reports zero devices must keep the
            # CPU-build-on-GPU-machine edge case above.
            try:
                import torch

                # ROCm also exposes devices through torch.cuda, so keep this
                # non-NVML fallback limited to non-HIP runtimes.
                is_cuda = (
                    getattr(torch.version, "hip", None) is None
                    and torch.cuda.is_available()
                    and torch.cuda.device_count() > 0
                )
                if is_cuda:
                    logger.debug("CUDA detected via torch (NVML unavailable)")
            except Exception as exc:
                logger.debug("torch CUDA detection failed: %s", exc)

    if is_cuda:
        logger.debug("CUDA is available")

    return _BUILTIN_PLATFORM_QUALNAMES["cuda"] if is_cuda else None


def mps_platform_plugin() -> str | None:
    """Detect if MPS (Metal Performance Shaders) is available on macOS."""
    is_mps = False

    try:
        import torch

        if torch.backends.mps.is_available():
            is_mps = True
            logger.debug("MPS (Metal Performance Shaders) is available")
    except Exception as e:
        logger.debug("MPS detection failed: %s", e)

    return _BUILTIN_PLATFORM_QUALNAMES["mps"] if is_mps else None


def cpu_platform_plugin() -> str:
    """Detect if CPU platform should be used."""
    # CPU is always available as a fallback
    return _BUILTIN_PLATFORM_QUALNAMES["cpu"]


def rocm_platform_plugin() -> str | None:
    is_rocm = False

    try:
        import amdsmi

        amdsmi.amdsmi_init()
        try:
            if len(amdsmi.amdsmi_get_processor_handles()) > 0:
                is_rocm = True
                logger.debug("ROCm platform is available")
        finally:
            amdsmi.amdsmi_shut_down()
    except Exception as e:
        logger.debug("ROCm platform is unavailable: %s", e)

    return _BUILTIN_PLATFORM_QUALNAMES["rocm"] if is_rocm else None


def npu_platform_plugin() -> str | None:
    is_npu = False

    try:
        import torch

        if torch.npu.is_available():
            is_npu = True
            logger.debug("NPU is available")
    except Exception as e:
        logger.debug("NPU detection failed: %s", e)
    return _BUILTIN_PLATFORM_QUALNAMES["npu"] if is_npu else None


def musa_platform_plugin() -> str | None:
    is_musa = False

    try:
        import pymtml

        pymtml.mtmlLibraryInit()
        try:
            is_musa = pymtml.mtmlLibraryCountDevice() > 0
        finally:
            pymtml.mtmlLibraryShutDown()
    except Exception as e:
        logger.debug("MUSA platform is unavailable: %s", e)

    return _BUILTIN_PLATFORM_QUALNAMES["musa"] if is_musa else None


def xpu_platform_plugin() -> str | None:
    """Detect if Intel XPU platform is available."""
    is_xpu = False

    try:
        import torch

        # Check if Intel Extension for PyTorch is available and XPU devices exist
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            device_count = torch.xpu.device_count()
            if device_count > 0:
                is_xpu = True
                logger.info(
                    "Intel XPU platform is available with %d device(s)", device_count
                )
    except Exception as e:
        logger.info("Intel XPU platform is unavailable: %s", e)

    return _BUILTIN_PLATFORM_QUALNAMES["xpu"] if is_xpu else None


builtin_platform_plugins = {
    "mps": mps_platform_plugin,
    "xpu": xpu_platform_plugin,
    "rocm": rocm_platform_plugin,
    "cuda": cuda_platform_plugin,
    "npu": npu_platform_plugin,
    "musa": musa_platform_plugin,
    "cpu": cpu_platform_plugin,
}


@dataclass(frozen=True)
class _PlatformSelection:
    qualname: str
    plugin_name: str | None = None
    distribution_name: str | None = None

    @property
    def is_external(self) -> bool:
        return self.plugin_name is not None


def _select_current_platform() -> _PlatformSelection:
    selected = envs.SGLANG_DIFFUSION_PLATFORM_OVERRIDE.strip()
    if selected:
        builtin_name = selected.lower()
        if builtin_name in BUILTIN_PLATFORM_NAMES:
            if builtin_name not in _BUILTIN_PLATFORM_OVERRIDE_NAMES:
                raise ValueError(
                    f"Unsupported SGLANG_DIFFUSION_PLATFORM_OVERRIDE={selected!r}"
                )
            return _PlatformSelection(_BUILTIN_PLATFORM_QUALNAMES[builtin_name])
        return _resolve_selected_platform(_discover_platform_plugin_entries(), selected)

    platform_selection = _resolve_automatic_platform(
        _discover_platform_plugin_entries()
    )
    if platform_selection is not None:
        return platform_selection

    for detect in builtin_platform_plugins.values():
        platform_cls_qualname = detect()
        if platform_cls_qualname is not None:
            return _PlatformSelection(platform_cls_qualname)
    raise RuntimeError("No platform plugin found. Please check your installation.")


def resolve_current_platform_cls_qualname() -> str:
    """Resolve the selected class name without mutating singleton state."""
    return _select_current_platform().qualname


def _discover_platform_plugin_entries() -> tuple[EntryPoint, ...]:
    entries = tuple(entry_points(group=PLATFORM_PLUGINS_GROUP))
    if entries:
        logger.info("Available diffusion platform plugins:")
        for entry_point in entries:
            logger.info("  - %s -> %s", entry_point.name, entry_point.value)
    return entries


def _reject_platform_names(names: Iterable[str], *, reason: str) -> None:
    # Sorted so the message does not depend on entry-point iteration order.
    offenders = sorted(names)
    if offenders:
        raise RuntimeError(f"{reason}: " + ", ".join(repr(name) for name in offenders))


def _validate_platform_entries(entries: tuple[EntryPoint, ...]) -> None:
    counts = Counter(entry_point.name for entry_point in entries)
    _reject_platform_names(
        (name for name, count in counts.items() if count > 1),
        reason="Diffusion platform entry-point names must be unique",
    )
    _reject_platform_names(
        (name for name in counts if name.lower() in BUILTIN_PLATFORM_NAMES),
        reason="Diffusion platform entry points cannot use built-in names",
    )


def _platform_selection(
    entry_point: EntryPoint, qualname: object
) -> _PlatformSelection | None:
    if qualname is None:
        return None
    # activate() is third-party, so a bad return is named rather than left to
    # surface as an AttributeError from some later attribute access.
    selected = qualname.strip() if isinstance(qualname, str) else ""
    if not selected:
        raise TypeError(
            f"Diffusion platform plugin {entry_point.name!r} must return a "
            "non-empty class qualname or None"
        )
    return _PlatformSelection(
        qualname=selected,
        plugin_name=entry_point.name,
        distribution_name=entry_point.dist.name if entry_point.dist else None,
    )


def _resolve_selected_platform(
    entries: tuple[EntryPoint, ...],
    selected: str,
) -> _PlatformSelection:
    matches = tuple(
        entry_point for entry_point in entries if entry_point.name == selected
    )
    if not matches:
        available = ", ".join(repr(entry_point.name) for entry_point in entries)
        raise ValueError(
            f"Unsupported SGLANG_DIFFUSION_PLATFORM_OVERRIDE={selected!r}; "
            "entry point not found in group "
            f"{PLATFORM_PLUGINS_GROUP!r} (available: "
            f"{available or 'none'})."
        )

    _validate_platform_entries(matches)
    logger.info(
        "Selecting platform plugin %s via SGLANG_DIFFUSION_PLATFORM_OVERRIDE",
        selected,
    )
    selection = _platform_selection(matches[0], matches[0].load()())
    if selection is None:
        raise RuntimeError(
            f"Platform plugin {selected!r} is installed but activate() "
            "returned None (hardware not available on this machine?)."
        )
    logger.info("OOT platform plugin activated: %s -> %s", selected, selection.qualname)
    return selection


def _resolve_automatic_platform(
    entries: tuple[EntryPoint, ...],
) -> _PlatformSelection | None:
    _validate_platform_entries(entries)
    activated: list[_PlatformSelection] = []
    for entry_point in entries:
        # A raising activate() propagates: silently skipping it would fall back
        # to a built-in platform and run the whole job on the wrong hardware.
        selection = _platform_selection(entry_point, entry_point.load()())
        if selection is not None:
            activated.append(selection)
            logger.info(
                "OOT platform plugin activated: %s -> %s",
                entry_point.name,
                selection.qualname,
            )

    if not activated:
        return None
    if len(activated) == 1:
        return activated[0]
    names = ", ".join(repr(selection.plugin_name) for selection in activated)
    raise RuntimeError(
        f"Multiple platform plugins activated: {names}. "
        "Set SGLANG_DIFFUSION_PLATFORM_OVERRIDE to select one."
    )


def _load_platform_class(
    qualname: str, *, external: bool | None = None
) -> type[Platform]:
    platform_cls = resolve_obj_by_qualname(qualname)
    if not isinstance(platform_cls, type) or not issubclass(platform_cls, Platform):
        raise TypeError(f"Expected a Platform subclass: {qualname}")
    if external is None:
        external = qualname not in _BUILTIN_PLATFORM_QUALNAMES.values()
    if external and platform_cls._enum is not PlatformEnum.OOT:
        raise TypeError(
            f"External diffusion platform {qualname} must set "
            "_enum = sglang.multimodal_gen.runtime.platforms.PlatformEnum.OOT"
        )
    return platform_cls


_current_platform: Platform | None = None
_current_platform_selection: _PlatformSelection | None = None
_init_trace: str = ""

_backend_init_done = False
_backend_init_error: BaseException | None = None

current_platform: Platform


def _resolve_current_platform() -> Platform:
    # Platform plugins import this module to subclass Platform, so resolution
    # must remain lazy.
    global _current_platform, _current_platform_selection, _init_trace

    if _current_platform is not None:
        return _current_platform

    selection = _select_current_platform()
    platform_cls = _load_platform_class(
        selection.qualname, external=selection.is_external
    )
    platform = platform_cls()
    if selection.is_external:
        for attribute in ("device_name", "device_type"):
            value = getattr(platform, attribute, None)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(
                    f"External diffusion platform {selection.qualname} must "
                    f"define a non-empty {attribute}"
                )

    # Publish the instance and its provenance together, only after the
    # complete external contract has passed validation.
    _current_platform_selection = selection
    _current_platform = platform
    _init_trace = "".join(traceback.format_stack())
    return platform


def get_selected_platform_dist() -> str | None:
    _resolve_current_platform()
    assert _current_platform_selection is not None
    return _current_platform_selection.distribution_name


def initialize_current_platform() -> None:
    """Run backend initialization once per process.

    Only worker entry points call this, so a launcher never marks itself
    initialized and every worker starts from clean module state.

    A failed initialization is terminal for that process: retrying arbitrary
    backend side effects can duplicate registrations and leave a worker in a
    state that reflects neither attempt.
    """
    global _backend_init_error, _backend_init_done

    if _backend_init_done:
        if _backend_init_error is not None:
            raise RuntimeError(
                "Diffusion platform backend initialization previously failed: "
                f"{_backend_init_error}"
            ) from _backend_init_error
        return

    try:
        _resolve_current_platform().init_backend()
    except BaseException as exc:
        # BaseException too: the finally below marks this attempt done, so an
        # unrecorded interrupt would let the next call report success.
        _backend_init_error = exc
        raise
    finally:
        _backend_init_done = True


def __getattr__(name: str):
    if name == "current_platform":
        return _resolve_current_platform()
    elif name in globals():
        return globals()[name]
    else:
        raise AttributeError(f"No attribute named '{name}' exists in {__name__}.")


__all__ = [
    "BUILTIN_PLATFORM_NAMES",
    "PLATFORM_PLUGINS_GROUP",
    "Platform",
    "PlatformEnum",
    "current_platform",
    "get_selected_platform_dist",
    "initialize_current_platform",
    "_init_trace",
]
