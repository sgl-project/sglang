"""Torch-compile configuration helpers for per-op local compilation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, ClassVar, Dict, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp

# Sentinel for "not set" — distinct from ``None`` which is a valid
# ``torch.compile(dynamic=None)`` value meaning "let Dynamo decide".
_UNSET = object()


@dataclass
class CompileConfig:
    """Per-op torch.compile configuration.

    Fields set to ``None`` inherit from the next level in the resolution
    chain: server-arg override > class default > global default.

    ``dynamic`` uses :data:`_UNSET` (not ``None``) to signal "inherit",
    because ``None`` is a meaningful ``torch.compile`` value.
    """

    mode: Optional[str] = None
    options: Optional[dict] = None
    dynamic: object = _UNSET


def get_default_compile_options() -> dict:
    options: dict = {}
    if hasattr(torch._inductor.config, "triton"):
        triton_cfg = torch._inductor.config.triton
        if hasattr(triton_cfg, "enable_pdl"):
            options["triton.enable_pdl"] = True
    return options


def get_default_compile_mode() -> str:
    return os.environ.get("SGLANG_TORCH_COMPILE_MODE", "default")


def merge_mode_options(
    mode: Optional[str] = None, options: Optional[dict] = None
) -> dict:
    """Merge a ``torch.compile`` *mode*'s defaults with explicit *options*.

    ``torch.compile`` raises if both ``mode`` and ``options`` are given.
    This helper expands the mode into its option defaults (via
    ``torch._inductor.list_mode_options``) and layers explicit options on
    top, returning a single merged dict suitable for ``options=``.
    """
    from torch._inductor import list_mode_options

    base = list_mode_options(mode or "default")
    if options:
        base.update(options)
    return base


def compile_callable(
    fn: Callable,
    compile_mode: Optional[str] = None,
    compile_options: Optional[dict] = None,
    compile_dynamic: object = False,
) -> Callable:
    """Compile a callable using SGLang's local torch.compile option policy."""
    return torch.compile(
        fn,
        options=merge_mode_options(compile_mode, compile_options),
        dynamic=compile_dynamic,
    )


def _resolve_base_compile_config(obj: object) -> tuple[str, dict, object]:
    """Resolve class-level + global default config for any object with a ``compile_config`` ClassVar."""
    mode = get_default_compile_mode()
    options = get_default_compile_options()
    dynamic = _UNSET

    cls_config = getattr(type(obj), "compile_config", None)
    if cls_config is not None:
        if cls_config.mode is not None:
            mode = cls_config.mode
        if cls_config.options is not None:
            options = cls_config.options
        if cls_config.dynamic is not _UNSET:
            dynamic = cls_config.dynamic

    return mode, options, dynamic


def _apply_override(
    mode: str, options: dict, dynamic: object, ovr: CompileConfig
) -> tuple[str, dict, object]:
    if ovr.mode is not None:
        mode = ovr.mode
    if ovr.options is not None:
        options = ovr.options
    if ovr.dynamic is not _UNSET:
        dynamic = ovr.dynamic
    return mode, options, dynamic


def resolve_compile_config(
    op: MultiPlatformOp,
    overrides: Optional[Dict[str, CompileConfig]] = None,
) -> CompileConfig:
    """Resolve compile config for *op*: server-arg override > class default > global default."""
    mode, options, dynamic = _resolve_base_compile_config(op)

    if overrides:
        for cls in type(op).__mro__:
            if cls.__name__ in overrides:
                mode, options, dynamic = _apply_override(
                    mode, options, dynamic, overrides[cls.__name__]
                )
                break

    return CompileConfig(mode=mode, options=options, dynamic=dynamic)


def resolve_region_compile_config(
    mixin: object,
    region_name: str,
    overrides: Optional[Dict[str, CompileConfig]] = None,
) -> CompileConfig:
    """Resolve compile config for a compilable region.

    Resolution: server-arg override (by region name) > class compile_config > global default.
    """
    mode, options, dynamic = _resolve_base_compile_config(mixin)

    if overrides and region_name in overrides:
        mode, options, dynamic = _apply_override(
            mode, options, dynamic, overrides[region_name]
        )

    return CompileConfig(mode=mode, options=options, dynamic=dynamic)


def parse_compile_op_config(
    raw: Optional[str],
) -> Optional[Dict[str, CompileConfig]]:
    """Parse ``--torch-compile-op-config`` JSON into a class-name dict."""
    if not raw:
        return None
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("expected a JSON object mapping names to config objects")

    result: Dict[str, CompileConfig] = {}
    allowed_keys = {"mode", "options", "dynamic"}
    for name, cfg in parsed.items():
        if not isinstance(name, str) or not name:
            raise ValueError("config names must be non-empty strings")
        if not isinstance(cfg, dict):
            raise ValueError(f"config for {name!r} must be a JSON object")

        unknown_keys = set(cfg) - allowed_keys
        if unknown_keys:
            raise ValueError(
                f"config for {name!r} has unknown keys: {sorted(unknown_keys)}"
            )

        mode = cfg.get("mode")
        if mode is not None and not isinstance(mode, str):
            raise ValueError(f"config for {name!r} has non-string mode")

        options = cfg.get("options")
        if options is not None and not isinstance(options, dict):
            raise ValueError(f"config for {name!r} has non-object options")

        dynamic = cfg["dynamic"] if "dynamic" in cfg else _UNSET
        if (
            dynamic is not _UNSET
            and dynamic is not True
            and dynamic is not False
            and dynamic is not None
        ):
            raise ValueError(
                f"config for {name!r} dynamic must be true, false, or null"
            )

        result[name] = CompileConfig(
            mode=mode,
            options=options,
            dynamic=dynamic,
        )
    return result


class CompilableRegionMixin:
    """Mixin for nn.Modules that expose named sub-regions for torch.compile
    under --torch-compile-scope local.

    Subclasses override ``get_compilable_regions`` to return a mapping from
    region names (used in --torch-compile-override-regions) to the method name
    that should be compiled.  The ``_to_torch`` walk in cuda_graph_runner
    discovers these regions alongside ``MultiPlatformOp`` handling.
    """

    compile_config: ClassVar[CompileConfig] = CompileConfig()

    # Model-author default for the ``dynamic`` kwarg passed to
    # ``torch.compile`` for each region.  This captures properties inherent
    # to the region's code (e.g. QKNorm must be static-shape for Inductor
    # to optimize) rather than user/operator preferences.
    # Takes precedence over the global ``compile_dynamic`` and
    # ``CompileConfig.dynamic`` resolved from ``--torch-compile-op-config``.
    # Example: ``_REGION_DYNAMIC = {"AlphaRegion": False, "BetaRegion": None}``
    _REGION_DYNAMIC: ClassVar[Dict[str, object]] = {}

    def get_compilable_regions(self) -> dict[str, str]:
        """Return {region_name: method_name} for compilable sub-functions."""
        return {}

    def _get_region_compile_method(
        self,
        region_name: str,
        method_name: str,
        compile_mode: Optional[str] = None,
        compile_options: Optional[dict] = None,
        compile_dynamic: bool = False,
    ) -> Callable:
        """Return the compiled callable for *method_name*.

        Uses ``_REGION_DYNAMIC[region_name]`` when present, falling back to
        *compile_dynamic*.  Override in subclasses for further customisation.
        """
        compile_dynamic = self._REGION_DYNAMIC.get(region_name, compile_dynamic)

        return compile_callable(
            getattr(self, method_name),
            compile_mode=compile_mode,
            compile_options=compile_options,
            compile_dynamic=compile_dynamic,
        )

    def enter_region_compile(
        self,
        region_name: str,
        compile_mode: Optional[str] = None,
        compile_options: Optional[dict] = None,
        compile_dynamic: bool = False,
    ):
        """Replace the named method with its ``torch.compile``d version."""
        if hasattr(self, "_compiled_region_originals") and (
            region_name in self._compiled_region_originals
        ):
            return  # already compiled

        regions = self.get_compilable_regions()
        method_name = regions[region_name]
        was_instance_attr = method_name in self.__dict__
        original = self.__dict__.get(method_name) if was_instance_attr else None
        compiled = self._get_region_compile_method(
            region_name,
            method_name,
            compile_mode=compile_mode,
            compile_options=compile_options,
            compile_dynamic=compile_dynamic,
        )
        if not hasattr(self, "_compiled_region_originals"):
            self._compiled_region_originals: dict[str, tuple[str, object, bool]] = {}
        self._compiled_region_originals[region_name] = (
            method_name,
            original,
            was_instance_attr,
        )
        setattr(self, method_name, compiled)

    def leave_region_compile(self, region_name: str):
        """Restore the original (un-compiled) method."""
        if not hasattr(self, "_compiled_region_originals"):
            return
        entry = self._compiled_region_originals.pop(region_name, None)
        if entry is None:
            return
        method_name, original, was_instance_attr = entry
        if was_instance_attr:
            setattr(self, method_name, original)
        elif method_name in self.__dict__:
            delattr(self, method_name)

    def is_region_compiled(self, region_name: str) -> bool:
        return hasattr(self, "_compiled_region_originals") and (
            region_name in self._compiled_region_originals
        )
