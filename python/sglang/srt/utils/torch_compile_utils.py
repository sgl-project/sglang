"""Torch-compile configuration helpers for per-op local compilation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, ClassVar, Dict, Optional

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.utils.multi_platform import MultiPlatformOp


@dataclass
class CompileConfig:
    """Per-op torch.compile configuration.

    Fields set to ``None`` inherit from the next level in the resolution
    chain: server-arg override > class default > global default.
    """

    mode: Optional[str] = None
    options: Optional[dict] = None


def get_default_compile_options() -> dict:
    options: dict = {}
    if hasattr(torch._inductor.config, "triton"):
        triton_cfg = torch._inductor.config.triton
        if hasattr(triton_cfg, "enable_pdl"):
            options["triton.enable_pdl"] = True
    return options


def get_default_compile_mode() -> str:
    return os.environ.get("SGLANG_TORCH_COMPILE_MODE", "default")


def _resolve_base_compile_config(obj: object) -> tuple[str, dict]:
    """Resolve class-level + global default config for any object with a ``compile_config`` ClassVar."""
    mode = get_default_compile_mode()
    options = get_default_compile_options()

    cls_config = getattr(type(obj), "compile_config", None)
    if cls_config is not None:
        if cls_config.mode is not None:
            mode = cls_config.mode
        if cls_config.options is not None:
            options = cls_config.options

    return mode, options


def _apply_override(mode: str, options: dict, ovr: CompileConfig) -> tuple[str, dict]:
    if ovr.mode is not None:
        mode = ovr.mode
    if ovr.options is not None:
        options = ovr.options
    return mode, options


def resolve_compile_config(
    op: MultiPlatformOp,
    overrides: Optional[Dict[str, CompileConfig]] = None,
) -> CompileConfig:
    """Resolve compile config for *op*: server-arg override > class default > global default."""
    mode, options = _resolve_base_compile_config(op)

    if overrides:
        for cls in type(op).__mro__:
            if cls.__name__ in overrides:
                mode, options = _apply_override(mode, options, overrides[cls.__name__])
                break

    return CompileConfig(mode=mode, options=options)


def resolve_region_compile_config(
    mixin: object,
    region_name: str,
    overrides: Optional[Dict[str, CompileConfig]] = None,
) -> CompileConfig:
    """Resolve compile config for a compilable region.

    Resolution: server-arg override (by region name) > class compile_config > global default.
    """
    mode, options = _resolve_base_compile_config(mixin)

    if overrides and region_name in overrides:
        mode, options = _apply_override(mode, options, overrides[region_name])

    return CompileConfig(mode=mode, options=options)


def parse_compile_op_config(
    raw: Optional[str],
) -> Optional[Dict[str, CompileConfig]]:
    """Parse ``--torch-compile-op-config`` JSON into a class-name dict."""
    if not raw:
        return None
    parsed = json.loads(raw)
    return {
        name: CompileConfig(
            mode=cfg.get("mode"),
            options=cfg.get("options"),
        )
        for name, cfg in parsed.items()
    }


class CompilableRegionMixin:
    """Mixin for nn.Modules that expose named sub-regions for torch.compile
    under --torch-compile-scope local.

    Subclasses override ``get_compilable_regions`` to return a mapping from
    region names (used in --torch-compile-override-layers) to the method name
    that should be compiled.  The ``_to_torch`` walk in cuda_graph_runner
    discovers these regions alongside ``MultiPlatformOp`` handling.
    """

    compile_config: ClassVar[CompileConfig] = CompileConfig()

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

        Override in subclasses to customise ``torch.compile`` kwargs per
        region (e.g. force ``dynamic=None``).
        """
        return torch.compile(
            getattr(self, method_name),
            mode=compile_mode,
            options=compile_options,
            dynamic=compile_dynamic,
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
            self._compiled_region_originals: dict[
                str, tuple[str, object, bool]
            ] = {}
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
