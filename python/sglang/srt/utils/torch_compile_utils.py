"""Torch-compile configuration helpers for per-op local compilation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

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


def resolve_compile_config(
    op: MultiPlatformOp,
    overrides: Optional[Dict[str, CompileConfig]] = None,
) -> CompileConfig:
    """Resolve compile config for *op*: server-arg override > class default > global default."""
    mode = get_default_compile_mode()
    options = get_default_compile_options()

    cls_config = getattr(type(op), "compile_config", None)
    if cls_config is not None:
        if cls_config.mode is not None:
            mode = cls_config.mode
        if cls_config.options is not None:
            options = cls_config.options

    if overrides:
        for cls in type(op).__mro__:
            if cls.__name__ in overrides:
                ovr = overrides[cls.__name__]
                if ovr.mode is not None:
                    mode = ovr.mode
                if ovr.options is not None:
                    options = ovr.options
                break

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
