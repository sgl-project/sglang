"""Environment fields owned by the optional AlphaMoE integration."""

from __future__ import annotations

import os

from sglang.srt.environ import EnvBool, EnvField, EnvStr, _exportable_value


class _AlphaMoEEnvs:
    def __init__(self):
        # Restricted Qwen TP4 routing A/B: Triton experts and AlphaMoE TopK.
        self.SGLANG_FLASHINFER_ALPHAMOE_ROUTER_ONLY = EnvBool(False)
        self.SGLANG_FLASHINFER_ALPHAMOE_TRACE_SHAPES = EnvBool(False)
        self.SGLANG_FLASHINFER_ALPHAMOE_TRACE_ARM_FILE = EnvStr("")
        # EnvField locks class-level descriptor naming after environ imports.
        # These fresh instance fields bind their names without changing that
        # global lock; parsing, defaults, overrides, and dynamic get are shared.
        for name, field in vars(self).items():
            field.name = name


alphamoe_envs = _AlphaMoEEnvs()


def exportable_alphamoe_env_vars() -> dict[str, str]:
    return {
        field.name: _exportable_value(os.environ[field.name])
        for field in sorted(
            (
                value
                for value in vars(alphamoe_envs).values()
                if isinstance(value, EnvField)
            ),
            key=lambda field: field.name,
        )
        if not field.secret and field.name in os.environ
    }
