# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.srt.utils.common import get_compiler_backend


def maybe_enable_inductor_compute_comm_overlap() -> None:
    try:
        import torch._inductor.config as _inductor_cfg

        _inductor_cfg.reorder_for_compute_comm_overlap = True
    except ImportError:
        pass


def build_torch_compile_kwargs(*, mode: str | None) -> dict[str, object]:
    compile_kwargs: dict[str, object] = {"fullgraph": False, "dynamic": None}
    if current_platform.is_npu():
        compile_kwargs["backend"] = get_compiler_backend()
        compile_kwargs["dynamic"] = False
    elif mode is not None:
        compile_kwargs["mode"] = mode
    return compile_kwargs


def resolve_torch_compile_mode(
    *env_names: str,
    config: object | None = None,
    default: str,
) -> str:
    for env_name in env_names:
        mode = os.environ.get(env_name)
        if mode:
            return mode
    mode = getattr(config, "torch_compile_mode", None)
    if mode:
        return mode
    return default


@dataclass
class CompiledModuleRegistry:
    module_ids: set[int] = field(default_factory=set)

    def is_compiled(self, module: nn.Module) -> bool:
        return id(module) in self.module_ids

    def compile_once(
        self,
        module: nn.Module,
        *,
        compile_kwargs: dict[str, object],
    ) -> bool:
        module_id = id(module)
        if module_id in self.module_ids:
            return False
        module.compile(**compile_kwargs)
        self.module_ids.add(module_id)
        return True


class CallableModule(nn.Module):
    """Module wrapper for compiling non-forward callables with module.compile"""

    def __init__(self, fn: Callable[..., Any]) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


@dataclass
class ActiveTargetCompiledCallable:
    """Cache one compiled callable module for the currently active target object"""

    target_id: int | None = None
    compiled_module: CallableModule | None = None

    def get_or_compile(
        self,
        target: object,
        fn: Callable[..., Any],
        *,
        compile_kwargs: dict[str, object],
    ) -> Callable[..., Any]:
        target_id = id(target)
        if self.target_id == target_id and self.compiled_module is not None:
            return self.compiled_module

        module = CallableModule(fn)
        module.compile(**compile_kwargs)
        self.target_id = target_id
        self.compiled_module = module
        return module
