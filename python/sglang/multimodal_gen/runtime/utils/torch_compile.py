# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch.nn as nn


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
