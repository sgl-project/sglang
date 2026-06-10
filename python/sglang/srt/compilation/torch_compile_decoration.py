"""torch.compile decoration helpers used by the decode-Full path under
``--enable-torch-compile``.

``patch_model`` wraps the model forward with ``torch.compile`` for batch
sizes that fall in the compile bucket list and returns the raw forward
otherwise. ``set_torch_compile_config`` flips the inductor/dynamo config
flags expected by that path.

Note: the prefill-tc_piecewise path (``TcPiecewiseCudaGraphBackend``) does NOT
use ``patch_model`` — it goes through ``compilation/compile.py``'s
``install_torch_compiled``. ``_to_torch`` here is duplicated by
tc_piecewise's local ``_toggle_multi_platform_ops``; the duplication is kept
because the two paths have different lifecycle requirements.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch

from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.utils import get_bool_env_var, is_hip
from sglang.srt.utils.patch_torch import monkey_patch_torch_compile

_is_hip = is_hip()


def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int) -> None:
    for sub in model._modules.values():
        if isinstance(sub, MultiPlatformOp):
            if reverse:
                sub.leave_torch_compile()
            else:
                sub.enter_torch_compile(num_tokens=num_tokens)
        if isinstance(sub, torch.nn.Module):
            _to_torch(sub, reverse, num_tokens)


@contextmanager
def patch_model(
    model: torch.nn.Module,
    enable_compile: bool,
    num_tokens: int,
    tp_group: GroupCoordinator,
):
    """Patch the model to make it compatible with torch.compile."""
    backup_ca_comm = None

    try:
        if enable_compile:
            _to_torch(model, reverse=False, num_tokens=num_tokens)
            backup_ca_comm = tp_group.ca_comm
            yield torch.compile(
                torch.no_grad()(model.forward),
                mode=os.environ.get(
                    "SGLANG_TORCH_COMPILE_MODE", "max-autotune-no-cudagraphs"
                ),
                dynamic=_is_hip and get_bool_env_var("SGLANG_TORCH_DYNAMIC_SHAPE"),
            )
        else:
            yield model.forward
    finally:
        if enable_compile:
            _to_torch(model, reverse=True, num_tokens=num_tokens)
            tp_group.ca_comm = backup_ca_comm


def set_torch_compile_config() -> None:
    import torch._dynamo.config
    import torch._inductor.config

    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.fx_graph_cache = True

    torch._dynamo.config.accumulated_cache_size_limit = 1024
    if hasattr(torch._dynamo.config, "cache_size_limit"):
        torch._dynamo.config.cache_size_limit = 1024

    monkey_patch_torch_compile()
