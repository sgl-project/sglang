from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_cache_canary.api import (
    attach,
    get_runner,
    maybe_perturb_req_to_token,
    run_head,
    run_tail,
)
from sglang.srt.kv_cache_canary.config import CanaryConfig

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_FORWARD_PATCHED_ATTR = "_kv_cache_canary_forward_patched"


def install_on_model_runner(
    *,
    model_runner: "ModelRunner",
    mode: Optional[str],
) -> None:
    """Attach the canary to the model runner's pool and wrap its forward method.

    Must be called AFTER ``init_memory_pool`` and BEFORE ``init_device_graphs``.
    Idempotent: a second call on the same ``model_runner`` is a no-op.
    """
    config = CanaryConfig.from_server_args(mode)
    if not config.enabled:
        return

    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    pool = getattr(model_runner, "token_to_kv_pool", None)
    if not isinstance(pool, MHATokenToKVPool):
        logger.warning(
            "kv-canary v1 only supports MHATokenToKVPool; got %s. Skipping.",
            type(pool).__name__ if pool is not None else None,
        )
        return

    if getattr(model_runner, _FORWARD_PATCHED_ATTR, False):
        return

    device = torch.device(model_runner.device)
    runner = attach(
        pool=pool,
        config=config,
        req_to_token_pool=model_runner.req_to_token_pool,
        device=device,
    )
    if runner is None:
        return

    _patch_forward(model_runner=model_runner)
    setattr(model_runner, _FORWARD_PATCHED_ATTR, True)


def _patch_forward(*, model_runner: "ModelRunner") -> None:
    original_forward = model_runner.forward

    def patched_forward(forward_batch, *args, **kwargs):
        pool = model_runner.token_to_kv_pool
        runner = get_runner(pool)
        if runner is None or not runner.config.enabled:
            return original_forward(forward_batch, *args, **kwargs)

        maybe_perturb_req_to_token(
            runner=runner,
            req_to_token_pool=model_runner.req_to_token_pool,
        )
        plan = run_head(runner=runner, forward_batch=forward_batch)
        output = original_forward(forward_batch, *args, **kwargs)
        run_tail(runner=runner, forward_batch=forward_batch, plan=plan)
        return output

    model_runner.forward = patched_forward
