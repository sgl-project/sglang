from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_cache_canary.api import (
    attach,
    get_runner,
    install_req_to_token_pool_free_hook,
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
    """Attach the canary to the model runner's pool and wire its hooks.

    Must be called AFTER ``init_memory_pool`` and BEFORE ``init_device_graphs``.
    Idempotent: a second call on the same ``model_runner`` is a no-op.

    The canary patches ``model_runner.model.forward`` (the bound method).
    SGLang calls ``self.model.forward(...)`` directly (not ``self.model(...)``)
    and ``cuda_graph_runner.patch_model`` yields ``model.forward`` for capture
    too — so patching the bound method is the single point that covers both
    the eager and the captured-into-cuda-graph paths.
    """
    config = CanaryConfig.from_server_args(mode)
    if not config.enabled:
        return

    if _server_args_use_disaggregation(model_runner):
        logger.warning(
            "kv-canary: PD disaggregation is not yet supported by v1; skipping install."
        )
        return

    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    pool = model_runner.token_to_kv_pool
    if not isinstance(pool, MHATokenToKVPool):
        logger.warning(
            "kv-canary v1 only supports MHATokenToKVPool; got %s. Skipping.",
            type(pool).__name__,
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

    install_req_to_token_pool_free_hook(
        runner=runner,
        req_to_token_pool=model_runner.req_to_token_pool,
    )
    _patch_model_forward(model_runner=model_runner)
    setattr(model_runner, _FORWARD_PATCHED_ATTR, True)


def _server_args_use_disaggregation(model_runner: "ModelRunner") -> bool:
    server_args = getattr(model_runner, "server_args", None)
    if server_args is None:
        return False
    mode = getattr(server_args, "disaggregation_mode", None)
    if mode is None or mode == "null":
        return False
    return True


def _patch_model_forward(*, model_runner: "ModelRunner") -> None:
    """Wrap ``model_runner.model.forward`` to run the canary kernel pair.

    The wrapped function:
      1. Pulls the canary runner off the pool (skip path if disabled);
      2. Optionally perturbs ``req_to_token_pool`` for the self-test;
      3. Builds a ``BatchPlan`` and launches the head kernel BEFORE the real
         model forward (on the current stream — same stream cuda graph capture
         records into);
      4. Runs the real model forward;
      5. Launches the tail kernel and triggers end-of-forward (counter
         polling + unconditional cross-rank allreduce of the error flag).
    """
    model = model_runner.model
    original_forward = model.forward

    def patched_model_forward(*args, **kwargs):
        forward_batch = _extract_forward_batch(args, kwargs)
        pool = model_runner.token_to_kv_pool
        runner = get_runner(pool)
        if forward_batch is None or runner is None or not runner.config.enabled:
            return original_forward(*args, **kwargs)

        # During cuda graph CAPTURE, the expected_* host-derived tensors would
        # be frozen into the graph and replays would re-use stale values.
        # v1 limitation: skip the canary kernel inside captured regions; the
        # eager (extend / prefill / out-of-graph decode) path still runs it,
        # and the §5 health check distinguishes "kernel never ran" from
        # "kernel ran only on eager".
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return original_forward(*args, **kwargs)

        rank = getattr(model_runner, "tp_rank", 0) or 0
        maybe_perturb_req_to_token(
            runner=runner,
            req_to_token_pool=model_runner.req_to_token_pool,
            rank=rank,
        )
        plan = run_head(runner=runner, forward_batch=forward_batch)
        output = original_forward(*args, **kwargs)
        run_tail(runner=runner, forward_batch=forward_batch, plan=plan)
        return output

    model.forward = patched_model_forward


def _extract_forward_batch(args, kwargs):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    if "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], ForwardBatch):
        return kwargs["forward_batch"]
    for arg in args:
        if isinstance(arg, ForwardBatch):
            return arg
    return None
