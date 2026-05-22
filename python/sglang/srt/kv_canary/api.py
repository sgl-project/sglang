from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.distributed.parallel_state import get_pp_group, get_tp_group
from sglang.srt.kv_canary.capacities import CanaryLaunchCapacities
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.pool_patch.api import attach_canary_buffers
from sglang.srt.kv_canary.pool_patch.utils import wrap_method
from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner
from sglang.srt.mem_cache.swa_memory_pool import SWATokenToKVPoolAllocator
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.kv_canary.token_oracle.oracle_manager import TokenOracleManager
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def install_canary(
    *,
    server_args: "ServerArgs",
    model_runner: "ModelRunner",
    token_oracle_manager: Optional["TokenOracleManager"] = None,
) -> Optional[CanaryRunner]:
    config = CanaryConfig.from_env(server_args)
    if config.mode == "off":
        return None

    device = torch.device(model_runner.device)
    buffer_groups = attach_canary_buffers(
        pool=model_runner.token_to_kv_pool,
        config=config,
        device=device,
    )
    allocator = model_runner.token_to_kv_pool_allocator
    swa_allocator = (
        allocator if isinstance(allocator, SWATokenToKVPoolAllocator) else None
    )
    runner = CanaryRunner(
        config=config,
        buffer_groups=buffer_groups,
        device=device,
        tp_group=get_tp_group(),
        pp_group=get_pp_group(),
        req_to_token_pool=model_runner.req_to_token_pool,
        launch_capacities=CanaryLaunchCapacities.from_args(
            server_args=model_runner.server_args,
            req_to_token_pool_size=model_runner.req_to_token_pool.size,
            max_seq_len_per_req=model_runner.req_to_token_pool.req_to_token.shape[1],
            pool_slot_count=model_runner.max_total_num_tokens,
        ),
        swa_window_size=model_runner.sliding_window_size or 0,
        token_oracle_manager=token_oracle_manager,
        swa_allocator=swa_allocator,
    )

    _patch_model_forward(model_runner=model_runner, runner=runner)

    logger.info("install_canary: config=%s", config)
    return runner


def get_canary_runner(model_runner: "ModelRunner") -> Optional[CanaryRunner]:
    """Return the runner attached to this ModelRunner, or None if canary was not installed."""
    return model_runner.canary_runner


def _patch_model_forward(*, model_runner: "ModelRunner", runner: CanaryRunner) -> None:
    def _with_canary_bracketing(original: Callable, *args: Any, **kwargs: Any) -> Any:
        forward_batch = _extract_forward_batch(args, kwargs)
        assert (
            forward_batch is not None
        ), "kv-canary: patched model.forward called without a ForwardBatch"

        runner.launch_head_kernels(forward_batch)
        output = original(*args, **kwargs)
        runner.launch_tail_kernels(forward_batch)
        return output

    wrap_method(model_runner.model, "forward", wrapper=_with_canary_bracketing)


def _extract_forward_batch(args, kwargs) -> Optional[ForwardBatch]:
    if "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], ForwardBatch):
        return kwargs["forward_batch"]
    for arg in args:
        if isinstance(arg, ForwardBatch):
            return arg
    return None
