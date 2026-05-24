from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.distributed.parallel_state import get_pp_group, get_tp_group
from sglang.srt.kv_canary.capacities import CanaryLaunchCapacities
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.perturb.config import PerturbConfig
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

    perturb_config = PerturbConfig.from_env()
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
    launch_capacities = CanaryLaunchCapacities.from_args(
        server_args=model_runner.server_args,
        req_to_token_pool_size=model_runner.req_to_token_pool.size,
        max_seq_len_per_req=model_runner.req_to_token_pool.req_to_token.shape[1],
        pool_slot_count=model_runner.max_total_num_tokens,
    )
    swa_window_size = model_runner.sliding_window_size or 0
    runner = CanaryRunner(
        config=config,
        perturb_config=perturb_config,
        buffer_groups=buffer_groups,
        device=device,
        tp_group=get_tp_group(),
        pp_group=get_pp_group(),
        req_to_token_pool=model_runner.req_to_token_pool,
        launch_capacities=launch_capacities,
        swa_window_size=swa_window_size,
        token_oracle_manager=token_oracle_manager,
        swa_allocator=swa_allocator,
    )

    _patch_model_forward(model_runner=model_runner, runner=runner)

    # Single-line summary of every knob that controls canary behavior at boot time.
    # Disaggregation mode is included so PD logs are unambiguous about which side this is.
    logger.info(
        "install_canary: disaggregation_mode=%s config=%s perturb_config=%s "
        "launch_capacities=%s n_buffer_groups=%d buffer_group_kinds=%s "
        "swa_window_size=%d",
        server_args.disaggregation_mode,
        config,
        perturb_config,
        launch_capacities,
        len(buffer_groups),
        [g.kind.name for g in buffer_groups],
        swa_window_size,
    )
    return runner


def get_canary_runner(model_runner: "ModelRunner") -> Optional[CanaryRunner]:
    """Return the runner attached to this ModelRunner, or None if canary was not installed."""
    return model_runner.canary_runner


def _patch_model_forward(*, model_runner: "ModelRunner", runner: CanaryRunner) -> None:
    # [PP-DIAG] expose rank info so wrapper prints identify which PP/TP rank is calling
    try:
        from sglang.srt.distributed.parallel_state import get_pp_group, get_tp_group

        _pp_rank = get_pp_group().rank_in_group if get_pp_group() is not None else -1
        _tp_rank = get_tp_group().rank_in_group if get_tp_group() is not None else -1
    except Exception:
        _pp_rank = -2
        _tp_rank = -2
    logger.warning(
        "[PP-DIAG] install_canary _patch_model_forward fired: pp_rank=%d tp_rank=%d",
        _pp_rank,
        _tp_rank,
    )

    _call_count = {"n": 0, "head": 0, "tail": 0}

    def _with_canary_bracketing(original: Callable, *args: Any, **kwargs: Any) -> Any:
        forward_batch = _extract_forward_batch(args, kwargs)
        assert (
            forward_batch is not None
        ), "kv-canary: patched model.forward called without a ForwardBatch"

        _call_count["n"] += 1
        if _call_count["n"] <= 3 or _call_count["n"] % 200 == 0:
            logger.warning(
                "[PP-DIAG] wrapper-entry pp=%d tp=%d call#%d before launch_head_kernels",
                _pp_rank,
                _tp_rank,
                _call_count["n"],
            )

        runner.launch_head_kernels(forward_batch)
        _call_count["head"] += 1

        if _call_count["n"] <= 3 or _call_count["n"] % 200 == 0:
            logger.warning(
                "[PP-DIAG] wrapper-mid pp=%d tp=%d call#%d head_done=%d",
                _pp_rank,
                _tp_rank,
                _call_count["n"],
                _call_count["head"],
            )

        output = original(*args, **kwargs)
        runner.launch_tail_kernels(forward_batch)
        _call_count["tail"] += 1

        if _call_count["n"] <= 3 or _call_count["n"] % 200 == 0:
            logger.warning(
                "[PP-DIAG] wrapper-exit pp=%d tp=%d call#%d head=%d tail=%d",
                _pp_rank,
                _tp_rank,
                _call_count["n"],
                _call_count["head"],
                _call_count["tail"],
            )

        return output

    wrap_method(model_runner.model, "forward", wrapper=_with_canary_bracketing)


def _extract_forward_batch(args, kwargs) -> Optional[ForwardBatch]:
    if "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], ForwardBatch):
        return kwargs["forward_batch"]
    for arg in args:
        if isinstance(arg, ForwardBatch):
            return arg
    return None
