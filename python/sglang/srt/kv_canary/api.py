from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.distributed.parallel_state import get_pp_group, get_tp_group
from sglang.srt.kv_canary.capacities import CanaryLaunchCapacities
from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.pool_patch.api import attach_canary_buffers
from sglang.srt.kv_canary.pool_patch.wrap_method import wrap_method
from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner
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
    """Build and install canary for a ModelRunner. Returns the CanaryRunner, or None when
    config.mode == "off". The caller is responsible for assigning the return value onto the
    ModelRunner (``model_runner.canary_runner = install_canary(...)``).

    Steps when enabled:

    1. Build CanaryConfig from server_args + env vars.
    2. attach_canary_buffers on model_runner.token_to_kv_pool.
    3. Build CanaryEndpoint tuple.
    4. Allocate CanaryDeviceState (violation log, counters, pump bufs).
    5. Construct CanaryRunner (which also allocates static per-forward PlanInput buffers and,
       when ``token_oracle_manager`` is provided, wires it into the per-forward input-check path
       so expected_input_* tensors are filled from the same oracle that drives sampling).
    6. Monkeypatch the model nn.Module's ``.forward`` to bracket the original with
       ``canary_runner.launch_head_kernels(forward_batch)`` +
       ``canary_runner.launch_tail_kernels(forward_batch)``. These two calls run kernel launches
       only — they execute inside cuda graph capture region and therefore get captured into the
       graph, auto-replaying every step.

    The host-side hooks are exposed as a single context manager
    ``canary_runner.with_forward_pass(forward_batch)``. ``ModelRunner.forward`` wraps its
    ``_forward_raw(...)`` call with that context (falling back to contextlib.nullcontext when
    no canary is installed).
    """
    config = CanaryConfig.from_env(server_args)
    if config.mode == "off":
        return None

    device = torch.device(model_runner.device)
    buffer_groups = attach_canary_buffers(
        pool=model_runner.token_to_kv_pool,
        config=config,
        device=device,
    )
    runner = CanaryRunner(
        config=config,
        buffer_groups=buffer_groups,
        device=device,
        tp_group=get_tp_group(),
        pp_group=get_pp_group(),
        req_to_token_pool=model_runner.req_to_token_pool,
        radix_cache=None,
        launch_capacities=CanaryLaunchCapacities.from_args(
            server_args=model_runner.server_args,
            req_to_token_pool_size=model_runner.req_to_token_pool.size,
            max_seq_len_per_req=model_runner.req_to_token_pool.req_to_token.shape[1],
            pool_slot_count=model_runner.max_total_num_tokens,
        ),
        swa_window_size=model_runner.sliding_window_size or 0,
        token_oracle_manager=token_oracle_manager,
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
