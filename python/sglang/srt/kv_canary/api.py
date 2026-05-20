from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Optional

import torch

from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.pool_patch.api import attach_canary_buffers
from sglang.srt.kv_canary.pool_patch.helpers import _wrap_method
from sglang.srt.kv_canary.runner.canary_runner import (
    CanaryLaunchCapacities,
    CanaryRunner,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.kv_canary.mock_model.oracle_manager import OracleSamplerHook
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY: int = 1_000_000


def install_canary(
    *,
    server_args: "ServerArgs",
    model_runner: "ModelRunner",
    oracle_sampler_hook: Optional["OracleSamplerHook"] = None,
) -> Optional[CanaryRunner]:
    """Build and install canary for a ModelRunner. Returns the CanaryRunner, or None when
    config.mode == "off". The caller is responsible for assigning the return value onto the
    ModelRunner (``model_runner.canary_runner = install_canary(...)``).

    Steps when enabled:

    1. Build CanaryConfig from server_args + env vars.
    2. attach_canary_buffers on model_runner.token_to_kv_pool.
    3. Build CanaryEndpoint tuple.
    4. Allocate CanaryDeviceState (violation log, counters, pump bufs).
    5. Construct CanaryRunner (which also allocates static per-forward PlanInput buffers).
    6. Bind ``oracle_sampler_hook`` (when provided) so the per-forward input-check path can
       fill expected_input_* tensors from the same oracle that drives sampling.
    7. Monkeypatch the model nn.Module's ``.forward`` to bracket the original with
       ``canary_runner.launch_head_kernels(forward_batch)`` +
       ``canary_runner.launch_tail_kernels(forward_batch)``. These two calls run kernel launches
       only — they execute inside cuda graph capture region and therefore get captured into the
       graph, auto-replaying every step.

    NO patching of CudaGraphRunner / EAGLEDraftCudaGraphRunner / PiecewiseCudaGraphRunner /
    BreakableCudaGraphRunner / any speculative graph runner subclass.

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
        allocator=getattr(model_runner, "token_to_kv_pool_allocator", None),
    )
    runner = CanaryRunner(
        config=config,
        buffer_groups=buffer_groups,
        device=device,
        tp_group=_resolve_tp_group(),
        req_to_token_pool=model_runner.req_to_token_pool,
        radix_cache=None,
        launch_capacities=_compute_launch_capacities(model_runner=model_runner),
        swa_window_size=int(model_runner.sliding_window_size or 0),
    )

    if oracle_sampler_hook is not None:
        runner.attach_oracle_sampler_hook(oracle_sampler_hook)

    _patch_model_forward(model_runner=model_runner, runner=runner)

    logger.info(
        "install_canary: mode=%s tags=%d sweep_cadence=%d",
        config.mode,
        runner.active_tag_count,
        config.sweep_interval,
    )
    return runner


def get_canary_runner(model_runner: "ModelRunner") -> Optional[CanaryRunner]:
    """Return the runner attached to this ModelRunner, or None if canary was not installed."""
    return model_runner.canary_runner


def _resolve_tp_group():
    from sglang.srt.distributed.parallel_state import get_tp_group

    try:
        return get_tp_group()
    except (AssertionError, RuntimeError, AttributeError):
        return None


def _compute_launch_capacities(
    *, model_runner: "ModelRunner"
) -> CanaryLaunchCapacities:
    server_args = model_runner.server_args
    cuda_graph_max_bs = server_args.cuda_graph_max_bs or 0
    spec_num_draft_tokens = server_args.speculative_num_draft_tokens
    num_tokens_per_bs = 1
    if spec_num_draft_tokens:
        num_tokens_per_bs = max(num_tokens_per_bs, int(spec_num_draft_tokens))
    max_running_requests = int(model_runner.req_to_token_pool.size)
    max_bs = max(int(cuda_graph_max_bs), max_running_requests)
    chunked_prefill_size = server_args.chunked_prefill_size
    max_prefill_tokens = int(server_args.max_prefill_tokens)
    if chunked_prefill_size is None or chunked_prefill_size < 0:
        max_extend_tokens_per_forward = max_prefill_tokens
    else:
        max_extend_tokens_per_forward = int(chunked_prefill_size)
    pool_slot_count = int(model_runner.max_total_num_tokens)
    write_entry_capacity = max(
        1, max(max_bs * num_tokens_per_bs, max_extend_tokens_per_forward)
    )
    max_seq_len_per_req = int(model_runner.req_to_token_pool.req_to_token.shape[1])

    return CanaryLaunchCapacities(
        per_forward_verify_capacity=max(1, max_seq_len_per_req),
        per_forward_write_req_capacity=max(1, max_bs),
        per_forward_write_entry_capacity=write_entry_capacity,
        sweep_verify_capacity=max(
            1, min(pool_slot_count, _MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY)
        ),
    )


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

    _wrap_method(model_runner.model, "forward", wrapper=_with_canary_bracketing)


def _extract_forward_batch(args, kwargs) -> Optional[ForwardBatch]:
    if "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], ForwardBatch):
        return kwargs["forward_batch"]
    for arg in args:
        if isinstance(arg, ForwardBatch):
            return arg
    return None
