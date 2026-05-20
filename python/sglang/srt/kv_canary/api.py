from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_canary.config import CanaryConfig
from sglang.srt.kv_canary.runner.canary_runner import CanaryRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY: int = 1_000_000


def install_canary(
    *,
    server_args: "ServerArgs",
    model_runner: "ModelRunner",
) -> Optional[CanaryRunner]:
    """Install canary on a ModelRunner. Returns None if config.mode == "off". Otherwise:

    1. Build CanaryConfig from server_args + env vars.
    2. For each KV pool on model_runner (token_to_kv_pool + any aux pools): attach_canary_buffers.
    3. Build CanaryEndpoint tuples per pool.
    4. Allocate CanaryDeviceState (violation log, counters, pump bufs).
    5. Construct CanaryRunner (which also allocates static per-forward PlanInput buffers) and
       stash on model_runner.canary_runner.
    6. Monkeypatch the model nn.Module's `.forward` to bracket the original with
       `canary_runner.launch_head_kernels(forward_batch)` + `canary_runner.launch_tail_kernels(
       forward_batch)`. These two calls run kernel launches only — they execute inside cuda graph
       capture region and therefore get captured into the graph, auto-replaying every step.

    NO patching of CudaGraphRunner / EAGLEDraftCudaGraphRunner / PiecewiseCudaGraphRunner /
    BreakableCudaGraphRunner / any speculative graph runner subclass.

    The host-side hooks `canary_runner.before_forward(forward_batch)` (perturb + plan-input fill)
    and `canary_runner.end_of_step()` (sweep + D2H pump + allreduce + raise) are NOT installed
    by install_canary. They are invoked by explicit source-level call sites in
    `ModelRunner.forward` — one call right before the `graph_runner.replay() / model.forward()`
    branch, one call right after it returns.

    Idempotent: second call on the same model_runner is an error.
    """
    if get_canary_runner(model_runner) is not None:
        raise RuntimeError(
            "kv-canary: install_canary called twice on the same ModelRunner"
        )

    config = CanaryConfig.from_env(server_args)
    if config.mode == "off":
        return None

    pools = _collect_pools(model_runner)
    device = torch.device(model_runner.device)
    tp_group = _resolve_tp_group()
    capacities = _compute_launch_capacities(model_runner=model_runner, config=config)
    swa_window_size = _resolve_swa_window_size(model_runner)
    allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)

    runner = CanaryRunner(
        config=config,
        pools=pools,
        device=device,
        tp_group=tp_group,
        req_to_token_pool=model_runner.req_to_token_pool,
        radix_cache=None,
        per_forward_verify_capacity=capacities.per_forward_verify_capacity,
        per_forward_write_req_capacity=capacities.per_forward_write_req_capacity,
        per_forward_write_entry_capacity=capacities.per_forward_write_entry_capacity,
        sweep_verify_capacity=capacities.sweep_verify_capacity,
        swa_window_size=swa_window_size,
        token_pool_allocator=allocator,
    )

    model_runner.canary_runner = runner

    _patch_model_forward(model_runner=model_runner, runner=runner)

    logger.info(
        "install_canary: mode=%s tags=%d sweep_cadence=%d",
        config.mode,
        runner.active_tag_count,
        config.sweep_every_n_steps,
    )
    return runner


def get_canary_runner(model_runner: "ModelRunner") -> Optional[CanaryRunner]:
    """Return the runner attached to this ModelRunner, or None if canary was not installed."""
    return getattr(model_runner, "canary_runner", None)


@dataclass(frozen=True, slots=True, kw_only=True)
class _LaunchCapacities:
    per_forward_verify_capacity: int
    per_forward_write_req_capacity: int
    per_forward_write_entry_capacity: int
    sweep_verify_capacity: int


def _collect_pools(model_runner: "ModelRunner") -> list:
    pools = [model_runner.token_to_kv_pool]
    return pools


def _resolve_tp_group():
    from sglang.srt.distributed.parallel_state import get_tp_group

    try:
        return get_tp_group()
    except (AssertionError, RuntimeError, AttributeError):
        return None


def _resolve_swa_window_size(model_runner: "ModelRunner") -> int:
    window = model_runner.sliding_window_size
    if window is None:
        return 0
    return int(window) if int(window) > 0 else 0


def _compute_launch_capacities(
    *, model_runner: "ModelRunner", config: CanaryConfig
) -> _LaunchCapacities:
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
    per_forward_verify_capacity = max(1, max_seq_len_per_req)
    sweep_verify_capacity = max(
        1, min(pool_slot_count, _MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY)
    )

    _ = config
    return _LaunchCapacities(
        per_forward_verify_capacity=per_forward_verify_capacity,
        per_forward_write_req_capacity=max(1, max_bs),
        per_forward_write_entry_capacity=write_entry_capacity,
        sweep_verify_capacity=sweep_verify_capacity,
    )


def _patch_model_forward(*, model_runner: "ModelRunner", runner: CanaryRunner) -> None:
    model = model_runner.model
    original_forward = model.forward

    @functools.wraps(original_forward)
    def patched_model_forward(*args, **kwargs):
        forward_batch = _extract_forward_batch(args, kwargs)
        if forward_batch is None:
            return original_forward(*args, **kwargs)

        runner.launch_head_kernels(forward_batch)
        output = original_forward(*args, **kwargs)
        runner.launch_tail_kernels(forward_batch)
        return output

    model.forward = patched_model_forward


def _extract_forward_batch(args, kwargs) -> Optional[ForwardBatch]:
    if "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], ForwardBatch):
        return kwargs["forward_batch"]
    for arg in args:
        if isinstance(arg, ForwardBatch):
            return arg
    return None
