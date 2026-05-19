from __future__ import annotations

import contextvars
import functools
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.kv_cache_canary.config import CanaryConfig
from sglang.srt.kv_cache_canary.runner import CanaryRunner
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

_MAX_CUDA_GRID_SAFE_VERIFY_CAPACITY: int = 1_000_000

_INSIDE_REPLAY: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "kv_cache_canary_inside_replay", default=False
)
_REPLAY_CLASS_PATCHED_ATTR = "_kv_cache_canary_replay_class_patched"
_OPTIONAL_GRAPH_RUNNER_CLASSES: tuple[tuple[str, str], ...] = (
    (
        "sglang.srt.speculative.eagle_draft_cuda_graph_runner",
        "EAGLEDraftCudaGraphRunner",
    ),
    (
        "sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner",
        "EAGLEDraftExtendCudaGraphRunner",
    ),
    (
        "sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner",
        "MultiLayerEagleDraftExtendCudaGraphRunner",
    ),
    (
        "sglang.srt.speculative.frozen_kv_mtp_cuda_graph_runner",
        "FrozenKVMTPCudaGraphRunner",
    ),
    (
        "sglang.srt.model_executor.piecewise_cuda_graph_runner",
        "PiecewiseCudaGraphRunner",
    ),
    (
        "sglang.srt.model_executor.breakable_cuda_graph_runner",
        "BreakableCudaGraphRunner",
    ),
)


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
    5. Construct CanaryRunner and stash on model_runner.canary_runner.
    6. Monkeypatch ModelRunner.forward to call runner.forward_step(forward_batch) wrapping the original.

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
    )

    model_runner.canary_runner = runner

    _patch_model_forward(model_runner=model_runner, runner=runner)
    _patch_cuda_graph_runner_replay_class_method()

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

        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return original_forward(*args, **kwargs)

        if _INSIDE_REPLAY.get():
            return original_forward(*args, **kwargs)

        runner.forward_step_before_model(forward_batch)
        output = original_forward(*args, **kwargs)
        runner.forward_step_after_model()
        runner.end_of_step()
        return output

    model.forward = patched_model_forward


def _patch_cuda_graph_runner_replay_class_method() -> None:
    classes_to_patch: list[type] = [CudaGraphRunner]
    for module_path, class_name in _OPTIONAL_GRAPH_RUNNER_CLASSES:
        optional_cls = _try_import_class(module_path, class_name)
        if optional_cls is not None:
            classes_to_patch.append(optional_cls)

    for cls in classes_to_patch:
        _patch_graph_runner_class_replay(cls)


def _try_import_class(module_path: str, class_name: str) -> Optional[type]:
    try:
        module = __import__(module_path, fromlist=[class_name])
    except ImportError:
        logger.debug("kv-canary: %s not available; skipping", class_name)
        return None
    return getattr(module, class_name, None)


def _patch_graph_runner_class_replay(cls: type) -> None:
    if getattr(cls, _REPLAY_CLASS_PATCHED_ATTR, False):
        return

    original_replay = cls.replay

    @functools.wraps(original_replay)
    def patched_replay(self, forward_batch, *args, **kwargs):
        runner = get_canary_runner(self.model_runner)
        if runner is None or runner.config.mode == "off":
            return original_replay(self, forward_batch, *args, **kwargs)

        runner.forward_step_before_model(forward_batch)
        token = _INSIDE_REPLAY.set(True)
        try:
            output = original_replay(self, forward_batch, *args, **kwargs)
        finally:
            _INSIDE_REPLAY.reset(token)
        runner.forward_step_after_model()
        runner.end_of_step()
        return output

    cls.replay = patched_replay
    setattr(cls, _REPLAY_CLASS_PATCHED_ATTR, True)


def _extract_forward_batch(args, kwargs) -> Optional[ForwardBatch]:
    if "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], ForwardBatch):
        return kwargs["forward_batch"]
    for arg in args:
        if isinstance(arg, ForwardBatch):
            return arg
    return None
