"""Install glue: attach a canary runner to a :class:`ModelRunner` and patch its forward hooks.

The install function:

- builds a :class:`CanaryConfig` from server args,
- detects pool type (MHA / MLA / SWA) and supplies the SWA window length if applicable,
- computes capacities for all three (per-forward, running-sweep, radix-orphan-sweep) plan slots,
- delegates to :func:`api.attach` to install canary buffers and create runners,
- patches ``model_runner.model.forward`` to drive head/tail launches,
- patches every cuda-graph runner class's ``replay`` so head/tail launches fire eagerly around the captured
  replay (the canary itself is NOT baked into cuda-graph capture).
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import threading
from typing import List, Optional, Tuple

import torch

from sglang.srt.kv_cache_canary.api import (
    attach,
    get_runners,
    run_head,
    run_tail,
)
from sglang.srt.kv_cache_canary.config import CanaryConfig
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_FORWARD_PATCHED_ATTR = "_kv_cache_canary_forward_patched"

# Thread-local flag set by ``patched_replay`` around the ``original_replay`` call so the inner
# ``patched_model_forward`` (invoked through the captured replay's inner ``model.forward``) knows the outer
# wrapper has already launched the canary kernel pair and skips its own eager-path launch.
_replay_in_flight = threading.local()


def _is_inside_replay() -> bool:
    return bool(getattr(_replay_in_flight, "active", False))


def install_on_model_runner(
    *,
    model_runner: ModelRunner,
    mode: Optional[str],
    real_kv_hash_mode: Optional[str] = None,
    real_data_sweep_every_n_steps: int = 0,
) -> None:
    """Attach the canary to the model runner's pool and wire its hooks.

    Must be called AFTER ``init_memory_pool`` and BEFORE ``init_device_graphs``. Idempotent: a second call
    on the same ``model_runner`` is a no-op.
    """
    config = CanaryConfig.from_server_args(
        mode,
        real_kv_hash_mode=real_kv_hash_mode,
        real_data_sweep_every_n_steps=real_data_sweep_every_n_steps,
    )
    if not config.enabled:
        return

    pool = model_runner.token_to_kv_pool
    if not isinstance(pool, (BaseSWAKVPool, MLATokenToKVPool, MHATokenToKVPool)):
        logger.warning(
            "kv-canary: unsupported pool type %s; skipping install.",
            type(pool).__name__,
        )
        return

    if getattr(model_runner, _FORWARD_PATCHED_ATTR, False):
        return

    if isinstance(pool, BaseSWAKVPool):
        window_size = model_runner.sliding_window_size
        if window_size is None or int(window_size) <= 0:
            logger.warning(
                "kv-canary: SWA pool detected but model_runner.sliding_window_size is %r; "
                "falling back to full-prefix verify for the SWA canary "
                "(may produce spurious violations on long prefixes).",
                window_size,
            )
        else:
            config = dataclasses.replace(config, swa_window_size=int(window_size))

    device = torch.device(model_runner.device)
    capacities = _compute_launch_capacities(model_runner=model_runner, config=config)
    # Radix cache is bound later by the scheduler via ``api.attach_radix_cache_to_pool`` once
    # ``tree_cache`` exists; at this point in ModelRunner init the radix cache has not been built yet
    # (SOT §6.2 sweep coverage).
    runners = attach(
        pool=pool,
        config=config,
        device=device,
        per_forward_verify_capacity=capacities.per_forward_verify_capacity,
        per_forward_write_req_capacity=capacities.per_forward_write_req_capacity,
        running_sweep_verify_capacity=capacities.running_sweep_verify_capacity,
        radix_sweep_verify_capacity=capacities.radix_sweep_verify_capacity,
        radix_sweep_extras_capacity=capacities.radix_sweep_extras_capacity,
        per_forward_extras_capacity=capacities.per_forward_extras_capacity,
        running_sweep_extras_capacity=capacities.running_sweep_extras_capacity,
        pseudo_token_capacity=capacities.pseudo_token_capacity,
        req_to_token_pool=model_runner.req_to_token_pool,
        radix_cache=None,
        tp_rank=model_runner.tp_rank,
    )
    if not runners:
        return

    _patch_model_forward(model_runner=model_runner)
    _patch_cuda_graph_runner_replay_class_method()
    setattr(model_runner, _FORWARD_PATCHED_ATTR, True)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _LaunchCapacities:
    """Fixed launch-buffer capacities sized for every plausible forward and sweep cycle."""

    per_forward_verify_capacity: int
    per_forward_write_req_capacity: int
    per_forward_extras_capacity: int
    running_sweep_verify_capacity: int
    running_sweep_extras_capacity: int
    radix_sweep_verify_capacity: int
    radix_sweep_extras_capacity: int
    pseudo_token_capacity: int


def _compute_launch_capacities(
    *, model_runner: ModelRunner, config: CanaryConfig
) -> _LaunchCapacities:
    """Pick fixed launch-buffer capacities that cover every forward shape and worst-case sweep.

    Padding rows past the active count carry ``num_valid == 0`` and the kernel short-circuits them with
    zero I/O, so oversizing is cheap.
    """
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

    return _LaunchCapacities(
        per_forward_verify_capacity=max(1, pool_slot_count),
        per_forward_write_req_capacity=max(1, max_bs),
        per_forward_extras_capacity=1,
        running_sweep_verify_capacity=max(1, pool_slot_count),
        running_sweep_extras_capacity=1,
        radix_sweep_verify_capacity=max(1, pool_slot_count),
        radix_sweep_extras_capacity=max(1, pool_slot_count),
        pseudo_token_capacity=max(
            1, max(max_bs * num_tokens_per_bs, max_extend_tokens_per_forward)
        ),
    )


def _patch_model_forward(*, model_runner: ModelRunner) -> None:
    """Wrap ``model_runner.model.forward`` to run the canary head/tail launches.

    Two execution paths, both routed through this single wrapper:

    1. **Eager** (prefill/extend/decode outside any cuda graph): builds a plan from the live
       ``ForwardBatch``, launches head, runs real forward, launches tail, ends forward.
    2. **Cuda graph capture** and **replay**: handled by :func:`_patch_cuda_graph_runner_replay_class_method`
       patching the graph runner's ``replay`` so the canary kernel pair runs **eagerly around**
       ``original_replay``. The canary is intentionally NOT recorded into any captured graph (see the
       runner's plan-input ABI: cuda-graph capture pins the plan tensor addresses, but the launches
       themselves stay eager so the pump can D2H without blocking inside capture).
    """
    model = model_runner.model
    original_forward = model.forward

    @functools.wraps(original_forward)
    def patched_model_forward(*args, **kwargs):
        forward_batch = _extract_forward_batch(args, kwargs)
        pool = model_runner.token_to_kv_pool
        runners = get_runners(pool)
        if forward_batch is None or not runners or not runners[0].config.enabled:
            return original_forward(*args, **kwargs)

        is_capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if is_capturing:
            return original_forward(*args, **kwargs)

        if _is_inside_replay():
            return original_forward(*args, **kwargs)

        run_head(runners=runners, forward_batch=forward_batch)
        output = original_forward(*args, **kwargs)
        run_tail(runners=runners, forward_batch=forward_batch)
        return output

    model.forward = patched_model_forward


_REPLAY_CLASS_PATCHED_ATTR = "_kv_cache_canary_replay_class_patched"


_OPTIONAL_GRAPH_RUNNER_CLASSES: Tuple[Tuple[str, str], ...] = (
    (
        "sglang.srt.speculative.eagle_draft_cuda_graph_runner",
        "EAGLEDraftCudaGraphRunner",
    ),
    (
        "sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner",
        "EAGLEDraftExtendCudaGraphRunner",
    ),
    (
        "sglang.srt.model_executor.piecewise_cuda_graph_runner",
        "PiecewiseCudaGraphRunner",
    ),
)


def _patch_cuda_graph_runner_replay_class_method() -> None:
    """Wrap ``replay`` at the CLASS level for every graph-runner family.

    SGLang has several independent graph-runner classes that each manage their own captured CUDA graphs and
    bypass ``model.forward`` by calling ``self.graphs[bs].replay()`` directly. Patching only the base
    ``CudaGraphRunner`` leaves the spec decoding draft worker and piecewise hot paths uninstrumented; we
    enumerate the full family.

    Why CLASS-level: the canary install runs BEFORE ``init_device_graphs``, so the per-instance graph
    runners don't exist yet at install time. Patching the class method covers every instance created
    afterwards.

    Idempotent at the class level: a second install call is a no-op.
    """
    classes_to_patch: List[type] = [CudaGraphRunner]
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
        model_runner = self.model_runner
        pool = model_runner.token_to_kv_pool
        runners = get_runners(pool)
        if not runners or not runners[0].config.enabled:
            return original_replay(self, forward_batch, *args, **kwargs)

        run_head(runners=runners, forward_batch=forward_batch)
        _replay_in_flight.active = True
        try:
            output = original_replay(self, forward_batch, *args, **kwargs)
        finally:
            _replay_in_flight.active = False
        run_tail(runners=runners, forward_batch=forward_batch)
        return output

    cls.replay = patched_replay
    setattr(cls, _REPLAY_CLASS_PATCHED_ATTR, True)


def _extract_forward_batch(args, kwargs):
    if "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], ForwardBatch):
        return kwargs["forward_batch"]
    for arg in args:
        if isinstance(arg, ForwardBatch):
            return arg
    return None
