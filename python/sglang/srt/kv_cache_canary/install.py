from __future__ import annotations

import dataclasses
import functools
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.jit_kernel.kv_cache_canary import KERNEL_KIND_HEAD, KERNEL_KIND_TAIL
from sglang.srt.kv_cache_canary.api import (
    attach,
    finalize_replay,
    get_runners,
    launch_canary_for_capture,
    prepare_replay,
    run_head,
    run_tail,
)
from sglang.srt.kv_cache_canary.config import CanaryConfig
from sglang.srt.kv_cache_canary.test_utils import maybe_perturb_hook

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_FORWARD_PATCHED_ATTR = "_kv_cache_canary_forward_patched"


def install_on_model_runner(
    *,
    model_runner: "ModelRunner",
    mode: Optional[str],
    real_kv_hash_mode: Optional[str] = None,
) -> None:
    """Attach the canary to the model runner's pool and wire its hooks.

    Must be called AFTER ``init_memory_pool`` and BEFORE ``init_device_graphs``.
    Idempotent: a second call on the same ``model_runner`` is a no-op.

    The canary patches ``model_runner.model.forward`` (the bound method).
    SGLang calls ``self.model.forward(...)`` directly (not ``self.model(...)``)
    and ``cuda_graph_runner.patch_model`` yields ``model.forward`` for capture
    too — so patching the bound method is the single point that covers both
    the eager and the captured-into-cuda-graph paths.

    ``real_kv_hash_mode`` (one of ``off`` / ``bit`` / ``all``, default
    ``off``) controls the canary-with-real-data fingerprint (UserInstr
    Fix 5 / part c).
    """
    config = CanaryConfig.from_server_args(mode, real_kv_hash_mode=real_kv_hash_mode)
    if not config.enabled:
        return

    pool = model_runner.token_to_kv_pool
    if not _supports_canary(pool):
        logger.warning(
            "kv-canary: unsupported pool type %s; skipping install.",
            type(pool).__name__,
        )
        return

    if getattr(model_runner, _FORWARD_PATCHED_ATTR, False):
        return

    if _is_swa_pool(pool):
        # SWA's req_to_token mapping only addresses the most recent
        # ``sliding_window_size`` slots; the verify range for the SWA
        # canary must be clipped accordingly. The FULL canary on the
        # same pool gets ``swa_window_size = None`` injected by
        # :func:`attach` and uses the full prefix.
        window_size = model_runner.sliding_window_size
        if window_size is None or int(window_size) <= 0:
            logger.warning(
                "kv-canary: SWA pool detected but model_runner.sliding_window_size "
                "is %r; falling back to full-prefix verify for the SWA canary "
                "(may produce spurious violations on long prefixes).",
                window_size,
            )
        else:
            config = dataclasses.replace(config, swa_window_size=int(window_size))

    device = torch.device(model_runner.device)
    verify_capacity, write_capacity, write_req_capacity = _compute_launch_capacities(
        model_runner=model_runner, config=config
    )
    runners = attach(
        pool=pool,
        config=config,
        device=device,
        verify_capacity=verify_capacity,
        write_capacity=write_capacity,
        write_req_capacity=write_req_capacity,
    )
    if not runners:
        return

    _patch_model_forward(model_runner=model_runner)
    _patch_cuda_graph_runner_replay_class_method()
    setattr(model_runner, _FORWARD_PATCHED_ATTR, True)


def _compute_launch_capacities(
    *, model_runner: "ModelRunner", config: CanaryConfig
) -> Tuple[int, int, int]:
    """Pick fixed launch-buffer capacities that cover every forward shape.

    Returns ``(verify_capacity, write_capacity, write_req_capacity)``. Each
    is sized off the largest plausible per-forward batch:

    - ``write_capacity`` = max_bs * num_tokens_per_bs (extend tokens or
      spec_num_draft_tokens).
    - ``write_req_capacity`` = max_bs.
    - ``verify_capacity`` = ``max_total_num_tokens`` — every forward
      verifies the full per-req prefix, and the sum of all reqs' K_req
      across one forward is bounded by the global slot pool size. This is
      the only upper bound that is independent of the (now removed)
      per-req verify cap.

    Padding rows past ``num_active_*`` carry ``*_active_mask == 0`` and the
    kernel short-circuits them with zero I/O, so oversizing is cheap.
    """
    server_args = model_runner.server_args
    cuda_graph_max_bs = server_args.cuda_graph_max_bs or 0
    spec_num_draft_tokens = server_args.speculative_num_draft_tokens
    num_tokens_per_bs = 1
    if spec_num_draft_tokens:
        num_tokens_per_bs = max(num_tokens_per_bs, int(spec_num_draft_tokens))
    max_running_requests = int(model_runner.req_to_token_pool.size)
    max_bs = max(int(cuda_graph_max_bs), max_running_requests)
    # Prefill batches write one entry per extend token, not per request, so
    # the per-forward write count is bounded by the chunked-prefill budget
    # (or ``max_prefill_tokens`` if chunking is disabled), not by ``max_bs``.
    chunked_prefill_size = server_args.chunked_prefill_size
    max_prefill_tokens = int(server_args.max_prefill_tokens)
    if chunked_prefill_size is None or chunked_prefill_size < 0:
        max_extend_tokens_per_forward = max_prefill_tokens
    else:
        max_extend_tokens_per_forward = int(chunked_prefill_size)
    write_capacity = max(max_bs * num_tokens_per_bs, max_extend_tokens_per_forward)
    write_req_capacity = max_bs
    verify_capacity = max(1, int(model_runner.max_total_num_tokens))
    return verify_capacity, write_capacity, write_req_capacity


def _supports_canary(pool: object) -> bool:
    """Return True if ``pool`` matches any of the known canary dispatch shapes."""
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool

    return isinstance(pool, (BaseSWAKVPool, MLATokenToKVPool, MHATokenToKVPool))


def _is_swa_pool(pool: object) -> bool:
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool

    return isinstance(pool, BaseSWAKVPool)


def _patch_model_forward(*, model_runner: "ModelRunner") -> None:
    """Wrap ``model_runner.model.forward`` to run the canary kernel pair.

    Three execution paths, all routed through this single wrapper:

    1. **Eager** (prefill/extend/decode outside cuda graph): builds a
       ``BatchPlan`` from the ``ForwardBatch``, launches head kernel,
       runs real forward, launches tail kernel, ends forward.
    2. **CUDA graph capture**: ``is_current_stream_capturing()`` is True.
       We CANNOT compute a real plan here. Instead, we launch head+tail
       kernels reading from the runner's fixed launch buffers — their
       default skip-sentinel makes the recorded kernel a no-op. Replay-time
       refill (see #3) turns the same recorded launches into real
       verify/write work.
    3. **CUDA graph replay**: handled by
       :func:`_patch_cuda_graph_runner_replay_class_method` patching
       ``CudaGraphRunner.replay``.
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
            launch_canary_for_capture(runners, kernel_kind=KERNEL_KIND_HEAD)
            output = original_forward(*args, **kwargs)
            launch_canary_for_capture(runners, kernel_kind=KERNEL_KIND_TAIL)
            return output

        maybe_perturb_hook(
            runner=runners[0], model_runner=model_runner, forward_batch=forward_batch
        )
        plans = run_head(runners=runners, forward_batch=forward_batch)
        output = original_forward(*args, **kwargs)
        run_tail(runners=runners, plans=plans)
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

    SGLang has several independent graph-runner classes that each manage
    their own captured CUDA graphs and bypass ``model.forward`` by calling
    ``self.graphs[bs].replay()`` directly. Patching only the base
    ``CudaGraphRunner`` leaves the spec decoding draft worker and piecewise
    hot paths uninstrumented; we enumerate the full family.

    Why CLASS-level: the canary install runs BEFORE ``init_device_graphs``,
    so the per-instance graph runners don't exist yet at install time.
    Patching the class method covers every instance created afterwards.

    Idempotent at the class level: a second install call is a no-op.
    """
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

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

        maybe_perturb_hook(
            runner=runners[0], model_runner=model_runner, forward_batch=forward_batch
        )
        plans = prepare_replay(runners=runners, forward_batch=forward_batch)
        output = original_replay(self, forward_batch, *args, **kwargs)
        finalize_replay(runners=runners, plans=plans)
        return output

    cls.replay = patched_replay
    setattr(cls, _REPLAY_CLASS_PATCHED_ATTR, True)


def _extract_forward_batch(args, kwargs):
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    if "forward_batch" in kwargs and isinstance(kwargs["forward_batch"], ForwardBatch):
        return kwargs["forward_batch"]
    for arg in args:
        if isinstance(arg, ForwardBatch):
            return arg
    return None
