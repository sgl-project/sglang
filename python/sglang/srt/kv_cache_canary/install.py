from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.jit_kernel.kv_cache_canary import KERNEL_KIND_HEAD, KERNEL_KIND_TAIL
from sglang.srt.kv_cache_canary.api import (
    attach,
    finalize_replay,
    get_runner,
    install_req_to_token_pool_free_hook,
    install_spec_allocator_free_hook,
    install_swa_eviction_hook,
    launch_canary_for_capture,
    maybe_perturb_req_to_token,
    prepare_replay,
    run_head,
    run_tail,
)
from sglang.srt.kv_cache_canary.config import CanaryConfig
from sglang.srt.kv_cache_canary.pool_patch import PoolKind

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

    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool

    pool = model_runner.token_to_kv_pool
    pool_kind = _select_pool_kind(model_runner=model_runner, pool=pool)
    if pool_kind is None:
        logger.warning(
            "kv-canary: unsupported pool type %s; skipping install.",
            type(pool).__name__,
        )
        return

    if getattr(model_runner, _FORWARD_PATCHED_ATTR, False):
        return

    device = torch.device(model_runner.device)
    launch_capacity = _compute_launch_capacity(model_runner=model_runner, config=config)
    runner = attach(
        pool=pool,
        config=config,
        req_to_token_pool=model_runner.req_to_token_pool,
        device=device,
        pool_kind=pool_kind,
        launch_capacity=launch_capacity,
    )
    if runner is None:
        return

    install_req_to_token_pool_free_hook(
        runner=runner,
        req_to_token_pool=model_runner.req_to_token_pool,
    )

    if isinstance(pool, BaseSWAKVPool):
        install_swa_eviction_hook(runner=runner, pool=pool)

    install_spec_allocator_free_hook(
        runner=runner,
        model_runner=model_runner,
    )

    _patch_model_forward(model_runner=model_runner)
    _patch_cuda_graph_runner_replay_class_method()
    setattr(model_runner, _FORWARD_PATCHED_ATTR, True)


def _compute_launch_capacity(
    *, model_runner: "ModelRunner", config: CanaryConfig
) -> int:
    """Pick a fixed launch-buffer capacity that covers every forward shape.

    Strategy: take the largest plausible per-forward batch (cuda-graph max-bs
    × tokens-per-bs for decode, OR ``max_running_requests`` × max sequence
    extension for prefill — whichever is larger), then add headroom for
    verify entries (``max_verify_per_req_per_forward`` per req).

    The launch buffers are allocated up-front (~ a few MB for typical
    configs) and the kernel always launches with ``num_slots == capacity``;
    padding rows carry ``verify_mask = -1`` and the kernel short-circuits
    them with zero I/O. So oversizing here is cheap and avoids the
    "BatchPlan exceeds launch capacity" runtime error in plan_batch.
    """
    server_args = model_runner.server_args
    cuda_graph_max_bs = getattr(server_args, "cuda_graph_max_bs", None) or 0
    spec_num_draft_tokens = getattr(server_args, "speculative_num_draft_tokens", None)
    num_tokens_per_bs = 1
    if spec_num_draft_tokens:
        num_tokens_per_bs = max(num_tokens_per_bs, int(spec_num_draft_tokens))
    max_running_requests = int(model_runner.req_to_token_pool.size)
    max_bs = max(int(cuda_graph_max_bs), max_running_requests)
    write_slots = max_bs * num_tokens_per_bs
    verify_slots = max_bs * max(1, int(config.max_verify_per_req_per_forward))
    return int(write_slots + verify_slots)


def _select_pool_kind(
    *, model_runner: "ModelRunner", pool: object
) -> Optional[PoolKind]:
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool

    is_draft = bool(model_runner.is_draft_worker)
    if isinstance(pool, BaseSWAKVPool):
        return PoolKind.SWA
    if isinstance(pool, MLATokenToKVPool):
        return PoolKind.MLA
    if isinstance(pool, MHATokenToKVPool):
        if is_draft:
            return PoolKind.DRAFT
        return PoolKind.FULL
    return None


def _patch_model_forward(*, model_runner: "ModelRunner") -> None:
    """Wrap ``model_runner.model.forward`` to run the canary kernel pair.

    Three execution paths, all routed through this single wrapper:

    1. **Eager** (prefill/extend/decode outside cuda graph): builds a
       ``BatchPlan`` from the ``ForwardBatch``, launches head kernel,
       runs real forward, launches tail kernel, ends forward.
    2. **CUDA graph capture**: ``is_current_stream_capturing()`` is True.
       We CANNOT compute a real plan here (forward_batch.input_ids is
       graph-buffer-backed dummy data; CPU sync is illegal mid-capture).
       Instead, we launch head+tail kernels reading from the runner's
       fixed launch buffers — their default skip-sentinel makes the
       recorded kernel a no-op. Replay-time refill (see #3) turns the same
       recorded launches into real verify/write work.
    3. **CUDA graph replay**: handled by
       :func:`_install_cuda_graph_runner_replay_hook` patching
       ``CudaGraphRunner.replay`` — the wrapper does NOT run for replays
       (sglang's replay calls ``graph.replay()`` directly, bypassing
       ``model.forward``).
    """
    model = model_runner.model
    original_forward = model.forward

    @functools.wraps(original_forward)
    def patched_model_forward(*args, **kwargs):
        forward_batch = _extract_forward_batch(args, kwargs)
        pool = model_runner.token_to_kv_pool
        runner = get_runner(pool)
        if forward_batch is None or runner is None or not runner.config.enabled:
            return original_forward(*args, **kwargs)

        is_capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        if is_capturing:
            launch_canary_for_capture(runner, kernel_kind=KERNEL_KIND_HEAD)
            output = original_forward(*args, **kwargs)
            launch_canary_for_capture(runner, kernel_kind=KERNEL_KIND_TAIL)
            return output

        _maybe_perturb(
            runner=runner, model_runner=model_runner, forward_batch=forward_batch
        )
        plan = run_head(runner=runner, forward_batch=forward_batch)
        output = original_forward(*args, **kwargs)
        run_tail(runner=runner, forward_batch=forward_batch, plan=plan)
        return output

    model.forward = patched_model_forward


def _maybe_perturb(
    *,
    runner,
    model_runner: "ModelRunner",
    forward_batch,
) -> None:
    """Shared eager + replay self-test perturb hook."""
    active_indices, active_seq_lens = _extract_active_rows(forward_batch)
    maybe_perturb_req_to_token(
        runner=runner,
        req_to_token_pool=model_runner.req_to_token_pool,
        rank=model_runner.tp_rank,
        active_req_pool_indices=active_indices,
        active_seq_lens=active_seq_lens,
    )


_REPLAY_CLASS_PATCHED_ATTR = "_kv_cache_canary_replay_class_patched"


def _patch_cuda_graph_runner_replay_class_method() -> None:
    """Wrap ``replay`` at the CLASS level for every graph-runner family.

    SGLang has several independent graph-runner classes that each manage
    their own captured CUDA graphs and bypass ``model.forward`` by calling
    ``self.graphs[bs].replay()`` directly (or by replaying a piecewise
    sub-graph). Patching only the base ``CudaGraphRunner`` leaves the spec
    decoding draft worker and piecewise hot paths uninstrumented — the
    canary kernel would never run during replay there. So we enumerate the
    full family.

    Why CLASS-level: the canary install runs BEFORE ``init_device_graphs``,
    so the per-instance graph runners don't exist yet at install time.
    Patching the class method covers every instance created afterwards.
    The patched body looks the canary runner up off
    ``self.model_runner.token_to_kv_pool`` per-call, so instances without a
    canary attached just delegate to the original method (zero cost).

    Each subclass below defines its own ``replay`` method (verified at
    import time), so patching the base class alone would NOT cover them —
    each class must be patched individually.

    Idempotent at the class level: a second install call is a no-op.
    """
    from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner

    classes_to_patch: list[type] = [CudaGraphRunner]
    try:
        from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
            EAGLEDraftCudaGraphRunner,
        )

        classes_to_patch.append(EAGLEDraftCudaGraphRunner)
    except ImportError:
        logger.debug("kv-canary: EAGLEDraftCudaGraphRunner not available; skipping")
    try:
        from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
            EAGLEDraftExtendCudaGraphRunner,
        )

        classes_to_patch.append(EAGLEDraftExtendCudaGraphRunner)
    except ImportError:
        logger.debug(
            "kv-canary: EAGLEDraftExtendCudaGraphRunner not available; skipping"
        )
    try:
        from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
            PiecewiseCudaGraphRunner,
        )

        classes_to_patch.append(PiecewiseCudaGraphRunner)
    except ImportError:
        logger.debug("kv-canary: PiecewiseCudaGraphRunner not available; skipping")

    for cls in classes_to_patch:
        _patch_graph_runner_class_replay(cls)


def _patch_graph_runner_class_replay(cls: type) -> None:
    if getattr(cls, _REPLAY_CLASS_PATCHED_ATTR, False):
        return

    original_replay = cls.replay

    @functools.wraps(original_replay)
    def patched_replay(self, forward_batch, *args, **kwargs):
        model_runner = self.model_runner
        pool = model_runner.token_to_kv_pool
        runner = get_runner(pool)
        if runner is None or not runner.config.enabled:
            return original_replay(self, forward_batch, *args, **kwargs)

        _maybe_perturb(
            runner=runner, model_runner=model_runner, forward_batch=forward_batch
        )
        plan = prepare_replay(runner=runner, forward_batch=forward_batch)
        output = original_replay(self, forward_batch, *args, **kwargs)
        finalize_replay(runner=runner, plan=plan)
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


def _extract_active_rows(
    forward_batch,
) -> tuple[Optional[list], Optional[list]]:
    """Pull (req_pool_indices, seq_lens) lists for active-row-aware perturb.

    Returns ``(None, None)`` when the data isn't available — perturb falls
    back to global random swap.
    """
    if forward_batch is None:
        return None, None
    if forward_batch.req_pool_indices is None or forward_batch.seq_lens is None:
        return None, None
    indices = forward_batch.req_pool_indices.detach().cpu().tolist()
    seq_lens = forward_batch.seq_lens.detach().cpu().tolist()
    return indices, seq_lens
