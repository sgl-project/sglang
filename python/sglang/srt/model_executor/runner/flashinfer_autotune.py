# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import annotations

import contextlib
import dataclasses
import datetime
import errno
import fcntl
import functools
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import IO, TYPE_CHECKING, Callable, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import (
    get_disagg,
    get_exec,
    get_model,
    get_parallel,
    get_schedule,
    get_spec,
    max_prefill_buffer_tokens,
)
from sglang.srt.utils import empty_context, log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.distributed.parallel_state import GroupCoordinator
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.runner.base_runner import BaseRunner

logger = logging.getLogger(__name__)

FLASHINFER_AUTOTUNE_WORKAROUND_SKIPS = frozenset()


def get_flashinfer_autotune_skip_ops(model_runner: ModelRunner) -> set[str]:
    skip_ops = set(get_exec().kernel.flashinfer_autotune_skip_ops or ())
    skip_ops.update(FLASHINFER_AUTOTUNE_WORKAROUND_SKIPS)
    return skip_ops


def should_run_flashinfer_autotune(
    model_runner: ModelRunner, *, for_speculative_draft: bool = False
) -> bool:
    """Check if flashinfer autotune should be run."""
    mr = model_runner
    if mr.device != "cuda":
        return False
    if get_exec().kernel.disable_flashinfer_autotune:
        return False
    if get_exec().deterministic.enable_deterministic_inference:
        # Tuned configs are per problem shape, so the reduction order would follow
        # the batch shape.
        return False

    if for_speculative_draft:
        backend_str = (
            get_spec().speculative_moe_runner_backend
            or get_exec().moe.moe_runner_backend
        )
        a2a_backend_str = (
            get_spec().speculative_moe_a2a_backend or get_exec().moe.moe_a2a_backend
        )
    else:
        backend_str = get_exec().moe.moe_runner_backend
        a2a_backend_str = get_exec().moe.moe_a2a_backend

    # Autotune can run before the MoE backend globals are initialized, so read
    # the configured backends -- the draft leaves (`get_spec()`) or the target
    # leaves (`get_exec().moe`) above. CuteDSL v1 bypasses MoeRunner, and its
    # dummy dispatch can exceed DeepEP low-latency's token limit.
    if backend_str == "flashinfer_cutedsl" and a2a_backend_str == "deepep":
        return False

    # TODO smor- support other cases for flashinfer autotune, such as, mamba backend

    moe_needs_autotune = backend_str in [
        "flashinfer_trtllm",
        "flashinfer_trtllm_routed",
        "flashinfer_mxfp4",
        "flashinfer_cutedsl",
        "flashinfer_cutlass",
    ]

    from sglang.srt.layers.quantization.fp4_utils import (
        get_fp4_gemm_runner_backend,
    )

    model_quantization = mr.model_config.quantization
    model_uses_fp4 = model_quantization in (
        "modelopt_fp4",
        "modelopt_mixed",
    )
    fp4_gemm_needs_autotune = model_uses_fp4 and (
        get_fp4_gemm_runner_backend().is_flashinfer_cutlass()
        or get_fp4_gemm_runner_backend().is_flashinfer_cutedsl()
    )

    from sglang.srt.layers.quantization.fp8_utils import (
        flashinfer_per_tensor_fp8_supported,
        resolve_mxfp8_dense_gemm_backend,
    )

    if model_quantization == "mxfp8":
        fp8_gemm_needs_autotune = resolve_mxfp8_dense_gemm_backend().is_flashinfer()
    elif model_quantization in ("modelopt", "modelopt_fp8", "modelopt_mixed"):
        fp8_gemm_needs_autotune = flashinfer_per_tensor_fp8_supported()
    else:
        fp8_gemm_needs_autotune = False

    if not (moe_needs_autotune or fp4_gemm_needs_autotune or fp8_gemm_needs_autotune):
        return False

    if torch.cuda.get_device_capability()[0] < 9:
        return False

    if mr.spec_algorithm.is_speculative():
        return mr.is_draft_worker if for_speculative_draft else not mr.is_draft_worker

    return True


def flashinfer_autotune_store_root(model_runner: ModelRunner) -> Path:
    """This rank's FlashInfer managed autotune store root.

    FlashInfer namespaces the store below the root by its environment identity
    (library versions, GPU) and keys each entry by op, runner and shape bucket
    plus whatever extras the runner adds, so the root separates deployments.
    It is resolved once per process, from the runner that attaches first.
    Per rank: a store shared across ranks would let one rank read a winner
    another published mid-tuning, so ranks of one TP group could split on a hit.
    """
    mr = model_runner
    model_key_parts = [
        str(get_model().model_path),
        str(mr.dtype),
        str(get_model().quantization),
        str(get_exec().moe.moe_runner_backend),
        str(get_parallel().tp_size),
        str(get_parallel().pp_size),
        str(get_parallel().attn_dp_size),
        str(get_parallel().moe_ep_size),
        str(mr.model_config.hf_config.__class__.__name__),
    ]
    # A different skip policy must not reuse previously tuned tactics.
    skip_ops = get_flashinfer_autotune_skip_ops(mr)
    model_key_parts.append("skip_ops=" + ",".join(sorted(skip_ops)))
    model_key = "|".join(model_key_parts)
    cache_key = hashlib.sha256(model_key.encode()).hexdigest()[:16]
    base = Path(envs.SGLANG_CACHE_DIR.get()) / "flashinfer" / "autotune" / "managed"
    if not envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.get():
        # Reuse disabled: tune from scratch into a fresh store, kept for inspection.
        base = base / "runs" / datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    parallel = get_parallel()
    return (
        base
        / cache_key
        / f"rank_tp{parallel.tp_rank}_pp{parallel.pp_rank}_dp{parallel.dp_rank or 0}"
    )


def _autotune_tactic_sync_group(
    tp_group: GroupCoordinator,
) -> Optional[torch.distributed.ProcessGroup]:
    """CPU group over the ranks that must agree on the tuned tactics.

    Per-rank timing noise alone makes each rank's ``argmin`` pick a different
    tactic for the same shape. FlashInfer all-reduces the timings over this
    group so every rank minimizes over the same numbers. TP is the scope: those
    ranks run the same dummy forward, and PP stages are already separate groups.
    """
    if tp_group.world_size <= 1:
        return None
    # The CPU group keeps the reduction of these scalars off the profiled stream.
    return tp_group.cpu_group


@contextlib.contextmanager
def _autotune_process_group(group: Optional[torch.distributed.ProcessGroup]):
    """Set FlashInfer's timing-reduction group, restoring the previous one after."""
    from flashinfer.autotuner import (
        get_autotune_process_group,
        set_autotune_process_group,
    )

    previous = get_autotune_process_group()
    set_autotune_process_group(group)
    try:
        yield
    finally:
        set_autotune_process_group(previous)


@dataclasses.dataclass(frozen=True)
class _AutotuneStore:
    """The store this process tunes into; ``root`` None means in memory only."""

    root: Optional[Path]
    # Held open for the process lifetime; the OS releases the lock on exit.
    lock: Optional[IO[bytes]] = None


_attached_store: Optional[_AutotuneStore] = None


def _lock_autotune_store(root: Path) -> Optional[IO[bytes]]:
    """Hold ``root`` exclusively, or return None (with the reason logged).

    FlashInfer snapshots the store when it attaches but reads a key published
    after that from disk on first lookup, so a second server publishing into
    the same root while this one tunes could split a TP group's ranks on a hit.
    """
    try:
        root.mkdir(parents=True, exist_ok=True)
        lock = open(root / ".lock", "ab")
    except OSError as e:
        logger.warning(
            "FlashInfer autotune: cannot create the store lock under %s (%s).", root, e
        )
        return None
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        lock.close()
        if e.errno in (errno.EWOULDBLOCK, errno.EAGAIN):
            reason = "another process holds it"
        else:
            reason = f"flock is not supported here: {e}"
        logger.warning("FlashInfer autotune: cannot lock store %s (%s).", root, reason)
        return None
    return lock


def _autotune_store_digest(root: Path) -> str:
    """Hash of every entry under ``root``, across all environment namespaces.

    Covers every namespace rather than recomputing FlashInfer's environment
    hash: a rank in a different environment reads a different namespace, and
    stale namespaces only cost a spurious wipe. An unreadable entry counts by
    name, so a peer without it disagrees rather than being silently matched.
    """
    digest = hashlib.sha256()
    for entry in sorted(root.glob("**/entries/*.json")):
        digest.update(str(entry.relative_to(root)).encode() + b"\0")
        try:
            digest.update(hashlib.sha256(entry.read_bytes()).digest())
        except OSError as e:
            logger.warning("FlashInfer autotune: cannot read %s (%s).", entry, e)
            digest.update(b"unreadable")
    return digest.hexdigest()


def _wipe_autotune_store(root: Path) -> bool:
    """Delete every entry under ``root`` except the lock; return whether it worked."""
    try:
        children = list(root.iterdir())
    except OSError as e:
        logger.warning("FlashInfer autotune: cannot list %s (%s).", root, e)
        return False
    for child in children:
        if child.name == ".lock":
            continue
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("FlashInfer autotune: cannot remove %s (%s).", child, e)
    left = list(root.glob("**/entries/*.json"))
    if left:
        logger.warning(
            "FlashInfer autotune: %d entries survived wiping %s.", len(left), root
        )
    return not left


def _agree_on_autotune_store(
    root: Path,
    locked: bool,
    group: Optional[torch.distributed.ProcessGroup],
    env: dict[str, str],
) -> bool:
    """Enter tuning with the same store on every rank, or with none at all.

    A store hit skips a profile, so stores that disagree desync the reduction:
    diverged stores are wiped on every rank. The environment is compared too:
    it picks the namespace a rank reads, so equal stores alone do not mean two
    ranks hit the same entries. A rank that could not lock, read
    or wipe its store cannot vouch for its contents, so then no rank uses one.
    Every branch is decided on gathered values, so all ranks take the same one.
    Returns whether this rank tunes into its on-disk store.
    """
    if group is None:
        return locked
    digest = ""
    if locked:
        try:
            digest = hashlib.sha256(
                json.dumps(env, sort_keys=True).encode()
                + _autotune_store_digest(root).encode()
            ).hexdigest()
        except OSError as e:
            logger.warning("FlashInfer autotune: cannot scan %s (%s).", root, e)
            locked = False
    world_size = torch.distributed.get_world_size(group)
    gathered: list = [None] * world_size
    torch.distributed.all_gather_object(gathered, (locked, digest), group=group)
    if not all(rank_locked for rank_locked, _ in gathered):
        log_info_on_rank0(
            logger,
            "FlashInfer autotune: a rank could not lock its autotune store (see "
            "that rank's warning); tuning in memory only on every rank.",
        )
        return False
    if len({rank_digest for _, rank_digest in gathered}) == 1:
        return True
    log_info_on_rank0(
        logger,
        "FlashInfer autotune: per-rank stores disagree, discarding them and "
        "tuning from scratch so all ranks agree on the tactics.",
    )
    wiped: list = [None] * world_size
    torch.distributed.all_gather_object(wiped, _wipe_autotune_store(root), group=group)
    if not all(wiped):
        log_info_on_rank0(
            logger,
            "FlashInfer autotune: a rank could not empty its autotune store (see "
            "that rank's warning); tuning in memory only on every rank.",
        )
        return False
    return True


def attach_flashinfer_autotune_store(model_runner: ModelRunner) -> _AutotuneStore:
    """Attach this rank's managed autotune store, once per process.

    ``autotune_v2`` attaches its store for the whole process (last attach wins),
    and serving then reads tuned winners from that store's partition, not ones
    tuned in memory under another store or before the attach. So every tuning
    pass in the process -- target, extend, draft, and the PCIe-IPC all-reduce
    -- must tune into one store attached before the first of them runs. The
    gate runs before the attach: attaching loads the store into memory, after
    which a wipe would no longer reach what the rank serves.
    """
    global _attached_store
    if _attached_store is not None:
        return _attached_store
    from flashinfer import autotune_v2
    from flashinfer.autotuner import _collect_metadata

    root = flashinfer_autotune_store_root(model_runner)
    lock = _lock_autotune_store(root)
    sync_group = _autotune_tactic_sync_group(get_parallel().tp_group)
    # The environment autotune_v2 namespaces the store by.
    env = _collect_metadata()
    if _agree_on_autotune_store(root, lock is not None, sync_group, env):
        # Attach and load only; tuning happens in flashinfer_autotune_context.
        with autotune_v2(mode="replay", cache_root=root):
            pass
        _attached_store = _AutotuneStore(root, lock)
    else:
        if lock is not None:
            lock.close()
        _attached_store = _AutotuneStore(None)
    return _attached_store


@contextlib.contextmanager
def flashinfer_autotune_context(model_runner: ModelRunner, *, run_lm_head: bool):
    from flashinfer import autotune_v2

    mr = model_runner
    store = attach_flashinfer_autotune_store(mr)
    sync_group = _autotune_tactic_sync_group(get_parallel().tp_group)
    if store.root is None:
        logger.info("Running FlashInfer autotune in memory only (no cache)")
    else:
        logger.info("Running FlashInfer autotune with cache: %s", store.root)

    # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
    # calls on default stream (unsupported by CUDA) when --enable-symm-mem is used.
    mr.forward_stream.wait_stream(torch.cuda.current_stream())
    with torch.get_device_module(mr.device).stream(mr.forward_stream):
        from sglang.srt.layers.logits_processor import autotune_dummy_run_mode

        with (
            _autotune_process_group(sync_group),
            autotune_v2(
                mode="tune",
                persistent_cache=store.root is not None,
                cache_root=store.root,
                skip_ops=get_flashinfer_autotune_skip_ops(mr),
            ),
            autotune_dummy_run_mode(run_lm_head=run_lm_head),
        ):
            yield
    torch.cuda.current_stream().wait_stream(mr.forward_stream)
    logger.info("FlashInfer autotune completed.")


def run_flashinfer_autotune_forward(
    model_runner: ModelRunner, forward_fn: Callable[[], None], *, run_lm_head: bool
) -> None:
    """Run flashinfer autotune forward."""
    with flashinfer_autotune_context(model_runner, run_lm_head=run_lm_head):
        forward_fn()


def maybe_flashinfer_autotune_speculative_draft(
    runner: BaseRunner,
    forward_fn: Callable[[], None],
    *,
    post_warmup_hook: Optional[Callable[[], None]] = None,
    run_lm_head: bool = True,
) -> None:
    """Run speculative draft flashinfer autotune."""
    mr = runner.model_runner
    phase_key = f"{runner.__class__.__module__}.{runner.__class__.__qualname__}"
    tuned_phases = getattr(mr, "_flashinfer_spec_draft_autotuned_phases", None)
    if tuned_phases is None:
        tuned_phases = set()
        mr._flashinfer_spec_draft_autotuned_phases = tuned_phases
    if phase_key in tuned_phases:
        return
    if (
        not mr.spec_algorithm.is_speculative()
        or not mr.is_draft_worker
        or not should_run_flashinfer_autotune(mr, for_speculative_draft=True)
    ):
        return

    def run_and_reset():
        forward_fn()
        if post_warmup_hook is not None:
            post_warmup_hook()

    run_flashinfer_autotune_forward(mr, run_and_reset, run_lm_head=run_lm_head)
    tuned_phases.add(phase_key)


def maybe_flashinfer_autotune_extend(
    runner: BaseRunner, *, decode_num_tokens: int
) -> None:
    """Also autotune kernels at the prefill token ceiling.

    The decode-shaped autotune only covers token counts up to the decode
    batch size, so larger prefill/extend batches fall outside the tuned
    buckets and run flashinfer's default heuristic — which can be far
    slower than the tuned tactic (e.g. trtllm-gen fp4 MoE is ~30% slower
    untuned at >=8k tokens on sm100). One extra forward at the largest
    per-rank extend token count tunes all buckets up to it.
    """
    mr = runner.model_runner
    # Prefer the per-rank scheduler buffer while preserving the legacy ceiling
    # when chunked prefill is disabled.
    num_tokens = max_prefill_buffer_tokens() or get_schedule().max_prefill_tokens
    if num_tokens <= (decode_num_tokens or 0):
        return  # decode-shaped autotune already covered these buckets
    # DSpark's dummy forward is TARGET_VERIFY-shaped and misses large prefill GEMMs.
    prefill_autotune = getattr(mr.model, "autotune_prefill_kernels", None)
    wants_prefill_autotune = getattr(mr.model, "wants_prefill_autotune", None)
    if wants_prefill_autotune is not None and not wants_prefill_autotune():
        # A model that declines has nothing to tune at the prefill ceiling, so
        # it skips the autotune context and its timing-reduction setup entirely.
        prefill_autotune = None
    if prefill_autotune is not None and mr.is_generation and not mr.is_draft_worker:
        with flashinfer_autotune_context(mr, run_lm_head=False):
            tuned = prefill_autotune(num_tokens, dtype=mr.dtype)
        if tuned:
            return

    if not envs.SGLANG_FLASHINFER_AUTOTUNE_EXTEND.get():
        return
    is_pd_prefill_target = (
        get_disagg().disaggregation_mode == "prefill" and not mr.is_draft_worker
    )
    if not mr.is_generation or (
        mr.spec_algorithm.is_speculative() and not is_pd_prefill_target
    ):
        # Ordinary speculative runners force TARGET_VERIFY; PD prefill targets
        # have no draft-side state and preserve the requested EXTEND mode.
        return
    # Multimodal generation wrappers can still run this text-only EXTEND dummy;
    # an incompatible model should fail the explicit opt-in visibly.

    if mr.attn_backend.extend_dummy_seqs_capped_by_req_pool:
        pool_size = mr.req_to_token_pool.size
        num_tokens_per_req = (num_tokens + pool_size - 1) // pool_size
    else:
        # Packed dummies tune measurably worse tactics for the same token
        # bucket, so pack only where the backend would otherwise crash. None
        # (not 1) keeps the backend's own seq_len_fill_value in _dummy_run.
        num_tokens_per_req = None
    per_req = num_tokens_per_req or 1
    batch_size = (num_tokens + per_req - 1) // per_req
    num_tokens = batch_size * per_req

    buffers = runner._alloc_dummy_decode_buffers(
        batch_size,
        num_tokens_per_req=per_req,
        allocate_logits_buffer=False,
    )
    canary_run_ctx = (
        c.with_active_single_forward_manager(0)
        if (c := mr.canary_manager) is not None
        else empty_context()
    )

    forward_fn = functools.partial(
        runner._dummy_run,
        batch_size=batch_size,
        buffers=buffers,
        run_ctx=canary_run_ctx,
        forward_mode_override=ForwardMode.EXTEND,
        extend_num_tokens_per_req=num_tokens_per_req,
    )

    log_info_on_rank0(
        logger,
        f"FlashInfer autotune: extra EXTEND pass at {num_tokens} tokens "
        f"({batch_size} seqs x {per_req} tokens).",
    )
    try:
        run_flashinfer_autotune_forward(mr, forward_fn, run_lm_head=False)
    except torch.OutOfMemoryError:
        if _autotune_tactic_sync_group(get_parallel().tp_group) is not None:
            # Tuning is collective: this rank has stopped reducing while its
            # peers wait on the next tactic, so skipping the pass would hang
            # them. Fail instead of degrading alone.
            raise
        # The pass is an optimization; without headroom for the extend-shaped
        # forward, fall back to untuned extend buckets instead of failing.
        log_info_on_rank0(
            logger,
            "FlashInfer extend autotune skipped: not enough free memory "
            f"for a {num_tokens}-token dummy forward.",
        )
    finally:
        # release dummy buffers before capture measures free memory
        del forward_fn, buffers
        torch.cuda.empty_cache()
