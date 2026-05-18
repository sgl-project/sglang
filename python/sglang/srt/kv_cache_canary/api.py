from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.kv_cache_canary.config import CanaryConfig
from sglang.srt.kv_cache_canary.host_state import BatchPlan
from sglang.srt.kv_cache_canary.runner import CanaryRunner

if TYPE_CHECKING:
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.mem_cache.req_to_token_pool import ReqToTokenPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

_GLOBAL_RUNNER_KEY = "_kv_cache_canary_runner"


def attach(
    *,
    pool: "MHATokenToKVPool",
    config: CanaryConfig,
    req_to_token_pool: "ReqToTokenPool",
    device: torch.device,
) -> Optional[CanaryRunner]:
    """Attach canary to ``pool`` and create a runner.

    Must be called AFTER ``init_memory_pool`` and BEFORE ``init_device_graphs``
    so that the canary kernel is captured into the CUDA graph and the shadow
    tensors are baked into the graph's pointer table.
    """
    if not config.enabled:
        return None
    if hasattr(pool, _GLOBAL_RUNNER_KEY):
        logger.warning("kv-canary: pool already has a runner attached; reusing it")
        return get_runner(pool)

    runner = CanaryRunner(
        config=config,
        pool=pool,
        num_req_slots=int(req_to_token_pool.size),
        device=device,
    )
    setattr(pool, _GLOBAL_RUNNER_KEY, runner)
    logger.info("kv-canary: attached runner in mode=%s", config.mode.value)
    return runner


def get_runner(pool: "MHATokenToKVPool") -> Optional[CanaryRunner]:
    return getattr(pool, _GLOBAL_RUNNER_KEY, None)


def install_req_to_token_pool_free_hook(
    *,
    runner: CanaryRunner,
    req_to_token_pool: "ReqToTokenPool",
) -> None:
    """Wrap ``ReqToTokenPool.free`` so canary state is dropped on req release.

    Without this, a reused ``req_pool_idx`` inherits the previous request's
    ``K_req`` / ``prev_hash_tail``, which would make the new req's first token
    look like a verify-then-mismatch (P0-1 of host review).
    """
    if getattr(req_to_token_pool, "_kv_canary_free_patched", False):
        return
    original_free = req_to_token_pool.free

    def patched_free(req) -> None:
        idx = req.req_pool_idx
        original_free(req)
        if idx is not None:
            runner.host_state.reset_request(int(idx))

    req_to_token_pool.free = patched_free
    setattr(req_to_token_pool, "_kv_canary_free_patched", True)


def maybe_perturb_req_to_token(
    *,
    runner: Optional[CanaryRunner],
    req_to_token_pool: "ReqToTokenPool",
    rank: int,
) -> None:
    """Self-test helper: probabilistically swap slot pointers in ``req_to_token``.

    Per-rank deterministic RNG so multi-rank perturbations don't interact in
    non-reproducible ways.
    """
    if runner is None:
        return
    prob = runner.config.perturb_req_to_token_prob
    if prob <= 0.0:
        return

    rng = random.Random(
        _rng_seed_for_rank(runner.config.perturb_req_to_token_seed, rank)
    )
    if rng.random() >= prob:
        return
    table = req_to_token_pool.req_to_token
    if not isinstance(table, torch.Tensor) or table.numel() == 0:
        return
    rows, cols = int(table.shape[0]), int(table.shape[1])
    if rows <= 1 or cols <= 1:
        return
    r = rng.randrange(1, rows)
    a = rng.randrange(cols)
    b = rng.randrange(cols)
    if a == b:
        return
    tmp = table[r, a].clone()
    table[r, a] = table[r, b]
    table[r, b] = tmp
    logger.warning(
        "kv-canary self-test: perturbed req_to_token_pool[%d] columns %d <-> %d (rank=%d)",
        r,
        a,
        b,
        rank,
    )


def _rng_seed_for_rank(base_seed: int, rank: int) -> int:
    return (
        (base_seed & 0xFFFFFFFF) * 0x9E3779B1 + rank * 0xBF58476D1CE4E5B9
    ) & 0xFFFFFFFFFFFFFFFF


def run_head(
    *,
    runner: Optional[CanaryRunner],
    forward_batch: "ForwardBatch",
) -> Optional[BatchPlan]:
    if runner is None:
        return None
    plan = _plan_from_forward_batch(runner=runner, forward_batch=forward_batch)
    if plan is None:
        return None
    runner.run_head(plan=plan, slot_indices=forward_batch.out_cache_loc)
    return plan


def run_tail(
    *,
    runner: Optional[CanaryRunner],
    forward_batch: "ForwardBatch",
    plan: Optional[BatchPlan],
) -> None:
    if runner is None or plan is None:
        return
    runner.run_tail(plan=plan, slot_indices=forward_batch.out_cache_loc)
    runner.host_state.commit_plan(plan)
    runner.end_of_forward()


def _plan_from_forward_batch(
    *,
    runner: CanaryRunner,
    forward_batch: "ForwardBatch",
) -> Optional[BatchPlan]:
    """Translate a ``ForwardBatch`` into the per-slot expectations the kernel needs."""

    if forward_batch.out_cache_loc is None or forward_batch.out_cache_loc.numel() == 0:
        return None

    req_pool_indices = forward_batch.req_pool_indices.detach().cpu().tolist()
    input_ids_list: List[int] = forward_batch.input_ids.detach().cpu().tolist()
    out_cache_loc_list: List[int] = forward_batch.out_cache_loc.detach().cpu().tolist()

    forward_mode = forward_batch.forward_mode
    if forward_mode is None:
        return None
    is_extend = forward_mode.is_extend() or forward_mode.is_mixed()
    if is_extend:
        if (
            forward_batch.extend_seq_lens is None
            or forward_batch.extend_prefix_lens is None
        ):
            return None
        seq_lens = forward_batch.extend_seq_lens.detach().cpu().tolist()
        prefix_lens = forward_batch.extend_prefix_lens.detach().cpu().tolist()
    elif forward_mode.is_decode() or forward_mode.is_target_verify():
        seq_lens = [1] * len(req_pool_indices)
        full_seq_lens = forward_batch.seq_lens.detach().cpu().tolist()
        prefix_lens = [int(s) - 1 for s in full_seq_lens]
    else:
        return None

    # Padding-aware truncation: cuda-graph padded batches set padding rows'
    # req_pool_indices to 0 (the dedicated padding row) and pad input_ids on
    # the tail. ``num_token_non_padded`` (when present) tells us how many
    # tokens are real.
    num_real_tokens = _num_real_tokens(forward_batch, len(input_ids_list))
    if sum(seq_lens) != num_real_tokens:
        return None
    if len(out_cache_loc_list) != num_real_tokens:
        return None

    cursor = 0
    tokens_per_req: List[List[int]] = []
    write_slots_per_req: List[List[int]] = []
    for n in seq_lens:
        tokens_per_req.append(input_ids_list[cursor : cursor + n])
        write_slots_per_req.append(out_cache_loc_list[cursor : cursor + n])
        cursor += n

    return runner.host_state.plan_batch(
        req_pool_indices=[int(x) for x in req_pool_indices],
        req_token_counts=[int(x) for x in seq_lens],
        req_start_positions=[int(x) for x in prefix_lens],
        input_tokens_per_req=tokens_per_req,
        write_slot_indices_per_req=write_slots_per_req,
    )


def _num_real_tokens(forward_batch: "ForwardBatch", total_input_len: int) -> int:
    if hasattr(forward_batch, "num_token_non_padded_cpu"):
        value = forward_batch.num_token_non_padded_cpu
        if value is not None:
            try:
                return int(value)
            except TypeError:
                pass
    return total_input_len
