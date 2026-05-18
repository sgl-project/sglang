from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.kv_cache_canary.config import CanaryConfig, CanaryMode
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
    """Attach canary to ``pool`` and wire it onto the running model.

    Must be called AFTER ``init_memory_pool`` and BEFORE ``init_device_graphs``
    so that the canary kernel is captured into the CUDA graph and the shadow
    tensors are baked into the graph's pointer table.
    """
    if not config.enabled:
        return None

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


def maybe_perturb_req_to_token(
    *,
    runner: Optional[CanaryRunner],
    req_to_token_pool: "ReqToTokenPool",
) -> None:
    """Self-test helper: probabilistically swap slot pointers in ``req_to_token``."""
    if runner is None:
        return
    prob = runner.config.perturb_req_to_token_prob
    if prob <= 0.0:
        return
    if random.random() >= prob:
        return
    table = req_to_token_pool.req_to_token
    if not isinstance(table, torch.Tensor) or table.numel() == 0:
        return
    rows, cols = int(table.shape[0]), int(table.shape[1])
    r = random.randrange(rows)
    a = random.randrange(cols)
    b = random.randrange(cols)
    if a == b:
        return
    tmp = table[r, a].clone()
    table[r, a] = table[r, b]
    table[r, b] = tmp
    logger.warning(
        "kv-canary self-test: perturbed req_to_token_pool[%d] columns %d <-> %d",
        r,
        a,
        b,
    )


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
    runner.commit_batch(plan)
    runner.poll_violations()


def _plan_from_forward_batch(
    *,
    runner: CanaryRunner,
    forward_batch: "ForwardBatch",
) -> Optional[BatchPlan]:
    """Translate a ``ForwardBatch`` into the per-slot expectations the kernel needs."""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    if forward_batch.out_cache_loc is None or forward_batch.out_cache_loc.numel() == 0:
        return None

    req_pool_indices = forward_batch.req_pool_indices.detach().cpu().tolist()
    input_ids_list: List[int] = forward_batch.input_ids.detach().cpu().tolist()

    forward_mode = forward_batch.forward_mode
    if forward_mode is None:
        return None
    is_extend = forward_mode.is_extend() or forward_mode.is_mixed()
    if is_extend:
        if forward_batch.extend_seq_lens is None or forward_batch.extend_prefix_lens is None:
            return None
        seq_lens = forward_batch.extend_seq_lens.detach().cpu().tolist()
        prefix_lens = forward_batch.extend_prefix_lens.detach().cpu().tolist()
    elif forward_mode.is_decode() or forward_mode.is_target_verify():
        seq_lens = [1] * len(req_pool_indices)
        full_seq_lens = forward_batch.seq_lens.detach().cpu().tolist()
        prefix_lens = [int(s) - 1 for s in full_seq_lens]
    else:
        return None

    if sum(seq_lens) != len(input_ids_list):
        return None

    cursor = 0
    tokens_per_req: List[List[int]] = []
    for n in seq_lens:
        tokens_per_req.append(input_ids_list[cursor : cursor + n])
        cursor += n

    return runner.plan_forward(
        req_pool_indices=[int(x) for x in req_pool_indices],
        req_token_counts=[int(x) for x in seq_lens],
        req_start_positions=[int(x) for x in prefix_lens],
        input_tokens_per_req=tokens_per_req,
    )
