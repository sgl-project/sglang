"""Pseudo-mode install glue.

Centralises the wiring between :class:`PseudoOracle` and the sglang
runtime. Three hook surfaces:

* ``install_sampler_override`` on ``model_runner.sample``.
* A wrap of :func:`plan_batch_from_forward_batch` (the host helper the
  canary calls before every kernel launch) so the produced
  :class:`BatchPlan` carries ``expected_write_token_ids`` /
  ``expected_write_positions`` populated by the oracle.
* Scheduler-side hooks (admit / req-pool mapping / commit / finish):

  - admit: wrap ``Scheduler._add_request_to_queue`` (the single choke
    point through which every ``Req`` enters the queue).
  - req-pool mapping: wrap ``ScheduleBatch.prepare_for_extend`` (sets
    ``req.req_pool_idx`` for every freshly-batched extend req).
  - chunk commit + step commit + finish: wrap
    ``Scheduler.process_batch_result`` (covers prefill / decode / chunked
    prefill / overlap delay-sample via the same post-step entry).
"""

from __future__ import annotations

import dataclasses
import functools
import logging
import types
from typing import TYPE_CHECKING, Dict, Optional, Set

from sglang.srt.kv_cache_canary import host_state as _canary_host_state
from sglang.srt.kv_cache_canary.api import get_runners
from sglang.srt.pseudo_mode.sampler_override import install_sampler_override

if TYPE_CHECKING:
    from sglang.srt.kv_cache_canary.host_state import BatchPlan
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.pseudo_mode.oracle import PseudoOracle

logger = logging.getLogger(__name__)

_INSTALLED_ATTR = "_pseudo_mode_installed"
_PLAN_PATCHED_ATTR = "_pseudo_mode_plan_patched"
_SCHED_ADMIT_PATCHED_ATTR = "_pseudo_mode_sched_admit_patched"
_SCHED_RESULT_PATCHED_ATTR = "_pseudo_mode_sched_result_patched"
_PREPARE_EXTEND_PATCHED_ATTR = "_pseudo_mode_prepare_extend_patched"


def install_on_model_runner(
    *,
    model_runner: "ModelRunner",
    oracle: "PseudoOracle",
    scheduler: Optional["Scheduler"] = None,
) -> None:
    """Wire all pseudo-mode hooks on ``model_runner`` (+ optional scheduler).

    The kv-cache canary must already be attached on
    ``model_runner.token_to_kv_pool`` before this call; otherwise the
    BatchPlan expected_write_* fields have no kernel consumer and the
    install is meaningless. ``scheduler`` is optional only for unit
    tests that exercise the model-runner-only surface; production
    callers must pass it so admit / commit / finish hooks fire.

    Re-entry is supported: each sub-installer maintains its own
    idempotency sentinel, so the scheduler can call this a second time
    once it has constructed the scheduler-side hooks.
    """
    pool = model_runner.token_to_kv_pool
    runners = get_runners(pool)
    if not runners:
        raise RuntimeError(
            "pseudo-mode install: kv-cache canary is not attached to "
            "model_runner.token_to_kv_pool; canary install_on_model_runner "
            "must run before pseudo_mode.install_on_model_runner"
        )

    install_sampler_override(model_runner=model_runner, oracle=oracle)
    _install_plan_patch(oracle=oracle)
    if scheduler is not None:
        _install_scheduler_hooks(scheduler=scheduler, oracle=oracle)
    elif not getattr(model_runner, _INSTALLED_ATTR, False):
        logger.warning(
            "pseudo-mode install: scheduler argument is None; admit / "
            "commit_step / finish hooks are NOT installed"
        )

    setattr(model_runner, _INSTALLED_ATTR, True)


def _install_plan_patch(*, oracle: "PseudoOracle") -> None:
    """Wrap ``host_state.plan_batch_from_forward_batch`` to fill expected_*.

    The canary api module imports ``plan_batch_from_forward_batch`` by
    name at module load, so we rebind on **both** modules to ensure
    ``run_head`` / ``prepare_replay`` pick up the wrapper.
    """
    if getattr(_canary_host_state, _PLAN_PATCHED_ATTR, False):
        return

    original_plan = _canary_host_state.plan_batch_from_forward_batch

    @functools.wraps(original_plan)
    def patched_plan(
        *,
        forward_batch: "ForwardBatch",
        config,
    ) -> Optional["BatchPlan"]:
        plan = original_plan(forward_batch=forward_batch, config=config)
        if plan is None or plan.num_write == 0:
            return plan
        try:
            expected_tokens, expected_positions = (
                oracle.predict_input_tokens_for_plan(
                    plan=plan, forward_batch=forward_batch
                )
            )
        except (KeyError, IndexError) as exc:
            # Unknown req_pool_idx / out-of-range history can happen
            # transiently around admit/finish boundaries; skip the
            # expected fill for this forward rather than crashing the
            # whole step. The canary still verifies slot identity.
            logger.warning(
                "pseudo-mode plan patch: skipping expected fill (%s)", exc
            )
            return plan

        return dataclasses.replace(
            plan,
            expected_write_token_ids=expected_tokens,
            expected_write_positions=expected_positions,
        )

    _canary_host_state.plan_batch_from_forward_batch = patched_plan
    # Refresh the name binding inside canary.api too — it did
    # ``from .host_state import plan_batch_from_forward_batch`` at load
    # time, so its local binding is the original function object.
    from sglang.srt.kv_cache_canary import api as _canary_api

    _canary_api.plan_batch_from_forward_batch = patched_plan
    setattr(_canary_host_state, _PLAN_PATCHED_ATTR, True)


def _install_scheduler_hooks(
    *,
    scheduler: "Scheduler",
    oracle: "PseudoOracle",
) -> None:
    _patch_add_request_to_queue(scheduler=scheduler, oracle=oracle)
    _patch_prepare_for_extend(oracle=oracle)
    _patch_process_batch_result(scheduler=scheduler, oracle=oracle)


def _patch_add_request_to_queue(
    *,
    scheduler: "Scheduler",
    oracle: "PseudoOracle",
) -> None:
    """Wrap ``Scheduler._add_request_to_queue`` to admit reqs to the oracle.

    Every code path that enqueues a request — normal, session, retracted —
    routes through this single method. Retracted reqs were already
    admitted on first entry; skip the second admit.
    """
    if getattr(scheduler, _SCHED_ADMIT_PATCHED_ATTR, False):
        return

    original = scheduler._add_request_to_queue

    def patched(self_scheduler, req, is_retracted: bool = False):
        if not is_retracted:
            _admit_req_to_oracle(oracle=oracle, req=req)
        return original(req, is_retracted=is_retracted)

    scheduler._add_request_to_queue = types.MethodType(patched, scheduler)
    setattr(scheduler, _SCHED_ADMIT_PATCHED_ATTR, True)


def _admit_req_to_oracle(*, oracle: "PseudoOracle", req) -> None:
    if oracle.has_req(req.rid):  # already admitted (e.g., re-entry path)
        return
    max_new_tokens = int(req.sampling_params.max_new_tokens)
    try:
        oracle.admit(
            req_id=req.rid,
            origin_input_ids=list(req.origin_input_ids),
            max_new_tokens=max_new_tokens,
        )
    except ValueError as exc:
        logger.warning("pseudo-mode admit: %s", exc)


def _patch_prepare_for_extend(*, oracle: "PseudoOracle") -> None:
    """Patch ``ScheduleBatch.prepare_for_extend`` at the class level.

    After the original allocates ``req.req_pool_idx`` for every extend
    req, register the ``req_pool_idx -> req_id`` mapping. Also record
    the chunk size for chunked prefill: prepare_for_extend installs the
    per-req extend lens on the batch via ``batch.extend_lens``; we call
    ``register_chunk_commit`` once the chunk has been consumed (we use
    the post-extend ``req.extend_input_len`` snapshot saved on
    ``req`` during prepare_for_extend).
    """
    from sglang.srt.managers.schedule_batch import ScheduleBatch

    if getattr(ScheduleBatch, _PREPARE_EXTEND_PATCHED_ATTR, False):
        return

    original = ScheduleBatch.prepare_for_extend

    @functools.wraps(original)
    def patched(self_batch: "ScheduleBatch", *args, **kwargs):
        result = original(self_batch, *args, **kwargs)
        for req in self_batch.reqs:
            if not oracle.has_req(req.rid):
                continue
            try:
                oracle.register_req_pool_mapping(
                    req_pool_idx=int(req.req_pool_idx), req_id=req.rid
                )
            except (ValueError, KeyError) as exc:
                logger.debug("pseudo-mode req-pool register skipped: %s", exc)
        return result

    ScheduleBatch.prepare_for_extend = patched
    setattr(ScheduleBatch, _PREPARE_EXTEND_PATCHED_ATTR, True)


def _patch_process_batch_result(
    *,
    scheduler: "Scheduler",
    oracle: "PseudoOracle",
) -> None:
    """Wrap ``Scheduler.process_batch_result`` for commit / chunk / finish.

    On the post-side of the original call:

    * Walk every ``req`` in ``batch.reqs``. For extend mode, advance
      the oracle's per-req ``committed_chunks`` by the chunk size that
      sglang just consumed (``req.extend_input_len`` reflects this
      forward's extend slice, set in ``prepare_for_extend``). For
      decode / final-extend, call ``commit_step`` for each newly
      appended output token.
    * For any req that ``req.finished()`` is now True, call
      ``oracle.finish`` and drop its tracking state.
    """
    if getattr(scheduler, _SCHED_RESULT_PATCHED_ATTR, False):
        return

    original = scheduler.process_batch_result
    last_output_lens: Dict[str, int] = {}

    def patched(self_scheduler, batch, result):
        rv = original(batch, result)
        _post_step_oracle_sync(
            oracle=oracle,
            batch=batch,
            last_output_lens=last_output_lens,
        )
        return rv

    scheduler.process_batch_result = types.MethodType(patched, scheduler)
    setattr(scheduler, _SCHED_RESULT_PATCHED_ATTR, True)


def _post_step_oracle_sync(
    *,
    oracle: "PseudoOracle",
    batch: "ScheduleBatch",
    last_output_lens: Dict[str, int],
) -> None:
    """Reconcile sglang per-req state with the oracle after one step.

    For each req:
    1. If the req still has uncommitted prompt chunks (oracle's
       ``committed_chunks < prefill_len``), advance them to match
       sglang's ``len(req.fill_ids) - len(req.output_ids)``.
    2. For any new ``output_ids`` entries appended since last step,
       call ``commit_step``.
    3. If ``req.finished()``, call ``oracle.finish`` and remove the
       per-req trace.
    """
    forward_mode = batch.forward_mode
    if forward_mode is not None and forward_mode.is_target_verify():
        # Spec target-verify produces no sampler output the oracle should
        # commit; accept/reject bookkeeping is the spec worker's job.
        return
    is_extend = forward_mode is not None and (
        forward_mode.is_extend() or forward_mode.is_mixed()
    )
    finished_rids: Set[str] = set()
    for req in batch.reqs:
        rid = req.rid
        if not oracle.has_req(rid):
            continue
        if is_extend:
            # ``req.fill_ids = origin_input_ids + output_ids`` after each
            # extend step. The committed prompt prefix is therefore
            # ``len(req.fill_ids) - len(req.output_ids)`` clipped to the
            # prompt length.
            prompt_len = oracle.prefill_len(rid)
            committed_prompt = max(
                0, min(prompt_len, len(req.fill_ids) - len(req.output_ids))
            )
            delta = committed_prompt - oracle.committed_chunks(rid)
            if delta > 0:
                try:
                    oracle.register_chunk_commit(req_id=rid, chunk_size=delta)
                except (ValueError, KeyError) as exc:
                    logger.debug("pseudo-mode chunk_commit skipped: %s", exc)

        new_len = len(req.output_ids)
        prev_len = last_output_lens.get(rid, 0)
        if oracle.is_in_decode(rid):
            for k in range(prev_len, new_len):
                try:
                    oracle.commit_step(
                        req_id=rid, output_token=int(req.output_ids[k])
                    )
                except (RuntimeError, KeyError) as exc:
                    logger.debug("pseudo-mode commit_step skipped: %s", exc)
                    break
        last_output_lens[rid] = new_len

        if req.finished():
            finished_rids.add(rid)

    for rid in finished_rids:
        try:
            oracle.finish(req_id=rid)
        except KeyError:
            pass
        last_output_lens.pop(rid, None)


# Re-export for tests; the public surface is the one-call install above.
__all__ = ["install_on_model_runner"]
