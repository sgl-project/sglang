"""Regression for the in-flight dLLM abort fix (PR #28255).

A dLLM request lives in ``dllm_manager.waiting_queue`` for its whole denoising
lifetime and is never merged into ``running_batch``. ``_process_dllm_batches``
processes prefill-phase requests first and only falls back to decode-phase
requests when no prefill-phase request is pending (the prefill-XOR-decode split),
so a decode-phase request is parked in ``dllm_manager.waiting_queue`` — absent
from ``running_batch`` and ``cur_batch`` — whenever a prefill-phase request is
pending. Before the fix ``abort_request`` scanned only the scheduler waiting
queue plus ``running_batch``/``cur_batch``, so such a parked request's abort was
silently dropped and it kept denoising to its natural ``length`` finish.

The scripted runtime drives this deterministically: start a victim and step it
into decode phase, start a long-prompt blocker whose chunked prefill keeps the
victim parked, abort the parked victim, then step to completion. With the fix the
parked ``to_finish`` is honored on the victim's next round and it finishes with
``FINISH_ABORT``; reverting the fix (``SGLANG_DEBUG_REVERT_PR=28255``) drops the
abort and the victim runs to its ``FINISH_LENGTH``.
"""

from __future__ import annotations

import os
import unittest
from typing import ClassVar, Dict, Optional

from sglang.srt.managers.schedule_batch import FINISH_ABORT, Req
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import base_engine_kwargs

register_cuda_ci(est_time=240, stage="base-b", runner_config="1-gpu-large")

_MODEL = "inclusionAI/LLaDA2.0-mini"
_CHUNK = 256
# Large enough that the victim cannot finish on its own during the few steps
# before it is parked, yet small enough that the reverted case still reaches its
# length finish quickly once unparked.
_VICTIM_MAX_NEW_TOKENS = 512


def _find_dllm_req(t: ScriptedContext, rid: str) -> Optional[Req]:
    for req in t.scheduler.dllm_manager.waiting_queue:
        if req.rid == rid:
            return req
    return None


def _script_abort_parked_decode_req(t: ScriptedContext, expect_abort: bool):
    # Victim: short prompt, reaches decode phase quickly; large max_new_tokens so
    # it stays in flight rather than finishing on its own.
    rv = t.start_req(
        prompt_len=8, max_new_tokens=_VICTIM_MAX_NEW_TOKENS, ignore_eos=True
    )
    victim: Optional[Req] = None
    for _ in range(100):
        yield
        victim = _find_dllm_req(t, rv.rid)
        if victim is not None and not victim.is_dllm_prefill():
            break
    assert (
        victim is not None and not victim.is_dllm_prefill()
    ), "victim never reached dLLM decode phase"

    # Blocker: long prompt → chunked prefill spans many rounds. While it is in
    # prefill phase the decode-phase victim is parked (prefill-XOR-decode split).
    rb = t.start_req(prompt_len=8 * _CHUNK, max_new_tokens=4, ignore_eos=True)
    blocker: Optional[Req] = None
    for _ in range(50):
        yield
        blocker = _find_dllm_req(t, rb.rid)
        if blocker is not None and blocker.is_dllm_prefill():
            break
    assert (
        blocker is not None and blocker.is_dllm_prefill()
    ), "blocker never reached dLLM prefill phase"
    assert not victim.finished(), "victim finished before it could be parked"
    assert not victim.is_dllm_prefill(), "victim left decode phase"

    # Abort the parked victim; the abort is processed on the next step while the
    # victim is still parked behind the prefilling blocker.
    t.abort(rv)
    yield

    # Step until the victim finishes (blocker drains, victim is scheduled again).
    for _ in range(2000):
        if victim.finished():
            break
        yield
    assert victim.finished(), "victim never finished after abort"

    aborted = isinstance(victim.finished_reason, FINISH_ABORT)
    assert aborted == expect_abort, (
        f"victim finished_reason={victim.finished_reason!r} "
        f"(output_len={len(victim.output_ids)}) expect_abort={expect_abort}"
    )


class _DllmAbortParkedBase(ScriptedTestCase):
    revert_pr: ClassVar[bool]
    ENGINE_KWARGS = base_engine_kwargs(
        model_path=_MODEL,
        chunked_prefill_size=_CHUNK,
        dllm_algorithm="LowConfidence",
        trust_remote_code=True,
        mem_fraction_static=0.4,
        max_running_requests=4,
        attention_backend="flashinfer",
        kv_canary="none",
        kv_canary_real_data="none",
        kv_canary_sweep_interval=0,
    )
    _env_backup: ClassVar[Dict[str, Optional[str]]] = {}

    @classmethod
    def setUpClass(cls) -> None:
        if cls is _DllmAbortParkedBase:
            raise unittest.SkipTest("abstract base; concrete subclasses set revert_pr")
        env: Dict[str, str] = {}
        if cls.revert_pr:
            env["SGLANG_DEBUG_REVERT_PR"] = "28255"
        cls._env_backup = {k: os.environ.get(k) for k in env}
        os.environ.update(env)
        super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            super().tearDownClass()
        finally:
            for key, old in cls._env_backup.items():
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old

    def _run(self, *, expect_abort: bool) -> None:
        self.server.execute_script(
            _script_abort_parked_decode_req, args=(expect_abort,)
        )


class TestDllmAbortParkedRegression(_DllmAbortParkedBase):
    """Revert PR #28255; the parked dLLM abort is dropped (finishes length)."""

    revert_pr = True

    def test_parked_decode_req_abort_dropped(self) -> None:
        self._run(expect_abort=False)


class TestDllmAbortParkedClean(_DllmAbortParkedBase):
    """With the PR #28255 fix, the parked dLLM request aborts."""

    revert_pr = False

    def test_parked_decode_req_aborts(self) -> None:
        self._run(expect_abort=True)


if __name__ == "__main__":
    unittest.main()
