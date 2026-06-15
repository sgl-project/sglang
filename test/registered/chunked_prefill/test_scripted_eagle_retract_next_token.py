from __future__ import annotations

import os
import unittest
from typing import ClassVar, Dict, List, Optional

import torch

from sglang.srt.kv_canary.config import CanaryMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.scripted_runtime.context import ScriptedContext
from sglang.test.scripted_runtime.test_case import ScriptedTestCase
from sglang.test.scripted_runtime_chunked_helpers import (
    advance_to_decode_step,
    base_engine_kwargs,
    run_until_finished,
)

register_cuda_ci(est_time=320, stage="extra-a", runner_config="1-gpu-small")

_CHUNK = 32
_PROMPT_LEN = 64
_DECODE_STEPS = 300
_MAX_NEW_TOKENS = 360
_TEMPERATURE = 2.0
_DECODE_MAX_STEPS = 900


def _eagle_kwargs() -> Dict[str, object]:
    return base_engine_kwargs(
        model_path="Qwen/Qwen3-0.6B",
        chunked_prefill_size=_CHUNK,
        speculative_algorithm="EAGLE",
        speculative_num_steps=3,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=4,
        mem_fraction_static=0.3,
        kv_canary=CanaryMode.LOG.value,
        kv_canary_real_data="none",
        kv_canary_sweep_interval=0,
        context_length=16384,
        max_total_tokens=65536,
    )


def _canary_violation_count(t: ScriptedContext) -> int:
    scheduler = t.scheduler
    managers: List[object] = []
    target = scheduler.tp_worker.model_runner.canary_manager
    if target is not None:
        managers.append(target)
    if scheduler.draft_worker is not None:
        draft = scheduler.draft_worker.draft_worker.draft_runner.canary_manager
        if draft is not None:
            managers.append(draft)
    torch.cuda.synchronize()
    return sum(
        int(m._device_state.violation_log.violation_write_index.cpu().item())
        for m in managers
    )


def _script_retract_resume_output_region(t: ScriptedContext, expect_violation: bool):
    r = t.start_req(
        prompt_len=_PROMPT_LEN,
        max_new_tokens=_MAX_NEW_TOKENS,
        ignore_eos=True,
        temperature=_TEMPERATURE,
    )
    yield from advance_to_decode_step(r, _DECODE_STEPS, max_steps=_DECODE_MAX_STEPS)
    chunks_before = r.chunks_done
    t.pause_generation(mode="retract")
    t.continue_generation()
    yield from run_until_finished(r, max_steps=2000)
    assert r.finished
    assert r.chunks_done - chunks_before >= 2, (
        f"resume did not re-chunk into the output region: chunks_done "
        f"{chunks_before} -> {r.chunks_done}"
    )

    violations = _canary_violation_count(t)
    if expect_violation:
        assert violations > 0, (
            "expected a kv_canary verify_token violation from the wrong tail token, "
            f"got {violations}"
        )
    else:
        assert violations == 0, f"unexpected kv_canary violation(s): {violations}"


class _EagleRetractNextTokenBase(ScriptedTestCase):
    revert_pr: ClassVar[bool]
    ENGINE_KWARGS = _eagle_kwargs()
    _env_backup: ClassVar[Dict[str, Optional[str]]] = {}

    @classmethod
    def setUpClass(cls) -> None:
        if cls is _EagleRetractNextTokenBase:
            raise unittest.SkipTest("abstract base; concrete subclasses set revert_pr")
        cls._env_backup = {
            k: os.environ.get(k)
            for k in (
                "SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT",
                "SGLANG_DEBUG_REVERT_PR",
            )
        }
        os.environ["SGLANG_KV_CANARY_ENABLE_VERIFY_TOKEN_ASSERT"] = "1"
        if cls.revert_pr:
            os.environ["SGLANG_DEBUG_REVERT_PR"] = "28254"
        else:
            os.environ.pop("SGLANG_DEBUG_REVERT_PR", None)
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

    def _run(self, *, expect_violation: bool) -> None:
        self.server.execute_script(
            _script_retract_resume_output_region, args=(expect_violation,)
        )


class TestEagleRetractNextTokenRegression(_EagleRetractNextTokenBase):
    revert_pr = True

    def test_retract_resume_output_region_fires_canary(self) -> None:
        self._run(expect_violation=True)


class TestEagleRetractNextTokenClean(_EagleRetractNextTokenBase):
    revert_pr = False

    def test_retract_resume_output_region_clean(self) -> None:
        self._run(expect_violation=False)


if __name__ == "__main__":
    unittest.main()
