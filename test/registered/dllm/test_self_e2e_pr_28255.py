"""Regression for PR #28255: abort in-flight dLLM requests in the waiting queue.

A dLLM request lives in ``dllm_manager.waiting_queue`` for its whole denoising
lifetime and is never merged into ``running_batch``. ``_process_dllm_batches``
processes prefill-phase requests first and only falls back to decode-phase
requests when no prefill-phase request is pending (the prefill-XOR-decode split),
so a decode-phase request is parked in ``dllm_manager.waiting_queue`` — absent
from ``running_batch`` and ``cur_batch`` — whenever a prefill-phase request is
pending. Before the fix, ``abort_request`` scanned only the (already-emptied)
scheduler waiting queue plus ``running_batch``/``cur_batch``, so such a parked
request's abort was silently dropped and it kept denoising to its natural
``length`` finish.

This test drives a short "victim" request into decode phase, then continuously
floods the server with long, *unique* (uncached) prompts so a prefill-phase
request is always pending. The decode-phase victim is therefore continuously
parked in ``dllm_manager.waiting_queue`` (absent from ``running_batch`` and
``cur_batch``) and barely denoises, so it does not finish on its own before the
abort. It aborts the victim mid-flood, then stops the flood so the victim is
scheduled again. With the fix the parked ``to_finish`` is honored on the victim's
next round and it finishes ``abort``; reverting the fix
(``SGLANG_DEBUG_REVERT_PR=28255``) drops the abort and the victim runs to its
``length`` finish.
"""

from __future__ import annotations

import os
import threading
import time
import unittest
from typing import ClassVar, Optional

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="1-gpu-large")

_MODEL = "inclusionAI/LLaDA2.0-mini"
_VICTIM_RID = "dllm-decode-victim"
# Large enough that the victim cannot reach its 'length' finish on its own before
# the abort (it only denoises briefly before the flood parks it), yet small
# enough that, once unparked, the reverted case still finishes within the join.
_VICTIM_MAX_NEW_TOKENS = 2048


def _unique_flood_prompt(index: int) -> str:
    # Long and unique per request so it does not hit the radix cache; each flood
    # request then prefills block-by-block for many rounds, keeping a
    # prefill-phase request continuously pending.
    return " ".join(
        f"flood-{index}-segment-{j}: distinct phrase number {index * 1000 + j}"
        for j in range(160)
    )


class _DllmAbortInFlightBase(CustomTestCase):
    revert_pr: ClassVar[bool]
    base_url: ClassVar[str] = DEFAULT_URL_FOR_TEST
    process: ClassVar[Optional[object]] = None

    @classmethod
    def setUpClass(cls) -> None:
        if cls is _DllmAbortInFlightBase:
            raise unittest.SkipTest("abstract base; concrete subclasses set revert_pr")
        env = os.environ.copy()
        if cls.revert_pr:
            env["SGLANG_DEBUG_REVERT_PR"] = "28255"
        cls.process = popen_launch_server(
            _MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp-size",
                "1",
                # Two subclasses launch their servers sequentially in one process;
                # a modest fraction lets the second start before the first's GPU
                # memory is fully released (avoids a cross-subclass OOM).
                "--mem-fraction-static",
                "0.4",
                # Victim + a few concurrent flood prefills keep a prefill-phase
                # request continuously pending.
                "--max-running-requests",
                "4",
                "--chunked-prefill-size",
                "256",
                "--attention-backend",
                "flashinfer",
                "--dllm-algorithm",
                "LowConfidence",
                "--cuda-graph-bs",
                "1",
                "2",
                "3",
                "4",
            ],
            env=env,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    def _post_generate(
        self, *, rid: str, prompt: str, max_new_tokens: int, result: dict
    ) -> None:
        try:
            resp = requests.post(
                f"{self.base_url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": 0.0,
                        "ignore_eos": True,
                    },
                    "rid": rid,
                    "stream": False,
                },
                timeout=180,
            )
            result["status_code"] = resp.status_code
            if resp.status_code == 200:
                result["finish_reason"] = (
                    resp.json().get("meta_info", {}).get("finish_reason")
                )
        except requests.RequestException as exc:
            result["error"] = repr(exc)

    def test_abort_parked_decode_phase_dllm_request(self) -> None:
        victim: dict = {}
        victim_thread = threading.Thread(
            target=self._post_generate,
            kwargs=dict(
                rid=_VICTIM_RID,
                prompt="Briefly say hello.",
                max_new_tokens=_VICTIM_MAX_NEW_TOKENS,
                result=victim,
            ),
        )
        victim_thread.start()
        # Briefly let the victim finish its short prefill and enter decode phase
        # (short, so it has barely denoised before the flood parks it).
        time.sleep(0.5)

        # Continuously submit unique long-prompt requests so a prefill-phase
        # request is always pending, parking the decode-phase victim.
        stop_flood = threading.Event()
        flood_threads: list[threading.Thread] = []

        def flood() -> None:
            index = 0
            while not stop_flood.is_set():
                thread = threading.Thread(
                    target=self._post_generate,
                    kwargs=dict(
                        rid=f"dllm-flood-{index}",
                        prompt=_unique_flood_prompt(index),
                        max_new_tokens=4,
                        result={},
                    ),
                )
                thread.start()
                flood_threads.append(thread)
                index += 1
                time.sleep(0.1)

        flood_driver = threading.Thread(target=flood)
        flood_driver.start()
        # Keep the victim parked for a while before aborting it.
        time.sleep(2.5)

        # Abort the victim while it is parked behind pending prefill requests.
        abort_resp = requests.post(
            f"{self.base_url}/abort_request", json={"rid": _VICTIM_RID}, timeout=10
        )
        self.assertEqual(abort_resp.status_code, 200)

        # Hold the victim parked a bit longer so the abort is processed while it
        # is still parked, then stop the flood so the victim is scheduled again.
        time.sleep(1.0)
        stop_flood.set()
        flood_driver.join(timeout=30)
        for thread in flood_threads:
            thread.join(timeout=180)
        victim_thread.join(timeout=180)

        self.assertEqual(victim.get("status_code"), 200, victim)
        finish_reason = victim.get("finish_reason")
        self.assertIsNotNone(finish_reason, f"missing victim finish_reason: {victim}")

        if self.revert_pr:
            # Bug reintroduced: the parked dLLM abort is dropped, so the victim
            # runs to its natural max_new_tokens finish.
            self.assertEqual(
                finish_reason.get("type"),
                "length",
                f"expected 'length' (abort silently dropped), got {finish_reason}",
            )
        else:
            self.assertEqual(
                finish_reason.get("type"),
                "abort",
                f"expected 'abort' (parked dLLM request aborted), got {finish_reason}",
            )


class TestDllmAbortInFlightRegression(_DllmAbortInFlightBase):
    """Revert PR #28255; the parked dLLM abort is dropped (finishes 'length')."""

    revert_pr = True


class TestDllmAbortInFlightClean(_DllmAbortInFlightBase):
    """With the PR #28255 fix, the parked dLLM request aborts ('abort')."""

    revert_pr = False


if __name__ == "__main__":
    unittest.main()
