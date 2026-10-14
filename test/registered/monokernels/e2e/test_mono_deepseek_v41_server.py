"""DeepSeek-V4.1-Flash with --enable-monokernel on MI35x, TP4 (RFC #43949), across
the switch points: verify steps with DSpark (8 requests of 6 rows at the largest
width) and decode steps without it (48 rows), each with batches that step past the
largest width and back (a short GSM8K run, for gross failures) and the tokens of
covered steps scored again by the normal prefill path (KL consistency); and a fault
on one rank that stops the server before any token of the failed step is sent.
Every backbone layer runs whole; the kernel checks cover each form of K1's input."""

import os
import re
import tempfile
import time
import unittest
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.kl_test_utils import (
    test_input_output_logprobs_match_helper as assert_logprobs_match,
)
from sglang.test.monokernels.deepseek_v41.server import (
    ARGS,
    DECODE_ARGS,
    ENV,
    MODEL,
)
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_amd_ci(est_time=3600, stage="stage-c", runner_config="large-8-gpu-amd-mi35x")
# the spec's nightly job (RFC #43949: a spec red for four weeks is refused)
register_amd_ci(est_time=3600, suite="nightly-amd-8-gpu-mi35x", nightly=True)

LAUNCH_TIMEOUT = 3600
# The normal path scores 0.88 to 0.91 on this recipe: the bound is for gross failures
GSM8K_BOUND = 0.85
# Decode against prefill on 8 LongBench-v2 prompts, mean KL: about 0.007 on the
# normal path, 0.011 with the monokernels (up to 0.022 for one prompt)
KL_BOUND = 0.03


def _launch(args=ARGS, extra_env=None, extra_args=(), log=None):
    env = {**os.environ, **ENV, **(extra_env or {})}
    return popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=LAUNCH_TIMEOUT,
        other_args=args + ["--enable-monokernel", *extra_args],
        env=env,
        return_stdout_stderr=log,
    )


class _Serving:
    """A server's switch points: ``PARALLEL`` requests at a time step past the
    largest covered width as the batch fills, and back as it drains."""

    SERVER_ARGS: list
    PARALLEL: int

    @classmethod
    def setUpClass(cls):
        cls.process = _launch(cls.SERVER_ARGS, extra_args=["--watchdog-timeout", "600"])

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "process", None) is not None:
            terminate_and_kill_process_tree(cls.process)

    def test_gsm8k_past_the_largest_width(self):
        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=200,
            parallel=self.PARALLEL,
            max_new_tokens=512,
            host="127.0.0.1",
            port=int(DEFAULT_URL_FOR_TEST.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], GSM8K_BOUND)

    def test_covered_steps_match_prefill(self):
        # 8 requests in one batch: every decode or verify step runs at a covered
        # width, and the prefill that scores the tokens again is the normal path
        assert_logprobs_match(
            DEFAULT_URL_FOR_TEST,
            {MODEL: {"kl_div": KL_BOUND}},
            MODEL,
            max_samples=8,
            max_new_tokens=256,
            trust_remote_code=True,
        )


class TestMonokernelVerify(_Serving, CustomTestCase):
    """DSpark verify steps: above 8 requests (48 rows) the normal path."""

    SERVER_ARGS, PARALLEL = ARGS, 12


class TestMonokernelDecode(_Serving, CustomTestCase):
    """Decode steps without DSpark: above 48 rows the normal path."""

    SERVER_ARGS, PARALLEL = DECODE_ARGS, 56


class TestMonokernelFault(CustomTestCase):
    def test_a_stalled_rank_stops_the_server_without_tokens(self):
        # past startup (5000 launches), rank 1 stalls at its first wait; at the
        # deadline (half the watchdog) it stops, and its peers stop on its flag
        with (
            tempfile.NamedTemporaryFile("w+", suffix=".out") as out,
            tempfile.NamedTemporaryFile("w+", suffix=".err") as err,
        ):
            process = _launch(
                extra_env={"SGLANG_TEST_MONO_FAULT_INJECT": "1:1:5000"},
                extra_args=["--watchdog-timeout", "20"],
                log=(out, err),
            )
            try:
                try:
                    r = requests.post(
                        DEFAULT_URL_FOR_TEST + "/generate",
                        json={
                            "text": "Write a long essay about lighthouses.",
                            "sampling_params": {
                                "max_new_tokens": 1024,
                                "temperature": 0,
                                "ignore_eos": True,
                            },
                        },
                        timeout=300,
                    )
                    ok = r.status_code == 200 and r.json().get("text")
                except requests.RequestException:
                    ok = False
                self.assertFalse(ok, "a failed step's tokens reached the client")
                deadline = time.time() + 300
                while process.poll() is None and time.time() < deadline:
                    time.sleep(2)
                self.assertIsNotNone(process.poll(), "the server kept running")
                err.seek(0)
                out.seek(0)
                log = err.read() + out.read()
                # a launch never runs on after a failed wait: it stops with a store
                # to the null page, so the only faults are at address 0x0, none on
                # words that never came
                faults = re.findall(r"faulting addr: (0x[0-9a-fA-F]+)", log)
                self.assertTrue(faults, log[-2000:])
                self.assertEqual(set(faults) - {"0x0"}, set(), faults)
            finally:
                terminate_and_kill_process_tree(process)


if __name__ == "__main__":
    unittest.main()
