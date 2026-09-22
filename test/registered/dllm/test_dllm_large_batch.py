"""Large-batch dLLM coverage.

The other dLLM e2e tests run at 1-4 concurrent requests. Several scheduler and
denoise paths only exist above that: the retained-KV gather batches above a row
count, the denoise reduction switches formulation above a row count, and mixed
prefill/decode rounds need enough concurrent requests for a prefill row and a
decode row to coexist at all. None of that is reachable at bs=4, so this runs
the same gsm8k check at a batch wide enough to enter them -- the RL-rollout
shape this model is served in.

The mixed-round flag is parametrized here rather than unit-tested: round
composition needs a live Scheduler, PrefillAdder and both memory pools.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=500, stage="base-b", runner_config="1-gpu-large")

import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

MODEL = "inclusionAI/LLaDA2.0-mini"
# 64 concurrent requests x a 32-token denoise block = 2048 rows per forward,
# past every row-count threshold on the dLLM path. Larger would be more
# representative of a rollout but would not cover anything more.
# Each concurrent request costs memory outside the static pool, so raising
# this means lowering --mem-fraction-static with it.
MAX_RUNNING_REQUESTS = 64
# 400 (not the 200 the bs=4 tests use): unmask order near LowConfidence's
# gate shifts at this batch, so halve the per-example score granularity.
GSM8K_NUM_EXAMPLES = 400
GSM8K_SCORE_THRESHOLD = 0.88


def assert_mixed_rounds(test, expected: bool):
    """Assert on what the scheduler actually built, not just what it resolved.

    ``dllm_mixed_batch_enabled`` is a startup constant: a server that resolved
    it to True but never put a prefill row and a decode row in the same round
    would satisfy it identically. ``dllm_num_mixed_rounds`` counts the rounds
    whose can_run_list held both phases, so it separates the two. Call this
    after traffic has flowed -- on an idle server every counter is 0.
    """
    states = requests.get(test.base_url + "/server_info").json()["internal_states"]
    test.assertTrue(states, "server reported no internal states")
    for state in states:
        test.assertEqual(state["dllm_mixed_batch_enabled"], expected)
        test.assertGreater(state["dllm_num_rounds"], 0, "no dLLM round was scheduled")
        if expected:
            test.assertGreater(
                state["dllm_num_mixed_rounds"], 0, "no round mixed the two phases"
            )
        else:
            test.assertEqual(state["dllm_num_mixed_rounds"], 0)


def _server_args():
    return [
        "--trust-remote-code",
        "--tp-size",
        "1",
        "--mem-fraction-static",
        "0.9",
        "--max-running-requests",
        str(MAX_RUNNING_REQUESTS),
        "--attention-backend",
        "flashinfer",
        "--dllm-algorithm",
        "LowConfidence",
        "--dllm-fdfo",
        "--cuda-graph-bs-decode",
        "1",
        "8",
        "16",
        "32",
        "64",
    ]


class _Gsm8kAtLargeBatch:
    """gsm8k at MAX_RUNNING_REQUESTS, shared by both server configurations.

    Deliberately not a CustomTestCase subclass: a shared base that defined
    test_gsm8k *and* inherited the case class would be collected on its own and
    launch a third server.
    """

    label = ""
    mixed_rounds = False

    def _run_gsm8k(self, label: str):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=GSM8K_NUM_EXAMPLES,
            num_threads=MAX_RUNNING_REQUESTS,
        )
        metrics = run_eval(args)
        print(f"[{label}] {metrics=}")
        if is_in_ci():
            write_github_step_summary(
                f"### {label} (llada2-mini, bs={MAX_RUNNING_REQUESTS})\n"
                f"score={metrics['score']:.4f}\n"
            )
        self.assertGreater(metrics["score"], GSM8K_SCORE_THRESHOLD)
        # After traffic, not before: the round counters need rounds to have run.
        assert_mixed_rounds(self, self.mixed_rounds)

    def test_gsm8k(self):
        self._run_gsm8k(self.label)


def _launch(cls):
    cls.model = MODEL
    cls.base_url = DEFAULT_URL_FOR_TEST
    cls.process = popen_launch_server(
        cls.model,
        cls.base_url,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=_server_args(),
    )


class TestDllmLargeBatch(_Gsm8kAtLargeBatch, CustomTestCase):
    """dLLM at a batch wide enough to reach the large-batch code paths."""

    label = "large batch"
    mixed_rounds = False

    @classmethod
    def setUpClass(cls):
        _launch(cls)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestDllmLargeBatchMixedRound(_Gsm8kAtLargeBatch, CustomTestCase):
    """Same batch with prefill and decode rows scheduled into one round.

    Accuracy must be unaffected: mixing changes which rows share a forward, not
    what any row denoises.
    """

    label = "large batch, mixed rounds"
    mixed_rounds = True

    @classmethod
    def setUpClass(cls):
        # The server has to be spawned inside the override so it inherits it.
        with envs.SGLANG_ENABLE_DLLM_MIXED_BATCH.override(True):
            _launch(cls)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main(verbosity=3)
