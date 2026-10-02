"""Server-level coverage for the async D2H result readback (AsyncD2HCopyWorker).

Under Confidential Computing the overlap scheduler hands its per-step readback
to a worker thread and stores a HostCopyDone in `copy_done`. CI GPUs never run
CC, so without the force override nothing exercises that path end to end:
`run_batch`, `launch_batch_sample_if_needed`, and every `copy_done.synchronize()`
consumer keep taking the inline branch.

Both scheduler submit sites are covered: the inline-sample path, and the
delay-sample path via SGLANG_ENABLE_DELAY_SAMPLE.
"""

import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")

N_REQUESTS = 16
MAX_NEW_TOKENS = 32
PROMPT = "The capital of France is"


class TestAsyncD2HCopyWorkerServer(CustomTestCase):
    """python -m unittest test_async_d2h_copy_worker_server.TestAsyncD2HCopyWorkerServer"""

    process = None
    # Subclasses add their own overrides on top of the force flag.
    extra_envs: dict = {}

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_FORCE_CONFIDENTIAL_COMPUTE.override(True))
            for name, value in cls.extra_envs.items():
                stack.enter_context(getattr(envs, name).override(value))
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--mem-fraction-static", "0.7"],
            )

    @classmethod
    def tearDownClass(cls):
        if cls.process is not None:
            kill_process_tree(cls.process.pid)

    def _generate(self, idx: int) -> dict:
        payload = {
            "text": PROMPT,
            "sampling_params": {
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "logprob_start_len": -1,
        }
        r = requests.post(f"{self.base_url}/generate", json=payload, timeout=600)
        r.raise_for_status()
        return r.json()

    def test_concurrent_decode_readback(self):
        # A readback that returned the wrong step's tokens would show up as
        # diverging text across identical greedy requests.
        with ThreadPoolExecutor(max_workers=N_REQUESTS) as pool:
            results = [
                f.result()
                for f in as_completed(
                    [pool.submit(self._generate, i) for i in range(N_REQUESTS)]
                )
            ]

        texts = {r["text"] for r in results}
        self.assertEqual(len(texts), 1, f"greedy decode diverged: {texts}")
        self.assertTrue(texts.pop().strip(), "empty completion")

        for r in results:
            meta = r["meta_info"]
            self.assertEqual(meta["completion_tokens"], MAX_NEW_TOKENS)
            # The logprob readback rides the same copy as next_token_ids.
            self.assertEqual(
                len(meta["output_token_logprobs"]), meta["completion_tokens"]
            )


class TestAsyncD2HCopyWorkerServerDelaySample(TestAsyncD2HCopyWorkerServer):
    """Covers the launch_batch_sample_if_needed submit site."""

    extra_envs = {"SGLANG_ENABLE_DELAY_SAMPLE": True}


if __name__ == "__main__":
    unittest.main()
