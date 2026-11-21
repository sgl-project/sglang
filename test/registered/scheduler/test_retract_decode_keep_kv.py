"""Retraction with --preempt-keep-kv: accuracy holds, and a retracted request
resumes from its kept KV instead of recomputing its decoded tokens."""

import asyncio
import os
import re
import time
import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.sgl_eval_utils import run_sgl_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    STDERR_FILENAME,
    STDOUT_FILENAME,
    CustomTestCase,
    popen_launch_server,
    send_concurrent_generate_requests_with_custom_params,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=360, stage="base-b", runner_config="1-gpu-small")

_PREFILL_LINE = re.compile(r"Prefill batch.*#new-seq: (\d+), #new-token: (\d+)")
_RETRACTION_LINE = re.compile(r"#retracted_reqs: (\d+)")


class TestRetractDecodeKeepKV(CustomTestCase):
    """Forced retraction every few steps, resuming from kept KV: the eval score
    must hold, and the server must survive the strict memory check."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with (
            envs.SGLANG_TEST_RETRACT.override(True),
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
        ):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--chunked-prefill-size", "128", "--preempt-keep-kv"],
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=256,
            num_threads=32,
        )

        metrics = run_sgl_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.64)
        time.sleep(1)  # wait for mem check

        assert self.process.poll() is None, "Server crashed during test"


class TestRetractDecodeKeepKVResume(CustomTestCase):
    """A retracted request's resume prefill recomputes at most its last decoded
    token. Without the kept KV it recomputes every decoded token."""

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.stdout = open(STDOUT_FILENAME, "w")
        cls.stderr = open(STDERR_FILENAME, "w")
        with envs.SGLANG_TEST_RETRACT.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--preempt-keep-kv"],
                return_stdout_stderr=(cls.stdout, cls.stderr),
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)
        for stream, name in (
            (cls.stdout, STDOUT_FILENAME),
            (cls.stderr, STDERR_FILENAME),
        ):
            stream.close()
            os.remove(name)

    def test_resumed_prefill_is_one_token(self):
        # Two requests, so each forced retraction keeps one and retracts one;
        # distinct prompts, so neither resumes from the other's prefix.
        responses = asyncio.run(
            send_concurrent_generate_requests_with_custom_params(
                self.base_url,
                [
                    {
                        "text": f"Write a long story about the number {i}:",
                        "sampling_params": {
                            "temperature": 0,
                            "max_new_tokens": 64,
                            "ignore_eos": True,
                        },
                    }
                    for i in (7, 42)
                ],
            )
        )
        for status, body in responses:
            self.assertEqual(status, 200)
            self.assertEqual(body["meta_info"]["completion_tokens"], 64)

        retractions, resumes = 0, []
        with open(STDERR_FILENAME) as lines:
            for line in lines:
                if (m := _RETRACTION_LINE.search(line)) and int(m.group(1)) > 0:
                    retractions += 1
                elif retractions and (m := _PREFILL_LINE.search(line)):
                    resumes.append((int(m.group(1)), int(m.group(2))))
        self.assertGreater(retractions, 0)
        self.assertGreater(len(resumes), 0)
        for new_seqs, new_tokens in resumes:
            # Overlap scheduling can leave the last sampled token uncommitted,
            # so a resume may recompute one or two tokens, never the output.
            self.assertLessEqual(new_tokens, 2 * new_seqs)


if __name__ == "__main__":
    unittest.main()
