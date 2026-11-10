"""P0 NPU tests: release/resume memory occupation for RL rollouts.

The trainer and the rollout engine are co-located, so the engine must hand HBM
back between RL steps. These tests assert that the memory is actually released
and that a resumed engine still produces the same output.
"""

import os
import unittest

import requests

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_rl_utils import (
    NPU_RL_ENV,
    checkpoint_weight_bytes,
    generate,
    generation_signature,
    get,
    health,
    npu_available,
    npu_free_bytes,
    npu_server,
    release_memory,
    resume_memory,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=900, suite="base-b-test-1-npu-a3")
register_npu_ci(est_time=900, suite="nightly-1-npu-a3", nightly=True)

MODEL = QWEN3_0_6B_WEIGHTS_PATH
PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
)
BASE_ARGS = (
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--disable-overlap-schedule",
    "--mem-fraction-static",
    "0.45",
    "--context-length",
    "512",
    "--max-running-requests",
    "4",
)
MEMORY_ARGS = BASE_ARGS + ("--enable-memory-saver", "--enable-weights-cpu-backup")
NO_MEMORY_ARGS = BASE_ARGS


@unittest.skipUnless(npu_available(), "requires an Ascend NPU")
@unittest.skipUnless(os.path.isdir(MODEL), f"model not found: {MODEL}")
class TestNpuMemoryOccupation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.weight_bytes = checkpoint_weight_bytes(MODEL)
        # One memory-saver server for all release/resume cases. Restarting a
        # mempool-backed server repeatedly is what stresses the NPU driver.
        cls._server = npu_server(MODEL, extra_args=MEMORY_ARGS, env=NPU_RL_ENV)
        cls.base_url = cls._server.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)

    def setUp(self):
        # Failure recovery for the shared server: a failed case may leave a
        # tag offloaded, and resume is idempotent.
        resume_memory(self.base_url, ["weights", "kv_cache"])

    def test_01_release_weights_frees_memory(self):
        generate(self.base_url, PROMPTS[0])
        before = npu_free_bytes()

        try:
            release_memory(self.base_url, ["weights"])
            freed = npu_free_bytes() - before
            self.assertGreater(
                freed,
                int(0.5 * self.weight_bytes),
                "release(weights) did not hand HBM back to the driver",
            )
            # /health generates a token, which is invalid while weights are
            # offloaded; /model_info only reads server metadata.
            get(self.base_url, "/model_info")
        finally:
            resume_memory(self.base_url, ["weights"])

    def test_release_all_then_resume_restores_output(self):
        golden = generation_signature(self.base_url, PROMPTS)

        release_memory(self.base_url, ["weights", "kv_cache"])
        resume_memory(self.base_url, ["weights", "kv_cache"])

        actual = generation_signature(self.base_url, PROMPTS)
        self.assertEqual(actual, golden, "output changed after release/resume")

    def test_staged_resume_restores_output(self):
        golden = generation_signature(self.base_url, PROMPTS)

        release_memory(self.base_url, ["weights", "kv_cache"])
        resume_memory(self.base_url, ["weights"])
        resume_memory(self.base_url, ["kv_cache"])

        actual = generation_signature(self.base_url, PROMPTS)
        self.assertEqual(actual, golden, "staged resume changed the output")

    def test_resume_twice_keeps_server_healthy(self):
        golden = generation_signature(self.base_url, PROMPTS)

        release_memory(self.base_url, ["weights", "kv_cache"])
        resume_memory(self.base_url, ["weights", "kv_cache"])
        second = requests.post(
            self.base_url + "/resume_memory_occupation",
            json={"tags": ["weights", "kv_cache"]},
            timeout=120,
        )

        self.assertEqual(second.status_code, 200, second.text)
        self.assertTrue(health(self.base_url))
        actual = generation_signature(self.base_url, PROMPTS)
        self.assertEqual(actual, golden, "double resume changed the output")


@unittest.skipUnless(npu_available(), "requires an Ascend NPU")
@unittest.skipUnless(os.path.isdir(MODEL), f"model not found: {MODEL}")
class TestNpuMemoryOccupationFreshServer(CustomTestCase):
    def test_release_resume_on_fresh_server(self):
        # Keep one case on a freshly-started engine: the shared-server cases
        # cover repeated cycles, but not the first release on a clean allocator.
        with npu_server(MODEL, extra_args=MEMORY_ARGS, env=NPU_RL_ENV) as base_url:
            golden = generation_signature(base_url, PROMPTS)

            # Nothing has been released yet: resume must be a 200 no-op.
            fresh_resume = requests.post(
                base_url + "/resume_memory_occupation",
                json={"tags": ["weights"]},
                timeout=120,
            )
            self.assertEqual(fresh_resume.status_code, 200, fresh_resume.text)
            self.assertEqual(
                generation_signature(base_url, PROMPTS),
                golden,
                "fresh-server resume changed the output",
            )

            release_memory(base_url, ["weights", "kv_cache"])
            resume_memory(base_url, ["weights", "kv_cache"])

            actual = generation_signature(base_url, PROMPTS)
            self.assertEqual(actual, golden, "fresh-server resume changed the output")


@unittest.skipUnless(npu_available(), "requires an Ascend NPU")
@unittest.skipUnless(os.path.isdir(MODEL), f"model not found: {MODEL}")
class TestNpuMemoryOccupationWithoutFlag(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # No --enable-memory-saver: this server is the negative control and
        # cannot share the mempool-backed server above.
        cls._server = npu_server(MODEL, extra_args=NO_MEMORY_ARGS, env=NPU_RL_ENV)
        cls.base_url = cls._server.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)

    def test_release_without_flag_is_noop(self):
        release_memory(self.base_url, ["weights"])

        # Without --enable-memory-saver the call must not offload weights:
        # a real offload would make the next forward fail.
        self.assertTrue(health(self.base_url))
        response = generate(self.base_url, PROMPTS[0], max_new_tokens=4)
        self.assertTrue(response.get("output_ids"))


if __name__ == "__main__":
    unittest.main()
