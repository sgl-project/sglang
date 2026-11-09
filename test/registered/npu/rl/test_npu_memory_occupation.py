"""P0 NPU tests: release/resume memory occupation for RL rollouts.

The trainer and the rollout engine are co-located, so the engine must hand HBM
back between RL steps. These tests assert that the memory is actually released
and that a resumed engine still produces the same output.
"""

import os
import unittest

import requests

from sglang.test.ascend.npu_rl_utils import (
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
    warmup_npu,
)
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

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
SMALL_DELTA_BYTES = 128 * 1024 * 1024


@unittest.skipUnless(npu_available(), "requires an Ascend NPU")
@unittest.skipUnless(os.path.isdir(MODEL), f"model not found: {MODEL}")
class TestNpuMemoryOccupation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        warmup_npu()
        cls.weight_bytes = checkpoint_weight_bytes(MODEL)

    def test_release_weights_frees_memory(self):
        with npu_server(MODEL, extra_args=MEMORY_ARGS) as base_url:
            generate(base_url, PROMPTS[0])
            before = npu_free_bytes()

            release_memory(base_url, ["weights"])

            freed = npu_free_bytes() - before
            self.assertGreater(
                freed,
                int(0.5 * self.weight_bytes),
                "release(weights) did not hand HBM back to the driver",
            )
            # /health generates a token, which is invalid while weights are
            # offloaded; /model_info only reads server metadata.
            get(base_url, "/model_info")

    def test_release_all_then_resume_restores_output(self):
        with npu_server(MODEL, extra_args=MEMORY_ARGS) as base_url:
            golden = generation_signature(base_url, PROMPTS)

            release_memory(base_url, ["weights", "kv_cache"])

            resume_memory(base_url, ["weights", "kv_cache"])
            actual = generation_signature(base_url, PROMPTS)
            self.assertEqual(actual, golden, "output changed after release/resume")

    def test_staged_resume_restores_output(self):
        with npu_server(MODEL, extra_args=MEMORY_ARGS) as base_url:
            golden = generation_signature(base_url, PROMPTS)

            release_memory(base_url, ["weights", "kv_cache"])
            resume_memory(base_url, ["weights"])
            resume_memory(base_url, ["kv_cache"])

            actual = generation_signature(base_url, PROMPTS)
            self.assertEqual(actual, golden, "staged resume changed the output")

    @unittest.skip(
        "blocked: a second resume_memory_occupation raises KeyError('weights') "
        "in the scheduler and kills the server; enable once resume is idempotent"
    )
    def test_resume_twice_keeps_server_healthy(self):
        with npu_server(MODEL, extra_args=MEMORY_ARGS) as base_url:
            golden = generation_signature(base_url, PROMPTS)

            release_memory(base_url, ["weights", "kv_cache"])
            resume_memory(base_url, ["weights", "kv_cache"])
            second = requests.post(
                base_url + "/resume_memory_occupation",
                json={"tags": ["weights", "kv_cache"]},
                timeout=120,
            )

            self.assertNotEqual(
                second.status_code, 500, "second resume crashed the server"
            )
            self.assertTrue(health(base_url))
            actual = generation_signature(base_url, PROMPTS)
            self.assertEqual(actual, golden, "double resume changed the output")

    def test_release_without_flag_is_noop(self):
        with npu_server(MODEL, extra_args=NO_MEMORY_ARGS) as base_url:
            generate(base_url, PROMPTS[0])
            before = npu_free_bytes()

            release_memory(base_url, ["weights"])

            delta = npu_free_bytes() - before
            self.assertLess(
                abs(delta),
                SMALL_DELTA_BYTES,
                "release(weights) changed memory even though --enable-memory-saver "
                "was not set; the negative control is no longer meaningful",
            )
            self.assertTrue(health(base_url))


if __name__ == "__main__":
    unittest.main()
