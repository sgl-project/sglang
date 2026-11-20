"""P0 NPU tests: release/resume memory occupation for RL rollouts.

The trainer and the rollout engine are co-located, so the engine must hand HBM
back between RL steps. These tests assert that the memory is actually released
and that a resumed engine still produces the same output.
"""

import unittest

from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ascend.test_npu_rl_utils import (
    generation_signature,
    get,
    npu_free_bytes,
    npu_rl_env,
    npu_server,
    post,
    release_memory,
    resume_memory,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=420, suite="extra-b-test-1-npu-a3")
register_npu_ci(est_time=420, suite="nightly-test-1-npu-a3", nightly=True)

MODEL = QWEN3_0_6B_WEIGHTS_PATH
PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
)
MEMORY_ARGS = (
    "--attention-backend",
    "ascend",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
    "--disable-overlap-schedule",
    "--mem-fraction-static",
    "0.45",
    "--context-length",
    "512",
    "--max-running-requests",
    "4",
    "--enable-memory-saver",
    "--enable-weights-cpu-backup",
)
# The sleep -> update case below must run WITHOUT the cpu backup: the backup's
# D2H copy on pause also drains the device, which would hide the unmap race it
# is meant to catch.
NO_BACKUP_ARGS = tuple(
    arg for arg in MEMORY_ARGS if arg != "--enable-weights-cpu-backup"
)


class _MemoryOccupationCases:
    def test_01_release_weights_frees_memory(self):
        before = npu_free_bytes()
        released = False

        try:
            release_memory(self.base_url, ["weights"])
            released = True
            after_release = npu_free_bytes()
            self.assertGreater(
                after_release - before,
                int(0.5 * 2**30),
                "release(weights) did not hand HBM back to the driver",
            )
            # /health generates a token, which is invalid while weights are
            # offloaded; /model_info only reads server metadata.
            get(self.base_url, "/model_info")
        finally:
            if released:
                resume_memory(self.base_url, ["weights"])

        reallocated = after_release - npu_free_bytes()
        self.assertGreater(
            reallocated,
            int(0.4 * 2**30),
            "resume(weights) did not reallocate HBM",
        )
        self.assertEqual(
            generation_signature(self.base_url, PROMPTS),
            self.golden,
            "output changed after release/resume of weights",
        )

    def test_release_all_then_resume_restores_output(self):
        before = npu_free_bytes()

        release_memory(self.base_url, ["weights", "kv_cache"])
        self.assertGreater(
            npu_free_bytes() - before,
            int(1.5 * 2**30),
            "release(weights, kv_cache) did not hand HBM back to the driver",
        )
        resume_memory(self.base_url, ["weights", "kv_cache"])

        actual = generation_signature(self.base_url, PROMPTS)
        self.assertEqual(actual, self.golden, "output changed after release/resume")

    def test_staged_resume_restores_output(self):
        release_memory(self.base_url, ["weights", "kv_cache"])
        resume_memory(self.base_url, ["weights"])
        resume_memory(self.base_url, ["kv_cache"])

        actual = generation_signature(self.base_url, PROMPTS)
        self.assertEqual(actual, self.golden, "staged resume changed the output")


class TestNpuMemoryOccupation(_MemoryOccupationCases, CustomTestCase):
    @classmethod
    def setUpClass(cls):
        # One memory-saver server for all release/resume cases. Restarting a
        # mempool-backed server repeatedly is what stresses the NPU driver.
        cls._server = npu_server(MODEL, extra_args=MEMORY_ARGS, env=npu_rl_env())
        cls.base_url = cls._server.__enter__()
        cls.golden = generation_signature(cls.base_url, PROMPTS)

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)


class _SleepUpdateCases:
    def _update_weights_from_disk(self):
        post(self.base_url, "/update_weights_from_disk", {"model_path": MODEL})

    def test_sleep_update_matches_no_sleep_reference(self):
        """release(all) -> resume -> update must not change the output.

        Without the CPU backup, resume hands back freshly committed pages: a
        device still touching them when they were unmapped corrupts the model,
        and the next weight update cannot recover it. This is the RL round trip
        that used to produce stable garbage on Ascend.
        """
        self._update_weights_from_disk()
        reference = generation_signature(self.base_url, PROMPTS)

        release_memory(self.base_url, ["weights", "kv_cache"])
        resume_memory(self.base_url, ["weights", "kv_cache"])
        self._update_weights_from_disk()
        actual = generation_signature(self.base_url, PROMPTS)

        self.assertEqual(
            actual, reference, "sleep -> resume -> update changed the output"
        )


class TestNpuSleepUpdateNoBackup(_SleepUpdateCases, CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls._server = npu_server(MODEL, extra_args=NO_BACKUP_ARGS, env=npu_rl_env())
        cls.base_url = cls._server.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._server.__exit__(None, None, None)


if __name__ == "__main__":
    unittest.main()
