"""Tests that release_memory_occupation and resume_memory_occupation surface
state-machine errors (resume-without-release, double-release) as a clean
RuntimeError on the Engine surface, instead of crashing the scheduler with
an AssertionError or KeyError.
"""

import unittest

import sglang as sgl
from sglang.srt.constants import (
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_cuda_ci(est_time=120, suite="stage-b-test-1-gpu-small")


class TestReleaseResumeStateErrors(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            random_seed=42,
            enable_memory_saver=True,
            mem_fraction_static=0.6,
            disable_cuda_graph=True,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "engine") and cls.engine is not None:
            cls.engine.shutdown()

    def test_resume_without_prior_release_errors(self):
        with self.assertRaises(RuntimeError) as cm:
            self.engine.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])
        self.assertIn("not currently released", str(cm.exception))

    def test_resume_other_tag_while_only_one_released_errors(self):
        self.engine.release_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])
        try:
            with self.assertRaises(RuntimeError) as cm:
                self.engine.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_WEIGHTS])
            self.assertIn("not currently released", str(cm.exception))
        finally:
            self.engine.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])

    def test_double_release_errors(self):
        self.engine.release_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])
        try:
            with self.assertRaises(RuntimeError) as cm:
                self.engine.release_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])
            self.assertIn("already released", str(cm.exception))
        finally:
            self.engine.resume_memory_occupation(tags=[GPU_MEMORY_TYPE_KV_CACHE])

    def test_release_then_resume_cycle_still_serves(self):
        # The error-path changes should not regress the existing happy path:
        # a paired release / resume cycle still leaves the engine able to
        # generate.
        self.engine.release_memory_occupation(
            tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS],
        )
        self.engine.resume_memory_occupation(
            tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS],
        )
        out = self.engine.generate(
            prompt="Hello",
            sampling_params={"temperature": 0, "max_new_tokens": 4},
        )
        self.assertIsInstance(out["text"], str)
        self.assertGreater(len(out["text"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=3)
