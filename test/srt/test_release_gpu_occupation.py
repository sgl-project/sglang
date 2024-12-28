import time
import unittest

import sglang as sgl
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
)

_DEBUG_SLEEP = False  # set to true to observe memory usage in nvidia-smi more clearly


class TestReleaseGPUOccupation(unittest.TestCase):
    def test_release_and_resume_occupation(self):
        prompt = "Today is a sunny day and I like"
        expect_output = ' to spend it outdoors. I decided to'
        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            random_seed=42,
        )

        outputs = engine.generate(prompt, sampling_params)["text"]
        self.assertEqual(outputs, expect_output)

        if _DEBUG_SLEEP:
            time.sleep(3)

        t = time.time()
        engine.release_gpu_occupation()
        print('release_gpu_occupation', time.time() - t)

        if _DEBUG_SLEEP:
            time.sleep(3)

        t = time.time()
        engine.resume_gpu_occupation()
        print('resume_gpu_occupation', time.time() - t)

        outputs = engine.generate(prompt, sampling_params)["text"]
        self.assertEqual(outputs, expect_output)

        if _DEBUG_SLEEP:
            time.sleep(5)

        engine.shutdown()


if __name__ == "__main__":
    unittest.main()
