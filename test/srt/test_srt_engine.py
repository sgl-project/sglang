import json
import unittest

import sglang as sgl
from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST


class TestSRTBackend(unittest.TestCase):

    def test_engine_runtime_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_MODEL_NAME_FOR_TEST

        # without sampling_backend="pytorch", flashinfer sampling introduces randomness
        engine = sgl.Engine(
            model_path=model_path, random_seed=42, sampling_backend="pytorch"
        )
        out1 = engine.generate(prompt, {"temperature": 0})["text"]
        engine.shutdown()

        runtime = sgl.Runtime(
            model_path=model_path, random_seed=42, sampling_backend="pytorch"
        )
        out2 = json.loads(runtime.generate(prompt, {"temperature": 0}))["text"]
        runtime.shutdown()

        assert out1 == out2, f"{out1} != {out2}"


if __name__ == "__main__":
    unittest.main()
