import json
import unittest

import sglang as sgl

# from sglang.test.test_utils import DEFAULT_MODEL_NAME_FOR_TEST

DEFAULT_MODEL_NAME_FOR_TEST = "/shared/public/elr-models/meta-llama/Meta-Llama-3.1-8B-Instruct/07eb05b21d191a58c577b4a45982fe0c049d0693/"


class TestSRTBackend(unittest.TestCase):

    def test_engine_runtime_consistency(self):
        prompt = "Today is a sunny day and I like"
        model_path = DEFAULT_MODEL_NAME_FOR_TEST

        engine = sgl.Engine(model_path=model_path, random_seed=42)
        out1 = engine.generate(prompt, {"temperature": 0})["text"]
        engine.shutdown()

        runtime = sgl.Runtime(model_path=model_path, random_seed=42)
        out2 = json.loads(runtime.generate(prompt, {"temperature": 0}))["text"]
        runtime.shutdown()

        print("==== Answer 1 ====")
        print(out1)

        print("==== Answer 2 ====")
        print(out2)
        assert out1 == out2, f"{out1} != {out2}"


if __name__ == "__main__":
    unittest.main()
