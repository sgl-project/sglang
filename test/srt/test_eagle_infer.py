import unittest

import sglang as sgl


class TestEAGLEEngine(unittest.TestCase):

    def test_eagle_accuracy(self):
        prompt = "Today is a sunny day and I like"
        target_model_path = "meta-llama/Llama-2-7b-chat-hf"
        speculative_draft_model_path = "lmzheng/sglang-EAGLE-llama2-chat-7B"

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=target_model_path,
            speculative_draft_model_path=speculative_draft_model_path,
            speculative_algorithm="EAGLE",
            speculative_num_steps=3,
            speculative_eagle_topk=4,
            speculative_num_draft_tokens=16,
        )
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        engine = sgl.Engine(model_path=target_model_path)
        out2 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        print("==== Answer 1 ====")
        print(out1)

        print("==== Answer 2 ====")
        print(out2)
        self.assertEqual(out1, out2)


if __name__ == "__main__":
    unittest.main()
