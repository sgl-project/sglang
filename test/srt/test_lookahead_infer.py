import unittest

import sglang as sgl


class TestLOOKAHEADEngine(unittest.TestCase):

    def test_lookahead_accuracy(self):
        prompt = "[INST] <<SYS>>\\nYou are a helpful assistant.\\n<</SYS>>\\nWho are you? [/INST]"
        target_model_path = "meta-llama/Llama-2-7b-chat-hf"

        sampling_params = {"temperature": 0.0001, "max_new_tokens": 20, "top_k": 1}

        engine = sgl.Engine(
            model_path=target_model_path,
            speculative_algorithm="LOOKAHEAD",
            speculative_num_draft_tokens=4,
            speculative_one_branch=True,
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
