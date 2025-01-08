import unittest

import sglang as sgl


class TestEngineWithNGramSpeculativeDecoding(unittest.TestCase):

    def test_ngram_accuracy(self):
        prompt = "Today is a sunny day and I like"
        target_model_path = "daryl149/llama-2-7b-chat-hf"

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(model_path=target_model_path, disable_cuda_graph=True)
        out2 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        engine = sgl.Engine(
            model_path=target_model_path,
            speculative_algorithm="NGRAM",
            speculative_ngram_window_size=3,
            speculative_num_draft_tokens=16,
            watchdog_timeout=10000000,
            disable_cuda_graph=True,
        )
        out1 = engine.generate(prompt, sampling_params)["text"]
        engine.shutdown()

        print("==== Answer 1 ====")
        print(out1)

        print("==== Answer 2 ====")
        print(out2)
        self.assertEqual(out1, out2)


if __name__ == "__main__":
    unittest.main()
