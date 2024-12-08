"""
Usage:
python3 -m unittest test_srt_engine.TestSRTEngine.test_3_sync_streaming_combination
"""

import asyncio
import gc
import json
import unittest
from types import SimpleNamespace

import torch

import sglang as sgl
from sglang.bench_offline_throughput import BenchArgs, throughput_test
from sglang.srt.server_args import ServerArgs


class TestEAGLEEngine(unittest.TestCase):

    def test_eagle_accuracy(self):
        prompt = "Today is a sunny day and I like"
        target_model_path = "meta-llama/Llama-2-7b-chat-hf"
        draft_model_path = "kavio/Sglang-EAGLE-llama2-chat-7B"

        sampling_params = {"temperature": 0, "max_new_tokens": 8}

        engine = sgl.Engine(
            model_path=target_model_path,
            draft_model_path=draft_model_path,
            speculative_algorithm="EAGLE",
            num_speculative_steps=3,
            eagle_topk=4,
            num_draft_tokens=16,
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
