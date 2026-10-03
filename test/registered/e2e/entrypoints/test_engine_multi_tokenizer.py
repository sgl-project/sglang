"""Engine API with --tokenizer-worker-num > 1 (issue #15157).

The parent Engine keeps generate() working through an in-process TokenizerWorker,
and other processes join the same engine with Engine.attach_tokenizer_worker().
"""

import multiprocessing as mp
import os
import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-large")

PROMPTS = ["The capital of France is", "1 + 1 ="]
SAMPLING = {"temperature": 0, "max_new_tokens": 8}


def _attached_generate(parent_pid, prompts, queue):
    engine = sgl.Engine.attach_tokenizer_worker(parent_pid)
    try:
        outs = engine.generate(prompts, SAMPLING)
        queue.put([(o["text"], o["meta_info"]["completion_tokens"]) for o in outs])
    finally:
        engine.shutdown()


class TestEngineMultiTokenizer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            random_seed=42,
            tokenizer_worker_num=2,
        )

    @classmethod
    def tearDownClass(cls):
        cls.engine.shutdown()

    def test_1_parent_generate_uses_worker(self):
        self.assertIsNotNone(self.engine.tokenizer_router)
        outs = self.engine.generate(PROMPTS, SAMPLING)
        self.assertEqual(len(outs), len(PROMPTS))
        self.assertTrue(all(o["meta_info"]["completion_tokens"] > 0 for o in outs))

    def test_2_attached_process_matches_parent(self):
        expected = [
            (o["text"], o["meta_info"]["completion_tokens"])
            for o in self.engine.generate(PROMPTS, SAMPLING)
        ]
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(
            target=_attached_generate, args=(os.getpid(), PROMPTS, queue)
        )
        proc.start()
        got = queue.get(timeout=600)
        proc.join(timeout=60)
        self.assertEqual(proc.exitcode, 0)
        self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
