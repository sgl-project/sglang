"""Compare serial and overlapped startup with CPU-offloaded MoE weights."""

import math
import tempfile
import unittest

import torch
from startup_weight_load_test_utils import capture_worker_stderr
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

import sglang as sgl
from sglang.test.test_utils import CustomTestCase


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestCPUOffloadEngine(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.checkpoint = tempfile.TemporaryDirectory(prefix="startup-overlap-offload-")
        torch.manual_seed(42)
        config = Qwen3MoeConfig(
            vocab_size=256,
            hidden_size=128,
            intermediate_size=256,
            moe_intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=64,
            num_experts=4,
            num_experts_per_tok=2,
            max_position_embeddings=512,
            tie_word_embeddings=False,
            bos_token_id=1,
            eos_token_id=2,
        )
        model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
        model.save_pretrained(
            cls.checkpoint.name, safe_serialization=True, max_shard_size="100KB"
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "checkpoint"):
            cls.checkpoint.cleanup()

    def _generate(self, mode):
        prompts = [[1, 11, 23, 37], [1, 17, 29, 41, 53], [1] + list(range(90, 122))]
        with capture_worker_stderr() as logs:
            with sgl.Engine(
                model_path=self.checkpoint.name,
                skip_tokenizer_init=True,
                dtype="bfloat16",
                cpu_offload_gb=1,
                moe_runner_backend="auto",
                moe_a2a_backend="none",
                startup_weight_load_mode=mode,
                random_seed=42,
                cuda_graph_max_bs_decode=4,
                max_total_tokens=512,
                mem_fraction_static=0.2,
                log_level="info",
                decode_log_interval=1,
            ) as engine:
                info = engine.get_server_info()
                self.assertGreater(info["startup_time"]["cuda_graph"]["decode"], 0)
                outputs = []
                for batch in (prompts[:1], prompts, prompts):
                    outputs.append(
                        engine.generate(
                            input_ids=batch,
                            sampling_params={
                                "temperature": 0,
                                "max_new_tokens": 16,
                                "ignore_eos": True,
                            },
                            return_logprob=True,
                            logprob_start_len=0,
                        )
                    )
            logs.seek(0)
            log_text = logs.read()

        self.assertEqual(
            log_text.count("startup overlap profile=qwen3_moe_ep,"),
            0 if mode == "serial" else 1,
        )
        for batch_size in (1, 3):
            self.assertRegex(
                log_text,
                rf"Decode batch[^\n]*#running-req: {batch_size},[^\n]*cuda graph: True",
            )
        return outputs

    def test_serial_vs_overlap(self):
        reference = self._generate("serial")
        candidate = self._generate("overlap")
        self.assertEqual([len(batch) for batch in reference], [1, 3, 3])
        self.assertEqual([len(batch) for batch in candidate], [1, 3, 3])
        for before_batch, after_batch in zip(reference, candidate):
            self.assertEqual(len(before_batch), len(after_batch))
            for before, after in zip(before_batch, after_batch):
                self.assertEqual(before["output_ids"], after["output_ids"])
                self.assertEqual(len(after["output_ids"]), 16)
                for field in ("input_token_logprobs", "output_token_logprobs"):
                    left, right = before["meta_info"][field], after["meta_info"][field]
                    self.assertEqual(len(left), len(right))
                    self.assertGreater(len(left), 0)
                    for expected, actual in zip(left, right):
                        self.assertEqual(expected[1], actual[1])
                        if expected[0] is None:
                            self.assertIsNone(actual[0])
                        else:
                            self.assertTrue(math.isfinite(expected[0]))
                            self.assertTrue(math.isfinite(actual[0]))
                            self.assertAlmostEqual(expected[0], actual[0], delta=1e-5)


if __name__ == "__main__":
    unittest.main()
