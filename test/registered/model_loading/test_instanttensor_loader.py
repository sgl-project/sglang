import multiprocessing as mp
import os
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.runners import SRTRunner, check_close_model_outputs
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="weekly", runner_config="1-gpu-small")

MODEL = os.getenv("SGLANG_INSTANTTENSOR_TEST_MODEL", "Qwen/Qwen2-0.5B")


class TestInstantTensorLoader(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def test_generation_matches_safetensors(self):
        prompts = ["The capital of France is"]
        runner_args = dict(
            torch_dtype=torch.float16,
            model_type="generation",
            disable_cuda_graph=True,
            disable_radix_cache=True,
            max_total_tokens=256,
        )

        with SRTRunner(MODEL, load_format="safetensors", **runner_args) as runner:
            expected = runner.forward(prompts, max_new_tokens=16)

        configs = [
            {},
            {
                "backend": "URING",
                "chunk_size": 8 * 1024 * 1024,
                "io_depth": 64,
            },
        ]
        for extra_config in configs:
            with self.subTest(extra_config=extra_config):
                with SRTRunner(
                    MODEL,
                    load_format="instanttensor",
                    model_loader_extra_config=extra_config,
                    **runner_args,
                ) as runner:
                    actual = runner.forward(prompts, max_new_tokens=16)

                self.assertEqual(expected.output_strs, actual.output_strs)
                check_close_model_outputs(
                    hf_outputs=expected,
                    srt_outputs=actual,
                    prefill_tolerance=1e-6,
                    decode_tolerance=1e-6,
                    rouge_l_tolerance=1.0,
                    debug_text="safetensors vs InstantTensor weight loading",
                )


if __name__ == "__main__":
    unittest.main()
