from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-a", runner_config="1-gpu-small")

import unittest

from sglang.benchmark.one_batch import (
    BenchArgs,
    PortArgs,
    ServerArgs,
    correctness_test,
)
from sglang.test.test_utils import CustomTestCase


class TestFinalLayerPrefillOptimization(CustomTestCase):
    def test_llama_correctness(self):
        """Verify exact token match and logit parity on LLaMA architecture."""
        server_args = ServerArgs(
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v0.4",
            load_format="dummy",
            device="cuda",
        )
        port_args = PortArgs()
        bench_args = BenchArgs(
            run_name="test_llama_parity",
            batch_size=[1],
            input_len=[256],
            output_len=[32],
            correct=True,
        )
        correctness_test(server_args, port_args, bench_args, gpu_id=0, tp_rank=0)

    def test_qwen_correctness(self):
        """Verify exact token match and logit parity on Qwen architecture."""
        server_args = ServerArgs(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            load_format="dummy",
            device="cuda",
        )
        port_args = PortArgs()
        bench_args = BenchArgs(
            run_name="test_qwen_parity",
            batch_size=[1],
            input_len=[256],
            output_len=[32],
            correct=True,
        )
        correctness_test(server_args, port_args, bench_args, gpu_id=0, tp_rank=0)

    def test_glm_correctness(self):
        """Verify exact token match and logit parity on GLM architecture."""
        server_args = ServerArgs(
            model_path="THUDM/glm-4-9b-chat-hf",
            load_format="dummy",
            device="cuda",
        )
        port_args = PortArgs()
        bench_args = BenchArgs(
            run_name="test_glm_parity",
            batch_size=[1],
            input_len=[256],
            output_len=[32],
            correct=True,
        )
        correctness_test(server_args, port_args, bench_args, gpu_id=0, tp_rank=0)

    def test_gemma_correctness(self):
        """Verify exact token match and logit parity on Gemma architecture."""
        server_args = ServerArgs(
            model_path="google/gemma-2-2b-it",
            load_format="dummy",
            device="cuda",
        )
        port_args = PortArgs()
        bench_args = BenchArgs(
            run_name="test_gemma_parity",
            batch_size=[1],
            input_len=[256],
            output_len=[32],
            correct=True,
        )
        correctness_test(server_args, port_args, bench_args, gpu_id=0, tp_rank=0)


if __name__ == "__main__":
    unittest.main()

