import unittest
from sglang.benchmark.one_batch import (
    BenchArgs,
    PortArgs,
    ServerArgs,
    correctness_test,
)


class TestFinalLayerPrefillOptimization(unittest.TestCase):
    def test_correctness_and_logit_parity(self):
        """Verify exact token match and logit parity on LLaMA architecture."""
        server_args = ServerArgs(
            model_path="TinyLlama/TinyLlama-1.1B-Chat-v0.4",
            load_format="dummy",
            device="cuda",
        )
        port_args = PortArgs()
        bench_args = BenchArgs(
            run_name="test_parity",
            batch_size=[1],
            input_len=[256],
            output_len=[32],
            correct=True,
        )
        # correctness_test asserts exact token match and verifies logit tolerances
        correctness_test(server_args, port_args, bench_args, gpu_id=0, tp_rank=0)


if __name__ == "__main__":
    unittest.main()
