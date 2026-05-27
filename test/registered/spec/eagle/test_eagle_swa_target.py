import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, suite="base-b-test-1-gpu-large")


GPTOSS_TARGET = "openai/gpt-oss-20b"
DOGACEL_DRAFT = "Dogacel/specdrift-gpt-oss-20b-eagle3"


class TestEAGLE3SWATarget(CustomTestCase):
    """Regression for #22679: EAGLE on SWA target."""

    BASE_CONFIG = {
        "model_path": GPTOSS_TARGET,
        "speculative_draft_model_path": DOGACEL_DRAFT,
        "speculative_algorithm": "EAGLE3",
        "speculative_num_steps": 5,
        "speculative_eagle_topk": 8,
        "speculative_num_draft_tokens": 16,
        "mem_fraction_static": 0.85,
        "cuda_graph_max_bs": 4,
        "dtype": "bfloat16",
    }

    BACKENDS = ["triton"]

    def setUp(self):
        self.prompt = "The capital of South Korea is"
        self.sampling_params = {"temperature": 0, "max_new_tokens": 32}

    def test_correctness_per_backend(self):
        for backend in self.BACKENDS:
            with self.subTest(backend=backend):
                config = {**self.BASE_CONFIG, "attention_backend": backend}
                print(f"{config=}")
                engine = sgl.Engine(**config, log_level="info", decode_log_interval=10)
                try:
                    output = engine.generate(self.prompt, self.sampling_params)
                    self.assertIn("text", output)
                    self.assertGreater(len(output["text"]), 0)

                    info = engine.get_server_info()
                    accept_len = info["internal_states"][0].get(
                        "avg_spec_accept_length", 0.0
                    )
                    print(f"backend={backend} accept_len={accept_len}")
                    self.assertGreater(accept_len, 1.0)
                finally:
                    engine.flush_cache()
                    engine.shutdown()


if __name__ == "__main__":
    unittest.main()
