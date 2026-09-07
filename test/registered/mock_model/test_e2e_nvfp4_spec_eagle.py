from __future__ import annotations

import unittest

from sglang.srt.utils.common import is_sm120_supported
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mock_model.utils import MOCK_MODEL_PATH, run_mock_model_bench_serving
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=180, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    is_sm120_supported(), "native NVFP4 speculative decode requires SM120"
)
class TestE2ENVFP4SpeculativeEagle(CustomTestCase):
    def test_first_request_and_cuda_graph_replay(self) -> None:
        run_mock_model_bench_serving(
            extra_server_args=[
                "--kv-cache-dtype",
                "nvfp4",
                "--prefill-attention-backend",
                "flashinfer",
                "--decode-attention-backend",
                "trtllm_mha",
                "--page-size",
                "64",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                MOCK_MODEL_PATH,
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--mem-fraction-static",
                "0.45",
            ],
            input_check_enabled=False,
            num_prompts=4,
            random_input_len=512,
            random_output_len=64,
        )


if __name__ == "__main__":
    unittest.main()
