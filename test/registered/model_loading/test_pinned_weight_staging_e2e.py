from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd")

import multiprocessing as mp
import unittest

import torch

from sglang.srt.environ import envs
from sglang.test.runners import SRTRunner, check_close_model_outputs
from sglang.test.test_utils import CustomTestCase

MODEL = "Qwen/Qwen2-0.5B"
PROMPTS = ["The capital of the United Kingdom is"]


class TestPinnedWeightStagingE2E(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        mp.set_start_method("spawn", force=True)

    def _generate(self, pinned_weight_staging: bool):
        with envs.SGLANG_PINNED_WEIGHT_STAGING.override(pinned_weight_staging):
            with SRTRunner(
                MODEL,
                torch_dtype=torch.float16,
                model_type="generation",
                disable_cuda_graph=True,
                disable_radix_cache=True,
                max_total_tokens=2048,
            ) as runner:
                return runner.forward(PROMPTS, max_new_tokens=32)

    def test_generation_matches_without_staging(self):
        plain = self._generate(pinned_weight_staging=False)
        staged = self._generate(pinned_weight_staging=True)
        check_close_model_outputs(
            hf_outputs=plain,
            srt_outputs=staged,
            prefill_tolerance=1e-6,
            decode_tolerance=1e-6,
            rouge_l_tolerance=1.0,
            debug_text="pinned weight staging off vs on",
        )


if __name__ == "__main__":
    unittest.main()
