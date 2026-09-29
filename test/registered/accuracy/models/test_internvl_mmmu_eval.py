"""Keep InternVL's tokenizer/server regression independently rerunnable."""

import unittest

import test_vlms_mmmu_eval as vlm_eval

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import ModelLaunchSettings

register_cuda_ci(est_time=600, stage="nightly", runner_config="2-gpu-large")


class TestInternVLMmmuEval(vlm_eval.TestNightlyVLMMmmuEval):
    model_thresholds = {
        ModelLaunchSettings("OpenGVLab/InternVL2_5-2B"): (0.300, 18.0),
    }


if __name__ == "__main__":
    unittest.main()
