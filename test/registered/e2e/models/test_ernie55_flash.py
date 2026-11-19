# Copyright 2026 Baidu ERNIE Team and The SGLang Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""GSM8K accuracy for ERNIE-5.5-Flash-Base (FP8) on four B200s, attention DP 4."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=900, stage="nightly", runner_config="4-gpu-b200")


class TestErnie5_5Flash(GSM8KMixin, DefaultServerBase):
    model = "baidu/ERNIE-5.5-Flash-Base"
    other_args = [
        "--tp",
        "4",
        "--attn-dp-size",
        "4",
        "--context-length",
        "131072",
        "--max-running-requests",
        "256",
    ]
    # Base model: sgl-eval sends chat requests only, so use the few-shot harness.
    gsm8k_backend = "run_eval"
    gsm8k_num_shots = 5
    gsm8k_score_threshold = 0.93


if __name__ == "__main__":
    unittest.main()
