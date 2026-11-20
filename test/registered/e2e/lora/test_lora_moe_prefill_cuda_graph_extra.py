# Copyright 2023-2025 SGLang Team
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
"""Nightly legacy Triton and full-prefill-graph cases, with their own eager reference;
the triton_v2 arm runs with the dual (lora/nolora) decode graph."""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.lora_prefill_graph_kit import (
    EAGER_REFERENCE,
    run_prefill_graph_matrix,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=235, stage="nightly", runner_config="1-gpu-large")


class TestMoELoRAPrefillCudaGraphMatrix(CustomTestCase):
    def test_prefill_graph_matches_eager(self):
        run_prefill_graph_matrix(
            self,
            [
                EAGER_REFERENCE,
                ("triton", "breakable", None),
                ("triton", "full", None),
                ("csgmv", "full", None),
                ("triton_v2", "full", "lora_cutedsl"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
