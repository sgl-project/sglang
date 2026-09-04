# Copyright 2023-2024 SGLang Team
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

import unittest

import torch

from sglang.srt.layers.quantization.fp8_utils import unshuffle_aiter_fp8_weight
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")


@unittest.skipUnless(is_hip(), "requires ROCm AITER")
class TestAiterFp8Utils(CustomTestCase):
    def test_unshuffle_weight_round_trip(self):
        from aiter.ops.shuffle import shuffle_weight

        for shape in ((32, 64), (2, 32, 64)):
            with self.subTest(shape=shape):
                logical = (
                    torch.arange(
                        torch.Size(shape).numel(), device="cuda", dtype=torch.float32
                    )
                    .remainder(7)
                    .to(torch.float8_e4m3fn)
                    .reshape(shape)
                )
                shuffled = shuffle_weight(logical, layout=(16, 16))

                torch.testing.assert_close(
                    unshuffle_aiter_fp8_weight(shuffled), logical
                )


if __name__ == "__main__":
    unittest.main()
