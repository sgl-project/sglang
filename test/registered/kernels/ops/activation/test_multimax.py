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
import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=4, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _reference(x, r, t):
    out = x.clone()
    out += t[0] * F.relu(r[0] - x)
    out += t[1] * F.relu(x - r[1])
    out += t[2] * F.relu(r[2] - x) ** 2
    out += t[3] * F.relu(x - r[3]) ** 2
    return out


@pytest.mark.parametrize("shape", [(1, 50304), (37, 201216), (3, 5)])
def test_multimax_matches_eager_bit_for_bit(shape):
    from sglang.kernels.ops.activation.multimax import multimax

    torch.manual_seed(0)
    x = (torch.randn(shape, device="cuda") * 8).bfloat16()
    r = torch.tensor([-3.0, 4.0, -6.0, 7.0], device="cuda").bfloat16()
    t = torch.tensor([0.3, -0.2, 0.05, -0.04], device="cuda").bfloat16()
    assert torch.equal(multimax(x, r, t), _reference(x, r, t))


def test_multimax_keeps_non_finite_logits():
    from sglang.kernels.ops.activation.multimax import multimax

    x = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 0.0, -5.0, 9.0], device="cuda"
    ).bfloat16()
    r = torch.tensor([-3.0, 4.0, -6.0, 7.0], device="cuda").bfloat16()
    t = torch.tensor([0.3, -0.2, 0.05, -0.04], device="cuda").bfloat16()
    got, want = multimax(x, r, t), _reference(x, r, t)
    assert torch.equal(got.isnan(), want.isnan())
    keep = ~want.isnan()
    assert torch.equal(got[keep], want[keep])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
