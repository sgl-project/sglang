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

from sglang.srt.utils import get_device_sm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

NUM_HEADS, HEAD_DIM, GROUPS = 64, 512, 8


@pytest.mark.skipif(get_device_sm() < 100, reason="UE8M0 packed scales are SM100+")
@pytest.mark.parametrize("num_tokens", [1, 5, 64, 2049])
def test_postmix_quant_matches_unfused(num_tokens):
    from sglang.kernels.ops.attention.postmix_fp8_wo_a import (
        postmix_fp8_wo_a_quant,
        postmix_quant_supported,
    )
    from sglang.kernels.ops.quantization.fp8_group_major_quant_ue8m0 import (
        sglang_per_token_group_quant_fp8_dsv4_wo_a,
    )

    torch.manual_seed(0)
    o = torch.randn(num_tokens, NUM_HEADS, HEAD_DIM, device="cuda").bfloat16()
    x = o.view(num_tokens, GROUPS, -1)
    eye = torch.eye(NUM_HEADS, device="cuda").bfloat16()
    assert postmix_quant_supported(x, eye)
    codes, scales = postmix_fp8_wo_a_quant(x, eye)
    ref_codes, ref_scales = sglang_per_token_group_quant_fp8_dsv4_wo_a(x.contiguous())
    assert torch.equal(codes.view(torch.uint8), ref_codes.view(torch.uint8))
    assert scales.stride() == ref_scales.stride() and torch.equal(scales, ref_scales)

    mix = torch.eye(NUM_HEADS, device="cuda") + 0.05 * torch.randn(
        NUM_HEADS, NUM_HEADS, device="cuda"
    )
    mix = mix.bfloat16()
    codes, scales = postmix_fp8_wo_a_quant(x, mix)
    mixed = torch.matmul(mix, o).view(num_tokens, GROUPS, -1).contiguous()
    ref_codes, ref_scales = sglang_per_token_group_quant_fp8_dsv4_wo_a(mixed)
    assert torch.equal(scales, ref_scales)
    # Codes may move by one fp8 step where the bf16 mix rounded differently.
    diff = (codes.float() - ref_codes.float()).abs()
    assert (diff > 0).float().mean() < 1e-3


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
