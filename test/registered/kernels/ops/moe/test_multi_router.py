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
import triton
import triton.language as tl

from sglang.kernels.ops.moe.multi_router import _sigmoid
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=6, stage="base-b-kernel-unit", runner_config="1-gpu-large")

NUM_EXPERTS = 512
TOPK = 10


def _reference(logits, bias, scale, renormalize, num_routers):
    scores = logits.float().view(logits.shape[0], num_routers, -1).sigmoid().sum(1)
    ids = torch.topk(scores + bias, TOPK, dim=-1).indices
    weights = scores.gather(1, ids)
    if renormalize:
        weights = weights / weights.sum(-1, keepdim=True)
    return weights * scale[ids], ids


@triton.jit
def _sigmoid_probe(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    tl.store(out_ptr + offs, _sigmoid(x), mask=mask)


def _dense(weights, ids):
    return torch.zeros(ids.shape[0], NUM_EXPERTS, device=ids.device).scatter(
        1, ids.long(), weights
    )


@pytest.mark.parametrize(
    "num_tokens, renormalize, num_routers",
    [(1, True, 2), (3000, False, 3), (3000, True, 2)],
)
def test_multi_router_topk_matches_reference(num_tokens, renormalize, num_routers):
    from sglang.kernels.ops.moe.multi_router import multi_router_topk

    torch.manual_seed(0)
    bias = torch.randn(NUM_EXPERTS, device="cuda") * 0.05
    scale = torch.rand(NUM_EXPERTS, device="cuda") + 0.5
    logits = torch.randn(num_tokens, num_routers * NUM_EXPERTS, device="cuda") * 2
    logits = logits.bfloat16()
    weights, ids = multi_router_topk(
        logits, bias, scale, TOPK, renormalize, num_routers=num_routers
    )
    ref_weights, ref_ids = _reference(logits, bias, scale, renormalize, num_routers)
    # Slot order is not part of the contract: compare the selected sets, keyed by expert.
    assert torch.equal(torch.sort(ids.long(), 1).values, torch.sort(ref_ids, 1).values)
    torch.testing.assert_close(
        _dense(weights, ids), _dense(ref_weights, ref_ids), rtol=1e-5, atol=1e-6
    )


def test_router_sigmoid_matches_torch_on_every_bf16_input():
    bits = torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32, device="cuda")
    x = bits.to(torch.int16).view(torch.bfloat16)
    out = torch.empty(x.shape, dtype=torch.float32, device="cuda")
    _sigmoid_probe[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)
    ref = torch.sigmoid(x.float())
    finite = torch.isfinite(x)
    assert torch.equal(out[finite].view(torch.int32), ref[finite].view(torch.int32))


def test_multi_router_topk_empty():
    from sglang.kernels.ops.moe.multi_router import multi_router_topk

    logits = torch.empty(0, 2 * NUM_EXPERTS, device="cuda", dtype=torch.bfloat16)
    zeros = torch.zeros(NUM_EXPERTS, device="cuda")
    weights, ids = multi_router_topk(
        logits, zeros, zeros + 1, TOPK, True, num_routers=2
    )
    assert weights.shape == ids.shape == (0, TOPK)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
