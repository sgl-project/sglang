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
import math
import sys

import pytest
import torch

from sglang.kernels.ops.attention.ernie55.hca_qkv_prep import hca_qkv_prep
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not get_platform().is_blackwell,
    reason="requires a Blackwell GPU",
)


def _ue8m0(x):
    """Biased exponent of the smallest power of two >= ``x > 0``."""
    mantissa, exponent = torch.frexp(x)
    return (exponent + 126 + (mantissa != 0.5)).to(torch.uint8)


def _rms_norm(x, weight, eps):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight.float()


def _k_reference(kv, weight, eps, freqs_cis, positions, dim_nope, tile):
    x = _rms_norm(kv, weight, eps)
    nope, rope = x[:, :dim_nope], x[:, dim_nope:]
    amax = nope.unflatten(-1, (-1, tile)).abs().amax(-1)
    exponent = _ue8m0(amax.clamp(min=1e-4) / 448)
    rope = torch.view_as_complex(rope.unflatten(-1, (-1, 2))) * freqs_cis[positions]
    return nope, exponent, torch.view_as_real(rope).flatten(-2).bfloat16()


def _split_pages(cache, page_size, row, scale_row):
    """A page holds every slot's data row, then every slot's scale row."""
    data = cache[:, : page_size * row].unflatten(-1, (page_size, row))
    scales = cache[:, page_size * row : page_size * (row + scale_row)]
    return data, scales.unflatten(-1, (page_size, scale_row))


@pytest.mark.parametrize(
    "num_tokens,q_lora_rank,page_size", [(1, 1024, 64), (6, 2048, 256)]
)
def test_matches_reference(
    num_tokens,
    q_lora_rank,
    page_size,
    eps=1e-6,
    num_pages=4,
    dim_nope=448,
    dim_rope=64,
    tile=64,
    scale_row=8,
):
    head_dim = dim_nope + dim_rope
    row = dim_nope + 2 * dim_rope
    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    qkv_a = torch.randn(
        (num_tokens, q_lora_rank + 2 * head_dim),
        device="cuda",
        dtype=torch.bfloat16,
        generator=g,
    )[:, : q_lora_rank + head_dim]
    q_weight = (torch.rand(q_lora_rank, device="cuda", generator=g) + 0.5).bfloat16()
    kv_weight = (torch.rand(head_dim, device="cuda", generator=g) + 0.5).bfloat16()
    angles = torch.rand(
        (num_pages * page_size, dim_rope // 2), device="cuda", generator=g
    )
    freqs_cis = torch.polar(torch.ones_like(angles), angles * 6)
    positions = torch.randperm(num_pages * page_size, device="cuda", generator=g)
    out_loc = torch.randperm(num_pages * page_size, device="cuda", generator=g)
    positions, out_loc = positions[:num_tokens], out_loc[:num_tokens].int()
    out_loc[1::3] = -1
    page_bytes = math.ceil(page_size * (row + scale_row) / row) * row
    cache = torch.zeros((num_pages, page_bytes), dtype=torch.uint8, device="cuda")

    q, sf = hca_qkv_prep(
        qkv_a,
        q_weight,
        eps,
        kv_weight,
        eps,
        torch.view_as_real(freqs_cis).flatten(-2),
        positions,
        out_loc,
        cache,
        page_size,
        q_lora_rank,
    )

    written = out_loc >= 0
    loc = out_loc[written].long()
    page, slot = loc // page_size, loc % page_size
    nope_ref, exponent_ref, rope_ref = _k_reference(
        qkv_a[written, q_lora_rank:],
        kv_weight,
        eps,
        freqs_cis,
        positions[written],
        dim_nope,
        tile,
    )
    data, scales = _split_pages(cache, page_size, row, scale_row)
    assert torch.equal(scales[page, slot, : dim_nope // tile], exponent_ref)
    scale = exponent_ref.view(torch.float8_e8m0fnu).float()
    nope = data[page, slot, :dim_nope].view(torch.float8_e4m3fn).float()
    nope = nope * scale.repeat_interleave(tile, -1)
    torch.testing.assert_close(nope, nope_ref, rtol=2**-4, atol=1e-4)
    rope = data[page, slot, dim_nope:].view(torch.bfloat16)
    torch.testing.assert_close(rope, rope_ref)
    touched = torch.zeros_like(cache, dtype=torch.bool)
    touched_data, touched_scales = _split_pages(touched, page_size, row, scale_row)
    touched_data[page, slot] = True
    touched_scales[page, slot, : dim_nope // tile] = True
    assert not cache[~touched].any()

    q_ref = _rms_norm(qkv_a[:, :q_lora_rank], q_weight, eps)
    scale = sf.view(torch.float8_e8m0fnu).float()
    q_deq = q.float() * scale.repeat_interleave(q_lora_rank // sf.shape[-1], -1)
    torch.testing.assert_close(q_deq, q_ref, rtol=2**-4, atol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
