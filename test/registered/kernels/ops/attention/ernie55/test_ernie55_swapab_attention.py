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
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.attention.ernie55.swapab_attention import (
    swapab_combine_local,
    swapab_partial,
    swapab_partial_mla,
)
from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not get_platform().is_blackwell,
    reason="the swap-AB Gluon partial kernel uses tcgen05",
)


def _fp8(shape, g):
    x = torch.randn(shape, device="cuda", generator=g) * 32
    return x.to(torch.float8_e4m3fn)


def _v4_cache(num_pages, page_size, layout, dim_nope, dim_rope, g):
    """Pages of per-slot rows (fp8 nope, bf16 rope), then per-slot UE8M0 bytes."""
    num_slots = num_pages * page_size
    row, scale_row, tile = layout.data_bytes, layout.scale_bytes, layout.tile_size
    nope = _fp8((num_slots, dim_nope), g)
    exponent = torch.randint(
        121, 124, (num_slots, dim_nope // tile), device="cuda", generator=g
    ).to(torch.uint8)
    rope = torch.randn(
        (num_slots, dim_rope), device="cuda", dtype=torch.bfloat16, generator=g
    )
    page_bytes = math.ceil(page_size * (row + scale_row) / row) * row
    cache = torch.zeros((num_pages, page_bytes), dtype=torch.uint8, device="cuda")
    data = cache[:, : page_size * row].unflatten(-1, (page_size, row))
    scales = cache[:, page_size * row : page_size * (row + scale_row)]
    scales = scales.unflatten(-1, (page_size, scale_row))
    data[..., :dim_nope] = nope.view(torch.uint8).unflatten(0, (num_pages, -1))
    data[..., dim_nope:] = rope.view(torch.uint8).unflatten(0, (num_pages, -1))
    scales[..., : dim_nope // tile] = exponent.unflatten(0, (num_pages, -1))
    scale = exponent.view(torch.float8_e8m0fnu).float().repeat_interleave(tile, -1)
    values = torch.cat((nope.float() * scale, rope.float()), -1).bfloat16()
    return data, values


def _mla_cache(num_slots, dim_nope, dim_rope, tile, g):
    """Rows of fp8 nope, one fp32 scale per tile, then bf16 rope."""
    nope = _fp8((num_slots, dim_nope), g)
    scale = (
        torch.rand((num_slots, dim_nope // tile), device="cuda", generator=g) * 0.05
        + 0.01
    )
    rope = torch.randn(
        (num_slots, dim_rope), device="cuda", dtype=torch.bfloat16, generator=g
    )
    kv = torch.cat([t.view(torch.uint8) for t in (nope, scale, rope)], -1)
    nope = nope.float() * scale.repeat_interleave(tile, -1)
    values = torch.cat((nope, rope.float()), -1).bfloat16()
    return kv.view(num_slots, 1, -1), values


def _indices(bs, width, num_slots, g):
    ids = torch.randint(
        0, num_slots, (bs, width), device="cuda", dtype=torch.int32, generator=g
    )
    ids[:, 1::7] = -1
    lengths = torch.randint(
        1, width + 1, (bs,), device="cuda", dtype=torch.int32, generator=g
    )
    return ids, lengths


def _reference(q, scale, sink, segments, dim_v):
    kv, valid = [], []
    for values, ids, lengths in segments:
        pos = torch.arange(ids.shape[-1], device=ids.device)
        mask = (ids >= 0) & (pos < lengths[:, None])
        kv.append(torch.where(mask[..., None], values[ids.clamp(min=0)].double(), 0))
        valid.append(mask)
    kv, valid = torch.cat(kv, 1), torch.cat(valid, 1)
    scores = q.double() @ kv.transpose(1, 2) * scale
    scores.masked_fill_(~valid[:, None, :], -torch.inf)
    sink = sink[None, :, None].double().expand(q.shape[0], -1, -1)
    prob = torch.softmax(torch.cat((scores, sink), -1), -1)[..., :-1]
    return prob @ kv[..., :dim_v]


@pytest.mark.parametrize("bs,num_heads,tokens_per_req", [(1, 16, 1), (6, 32, 2)])
def test_hca_matches_reference(
    bs,
    num_heads,
    tokens_per_req,
    topk=100,
    extra_topk=72,
    compress_ratio=4,
    eps=1e-6,
    dim_nope=448,
    dim_rope=64,
):
    head_dim = dim_nope + dim_rope
    layout = SimpleNamespace(
        tile_size=64, data_bytes=dim_nope + 2 * dim_rope, scale_bytes=8
    )
    g = torch.Generator(device="cuda").manual_seed(bs)
    kv, values = _v4_cache(4, 64, layout, dim_nope, dim_rope, g)
    extra_kv, extra_values = _v4_cache(2, 128, layout, dim_nope, dim_rope, g)
    indices, lengths = _indices(bs, topk, values.shape[0], g)
    extra_indices, extra_lengths = _indices(bs, extra_topk, extra_values.shape[0], g)
    late = torch.arange(bs, device="cuda") % 2 == 0
    positions = extra_lengths * compress_ratio + torch.where(late, -1, tokens_per_req)
    angles = torch.rand(
        (int(positions.max()) + 1, dim_rope // 2), device="cuda", generator=g
    )
    freqs_cis = torch.polar(torch.ones_like(angles), angles * 6)
    q = torch.randn(
        (bs, num_heads, head_dim), device="cuda", dtype=torch.bfloat16, generator=g
    )
    sink = torch.randn(num_heads, device="cuda", generator=g)

    args = (q, kv, indices, lengths, head_dim**-0.5)
    kwargs = dict(
        extra_kv=extra_kv,
        extra_indices=extra_indices,
        extra_lengths=extra_lengths,
        kv_layout=layout,
        freqs=torch.view_as_real(freqs_cis).flatten(-2),
        positions=positions,
        q_eps=eps,
        tokens_per_req=tokens_per_req,
        compress_ratio=compress_ratio,
    )
    partial, maximum, sums, nt = swapab_partial(*args, **kwargs)
    split = swapab_partial(*args, tile_mode=1, **kwargs)
    swapab_partial(*args, tile_mode=2, out=split[:3], **kwargs)
    for a, b in zip(split, (partial, maximum, sums)):
        assert torch.equal(a, b)
    out = swapab_combine_local(partial, maximum, sums, sink, nt)

    x = q.float()
    x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
    rope = torch.view_as_complex(x[..., -dim_rope:].unflatten(-1, (-1, 2)))
    rope = torch.view_as_real(rope * freqs_cis[positions][:, None]).flatten(-2)
    q_ref = torch.cat((x[..., :-dim_rope], rope), -1).bfloat16()
    segments = (
        (values, indices, lengths),
        (extra_values, extra_indices, extra_lengths),
    )
    ref = _reference(q_ref, head_dim**-0.5, sink, segments, head_dim)
    torch.testing.assert_close(out.double(), ref, atol=2e-5, rtol=4e-3)


@pytest.mark.parametrize("bs,num_heads", [(1, 16), (5, 64)])
def test_mla_matches_reference(
    bs,
    num_heads,
    dim_nope=512,
    dim_rope=64,
    tile=128,
    topk=200,
    window=64,
    num_slots=1024,
):
    g = torch.Generator(device="cuda").manual_seed(bs)
    kv, values = _mla_cache(num_slots, dim_nope, dim_rope, tile, g)
    topk_ids, topk_lengths = _indices(bs, topk, num_slots, g)
    window_ids, window_lengths = _indices(bs, window, num_slots, g)
    q = torch.randn(
        (bs, num_heads, dim_nope + dim_rope),
        device="cuda",
        dtype=torch.bfloat16,
        generator=g,
    )
    sink = torch.randn(num_heads, device="cuda", generator=g)
    scale = (dim_nope + dim_rope) ** -0.5

    partial, maximum, sums, nt = swapab_partial_mla(
        q[..., :dim_nope],
        q[..., dim_nope:],
        kv,
        topk_ids,
        topk_lengths,
        window_ids,
        window_lengths,
        scale,
    )
    out = swapab_combine_local(partial, maximum, sums, sink, nt)

    segments = (
        (values, topk_ids, topk_lengths),
        (values, window_ids, window_lengths),
    )
    ref = _reference(q, scale, sink, segments, dim_nope)
    torch.testing.assert_close(out.double(), ref, atol=2e-5, rtol=4e-3)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
