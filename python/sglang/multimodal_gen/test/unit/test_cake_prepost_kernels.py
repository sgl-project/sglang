# SPDX-License-Identifier: Apache-2.0
"""GPU regressions against the unfused prepared-Sage glue."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or torch.cuda.get_device_capability() != (12, 0),
    reason="requires an SM120 CUDA GPU",
)


@pytest.mark.parametrize("used", [1, 64, 130, 37765])
@pytest.mark.parametrize("nq,nk", [(1, 1), (4, 4), (8, 2)])
def test_pack_pool_and_output_are_exact(used, nq, nk):
    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse import (
        SubBlockRouter,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse.cake_prepost_kernels import (
        pack_cake_operands,
        route_prequantized,
        write_cake_output,
    )
    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse.prepared_sage_attention import (
        _dequantize_for_routing,
    )

    b, h, d = 2, 2, 128
    total = (used + 511) // 512 * 512
    torch.manual_seed(9917 + used)
    q = torch.randint(-127, 128, (b, total, h, d), device="cuda", dtype=torch.int8)
    k = torch.randint(-127, 128, q.shape, device="cuda", dtype=torch.int8)
    v = torch.randn((b, d, h, total), device="cuda").to(torch.float8_e4m3fn)
    qs = torch.rand((b, h, (total + 127) // 128 * 4), device="cuda") * 0.02
    ks = torch.rand((b, h, (total + 63) // 64), device="cuda") * 0.02
    blocks = (used + 63) // 64
    packed = pack_cake_operands(q, k, v, qs, ks, used)
    reference = (
        q[:, :used].transpose(1, 2).contiguous(),
        k[:, :used].transpose(1, 2).contiguous(),
        v[..., : blocks * 64].transpose(1, 2).contiguous(),
        qs[:, :, : ((used + 127) // 128) * 4].contiguous(),
        ks[:, :, :blocks].contiguous(),
    )
    for got, want in zip(packed, reference):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
    qr, kr = _dequantize_for_routing(reference[0], reference[1], reference[4])
    plan = SubBlockRouter(n_q=nq, n_k=nk).route(
        qr.transpose(1, 2), kr.transpose(1, 2), 0.75, d**-0.5
    )
    index, topk = route_prequantized(q, k, ks, used, nq, nk, 0.75, d**-0.5)
    assert topk == plan.topk and torch.equal(index, plan.index)
    attn = torch.randn((b, h, used, d), device="cuda", dtype=torch.bfloat16)
    out = torch.empty((b, total, h, d), device="cuda", dtype=torch.bfloat16)
    write_cake_output(attn, out, used, total)
    assert torch.equal(out[:, :used], attn.transpose(1, 2))
    assert not out[:, used:].count_nonzero().item()


@pytest.mark.parametrize("mask_kind", ["mixed", "dense", "sparse"])
@pytest.mark.parametrize("stride", [1, 2])
def test_mixed_rows_preserve_dense_modalities(mask_kind, stride):
    from sglang.multimodal_gen.runtime.layers.attention.backends.subblock_sparse.cake_prepost_kernels import (
        mixed_row_indices,
    )

    b, h, blocks, topk = 2, 3, 17, 8
    index = torch.randint(
        0, blocks, (b, h, blocks, topk), device="cuda", dtype=torch.int32
    )
    storage = torch.arange(blocks * stride, device="cuda") % 3 != 0
    mask = storage[::stride]
    if mask_kind == "dense":
        mask.zero_()
    elif mask_kind == "sparse":
        mask.fill_(True)
    dense, counts = mixed_row_indices(index, mask, blocks)
    expected = (
        torch.arange(blocks, device="cuda", dtype=torch.int32)
        .view(1, 1, 1, -1)
        .expand(b, h, blocks, -1)
        .clone()
    )
    expected[..., :topk] = torch.where(
        mask.view(1, 1, -1, 1), index, expected[..., :topk]
    )
    assert torch.equal(dense, expected)
    assert torch.equal(counts, torch.where(mask, topk, blocks).int().expand(b, h, -1))
