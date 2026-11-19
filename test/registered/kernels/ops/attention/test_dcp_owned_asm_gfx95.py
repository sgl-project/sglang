"""``asm_sparse_mla_dcp`` against the Triton owned-count sparse MLA on gfx950.

The aiter non-persistent fp8 qh64 qseqlen1 MLA kernel ships only for gfx950, so
this runs on MI355X; the rest of the owned DCP route stays in
test_dcp_owned_asm_hip.py.
"""

import math

import pytest
import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=20, stage="base-b", runner_config="1-gpu-small-amd-mi35x")

if not (torch.cuda.is_available() and is_hip() and is_gfx95_supported()):
    pytest.skip("gfx950 fp8 asm MLA kernel", allow_module_level=True)

FP8 = torch.float8_e4m3fn
H, KV_LORA_RANK, ROPE_DIM, TOPK, DCP = 64, 512, 64, 2048, 4


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _owned_table(rows: int, ctx: int, rank: int):
    from sglang.kernels.ops.attention.dsa.transform_index import (
        transform_index_page_table_dcp,
    )

    table = torch.stack(
        [torch.randperm(ctx, device="cuda")[:TOPK] for _ in range(rows)]
    ).int()
    # Row 0 owns nothing on rank 0; row 1 has a short top-k.
    table[0, :] = -1
    table[0, 0] = 1
    table[1, 100:] = -1
    return transform_index_page_table_dcp(table, DCP, rank)


@pytest.mark.parametrize("T", [24, 144, 2304])
def test_asm_sparse_mla_matches_triton(T):
    from sglang.kernels.ops.attention.dcp_kernels import dcp_lse_combine_triton
    from sglang.kernels.ops.attention.dsa.asm_sparse_mla_dcp import (
        asm_sparse_mla_dcp,
        asm_sparse_mla_dcp_available,
        build_dcp_csr,
        is_asm_q8,
    )
    from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
        triton_sparse_mla_decode_splitk,
    )

    if not asm_sparse_mla_dcp_available():
        pytest.skip("aiter has no non-persistent fp8 gqa64 MLA kernel")
    torch.manual_seed(1)
    ctx, sm_scale = 60000, 0.1
    kv = (
        torch.randn(ctx // DCP + 1000, 1, KV_LORA_RANK + ROPE_DIM, device="cuda") * 0.5
    ).to(FP8)
    q_nope = torch.randn(T, H, 192, device="cuda").bfloat16()
    q_pe = torch.randn(T, H, ROPE_DIM, device="cuda").bfloat16()
    w_kc = (torch.randn(H, 192, KV_LORA_RANK, device="cuda") * 0.08).bfloat16()
    local, counts = _owned_table(T, ctx, rank=0)
    indptr, indices = build_dcp_csr(local, counts)
    one = torch.ones(1, dtype=torch.float32, device="cuda")

    # The query prepare packs: one fp8 [q_nope | q_pe] row.
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), w_kc).transpose(0, 1)
    q8 = torch.cat([q_nope_out, q_pe], -1).to(FP8)
    assert is_asm_q8(*q8.split([KV_LORA_RANK, ROPE_DIM], -1))
    out, lse = asm_sparse_mla_dcp(q8, kv, indptr, indices, sm_scale, KV_LORA_RANK, one)

    # The Triton kernel casts q to the fp8 KV dtype on load: feed it the same q.
    ref_out, ref_lse = triton_sparse_mla_decode_splitk(
        q_nope=q8[..., :KV_LORA_RANK].to(torch.bfloat16).contiguous(),
        q_rope=q8[..., KV_LORA_RANK:].to(torch.bfloat16).contiguous(),
        kv=kv,
        indices=local.unsqueeze(1),
        sm_scale=sm_scale,
        d_v=KV_LORA_RANK,
        workspace=[],
        return_lse=True,
        lengths=counts,
    )
    triton_out = ref_out.view(T, H, KV_LORA_RANK)
    triton_lse = ref_lse.view(T, H)
    owned = counts > 0

    # Both kernels round on their own way to the same fp32 result: the asm
    # one must be as close to it as the Triton one.
    qf, kvf = q8.float(), kv.float()[:, 0]
    exact_out = torch.zeros(T, H, KV_LORA_RANK, device="cuda")
    exact_lse = torch.full((T, H), float("-inf"), device="cuda")
    for t in owned.nonzero().flatten().tolist():
        sel = kvf[local[t, : counts[t]].long()]
        s = qf[t] @ sel.T * sm_scale
        exact_lse[t] = torch.logsumexp(s, -1)
        exact_out[t] = torch.softmax(s, -1) @ sel[:, :KV_LORA_RANK]
    asm_err = _rel_err(out[owned], exact_out[owned])
    triton_err = _rel_err(triton_out[owned], exact_out[owned])
    assert asm_err < 1.1 * triton_err + 1e-3, (asm_err, triton_err)
    torch.testing.assert_close(lse[owned], exact_lse[owned], rtol=0, atol=1e-2)
    torch.testing.assert_close(
        triton_lse[owned], exact_lse[owned] / math.log(2), rtol=0, atol=1e-2
    )
    # A row that owns no slot drops out of the merge with another rank's row.
    assert counts[0] == 0
    assert out[0].eq(0).all() and lse[0].eq(float("-inf")).all()
    other_out = torch.randn(1, H, KV_LORA_RANK, device="cuda").bfloat16()
    other_lse = torch.randn(1, H, device="cuda")
    merged, _ = dcp_lse_combine_triton(
        torch.stack([out[:1], other_out]),
        torch.stack([lse[:1], other_lse]),
        is_lse_base_on_e=True,
    )
    torch.testing.assert_close(merged, other_out, rtol=0, atol=0)

    # Row chunks of the batch read their slice of the same CSR.
    lo, hi = T // 3, T
    out_c, lse_c = asm_sparse_mla_dcp(
        q8[lo:hi].contiguous(),
        kv,
        indptr[lo : hi + 1],
        indices,
        sm_scale,
        KV_LORA_RANK,
        one,
    )
    torch.testing.assert_close(out_c, out[lo:hi], rtol=0, atol=0)
    torch.testing.assert_close(lse_c, lse[lo:hi], rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
