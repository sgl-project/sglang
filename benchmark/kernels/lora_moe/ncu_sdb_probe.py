"""One-off NCU probe: chunked expand c128 vs unchunked sgmv BS128 at the
SAME tile (128x256, BK64, w8, s2), dense-validity r64 prefill T=2048.
Documents the chunked kernel's register-spill cliff (plan §64.8). The
chunked launcher lives INLINE here so this probe stays runnable without
depending on which chunked arms the evolving bench carries. Lab driver
only."""

import argparse

import torch
import triton

from benchmark.kernels.lora_moe.bench_sgmv_real import synthesize_chunked_batch_info
from benchmark.kernels.lora_moe.bench_shared_down_b import _build_case, _DownBFixture
from sglang.kernels.ops.gemm.chunked_sgmv_expand import _chunked_lora_expand_kernel

parser = argparse.ArgumentParser()
parser.add_argument("--arm", required=True, choices=("chunked", "sgmv"))
args = parser.parse_args()
device = torch.device("cuda")
case = _build_case(
    device,
    preset="dense",
    num_tokens=2048,
    rank=64,
    seed=11,
    source_revision="ncu-sdb-probe",
)
fixture = _DownBFixture(case, device)
CONFIG = {"BLOCK_N": 256, "BLOCK_K": 64, "num_warps": 8, "num_stages": 2}


def chunked_c128(output: torch.Tensor) -> None:
    topk = fixture.leg.topk_ids
    valid = (topk >= 0) & (topk < case.num_experts_local)
    pair_slots = torch.where(
        valid,
        fixture.leg.token_slots[:, None].expand_as(topk),
        torch.full_like(topk, -1),
    ).reshape(-1)
    info = synthesize_chunked_batch_info(
        pair_slots,
        max_loras=case.slot_capacity,
        physical_rank=case.physical_rank,
        device=device,
        chunk=128,
    )
    hidden = case.moe_hidden_size
    slice_offsets = torch.tensor([0, hidden], dtype=torch.int32, device=device)
    grid = (triton.cdiv(hidden, CONFIG["BLOCK_N"]), 1, info.num_segments)
    _chunked_lora_expand_kernel[grid](
        x=fixture.leg.down_rank_out,
        weights=fixture.b_down_3d,
        output=output,
        output_stride_0=output.stride(0),
        output_stride_1=output.stride(1),
        seg_indptr=info.seg_indptr,
        weight_indices=info.weight_indices,
        lora_ranks=info.lora_ranks,
        permutation=info.permutation,
        num_segs=info.num_segments,
        scalings=info.scalings,
        slice_offsets=slice_offsets,
        NUM_SLICES=1,
        OUTPUT_DIM=hidden,
        MAX_RANK=case.physical_rank,
        BLOCK_M=info.max_len,  # == 128 (synthesis contract)
        BLOCK_N=CONFIG["BLOCK_N"],
        BLOCK_K=CONFIG["BLOCK_K"],
        num_warps=CONFIG["num_warps"],
        num_stages=CONFIG["num_stages"],
    )


torch.cuda.synchronize()
for _ in range(12):
    if args.arm == "chunked":
        chunked_c128(fixture.accum_buffer)
    else:
        fixture.sgmv(
            fixture.unchunked_info,
            {"BLOCK_S": 128, **CONFIG},
            fixture.accum_buffer,
        )
torch.cuda.synchronize()
print(f"{args.arm}: done")
