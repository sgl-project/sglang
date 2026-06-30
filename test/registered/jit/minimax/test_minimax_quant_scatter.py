"""Correctness for the fused MiniMax-M3 quant+scatter kernel.

`per_token_quant_fp8_ue8m0_scatter` computes the per-token fp8 + int32-packed
UE8M0 scale once and scatters them straight into the permuted grouped-GEMM input
(each token replicated to its top_k expert rows + the scale written MN-major).
This verifies it is byte-identical to the two-kernel reference it replaces:
`per_token_quant_fp8_ue8m0` (quant) followed by `fill_gateup_input_triton_kernel`
(scatter), on every written destination row.
"""

import random
import sys

import pytest
import torch

from sglang.jit_kernel.minimax_quant_ue8m0 import (
    per_token_quant_fp8_ue8m0,
    per_token_quant_fp8_ue8m0_scatter,
)
from sglang.srt.layers.moe.ep_moe.kernels import fill_gateup_input_triton_kernel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="base-b-kernel-unit-1-gpu-b200")

dev = "cuda"


@pytest.mark.parametrize("num_tokens", [1, 7, 64, 256])
@pytest.mark.parametrize("topk", [4, 5, 8])
@pytest.mark.parametrize("hidden,group", [(6144, 32), (2048, 32), (4096, 128)])
def test_quant_scatter_matches_quant_plus_fill(num_tokens, topk, hidden, group):
    arch_major, _ = torch.cuda.get_device_capability(torch.cuda.current_device())
    if arch_major <= 9:
        pytest.skip("UE8M0 fusion is Blackwell-only")

    E = 129  # 128 routed + 1 fused shared
    G4 = (hidden // group) // 4
    m_max = (num_tokens // 256 + 1) * 256
    torch.manual_seed(num_tokens * 91 + topk + hidden)
    random.seed(num_tokens)

    x = (torch.randn(num_tokens, hidden, device=dev, dtype=torch.bfloat16)) * 4.0

    # Dispatch indices: each token -> topk distinct experts (atomic-cursor style).
    tids = torch.empty(num_tokens, topk, dtype=torch.int32, device=dev)
    tids_cpu = torch.empty(num_tokens, topk, dtype=torch.int32)
    s2d = [0] * (num_tokens * topk)
    cur = [0] * E
    for t in range(num_tokens):
        for j, e in enumerate(random.sample(range(E), topk)):
            tids_cpu[t, j] = e
            s2d[t * topk + j] = e * m_max + cur[e]
            cur[e] += 1
    tids.copy_(tids_cpu)
    s2d = torch.tensor(s2d, dtype=torch.int32, device=dev)

    # Reference: quant + fill (two kernels).
    x_q, x_sf = per_token_quant_fp8_ue8m0(x, group)
    gi_ref = torch.zeros(E, m_max, hidden, device=dev, dtype=torch.float8_e4m3fn)
    gs_ref = torch.zeros(E, G4, m_max, device=dev, dtype=torch.int32)
    fill_gateup_input_triton_kernel[(num_tokens,)](
        x_q,
        x_sf,
        gi_ref,
        gs_ref,
        s2d,
        tids,
        topk,
        hidden,
        G4,
        m_max,
        x_sf.stride(0),
        x_sf.stride(1),
        BLOCK_SIZE=1024,
        SCALE_MN_MAJOR=True,
    )

    # Fused: quant + scatter (one kernel).
    gi_new = torch.zeros(E, m_max, hidden, device=dev, dtype=torch.float8_e4m3fn)
    gs_new = torch.zeros(E, G4, m_max, device=dev, dtype=torch.int32)
    per_token_quant_fp8_ue8m0_scatter(x, gi_new, gs_new, s2d, tids, topk, m_max, group)

    for t in range(num_tokens):
        for j in range(topk):
            e = int(tids_cpu[t, j])
            m = int(s2d[t * topk + j]) % m_max
            assert torch.equal(
                gi_new[e, m].view(torch.uint8), gi_ref[e, m].view(torch.uint8)
            ), f"fp8 mismatch token={t} slot={j} expert={e}"
            assert torch.equal(
                gs_new[e, :, m], gs_ref[e, :, m]
            ), f"scale mismatch token={t} slot={j} expert={e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-x"]))
