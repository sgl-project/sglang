"""Frozen410 BF16 TCGEN05 attention with gate fused into the merge epilogue.

QK/PV and FP32 PART/MAX/SUM unchanged. Preserve BF16 rounding before gate.
Native TGV o_proj remains the same consumer, with its activation PDL wait.
"""
import os

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.attention.qsa_tcgen05_v1 import _store, partial_tcgen05

_Q38C_QSA_PDL_FIX = os.environ.get("SGLANG_Q38C_QSA_PDL_FIX", "0") == "1"

@triton.jit
def _merge_gate_pdl_v2(P, MAX, SUM, OUT, GATE, GS: tl.constexpr, GH: tl.constexpr, NT: tl.constexpr, ST: tl.constexpr, WAIT_PRIMARY: tl.constexpr = False):
    # Release native o_proj weight DMA before reduction/gate; activation DMA
    # keeps its original griddepcontrol.wait until this entire producer ends.
    tl.extra.cuda.gdc_launch_dependents()
    if WAIT_PRIMARY:
        # The entry trigger releases only the consumer's independent work.
        # Our partial states must be ready before any reduction reads them.
        tl.extra.cuda.gdc_wait()
    row, head, dim_tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = tl.arange(0, ST)
    d = dim_tile * 64 + tl.arange(0, 64)
    kv_head = head // 12
    hh = head % 12
    state = (row * 2 + kv_head) * NT + t
    mx = tl.load(MAX + state * 16 + hh, t < NT, -float('inf'))
    denom = tl.load(SUM + state * 16 + hh, t < NT, 0.)
    maximum = tl.max(mx, 0)
    safe_max = tl.where(maximum > -float('inf'), maximum, 0.)
    weights = tl.exp(mx - safe_max)
    total = tl.sum(denom * weights, 0)
    partial = tl.load(
        P + (state[:, None] * 16 + hh) * 256 + d[None, :],
        (t < NT)[:, None], 0.,
    )
    value = tl.sum(partial * weights[:, None], 0) / tl.where(total > 0, total, 1.)
    # Retain frozen410's BF16 merge-output boundary before FP32 sigmoid/mul.
    rounded = value.to(tl.bfloat16).to(tl.float32)
    gate = tl.load(GATE + row * GS + head * GH + d).to(tl.float32)
    gated = rounded * tl.sigmoid(gate)
    tl.store(OUT + (row * 24 + head) * 256 + d, gated)


def run_gated(q, k_pool, v_pool, k_new, v_new, loc, req_to_token, row_requests,
        block_indices, lengths, qpos, scale, gate, width=4):
    """Original v11 ABI, chain TARGET_VERIFY M4 only; BF16 [4,24,256].

    Native top512-block order, graph padding, page-map, chain fresh/tail and
    current-KV stores are unchanged. No host reads of runtime sequence lengths.
    Other modes and shapes retain the original caller path.
    """
    assert gate.dtype == torch.bfloat16 and gate.stride(-1) == 1
    assert tuple(gate.shape) in ((4, 24, 256), (4, 6144))
    m, nh, d = q.shape
    h = k_pool.shape[1]
    assert m == width == 4 and nh == 24 and h == 2 and d == 256
    assert block_indices.shape == (m, 512)
    assert all(t.dtype == torch.bfloat16 for t in (q, k_pool, v_pool, k_new, v_new))
    assert all(t.stride(-1) == 1 and t.stride(-2) == d for t in (q, k_pool, v_pool, k_new, v_new))
    assert block_indices.stride(1) == 1
    # 32 complete 64-token tiles plus one tile for the native <=3-token tail.
    # This is a fixed mature TCGEN05 tile, not a tuned split-KV configuration.
    nt = triton.cdiv(512 * 4 + 3, 64)
    partial = torch.empty((m, h, nt, 16, d), dtype=torch.float32, device=q.device)
    maximum = torch.empty((m, h, nt, 16), dtype=torch.float32, device=q.device)
    sums = torch.empty_like(maximum)
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    _store[(m, h)](
        k_pool, v_pool, k_new, v_new, loc,
        k_pool.stride(0), v_pool.stride(0), k_new.stride(0), v_new.stride(0), h, d,
        num_warps=4,
    )
    partial_tcgen05[(m, nt, h)](
        q, k_pool, v_pool, k_new, v_new, loc, req_to_token, row_requests,
        block_indices, lengths, qpos, partial, maximum, sums,
        QS=q.stride(0), QH=q.stride(1), KS=k_pool.stride(0), VS=v_pool.stride(0),
        NKS=k_new.stride(0), NVS=v_new.stride(0), MS=req_to_token.stride(0),
        IS=block_indices.stride(0), NT=nt, GROUP=12, WIDTH=width, SCALE=float(scale),
        num_warps=4,
    )
    _merge_gate_pdl_v2[(m, nh, triton.cdiv(d, 64))](
        partial, maximum, sums, output, gate, gate.stride(0), gate.stride(1) if gate.ndim == 3 else 256, nt, triton.next_power_of_2(nt), WAIT_PRIMARY=_Q38C_QSA_PDL_FIX, num_warps=4, launch_pdl=True,
    )
    return output


