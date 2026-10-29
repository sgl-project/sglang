"""Small-M router-gate logits as a single Triton matvec launch."""

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl


@triton.jit
def router_gate_matvec_kernel(
    x_ptr,  # (M, K) bf16/fp16/fp32, row-major
    w_ptr,  # (E, K) fp32/bf16/fp16, k-major
    out_ptr,  # (M, E) fp32
    K,
    E,
    stride_xm,
    stride_we,
    BLOCK_E: tl.constexpr,
    BLOCK_K: tl.constexpr,
    USE_GDC: tl.constexpr = False,
):
    """Router-gate logits as a single matvec launch, fp32 accumulation for
    any float weight dtype. BLOCK_K covers the whole K in one masked load
    (single iteration for K <= BLOCK_K): a cold gate weight then costs one
    HBM round trip per CTA instead of a serial dependent-load chain, which
    is what dominates in the real model where ~94MB/layer of expert traffic
    flushes L2 between gate calls.
    """
    pid_m = tl.program_id(0)
    pid_e = tl.program_id(1)
    e_offs = pid_e * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_offs < E

    # First K tile, weight load ahead of the PDL wait (see docstring).
    k_offs = tl.arange(0, BLOCK_K)
    k_mask = k_offs < K
    w = tl.load(
        w_ptr + e_offs[:, None] * stride_we + k_offs[None, :],
        mask=e_mask[:, None] & k_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    if USE_GDC:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    x = tl.load(x_ptr + pid_m * stride_xm + k_offs, mask=k_mask, other=0.0).to(
        tl.float32
    )
    acc = tl.sum(w * x[None, :], axis=1)

    for k0 in range(BLOCK_K, K, BLOCK_K):
        k_offs = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K
        x = tl.load(x_ptr + pid_m * stride_xm + k_offs, mask=k_mask, other=0.0).to(
            tl.float32
        )
        w = tl.load(
            w_ptr + e_offs[:, None] * stride_we + k_offs[None, :],
            mask=e_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)

    tl.store(
        out_ptr + pid_m * E + e_offs, acc.to(out_ptr.dtype.element_ty), mask=e_mask
    )


# Cold-cache tuned on H20-3e (41 rotating gate weights so each call misses L2,
# like the real model); expected to carry to B200 (more SMs favor the wide
# grid even more) — re-tune with benchmark/kernels/bench_router_gate_matvec.py.
ROUTER_GATE_MATVEC_BLOCK_E = 4
ROUTER_GATE_MATVEC_NUM_WARPS = 8
# Beyond this M the per-M re-reads of the gate weight outgrow the library
# GEMM (cold H20-3e: bf16 wins to M=12, fp32 to M=8; cap at the lower).
ROUTER_GATE_MATVEC_MAX_M = 8


def router_gate_matvec(
    hidden_states: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Small-M router-gate logits: one triton launch replacing the library
    path — for fp32 gate weights the eager upcast + fp32 GEMM + splitKreduce
    triple, for bf16 the F.linear GEMV. Returns fp32 (M, E) logits with fp32
    accumulation (deterministic; for fp32 weights 0 top-8 routing flips over
    30104 random draws vs the fp32 reference; for bf16 weights this is
    slightly MORE precise than the library GEMV, so near-tie logits can
    round-trip differently — same order as the bf16-vs-fp32 gate change).

    Cold-cache (41 rotating weights, in-graph, H20-3e, E=513, K=2560), us/call:

        M      lib bf16   matvec bf16   lib fp32 chain   matvec fp32
        1        4.2         4.3            6.4              6.2
        2       13.8         4.5           14.6              6.4
        4       14.0         5.4           20.7             10.4
        8       14.6        10.4           20.1             18.1
        16      14.6        18.9  (lib)    20.9             30.9  (lib)

    Callers must gate on M <= ROUTER_GATE_MATVEC_MAX_M; prefill-sized M
    keeps the library GEMM."""
    assert (
        weight.dtype
        in (
            torch.float32,
            torch.bfloat16,
            torch.float16,
        )
        and weight.is_contiguous()
    )
    M, K = hidden_states.shape
    E = weight.shape[0]
    out = torch.empty((M, E), dtype=torch.float32, device=hidden_states.device)
    block_e = ROUTER_GATE_MATVEC_BLOCK_E
    # Single k-iteration whenever K fits one block: no serial dependent-load
    # chain on a cold weight.
    block_k = min(4096, triton.next_power_of_2(K))
    pdl_kwargs = {"USE_GDC": True, "launch_pdl": True} if is_arch_support_pdl() else {}
    router_gate_matvec_kernel[(M, triton.cdiv(E, block_e))](
        hidden_states,
        weight,
        out,
        K,
        E,
        hidden_states.stride(0),
        weight.stride(0),
        BLOCK_E=block_e,
        BLOCK_K=block_k,
        num_warps=ROUTER_GATE_MATVEC_NUM_WARPS,
        **pdl_kwargs,
    )
    return out
