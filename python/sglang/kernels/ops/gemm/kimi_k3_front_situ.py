"""Kimi-K3 latent-MoE input step in one Triton GEMM with the shared experts' SiTU in its epilogue.

One launch over the merged front weight [shared gate | shared up | router | latent down] (rows).
A gated tile pairs gate row j with up row N_HALF + j in adjacent columns, so it streams the same
weight bytes as any other tile and writes SiTU(gate) * up; the rest write router logits and the
latent input. Replaces the merged-front GEMM followed by a separate SiTU kernel for the prefill
batch sizes covered() admits.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# (first, last tokens, (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages, waves_per_eu)):
# the sizes where it beats the tuned merged-front GEMM + SiTU kernel on MI350X, each with the
# fastest config of a sweep; at the GEMM's tuned sizes and between these, the GEMM wins.
_RANGES = (
    (129, 160, (32, 128, 128, 8, 4, 3, 4)),
    (193, 256, (128, 64, 128, 8, 4, 3, 1)),
    (257, 320, (64, 128, 128, 8, 4, 3, 2)),
    (1152, 1280, (256, 128, 64, 8, 8, 3, 1)),
)


def covered(num_tokens: int) -> bool:
    return any(first <= num_tokens <= last for first, last, _ in _RANGES)


def _config(num_tokens: int):
    for first, last, config in _RANGES:
        if first <= num_tokens <= last:
            return config
    raise ValueError(f"no front SiTU config for {num_tokens} tokens")


@triton.jit
def _tanh(x):
    return 2.0 / (1.0 + tl.exp(-2.0 * x)) - 1.0


@triton.jit
def _remap_xcd(pid, num_pids, NUM_XCDS: tl.constexpr):
    # Hardware deals pids round-robin over the XCDs; give each XCD (one L2) a run of
    # consecutive pids, so the programs that share a weight panel share its L2.
    pids_per_xcd = (num_pids + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = num_pids % NUM_XCDS
    tall_xcds = tl.where(tall_xcds == 0, NUM_XCDS, tall_xcds)
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    return tl.where(
        xcd < tall_xcds,
        xcd * pids_per_xcd + local_pid,
        tall_xcds * pids_per_xcd + (xcd - tall_xcds) * (pids_per_xcd - 1) + local_pid,
    )


@triton.jit
def _grouped_pid(pid, num_pid_m, num_pid_n, GROUP_M: tl.constexpr):
    if GROUP_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_M * num_pid_n
        first_pid_m = pid // num_pid_in_group * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + pid % group_size_m
        pid_n = pid % num_pid_in_group // group_size_m
    return pid_m, pid_n


@triton.jit
def _front_situ_kernel(
    x_ptr,
    w_ptr,
    act_ptr,
    rest_ptr,
    M,
    stride_xm,
    stride_am,
    stride_rm,
    beta,
    linear_beta,
    K: tl.constexpr,
    N_HALF: tl.constexpr,
    N_REST: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    num_pid_m = tl.cdiv(M, BLOCK_M)
    n_gated: tl.constexpr = 2 * N_HALF // BLOCK_N
    num_pid_n: tl.constexpr = n_gated + N_REST // BLOCK_N
    # gfx950 parts carry 8 XCDs
    pid = _remap_xcd(tl.program_id(0), num_pid_m * num_pid_n, NUM_XCDS=8)
    pid_m, pid_n = _grouped_pid(pid, num_pid_m, num_pid_n, GROUP_M)
    gated = pid_n < n_gated
    cols = tl.arange(0, BLOCK_N)
    # Gated tile: column 2j is gate row j, column 2j + 1 the matching up row.
    rows = tl.where(
        gated,
        pid_n * (BLOCK_N // 2) + cols // 2 + (cols % 2) * N_HALF,
        2 * N_HALF + (pid_n - n_gated) * BLOCK_N + cols,
    )
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + (offs_m % M)[:, None] * stride_xm + offs_k[None, :]
    w_ptrs = w_ptr + rows[None, :].to(tl.int64) * K + offs_k[:, None]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(x_ptrs)
        b = tl.load(w_ptrs, cache_modifier=".cg")
        acc = tl.dot(a, b, acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
    mask_m = offs_m[:, None] < M
    if gated:
        # The gate and up projections are bf16 before SiTU reads them, as in the unfused pair.
        acc = acc.to(tl.bfloat16).to(tl.float32)
        gate, up = tl.split(tl.reshape(acc, [BLOCK_M, BLOCK_N // 2, 2]))
        gate = beta * _tanh(gate / beta) * tl.sigmoid(gate)
        up = linear_beta * _tanh(up / linear_beta)
        offs_a = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)
        tl.store(
            act_ptr + offs_m[:, None] * stride_am + offs_a[None, :],
            (gate * up).to(act_ptr.dtype.element_ty),
            mask=mask_m,
        )
    else:
        offs_r = (pid_n - n_gated) * BLOCK_N + cols
        tl.store(
            rest_ptr + offs_m[:, None] * stride_rm + offs_r[None, :],
            acc.to(rest_ptr.dtype.element_ty),
            mask=mask_m,
        )


def front_situ(
    x: torch.Tensor,
    weight: torch.Tensor,
    gate_up_rows: int,
    beta: float,
    linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(SiTU(x @ gate.T) * (x @ up.T), x @ rest.T) for weight = [gate; up; rest] rows, gate and
    up gate_up_rows / 2 rows each; x [M, K] bf16 (row stride free), weight contiguous."""
    m, k = x.shape
    block_m, block_n, block_k, group_m, num_warps, num_stages, waves_per_eu = _config(m)
    n_half = gate_up_rows // 2
    n_rest = weight.shape[0] - gate_up_rows
    assert gate_up_rows % block_n == 0 and n_rest % block_n == 0 and k % block_k == 0
    assert x.stride(-1) == 1 and weight.is_contiguous()
    act = torch.empty((m, n_half), dtype=x.dtype, device=x.device)
    rest = torch.empty((m, n_rest), dtype=x.dtype, device=x.device)
    grid = (triton.cdiv(m, block_m) * ((gate_up_rows + n_rest) // block_n),)
    _front_situ_kernel[grid](
        x,
        weight,
        act,
        rest,
        m,
        x.stride(0),
        act.stride(0),
        rest.stride(0),
        beta,
        linear_beta,
        K=k,
        N_HALF=n_half,
        N_REST=n_rest,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
        matrix_instr_nonkdim=16,
    )
    return act, rest


def warmup(
    weight: torch.Tensor, gate_up_rows: int, beta: float, linear_beta: float
) -> None:
    """Compile every config before serving: prefill sizes are not graph-captured."""
    for first, _, _ in _RANGES:
        x = weight.new_zeros((first, weight.shape[1]))
        front_situ(x, weight, gate_up_rows, beta, linear_beta)
