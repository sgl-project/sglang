# SPDX-License-Identifier: Apache-2.0
"""cuTile router GEMM: BF16 activations × FP32 weights → FP32 logits.

The caller splits each FP32 weight into three BF16 components before serving.
Main-product K tiles are combined with explicit FP32 rounding for accuracy.
Split-K configurations use a separate reduction kernel.
"""

import functools

import cuda.tile as ct
import torch

from sglang.srt.utils.custom_op import register_custom_op


@ct.kernel(num_ctas=1)
def _router_gemm_bf16x3_kernel(
    X,
    W,
    WR,
    Y,
    BM: ct.Constant[int],
    BN: ct.Constant[int],
    BK: ct.Constant[int],
    SPLIT_K: ct.Constant[int],
    EXPERT_MAJOR: ct.Constant[bool],
    PDL: ct.Constant[bool],
):
    col, row, part = ct.bid(0), ct.bid(1), ct.bid(2)
    if PDL:
        ct.grid_dependency_control_launch_dependents()
        ct.grid_dependency_control_wait()
    steps = ct.cdiv(ct.cdiv(X.shape[1], BK), SPLIT_K)
    if EXPERT_MAJOR:
        total = ct.full((4 * BN, BM), 0, ct.float32)
    else:
        total = ct.full((BM, BN), 0, ct.float32)
        residual = ct.full((BM, 2 * BN), 0, ct.float32)
    for step in range(steps):
        k = part * steps + step
        x = ct.load(
            X,
            (row, k),
            (BM, BK),
            padding_mode=ct.PaddingMode.ZERO,
            latency=4 if EXPERT_MAJOR or BK == 128 else 8,
        )
        if EXPERT_MAJOR:
            w = ct.load(
                W, (0, col, k), (4, BN, BK), padding_mode=ct.PaddingMode.ZERO, latency=4
            )
            local = ct.mma(
                ct.reshape(w, (4 * BN, BK)),
                ct.transpose(x),
                ct.full((4 * BN, BM), 0, ct.float32),
            )
        else:
            w0 = ct.load(
                W,
                (0, col, k),
                (1, BN, BK),
                padding_mode=ct.PaddingMode.ZERO,
                latency=4 if BK == 128 else 8,
            )
            wr = ct.load(
                WR,
                (0, col, k),
                (2, BN, BK),
                padding_mode=ct.PaddingMode.ZERO,
                latency=4 if BK == 128 else 8,
            )
            local = ct.mma(
                x,
                ct.transpose(ct.reshape(w0, (BN, BK))),
                ct.full((BM, BN), 0, ct.float32),
            )
            residual = ct.mma(x, ct.transpose(ct.reshape(wr, (2 * BN, BK))), residual)
        # Round each main-product tile before adding the next tile.
        total = ct.add(total, local, rounding_mode=ct.RoundingMode.RN)
    if EXPERT_MAJOR:
        main = ct.extract(total, (0, 0), (BN, BM))
        mid = ct.extract(total, (1, 0), (BN, BM))
        low = ct.extract(total, (2, 0), (BN, BM))
        value = ct.transpose(
            ct.add(
                main,
                ct.add(mid, low, rounding_mode=ct.RoundingMode.RN),
                rounding_mode=ct.RoundingMode.RN,
            )
        )
    else:
        correction = ct.sum(
            ct.reshape(residual, (BM, 2, BN)), axis=1, rounding_mode=ct.RoundingMode.RN
        )
        value = ct.add(total, correction, rounding_mode=ct.RoundingMode.RN)
    ct.store(
        Y,
        (part, row, col),
        ct.reshape(value, (1, BM, BN)),
        allow_tma=not (BM == 256 and SPLIT_K > 1),
    )


@ct.kernel
def _split_k_reduce_kernel(
    P, Y, SPLIT_K: ct.Constant[int], PDL: ct.Constant[bool] = False
):
    if PDL:
        ct.grid_dependency_control_wait()
    if SPLIT_K == 2:
        values = ct.load(
            P,
            (0, ct.bid(0)),
            (SPLIT_K, 1024),
            padding_mode=ct.PaddingMode.ZERO,
            allow_tma=False,
        )
        total = ct.sum(values, axis=0, rounding_mode=ct.RoundingMode.RN)
    else:
        total = ct.full((1024,), 0, ct.float32)
        for part in range(SPLIT_K):
            value = ct.load(
                P,
                (part, ct.bid(0)),
                (1, 1024),
                padding_mode=ct.PaddingMode.ZERO,
                allow_tma=False,
            )
            total = ct.add(
                total, ct.reshape(value, (1024,)), rounding_mode=ct.RoundingMode.RN
            )
    ct.store(Y, (ct.bid(0),), total, allow_tma=False)


_GEMM_KERNELS = {
    1: _router_gemm_bf16x3_kernel,
    2: _router_gemm_bf16x3_kernel.replace_hints(num_ctas=2),
}
# (max M, (BM, BN, BK, SPLIT_K, CTAS)); larger M uses the fallback below.
_BLACKWELL_CONFIGS = (
    (64, (32, 16, 256, 4, 1)),
    (128, (64, 32, 256, 4, 1)),
    (512, (128, 64, 256, 8, 1)),
    (1024, (128, 64, 128, 4, 1)),
    (2048, (128, 64, 128, 2, 1)),
    (4096, (256, 128, 256, 2, 2)),
)
# Use smaller tiles on Hopper to limit register pressure.
_HOPPER_CONFIGS = (
    (256, (64, 64, 128, 8, 1)),
    (512, (64, 64, 128, 4, 1)),
    (1024, (64, 64, 128, 2, 1)),
)


@functools.cache
def _is_hopper(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index)[0] == 9


def _select_launch_config(m: int, device: torch.device) -> tuple:
    if m > 128 and _is_hopper(device.index):
        configs, fallback = _HOPPER_CONFIGS, (64, 64, 128, 1, 1)
    else:
        configs, fallback = _BLACKWELL_CONFIGS, (256, 128, 256, 1, 2)
    return next((config for limit, config in configs if m <= limit), fallback)


def _router_gemm_bf16x3_fake(x: torch.Tensor, parts: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], parts.shape[0] // 3), dtype=torch.float32)


@register_custom_op(fake_impl=_router_gemm_bf16x3_fake)
def router_gemm_bf16x3_cutile(x: torch.Tensor, parts: torch.Tensor) -> torch.Tensor:
    """Multiply BF16 [M, K] by prepared BF16 [3N, K], producing FP32 [M, N]."""
    m, k = x.shape
    n = parts.shape[0] // 3
    bm, bn, bk, splits, ctas = _select_launch_config(m, x.device)
    pdl = bk == 128 and splits == 4
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    if m == 0 or n == 0:
        return out
    partial = (
        torch.empty((splits, m, n), device=x.device, dtype=torch.float32)
        if splits > 1
        else out.unsqueeze(0)
    )
    with torch.cuda.device(x.device):
        stream = torch.cuda.current_stream()
        ct.launch(
            stream,
            (ct.cdiv(n, bn), ct.cdiv(m, bm), splits),
            _GEMM_KERNELS[ctas],
            (
                x,
                parts.view(3, n, k),
                parts[n:].view(2, n, k),
                partial,
                bm,
                bn,
                bk,
                splits,
                m <= 64,
                pdl,
            ),
            programmatic_dependent_launch=pdl,
        )
        if splits > 1:
            ct.launch(
                stream,
                (ct.cdiv(out.numel(), 1024),),
                _split_k_reduce_kernel,
                (partial.view(splits, -1), out.view(-1), splits, pdl),
                programmatic_dependent_launch=pdl,
            )
    return out
