"""Hy4 iHC post + pre + RMSNorm for small token counts, on thread-block clusters.

Same math and bf16 rounding points as `hy4_ihc_post_pre_norm` (the previous block's
output RMSNorm folded in, at every token count it serves), optionally with the
normed output also quantized to fp8 e4m3 with one fp32 scale per 128 columns,
bit-identical to `sglang_per_token_group_quant_fp8`.

One token row per cluster of `split` CTAs, `tokens_per_cluster` tokens per cluster;
the row reductions go through distributed shared memory (csrc/hunyuan/ihc_cluster.cuh).
Coverage: SM90 / SM100 / SM103, bf16 activations, hc_mult 1 / 2 / 4 / 8, hidden a
multiple of 128 that some cluster size in `_SPLITS` divides into 128-column groups.
Measured on H200 only. Larger token counts belong to the Triton kernels: the
dispatcher serves a token count only while one wave of clusters holds it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import msgspec
import torch

from sglang.kernels.arch import current_arch
from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit
from sglang.kernels.jit.utils.compile.parallel import parallel_compile
from sglang.kernels.ops.quantization.fp8_kernel import (
    create_per_token_group_quant_fp8_output_scale,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_GROUP = 128
_CAPABILITIES = ((9, 0), (10, 0), (10, 3))
_HC_MULTS = (1, 2, 4, 8)
# Portable cluster sizes; a split must divide hidden / 128.
_SPLITS = (2, 3, 4, 6, 8)
_MIN_PREFERRED_SPLIT = 4
_TOKENS_PER_CLUSTER = (1, 2, 4)
_MAX_THREADS = 1024
_MAX_STATIC_SMEM = 48 * 1024


class ClusterConfig(msgspec.Struct, frozen=True):
    split: int
    tokens_per_cluster: int


class _Variant(msgspec.Struct, frozen=True):
    hc_mult: int
    hidden_size: int
    has_output: bool
    out_norm: bool
    quant: bool


def _round_up(x: int, m: int) -> int:
    return (x + m - 1) // m * m


def _static_smem_bytes(hc: int, hidden: int, split: int, tb: int) -> int:
    """Upper bound of the kernel's `Smem` struct (same field list, 16-byte padding)."""
    groups = hidden // _GROUP
    gpc = groups // split
    nq = _round_up(2 * hc + 1, 4)
    fields = (
        tb * hc * gpc * _GROUP * 2,
        tb * hc * gpc * nq * 4,
        tb * groups * nq * 4,
        tb * groups * 4,
        tb * groups * 4,
        tb * 4,
        tb * hc * 4,
        16,
    )
    return sum(_round_up(f, 16) for f in fields)


def _candidate_configs(hc: int, hidden: int) -> tuple[ClusterConfig, ...]:
    """The (split, tokens per cluster) configs the kernel compiles for this shape,
    in dispatch preference order: fewer tokens per cluster first (each extra token
    is another serial pass over the CTA's columns), then the larger cluster (less
    work per CTA). Splits below 4 only when no larger one divides the row: on H200
    they were slower than every config with a split of 4 or more."""
    if hidden % _GROUP:
        return ()
    groups = hidden // _GROUP
    splits = [
        s
        for s in _SPLITS
        if groups % s == 0
        and (16 * hc * (groups // s)) % 32 == 0
        and 16 * hc * (groups // s) <= _MAX_THREADS
    ]
    if any(s >= _MIN_PREFERRED_SPLIT for s in splits):
        splits = [s for s in splits if s >= _MIN_PREFERRED_SPLIT]
    return tuple(
        ClusterConfig(s, tb)
        for tb in _TOKENS_PER_CLUSTER
        for s in sorted(splits, reverse=True)
        if _static_smem_bytes(hc, hidden, s, tb) <= _MAX_STATIC_SMEM
    )


def hy4_ihc_cluster_supported(
    *, hc_mult: int, hidden_size: int, dtype: torch.dtype
) -> bool:
    """Whether the local GPU and this shape take the cluster kernel. Initializes
    CUDA only on an NVIDIA platform."""
    arch = current_arch()
    return (
        any(arch.is_sm(major, minor) for major, minor in _CAPABILITIES)
        and dtype == torch.bfloat16
        and hc_mult in _HC_MULTS
        and len(_candidate_configs(hc_mult, hidden_size)) > 0
    )


@cache_once
def _module(variant: _Variant, config: ClusterConfig) -> Module:
    args = (
        f"{variant.hc_mult}, {variant.hidden_size}, {config.split}, "
        f"{config.tokens_per_cluster}, {str(variant.has_output).lower()}, "
        f"{str(variant.out_norm).lower()}, {str(variant.quant).lower()}, "
        f"{str(is_arch_support_pdl()).lower()}"
    )
    return load_jit(
        "hy4_ihc_cluster",
        *args.split(", "),
        cuda_files=["hunyuan/ihc_cluster.cuh"],
        cuda_wrappers=[
            ("run", f"Hy4IhcCluster<{args}>::run"),
            ("max_active_clusters", f"Hy4IhcCluster<{args}>::max_active_clusters"),
        ],
    )


@cache_once
def _plan(variant: _Variant, device_index: int) -> tuple[ClusterConfig, ...]:
    """plan[T] is the config for T tokens; len(plan) - 1 is the largest T served.

    T takes the first config in preference order whose ceil(T / tokens per cluster)
    clusters fit in one wave (cudaOccupancyMaxActiveClusters on this device). Past
    one wave of every config the clusters queue, and the Triton kernels, which stream
    hc_fn through tensor cores, were faster on H200 at every such T.
    """
    configs = _candidate_configs(variant.hc_mult, variant.hidden_size)
    modules = parallel_compile((lambda c=c: _module(variant, c)) for c in configs)
    with torch.cuda.device(device_index):
        capacity = [int(m.max_active_clusters()) for m in modules]
    plan = [ClusterConfig(0, 0)]
    while True:
        T = len(plan)
        for config, clusters in zip(configs, capacity):
            if -(-T // config.tokens_per_cluster) <= clusters:
                plan.append(config)
                break
        else:
            return tuple(plan)


def hy4_ihc_cluster_max_tokens(
    *,
    hc_mult: int,
    hidden_size: int,
    has_output: bool,
    out_norm: bool,
    quant: bool,
    device: torch.device,
) -> int:
    """Largest token count `hy4_ihc_cluster` serves for this variant on `device`."""
    variant = _Variant(hc_mult, hidden_size, has_output, out_norm, quant)
    return len(_plan(variant, device.index)) - 1


def prepare_hy4_ihc_cluster(
    *,
    hc_mult: int,
    hidden_size: int,
    device: torch.device,
    variants: tuple[tuple[bool, bool, bool], ...],
) -> None:
    """Compile every config of each (has_output, out_norm, quant) variant, all in
    parallel, and build the dispatch tables, so serving never compiles."""
    variants = [_Variant(hc_mult, hidden_size, *v) for v in variants]
    configs = _candidate_configs(hc_mult, hidden_size)
    parallel_compile(
        (lambda v=v, c=c: _module(v, c)) for v in variants for c in configs
    )
    for variant in variants:
        _plan(variant, device.index)


def hy4_ihc_cluster(
    output: torch.Tensor | None,
    residual: torch.Tensor,
    post: torch.Tensor | None,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    magnitude: float,
    norm_eps: float,
    hc_eps: float,
    rms_weight: torch.Tensor,
    rms_eps: float,
    out_norm_weight: torch.Tensor | None = None,
    out_norm_eps: float = 0.0,
    quant: bool = False,
    config: ClusterConfig | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor] | None,
]:
    """Same contract as `hy4_ihc_post_pre_norm`, for 1 <= T <=
    `hy4_ihc_cluster_max_tokens`. Returns (new residual, y, next post, y_fp8):
    y_fp8 is (fp8 [T, hidden], fp32 scales [T, hidden / 128] with token stride 1
    and group stride ceil(T / 4) * 4, as DeepGEMM reads them) when `quant`, else None.
    `config` overrides the dispatch table (benchmarks and tests)."""
    assert residual.dtype == torch.bfloat16, residual.dtype
    T, hc, hidden = residual.shape
    has_output = output is not None
    variant = _Variant(hc, hidden, has_output, out_norm_weight is not None, quant)
    if config is None:
        config = _plan(variant, residual.device.index)[T]
    y = torch.empty((T, hidden), dtype=residual.dtype, device=residual.device)
    next_post = torch.empty((T, hc), dtype=torch.float32, device=residual.device)
    new_residual = torch.empty_like(residual) if has_output else None
    y_fp8 = None
    if quant:
        y_fp8 = (
            torch.empty((T, hidden), dtype=torch.float8_e4m3fn, device=y.device),
            create_per_token_group_quant_fp8_output_scale(
                x_shape=(T, hidden),
                device=y.device,
                group_size=_GROUP,
                column_major_scales=True,
                scale_tma_aligned=True,
                scale_ue8m0=False,
            ),
        )
    _module(variant, config).run(
        residual,
        output,
        post,
        out_norm_weight,
        hc_fn,
        hc_scale,
        hc_base,
        rms_weight,
        new_residual,
        y,
        next_post,
        *(y_fp8 if quant else (None, None)),
        out_norm_eps,
        norm_eps,
        hc_eps,
        rms_eps,
        magnitude,
    )
    return (residual if new_residual is None else new_residual), y, next_post, y_fp8
