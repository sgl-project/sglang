"""Fused post-norm + residual-add + pre-norm for MuseSpark v1.2.

CUDA replacement for ``_fused_norm_residual_triton_kernel`` in
``sglang.kernels.ops.muse_spark.fused_norm_residual``. It dispatches between the original
one-CTA kernel and the measured multi-CTA cluster kernel from the row count.

The bf16 round-trips reproduced here are load-bearing for parity with the torch
reference: the MuseSpark1x norm modules return the input dtype, so the
normalized branch is rounded through bf16 before the residual add, and the
pre-norm input/output are rounded when ``pre_norm_bf16`` is set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# 16B vectorized bf16 access per thread; hidden_size must be a multiple of
# num_threads * 8 (static_assert in the .cuh).
_VEC = 8
_THREAD_CHOICES = (1024, 512, 256, 128)
_CLUSTER_SIZES = (1, 2, 4, 8, 16)
# Measured on B300 / hidden_size 8192: below this the launch is latency bound and
# up-front weight loads win; above it they cost occupancy. 256 threads beat 1024
# by 0.3-0.5 us per call at M<=32.
_SMALL_M = 192
_SMALL_M_THREAD_CHOICES = (256, 512, 1024, 128)


def _launch_config(hidden_size: int, num_tokens: int) -> tuple[int, bool]:
    small = num_tokens < _SMALL_M
    for threads in _SMALL_M_THREAD_CHOICES if small else _THREAD_CHOICES[1:]:
        if hidden_size % (threads * _VEC) == 0:
            return threads, small
    raise RuntimeError(
        f"muse_spark norm_residual: hidden_size={hidden_size} must be a multiple of "
        f"{_THREAD_CHOICES[-1] * _VEC}"
    )


def cluster_size_for_num_tokens(num_tokens: int) -> int:
    """Return the measured GB300 decode dispatch policy."""
    if num_tokens < 0:
        raise ValueError("num_tokens must be non-negative")
    if num_tokens == 0:
        return 1
    if num_tokens <= 8:
        return 16
    if num_tokens <= 16:
        return 8
    if num_tokens <= 32:
        return 4
    if num_tokens <= 64:
        return 2
    return 1


def _cluster_num_threads(hidden_size: int, cluster_size: int) -> Optional[int]:
    if cluster_size == 1:
        return None
    num_threads = min(256, hidden_size // (cluster_size * _VEC))
    num_threads -= num_threads % 32
    if num_threads < 32 or hidden_size % (cluster_size * num_threads * _VEC):
        return None
    return num_threads


def _cluster_size_for_shape(hidden_size: int, num_tokens: int) -> int:
    cluster_size = cluster_size_for_num_tokens(num_tokens)
    while cluster_size > 1 and _cluster_num_threads(hidden_size, cluster_size) is None:
        cluster_size //= 2
    return cluster_size


def norm_residual_launch_config(
    hidden_size: int, num_tokens: int
) -> tuple[int, int, bool]:
    """Return (cluster size, threads, prefetch) for the automatic shape policy."""
    cluster_size = _cluster_size_for_shape(hidden_size, num_tokens)
    threads, prefetch = _cluster_launch_config(hidden_size, num_tokens, cluster_size)
    return cluster_size, threads, prefetch


def _cluster_launch_config(
    hidden_size: int, num_tokens: int, cluster_size: int
) -> tuple[int, bool]:
    if cluster_size not in _CLUSTER_SIZES:
        raise ValueError(f"cluster_size must be one of {_CLUSTER_SIZES}")
    if cluster_size == 1:
        return _launch_config(hidden_size, num_tokens)

    # Every lane owns complete 16-byte vectors. C=8/16 reduce the CTA width
    # rather than introducing partial vector accesses.
    num_threads = _cluster_num_threads(hidden_size, cluster_size)
    if num_threads is None:
        raise ValueError(
            f"unsupported cluster shape: H={hidden_size}, cluster={cluster_size}"
        )
    return num_threads, num_tokens < _SMALL_M


@cache_once
def _jit_module(
    hidden_size: int,
    num_threads: int,
    prefetch_weights: bool,
    cluster_size: int,
) -> Module:
    args = make_cpp_args(
        hidden_size,
        num_threads,
        is_arch_support_pdl(),
        prefetch_weights,
        cluster_size,
    )
    return load_jit(
        "muse_spark_norm_residual",
        *args,
        cuda_files=["muse_spark/norm_residual.cuh"],
        cuda_wrappers=[
            ("run_std", f"MuseSparkNormResidual<{args}, false, false>::run"),
            ("run_std_router", f"MuseSparkNormResidual<{args}, false, true>::run"),
            ("run_scale", f"MuseSparkNormResidual<{args}, true, false>::run"),
            ("run_scale_router", f"MuseSparkNormResidual<{args}, true, true>::run"),
        ],
    )


def fused_norm_residual(
    branch: torch.Tensor,
    stream: torch.Tensor,
    post_eps: float,
    pre_eps: float,
    post_norm_is_input_scale: bool = False,
    post_weight: Optional[torch.Tensor] = None,
    pre_weight: Optional[torch.Tensor] = None,
    alpha: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    want_router: bool = False,
    pre_norm_bf16: bool = False,
    output: Optional[torch.Tensor] = None,
    new_stream_out: Optional[torch.Tensor] = None,
    router_out: Optional[torch.Tensor] = None,
    launch_config: Optional[tuple[int, bool]] = None,
    cluster_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return ``(output, new_stream, router_out)`` for a ``[M, H]`` row batch.

    ``branch`` is bf16, ``stream`` is fp32, both contiguous. ``post_weight`` /
    ``pre_weight`` are the already-resolved effective weights (bf16 ``[H]``);
    ``alpha`` / ``beta`` are the residual-gate coefficients (fp32, ``[H]`` for
    channel-wise or ``[1]`` for scalar). Passing ``alpha=None`` selects the plain
    ``stream + normed_branch`` add. ``launch_config`` overrides the measured
    ``(num_threads, prefetch_weights)`` choice, for benchmarking.
    """
    assert branch.dtype == torch.bfloat16 and stream.dtype == torch.float32
    assert branch.dim() == 2 and stream.shape == branch.shape
    assert branch.is_contiguous() and stream.is_contiguous()

    m, hidden_size = branch.shape

    if output is None:
        output = torch.empty_like(branch)
    if new_stream_out is None:
        new_stream_out = torch.empty_like(stream)
    if want_router and router_out is None:
        router_out = torch.empty_like(stream)
    if not want_router:
        router_out = None

    if m == 0:
        return output, new_stream_out, router_out

    auto_cluster = cluster_size is None
    if auto_cluster:
        supports_clusters = torch.cuda.get_device_capability(branch.device)[0] >= 10
        cluster_size = (
            _cluster_size_for_shape(hidden_size, m)
            if supports_clusters and not pre_norm_bf16
            else 1
        )

    assert cluster_size is not None
    num_threads, prefetch_weights = _cluster_launch_config(hidden_size, m, cluster_size)
    if cluster_size == 1:
        num_threads, prefetch_weights = launch_config or (
            num_threads,
            prefetch_weights,
        )
    else:
        if launch_config is not None:
            raise ValueError("launch_config only applies to cluster_size=1")
        if pre_norm_bf16:
            raise ValueError("cluster kernel requires FP32 pre-norm input")
    module = _jit_module(hidden_size, num_threads, prefetch_weights, cluster_size)
    fn = getattr(
        module,
        ("run_scale" if post_norm_is_input_scale else "run_std")
        + ("_router" if want_router else ""),
    )
    fn(
        branch,
        stream,
        output,
        new_stream_out,
        router_out,
        post_weight,
        pre_weight,
        alpha,
        beta,
        float(post_eps),
        float(pre_eps),
        bool(pre_norm_bf16),
    )
    return output, new_stream_out, router_out
