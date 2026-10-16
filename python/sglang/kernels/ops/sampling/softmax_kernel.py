from __future__ import annotations

import bisect
import functools
from typing import TYPE_CHECKING, NamedTuple, Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    load_jit,
    make_cpp_args,
    parallel_compile,
)

if TYPE_CHECKING:
    from tvm_ffi.module import Module

# NOTE: must match `kSoftmaxMaxCluster` and `kSoftmaxPairedElems` in `softmax.cuh`
_MAX_CLUSTER = 16
_PAIRED_ELEMS = 32
# Cached fp32 per thread before a row is split over more CTAs: the most that fits 128
# registers without spilling on SM100/SM103 (11 vectors of 8; 12 spills).
_MAX_ELEMS_PER_THREAD = 88
_THREADS_OPTIONS = (512, 256)
# Per-wave fixed cost in units of one cached vector per thread, and the per-wave factor for
# 1 / 2 / 4 CTAs sharing an SM; fit on GB300 (see `_select_config`).
_WAVE_OVERHEAD = 4
_SHARED_SM_FACTOR = {1: 1.0, 2: 0.7, 4: 0.6}
# Rows the dispatch table is tabulated for; past it the choice no longer changes in practice
_MAX_TABULATED_ROWS = 4096
_MAX_SUPPORTED_VOCAB_SIZE = _MAX_CLUSTER * max(_THREADS_OPTIONS) * _MAX_ELEMS_PER_THREAD


class _Config(NamedTuple):
    cluster_size: int
    vecs_per_thread: int
    threads: int

    @property
    def max_cluster(self) -> int:
        # Only the pool merge's unroll depends on the cluster size, so round it up to a power of two
        return 1 << (self.cluster_size - 1).bit_length()


@cache_once
def _jit_softmax_module(threads: int, vecs_per_thread: int, max_cluster: int) -> Module:
    args = make_cpp_args(is_arch_support_pdl(), threads, vecs_per_thread, max_cluster)
    return load_jit(
        "sampling_softmax",
        *args,
        cuda_files=["sampling/softmax.cuh"],
        cuda_wrappers=[("softmax", f"SoftmaxKernel<{args}>::run")],
    )


@cache_once
def _device_info() -> tuple[int, int]:
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    return props.major, props.multi_processor_count


def _vec_size(major: int) -> int:
    # 256-bit vectors on Blackwell, 128-bit on Hopper; mirrors `kMaxVecBytes`
    return 8 if major >= 10 else 4


def can_softmax(logits: torch.Tensor, out: Optional[torch.Tensor] = None) -> bool:
    if out is not None and out is not logits and not can_softmax(out):
        return False
    if logits.dtype != torch.float32 or logits.ndim != 2:
        return False
    vec_size = _vec_size(_device_info()[0])
    _, vocab_size = logits.shape
    return (
        0 < vocab_size <= _MAX_SUPPORTED_VOCAB_SIZE
        and vocab_size % vec_size == 0
        and logits.stride(0) % vec_size == 0
        and logits.stride(1) == 1
    )


def _blocks_per_sm(threads: int, vecs_per_thread: int, vec_size: int) -> int:
    # Mirrors `kSoftmaxBlocksPerSM`: 64 registers per thread up to `_PAIRED_ELEMS`, else 128
    return (512 // threads) * (2 if vecs_per_thread * vec_size <= _PAIRED_ELEMS else 1)


@cache_once
def _cluster_capacity(cluster_size: int, blocks_per_sm: int) -> int:
    """Clusters resident at once with `blocks_per_sm` CTAs per SM; GPC placement included."""
    if cluster_size == 1:
        return _device_info()[1] * blocks_per_sm
    from sglang.kernels.jit.utils.occupancy import (
        NoSchedulableClustersError,
        get_max_active_clusters,
    )

    try:
        return get_max_active_clusters(cluster_size, blocks_per_sm)
    except NoSchedulableClustersError:
        return 0


def _select_config(num_rows: int, vocab_size: int) -> _Config:
    """The ``(cluster_size, vecs_per_thread, threads)`` costed fastest for ``[num_rows, vocab_size]``.

    Every (threads, cluster size) whose CTAs can hold the row is costed as
    waves x (`_WAVE_OVERHEAD` + vecs_per_thread x share) x `_SHARED_SM_FACTOR[blocks]`, where
    `blocks` is the CTAs per SM the kernel's registers allow, waves come from the device's real
    cluster capacity at that occupancy (GPC placement included), `share` is the SM's load in
    512-thread CTAs, and CTAs sharing an SM hide each other's exchange and store phases. Ties
    go to fewer threads per row. Fit on GB300 against an exhaustive sweep (V 32000-262144,
    1-4096 rows): mean regret 2.1%.
    """
    major, _ = _device_info()
    max_cluster = _MAX_CLUSTER if major >= 9 else 1  # clusters start at Hopper
    vec_size = _vec_size(major)
    num_vecs = vocab_size // vec_size
    max_vecs_per_thread = _MAX_ELEMS_PER_THREAD // vec_size
    best = None
    for threads in _THREADS_OPTIONS:
        for cluster in range(1, max_cluster + 1):
            vecs_per_thread = -(-num_vecs // (cluster * threads))
            if vecs_per_thread > max_vecs_per_thread:
                continue
            if cluster > 1 and cluster * threads > num_vecs:
                break  # some threads would hold nothing
            blocks = _blocks_per_sm(threads, vecs_per_thread, vec_size)
            capacity = _cluster_capacity(cluster, blocks)
            if capacity == 0:
                continue
            waves = -(-num_rows // capacity)
            share = blocks * threads / 512
            cost = (
                waves
                * (_WAVE_OVERHEAD + vecs_per_thread * share)
                * _SHARED_SM_FACTOR[blocks]
            )
            key = (cost, cluster * threads)
            if best is None or key < best[0]:
                best = (key, _Config(cluster, vecs_per_thread, threads))
    if best is None:
        raise ValueError(
            f"softmax: vocab_size {vocab_size} does not fit any cluster on this device"
        )
    return best[1]


class _SoftmaxModuleDispatch:
    """Maps a row count to the cluster size and module `_select_config` picks for it."""

    def __init__(self, max_rows: list[int], configs: list[_Config], mods: list[Module]):
        self._max_rows = max_rows
        self._entries = tuple(
            (config.cluster_size, mod)
            for config, mod in zip(configs, mods, strict=True)
        )

    def __call__(self, rows: int) -> tuple[int, Module]:
        index = min(bisect.bisect_left(self._max_rows, rows), len(self._entries) - 1)
        return self._entries[index]


@cache_once
def _precompile(dtype: torch.dtype, vocab_size: int) -> _SoftmaxModuleDispatch:
    device_index = torch.cuda.current_device()
    if dtype != torch.float32:
        raise ValueError(f"softmax: logits must be float32, got {dtype}")
    # Tabulate the choice per row count into (last row of a run, config) tiers
    max_rows: list[int] = []
    configs: list[_Config] = []
    for rows in range(1, _MAX_TABULATED_ROWS + 1):
        config = _select_config(rows, vocab_size)
        if configs and configs[-1] == config:
            max_rows[-1] = rows
        else:
            max_rows.append(rows)
            configs.append(config)
    # Neighbouring tiers often alternate between two configs, so build each distinct module once
    keys = set((c.threads, c.vecs_per_thread, c.max_cluster) for c in configs)
    mods = parallel_compile(functools.partial(_jit_softmax_module, *k) for k in keys)
    built = dict(zip(keys, mods))
    mods = [built[(c.threads, c.vecs_per_thread, c.max_cluster)] for c in configs]
    return _SoftmaxModuleDispatch(max_rows, configs, mods)


def softmax(
    logits: torch.Tensor,
    temperatures: Optional[torch.Tensor] = None,
    *,
    rows_per_request: int = 1,
    out: Optional[torch.Tensor] = None,
    config: Optional[tuple[int, int, int]] = None,
) -> torch.Tensor:
    """``softmax(logits / T)`` over fp32 ``[rows, vocab]``; in place when ``out is logits``.

    Rows must be whole aligned vectors (``vocab`` and the row stride multiples of 8 fp32 on
    Blackwell, 4 on Hopper; a 32 / 16 B aligned base), as every mainstream vocabulary is;
    anything else raises. ``temperatures`` holds one positive value per request (``rows //
    rows_per_request`` of them, any shape that flattens to that). ``config`` overrides the
    ``(cluster_size, vecs_per_thread, threads)`` choice, for benchmarking.
    """
    if out is None:
        out = torch.empty_like(logits)
    num_rows, vocab_size = logits.shape
    if config is None:
        cluster_size, module = _precompile(logits.dtype, vocab_size)(num_rows)
    else:
        config = _Config(*config)
        cluster_size = config.cluster_size
        module = _jit_softmax_module(
            config.threads, config.vecs_per_thread, config.max_cluster
        )
    if temperatures is not None:
        temperatures = temperatures.view(-1)
    module.softmax(logits, out, temperatures, rows_per_request, cluster_size)
    return out
