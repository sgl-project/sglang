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
# Any cluster launch caps an SM at 8 CTAs; the occupancy query does not know about clusters.
_MAX_CLUSTER_CTAS_PER_SM = 8
# Per-wave fixed cost in units of one cached vector per thread, and the factor for 1-4 CTAs
# sharing the busiest SM; fit on B300 (see `_select_config`).
_WAVE_OVERHEAD = 1
_SHARED_SM_FACTOR = (1.0, 0.9, 0.8, 0.7)
# 16-bit logits: discount for the 64-register configs (1024 threads per SM); fit on B300.
_HALF_INPUT_PAIRED_FACTOR = 0.9
# Rows the dispatch table is tabulated for; past it the choice no longer changes in practice
_MAX_TABULATED_ROWS = 4096
# Rows chosen one by one; past it, tiers end at powers of two and their 1.5x, since the
# choice there moves by a few percent at most.
_EXACT_ROWS = 64
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
def _jit_softmax_module(
    threads: int,
    vecs_per_thread: int,
    max_cluster: int,
    dtype: torch.dtype,
) -> Module:
    args = make_cpp_args(
        is_arch_support_pdl(), threads, vecs_per_thread, max_cluster, dtype
    )
    return load_jit(
        "sampling_softmax",
        *args,
        cuda_files=["sampling/softmax.cuh"],
        cuda_wrappers=[
            ("softmax", f"SoftmaxKernel<{args}>::run"),
            ("occupancy", f"SoftmaxKernel<{args}>::occupancy"),
        ],
    )


@cache_once
def _device_info() -> tuple[int, int]:
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    return props.major, props.multi_processor_count


def _vec_size(major: int) -> int:
    # 256-bit vectors on Blackwell, 128-bit on Hopper; mirrors `kMaxVecBytes`
    return 8 if major >= 10 else 4


_INPUT_DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def can_softmax(logits: torch.Tensor, out: Optional[torch.Tensor] = None) -> bool:
    if out is not None:
        if out.dtype != torch.float32 or out.shape != logits.shape:
            return False
        if out is not logits and not can_softmax(out):
            return False
    if logits.dtype not in _INPUT_DTYPES or logits.ndim != 2:
        return False
    vec_size = _vec_size(_device_info()[0])
    _, vocab_size = logits.shape
    return (
        0 < vocab_size <= _MAX_SUPPORTED_VOCAB_SIZE
        and vocab_size % vec_size == 0
        and logits.stride(0) % vec_size == 0
        and logits.stride(1) == 1
        and logits.data_ptr() % (vec_size * logits.element_size()) == 0
    )


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


def _candidates(vocab_size: int) -> list[_Config]:
    """Every (threads, cluster size) whose CTAs hold the row in registers with no idle thread."""
    major, _ = _device_info()
    max_cluster = _MAX_CLUSTER if major >= 9 else 1  # clusters start at Hopper
    num_vecs = vocab_size // _vec_size(major)
    max_vecs_per_thread = _MAX_ELEMS_PER_THREAD // _vec_size(major)
    configs = []
    for threads in _THREADS_OPTIONS:
        for cluster in range(1, max_cluster + 1):
            vecs_per_thread = -(-num_vecs // (cluster * threads))
            if vecs_per_thread > max_vecs_per_thread:
                continue
            if cluster > 1 and cluster * threads > num_vecs:
                break  # some threads would hold nothing
            if _cluster_capacity(cluster, 1) > 0:
                configs.append(_Config(cluster, vecs_per_thread, threads))
    if not configs:
        raise ValueError(
            f"softmax: vocab_size {vocab_size} does not fit any cluster on this device"
        )
    return configs


def _config_cost(
    config: _Config, num_rows: int, input_bytes: int, occupancy: int
) -> float:
    """Busiest-SM work of ``num_rows`` rows at the compiled kernel's ``occupancy``.

    Full waves stack ``occupancy`` CTAs on every SM; the last wave stacks the fewest CTAs per SM
    whose cluster capacity holds its rows (clusters never span GPCs, so this is not a division).
    Cost = (waves x `_WAVE_OVERHEAD` + layers x vecs_per_thread x threads / 512) x
    `_SHARED_SM_FACTOR[stacked - 1]`: co-resident CTAs hide each other's exchange and store
    phases. 16-bit logits favour the 1024-thread-per-SM configs by `_HALF_INPUT_PAIRED_FACTOR`.
    """
    cluster, vecs_per_thread, threads = config
    capacity = _cluster_capacity(cluster, occupancy)
    waves = -(-num_rows // capacity)
    tail = num_rows - (waves - 1) * capacity
    tail_layers = next(
        k for k in range(1, occupancy + 1) if _cluster_capacity(cluster, k) >= tail
    )
    layers = (waves - 1) * occupancy + tail_layers
    factor = _SHARED_SM_FACTOR[min(layers, occupancy, len(_SHARED_SM_FACTOR)) - 1]
    if (
        input_bytes < 4
        and vecs_per_thread * _vec_size(_device_info()[0]) <= _PAIRED_ELEMS
    ):
        factor *= _HALF_INPUT_PAIRED_FACTOR
    return (waves * _WAVE_OVERHEAD + layers * vecs_per_thread * threads / 512) * factor


def _select_config(
    rows: range,
    candidates: list[_Config],
    input_bytes: int,
    occupancy: dict[_Config, int],
) -> _Config:
    """The candidate with the least `_config_cost` summed over ``rows``, each row's cost taken
    relative to its cheapest candidate; ties go to fewer threads per row.

    Fit on B300 against an exhaustive sweep (V 129280 / 163840, 1-4096 rows, fp32 and bf16).
    """
    costs = {
        c: [_config_cost(c, r, input_bytes, occupancy[c]) for r in rows]
        for c in candidates
    }
    cheapest = [min(col) for col in zip(*costs.values())]
    return min(
        candidates,
        key=lambda c: (
            sum(x / m for x, m in zip(costs[c], cheapest)),
            c.cluster_size * c.threads,
        ),
    )


def _tier_bounds() -> list[int]:
    """Last row of each dispatch tier: every row up to `_EXACT_ROWS`, then 1.5x / 4/3x steps."""
    bounds = list(range(1, _EXACT_ROWS + 1))
    while bounds[-1] < _MAX_TABULATED_ROWS:
        edge = bounds[-1]
        edge = edge * 3 // 2 if edge & (edge - 1) == 0 else edge * 4 // 3
        bounds.append(min(edge, _MAX_TABULATED_ROWS))
    return bounds


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
    if dtype not in _INPUT_DTYPES:
        raise ValueError(f"softmax: logits must be one of {_INPUT_DTYPES}, got {dtype}")
    # Build every candidate: the cost needs the occupancy its registers actually reach
    candidates = _candidates(vocab_size)
    keys = list(dict.fromkeys(_module_key(c) for c in candidates))
    mods = parallel_compile(
        functools.partial(_jit_softmax_module, *k, dtype) for k in keys
    )
    built = dict(zip(keys, mods))
    occupancy = {}
    for c in candidates:
        occupancy[c] = built[_module_key(c)].occupancy()
        if c.cluster_size > 1:
            occupancy[c] = min(occupancy[c], _MAX_CLUSTER_CTAS_PER_SM)
    # Tabulate the choice per row count into (last row of a run, config) tiers
    max_rows: list[int] = []
    configs: list[_Config] = []
    first = 1
    for last in _tier_bounds():
        rows = range(first, last + 1)
        config = _select_config(rows, candidates, dtype.itemsize, occupancy)
        first = last + 1
        if configs and configs[-1] == config:
            max_rows[-1] = last
        else:
            max_rows.append(last)
            configs.append(config)
    mods = [built[_module_key(c)] for c in configs]
    return _SoftmaxModuleDispatch(max_rows, configs, mods)


def _module_key(config: _Config) -> tuple[int, int, int]:
    return config.threads, config.vecs_per_thread, config.max_cluster


def softmax(
    logits: torch.Tensor,
    temperatures: Optional[torch.Tensor] = None,
    *,
    rows_per_request: int = 1,
    out: Optional[torch.Tensor] = None,
    config: Optional[tuple[int, int, int]] = None,
) -> torch.Tensor:
    """``softmax(logits / T)`` over ``[rows, vocab]`` fp32 / bf16 / fp16 logits, into fp32
    probabilities; in place when ``out is logits`` (fp32 logits only).

    Rows must be whole aligned vectors (``vocab`` and the row stride multiples of 8 fp32 on
    Blackwell, 4 on Hopper; a 32 / 16 B aligned base), as every mainstream vocabulary is;
    anything else raises. ``temperatures`` holds one positive value per request (``rows //
    rows_per_request`` of them, any shape that flattens to that). ``config`` overrides the
    ``(cluster_size, vecs_per_thread, threads)`` choice, for benchmarking.
    """
    if out is None:
        out = torch.empty_like(logits, dtype=torch.float32)
    num_rows, vocab_size = logits.shape
    if config is None:
        cluster_size, module = _precompile(logits.dtype, vocab_size)(num_rows)
    else:
        config = _Config(*config)
        cluster_size = config.cluster_size
        module = _jit_softmax_module(
            config.threads, config.vecs_per_thread, config.max_cluster, logits.dtype
        )
    if temperatures is not None:
        temperatures = temperatures.view(-1)
    module.softmax(logits, out, temperatures, rows_per_request, cluster_size)
    return out
