# Copy from https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gen_ai
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import functools
import inspect
import sys
import warnings
from typing import Optional

import torch
import triton  # @manual
import triton.language as tl  # @manual
from triton.runtime import driver  # @manual


def map_dtype_to_triton(dtype: torch.dtype) -> tl.dtype:
    """
    Maps torch dtype to triton dtype.

    Args:
        dtype (torch.dtype): input dtype.

    Returns:
        tl.dtype: triton dtype.
    """
    if dtype == torch.float16:
        return tl.float16
    elif dtype == torch.bfloat16:
        return tl.bfloat16
    elif dtype == torch.float32:
        return tl.float32
    elif dtype == torch.int32:
        return tl.int32
    elif dtype == torch.float8_e4m3fn and torch.version.hip is None:
        return tl.float8e4nv
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


# check if we have the TMA version in Triton PR #4498 (https://github.com/triton-lang/triton/pull/4498).
HAS_TMA_DESC = "nv_tma_desc_type" in dir(tl)

if HAS_TMA_DESC:
    print(
        "TMA benchmarks will be running with experimental grid constant TMA descriptor.",
        file=sys.stderr,
    )
else:
    print(
        "TMA benchmarks will be running without grid constant TMA descriptor.",
        file=sys.stderr,
    )


class TmaAutoTuneHelper:

    # duck typing wrapper to implement the same interface as TmaDescKernelParam in Triton PR #4498
    class KernelParamWrapper:
        def __init__(self, desc):
            self.desc = desc

        def tma_desc_cpu_ptr(self):
            return self.desc.data_ptr()

    TMA_SIZE = 128

    def __init__(self):
        self.fill_1d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_1d_tma_descriptor
        )
        self.fill_2d_tma_descriptor_inner = (
            triton.runtime.driver.active.utils.fill_2d_tma_descriptor
        )
        if HAS_TMA_DESC:
            self.descriptors = {}
        else:
            self.cuda_descriptors = {}

    # Call this method outside of the lambda function for grid size
    def init_tma_descriptor(self, name):
        if HAS_TMA_DESC:
            self.descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cpu", dtype=torch.int8
            )
        else:
            self.cuda_descriptors[name] = torch.empty(
                TmaAutoTuneHelper.TMA_SIZE, device="cuda", dtype=torch.int8
            )

    # Call this method inside the lambda function for grid size
    def fill_1d_tma_descriptor(self, name, ptr, dim, block_dim, element_size):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_1d_tma_descriptor_inner(
                ptr, dim, block_dim, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    # Call this method inside the lambda function for grid size
    def fill_2d_tma_descriptor(
        self, name, ptr, dim1, dim0, block_dim1, block_dim0, element_size
    ):
        if HAS_TMA_DESC:
            desc_x = self.descriptors[name]
            assert desc_x.data_ptr() % 64 == 0
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, desc_x.data_ptr()
            )
        else:
            desc_x = self.cuda_descriptors[name]
            buf_x = torch.empty_like(desc_x, device="cpu", pin_memory=True)
            self.fill_2d_tma_descriptor_inner(
                ptr, dim1, dim0, block_dim1, block_dim0, element_size, buf_x.data_ptr()
            )
            desc_x.copy_(buf_x, non_blocking=True)

    def get_tma_descriptor_kernel_param(self, name):
        if HAS_TMA_DESC:
            assert self.descriptors[name] is not None
            return self.KernelParamWrapper(self.descriptors[name])
        else:
            assert self.cuda_descriptors[name] is not None
            return self.cuda_descriptors[name]


_NV_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "NUM_CONSUMER_GROUPS": 1,
        },
        num_stages=num_stages,
        num_warps=num_warps,
        num_ctas=num_ctas,
    )
    for block_size_m in [64, 128]
    for block_size_n in [64, 128, 256]
    for block_size_k in [64, 128, 256]
    for num_stages in [3, 4]
    for num_warps in [4, 8]
    for num_ctas in [1]
]

_HAS_WS_SUPPORT = None


def _check_ws_support():
    if not hasattr(tl, "async_task"):
        return False
    config_signature = inspect.signature(triton.Config).parameters
    if (
        "num_consumer_groups" not in config_signature
        or "num_buffers_warp_spec" not in config_signature
    ):
        return False
    if not HAS_TMA_DESC:
        return False
    return True


def _set_ws_support():
    global _HAS_WS_SUPPORT
    if _HAS_WS_SUPPORT is None:
        _HAS_WS_SUPPORT = _check_ws_support()


_set_ws_support()

if _HAS_WS_SUPPORT:
    _NV_WS_CONFIGS = [
        triton.Config(
            {
                "BLOCK_SIZE_M": block_size_m,
                "BLOCK_SIZE_N": block_size_n,
                "BLOCK_SIZE_K": block_size_k,
                "NUM_CONSUMER_GROUPS": max(1, num_consumer_groups),
                "USE_TMA_LOAD_ON_SCALES": use_tma_load_on_scales,
                "USE_TMA_STORE": use_tma_store,
            },
            num_stages=num_stages,
            num_warps=num_warps,
            num_ctas=num_ctas,
            num_consumer_groups=num_consumer_groups,
            num_buffers_warp_spec=num_stages,
        )
        for block_size_m in [64, 128, 256]
        for block_size_n in [64, 128, 256]
        for block_size_k in [64, 128, 256]
        for num_stages in [2, 3, 4]
        for num_warps in [4, 8, 16]
        # TODO(shikaili): Resolve LLVM error.
        for num_ctas in [1]
        for num_consumer_groups in [0, 2]
        for use_tma_load_on_scales in [True, False]
        # TODO(shikaili): Resolve compatibility with ws.
        for use_tma_store in [False]
    ]
else:
    _NV_WS_CONFIGS = _NV_CONFIGS


_AMD_CONFIGS = [
    triton.Config(
        {
            "BLOCK_SIZE_M": block_size_m,
            "BLOCK_SIZE_N": block_size_n,
            "BLOCK_SIZE_K": block_size_k,
            "waves_per_eu": waves_per_cu,
            "matrix_instr_nonkdim": matrix_instr_nonkdim,
            "NUM_CONSUMER_GROUPS": 1,
        },
        num_stages=num_stages,
        num_warps=num_warps,
    )
    for block_size_m in [32, 64, 128]
    for block_size_n in [32, 64, 128, 256]
    for block_size_k in [128, 256]
    for num_stages in [1, 2]
    for num_warps, waves_per_cu in [(4, 1), (8, 2), (16, 4)]
    for matrix_instr_nonkdim in [16]
]


def early_config_prune(configs, named_args, dtsize=None, dtype=None, **kwargs):
    device = torch.cuda.current_device()
    # BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages
    if dtsize is None:
        dtsize = named_args["c_ptr"].element_size()
    if dtype is None:
        dtype = named_args["c_ptr"].dtype

    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        (
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            num_stages,
            num_warps,
            num_consumer_groups,
            use_tma_load_on_scales,
        ) = (
            kw["BLOCK_SIZE_M"],
            kw["BLOCK_SIZE_N"],
            kw["BLOCK_SIZE_K"],
            config.num_stages,
            config.num_warps,
            config.num_consumer_groups,
            kw.get("USE_TMA_LOAD_ON_SCALES", False),
        )
        G, M, N, K = (
            named_args["G"],
            named_args["M_BUCKET"],
            named_args["N"],
            named_args["K"],
        )

        # 1. make sure we have enough smem
        max_shared_memory = driver.active.utils.get_device_properties(device)[
            "max_shared_mem"
        ]
        if torch.version.hip:
            required_shared_memory = BLOCK_N * BLOCK_K * num_stages * dtsize
        else:
            required_shared_memory = (BLOCK_M + BLOCK_N) * BLOCK_K * num_stages * dtsize
        if required_shared_memory > max_shared_memory:
            continue

        use_warp_specialization = num_consumer_groups >= 1

        M_PER_GROUP = M // G
        MIN_M_TILES = 32 if torch.version.hip else 64
        # 2. make sure we don't load M tiles that are too big
        if (
            not use_warp_specialization
            and BLOCK_M > MIN_M_TILES
            and BLOCK_M > (M_PER_GROUP * 2)
        ):
            continue
        # 3. make sure we don't load N tiles that are too small
        if BLOCK_M < 128 and BLOCK_M < (M_PER_GROUP // 2):
            continue

        num_sm = driver.active.utils.get_device_properties(device)[
            "multiprocessor_count"
        ]
        N_TILES = N // BLOCK_N
        MIN_N_TILES = 32 if torch.version.hip else 64
        # 4. make sure we don't load N tiles that are too big
        if (
            not use_warp_specialization
            and BLOCK_N > MIN_N_TILES
            and M * N_TILES < num_sm
        ):
            continue
        # 5. make sure we don't load N tiles that are too small
        if BLOCK_N < 128 and M * N_TILES > 2 * num_sm:
            continue

        # 6. make sure K can be evenly divided
        if K % BLOCK_K != 0:
            continue

        # 7. make sure we can partition for ws
        if use_warp_specialization:
            if num_warps != 4:
                continue

            # "tritongpu-warp-spec-data-partition"
            m_slice = BLOCK_M // num_consumer_groups
            n_slice = BLOCK_N // num_consumer_groups
            if m_slice < 64 and n_slice < 256:
                continue

        if dtsize >= 2:
            if use_tma_load_on_scales:
                continue
        pruned_configs.append(config)

    return pruned_configs


@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
    restore_value=["c_ptr"],  # restore for scatter_add fusion
)
@triton.jit
def _fbgemm_grouped_gemm(
    a_desc_ptr,
    b_desc_ptr,
    c_ptr,
    workspace,
    scatter_add_indices,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FUSE_SCATTER_ADD: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
) -> None:
    tl.static_assert(
        not (FUSE_SCATTER_ADD and USE_TMA_STORE),
        "Cannot fuse scatter add with TMA store!",
    )

    tidx = tl.program_id(0)

    dtype: tl.dtype = c_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    M_end_offset = M_end_offset.to(tl.int64)
    iterated_tiles = 0
    iterated_tiles = iterated_tiles.to(tl.int64)
    for g in tl.range(G):
        # Move across groups
        m_size = tl.load(m_sizes + g)

        if m_size > 0:
            M_start_offset = M_end_offset
            M_end_offset = M_start_offset + m_size
            N_start_offset = g.to(tl.int64) * N
            n_size = N

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                # pyre-ignore
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=c_ptr + M_start_offset * N,
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    global_size=[m_size, n_size],
                    element_ty=c_ptr.dtype.element_ty,
                )
                # pyre-ignore
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # Split M first and N second.
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                tl.static_assert(K % BLOCK_SIZE_K == 0)
                if USE_TMA_LOAD:
                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = tl._experimental_descriptor_load(
                            a_desc_ptr,
                            [m_offset, k_offset],
                            [BLOCK_SIZE_M, BLOCK_SIZE_K],
                            dtype,
                        )
                        b = tl._experimental_descriptor_load(
                            b_desc_ptr,
                            [n_offset, k_offset],
                            [BLOCK_SIZE_N, BLOCK_SIZE_K],
                            dtype,
                        )
                        if USE_FAST_ACCUM:
                            accumulator = tl.dot(a, b.T, accumulator)
                        else:
                            accumulator += tl.dot(a, b.T)
                else:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_k = tl.arange(0, BLOCK_SIZE_K)
                    a_ptrs = (
                        a_desc_ptr
                        + (M_start_offset + offs_am[:, None]) * K
                        + offs_k[None, :]
                    )
                    b_ptrs = (
                        b_desc_ptr
                        + (N_start_offset + offs_bn[:, None]) * K
                        + offs_k[None, :]
                    )
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = tl.load(a_ptrs, mask=offs_am[:, None] < m_size)
                        b = tl.load(b_ptrs, mask=offs_bn[:, None] < n_size)
                        accumulator += tl.dot(a, b.T)
                        a_ptrs += BLOCK_SIZE_K
                        b_ptrs += BLOCK_SIZE_K

                if USE_TMA_STORE:
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    tl._experimental_descriptor_store(
                        c_desc_ptr,
                        accumulator.to(c_ptr.dtype.element_ty),
                        [m_offset, n_offset],
                    )
                elif FUSE_SCATTER_ADD:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    mask = offs_am < m_size
                    m_offsets = tl.load(
                        scatter_add_indices + M_start_offset + offs_am,
                        mask=mask,
                        cache_modifier=".ca",
                    )
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    c = accumulator.to(c_ptr.dtype.element_ty)
                    tl.atomic_add(
                        c_ptr + m_offsets[:, None] * N + offs_bn[None, :],
                        c,
                        mask=mask[:, None] and offs_bn[None, :] < n_size,
                        sem="relaxed",
                    )
                else:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    c = accumulator.to(c_ptr.dtype.element_ty)
                    tl.store(
                        c_ptr
                        + (M_start_offset + offs_am[:, None]) * N
                        + offs_bn[None, :],
                        c,
                        mask=offs_am[:, None] < m_size and offs_bn[None, :] < n_size,
                    )
                tidx += NUM_SMS

            iterated_tiles += num_tiles


# TODO(shikaili): Too much code duplication. Need to refactor.
@triton.autotune(
    configs=_NV_WS_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={"early_config_prune": early_config_prune},
    restore_value=["c_ptr"],  # restore for scatter_add fusion
)
@triton.jit
def _fbgemm_grouped_gemm_ws(
    a_desc_ptr,
    b_desc_ptr,
    c_ptr,
    workspace,
    scatter_add_indices,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FUSE_SCATTER_ADD: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    USE_TMA_LOAD_ON_SCALES: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
) -> None:
    tl.static_assert(USE_TMA_LOAD, "Always use TMA load with warp specialziation!")
    tl.static_assert(not USE_TMA_LOAD_ON_SCALES, "Not supported!")
    tl.static_assert(
        not (FUSE_SCATTER_ADD and USE_TMA_STORE),
        "Cannot fuse scatter add with TMA store!",
    )

    tidx = tl.program_id(0)

    dtype: tl.dtype = c_ptr.dtype.element_ty
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    M_end_offset = M_end_offset.to(tl.int64)
    iterated_tiles = 0
    iterated_tiles = iterated_tiles.to(tl.int64)
    for g in tl.range(G):
        # Move across groups
        m_size = tl.load(m_sizes + g, cache_modifier=".ca")

        if m_size > 0:
            M_start_offset = M_end_offset
            M_end_offset = M_start_offset + m_size
            N_start_offset = g.to(tl.int64) * N

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            tl.static_assert(N % BLOCK_SIZE_N == 0)
            NUM_N_TILES: tl.constexpr = N // BLOCK_SIZE_N
            num_tiles = num_m_tiles * NUM_N_TILES

            if USE_TMA_STORE:
                with tl.async_task([0]):
                    # pyre-ignore
                    tl.extra.cuda.experimental_device_tensormap_create2d(
                        desc_ptr=c_desc_ptr,
                        global_address=c_ptr + M_start_offset * N,
                        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                        global_size=[m_size, N],
                        element_ty=c_ptr.dtype.element_ty,
                    )
                    # pyre-ignore
                    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            next_iterated_tiles = iterated_tiles + num_tiles
            if (tidx >= iterated_tiles) and (tidx < next_iterated_tiles):
                for i in range(tidx, next_iterated_tiles, NUM_SMS):
                    gidx = i - iterated_tiles
                    # Split M first and N second.
                    tile_m_idx = gidx % num_m_tiles
                    tile_n_idx = gidx // num_m_tiles

                    accumulator = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                    )
                    tl.static_assert(K % BLOCK_SIZE_K == 0)

                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        with tl.async_task([0]):
                            a = tl._experimental_descriptor_load(
                                a_desc_ptr,
                                [m_offset, k_offset],
                                [BLOCK_SIZE_M, BLOCK_SIZE_K],
                                dtype,
                            )
                            b = tl._experimental_descriptor_load(
                                b_desc_ptr,
                                [n_offset, k_offset],
                                [BLOCK_SIZE_N, BLOCK_SIZE_K],
                                dtype,
                            )
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            if USE_FAST_ACCUM:
                                accumulator = tl.dot(a, b.T, accumulator)
                            else:
                                accumulator += tl.dot(a, b.T)

                    if USE_TMA_STORE:
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                            n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                            tl._experimental_descriptor_store(
                                c_desc_ptr,
                                accumulator.to(c_ptr.dtype.element_ty),
                                [m_offset, n_offset],
                            )
                    elif FUSE_SCATTER_ADD:
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(
                                0, BLOCK_SIZE_M
                            )
                            mask = offs_am < m_size
                            m_offsets = tl.load(
                                scatter_add_indices + M_start_offset + offs_am,
                                mask=mask,
                                cache_modifier=".ca",
                            )
                            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(
                                0, BLOCK_SIZE_N
                            )
                            c = accumulator.to(c_ptr.dtype.element_ty)
                            tl.atomic_add(
                                c_ptr + m_offsets[:, None] * N + offs_bn[None, :],
                                c,
                                mask=mask[:, None],
                                sem="relaxed",
                            )
                    else:
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(
                                0, BLOCK_SIZE_M
                            )
                            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(
                                0, BLOCK_SIZE_N
                            )
                            c = accumulator.to(c_ptr.dtype.element_ty)
                            tl.store(
                                c_ptr
                                + (M_start_offset + offs_am[:, None]) * N
                                + offs_bn[None, :],
                                c,
                                mask=offs_am[:, None] < m_size,
                                cache_modifier=".cs",
                            )
                    tidx += NUM_SMS

            iterated_tiles += num_tiles


TT_FP8_DTYPE = tl.float8e4b8 if torch.version.hip else tl.float8e4nv


# TODO(shikaili): clean up redundant 'b_scale_desc_ptr' argument.
@triton.autotune(
    configs=_AMD_CONFIGS if torch.version.hip else _NV_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={
        "early_config_prune": functools.partial(
            early_config_prune, dtype=TT_FP8_DTYPE, dtsize=1
        )
    },
    restore_value=["c_ptr"],  # restore for scatter_add fusion
)
@triton.jit
def _fbgemm_grouped_gemm_fp8_rowwise(
    a_desc_ptr,
    a_scale_ptr,
    b_desc_ptr,
    b_scale_ptr,
    b_scale_desc_ptr,
    c_ptr,
    workspace,
    scatter_add_indices,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FUSE_SCATTER_ADD: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
) -> None:
    tl.static_assert(
        not (FUSE_SCATTER_ADD and USE_TMA_STORE),
        "Cannot fuse scatter add with TMA store!",
    )

    tidx = tl.program_id(0)

    dtype = TT_FP8_DTYPE
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    M_end_offset = M_end_offset.to(tl.int64)
    iterated_tiles = 0
    iterated_tiles = iterated_tiles.to(tl.int64)
    for g in tl.range(G):
        # Move across groups
        m_size = tl.load(m_sizes + g)

        if m_size > 0:
            M_start_offset = M_end_offset
            M_end_offset = M_start_offset + m_size
            N_start_offset = g.to(tl.int64) * N
            n_size = N

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            num_n_tiles = tl.cdiv(n_size, BLOCK_SIZE_N)
            num_tiles = num_m_tiles * num_n_tiles

            if USE_TMA_STORE:
                # pyre-ignore
                tl.extra.cuda.experimental_device_tensormap_create2d(
                    desc_ptr=c_desc_ptr,
                    global_address=c_ptr + M_start_offset * N,
                    load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                    global_size=[m_size, n_size],
                    element_ty=c_ptr.dtype.element_ty,
                )
                # pyre-ignore
                tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            while tidx >= iterated_tiles and tidx < iterated_tiles + num_tiles:
                gidx = tidx - iterated_tiles
                # Split M first and N second.
                tile_m_idx = gidx % num_m_tiles
                tile_n_idx = gidx // num_m_tiles

                accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                tl.static_assert(K % BLOCK_SIZE_K == 0)
                if USE_TMA_LOAD:
                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = tl._experimental_descriptor_load(
                            a_desc_ptr,
                            [m_offset, k_offset],
                            [BLOCK_SIZE_M, BLOCK_SIZE_K],
                            dtype,
                        )
                        b = tl._experimental_descriptor_load(
                            b_desc_ptr,
                            [n_offset, k_offset],
                            [BLOCK_SIZE_N, BLOCK_SIZE_K],
                            dtype,
                        )
                        if USE_FAST_ACCUM:
                            accumulator = tl.dot(a, b.T, accumulator)
                        else:
                            accumulator += tl.dot(a, b.T)
                else:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    offs_k = tl.arange(0, BLOCK_SIZE_K)
                    a_ptrs = (
                        a_desc_ptr
                        + (M_start_offset + offs_am[:, None]) * K
                        + offs_k[None, :]
                    )
                    b_ptrs = (
                        b_desc_ptr
                        + (N_start_offset + offs_bn[:, None]) * K
                        + offs_k[None, :]
                    )
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        a = tl.load(a_ptrs, mask=offs_am[:, None] < m_size)
                        b = tl.load(b_ptrs, mask=offs_bn[:, None] < n_size)
                        accumulator += tl.dot(a, b.T)
                        a_ptrs += BLOCK_SIZE_K
                        b_ptrs += BLOCK_SIZE_K

                offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                a_scale = tl.load(
                    a_scale_ptr + M_start_offset + offs_am[:, None],
                    mask=offs_am[:, None] < m_size,
                )
                b_scale = tl.load(
                    b_scale_ptr + N_start_offset + offs_bn[None, :],
                    mask=offs_bn[None, :] < n_size,
                )
                c = accumulator.to(tl.float32) * a_scale * b_scale

                if USE_TMA_STORE:
                    m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    tl._experimental_descriptor_store(
                        c_desc_ptr,
                        c.to(c_ptr.dtype.element_ty),
                        [m_offset, n_offset],
                    )
                elif FUSE_SCATTER_ADD:
                    offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
                    mask = offs_am < m_size
                    m_offsets = tl.load(
                        scatter_add_indices + M_start_offset + offs_am,
                        mask=mask,
                        cache_modifier=".ca",
                    )
                    offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
                    tl.atomic_add(
                        c_ptr + m_offsets[:, None] * N + offs_bn[None, :],
                        c.to(c_ptr.dtype.element_ty),
                        mask=mask[:, None] and offs_bn[None, :] < n_size,
                        sem="relaxed",
                    )
                else:
                    tl.store(
                        c_ptr
                        + (M_start_offset + offs_am[:, None]) * N
                        + offs_bn[None, :],
                        c,
                        mask=offs_am[:, None] < m_size and offs_bn[None, :] < n_size,
                    )
                tidx += NUM_SMS

            iterated_tiles += num_tiles


# TODO(shikaili): Too much code duplication. Need to refactor.
@triton.autotune(
    configs=_NV_WS_CONFIGS,
    key=["G", "M_BUCKET", "N", "K"],
    prune_configs_by={
        "early_config_prune": functools.partial(
            early_config_prune, dtype=TT_FP8_DTYPE, dtsize=1
        )
    },
    restore_value=["c_ptr"],  # restore for scatter_add fusion
)
@triton.jit
def _fbgemm_grouped_gemm_fp8_rowwise_ws(
    a_desc_ptr,
    a_scale_ptr,
    b_desc_ptr,
    b_scale_ptr,
    b_scale_desc_ptr,
    c_ptr,
    workspace,
    scatter_add_indices,
    m_sizes,
    # problem sizes
    G: tl.constexpr,
    M_BUCKET: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    NUM_SMS: tl.constexpr,
    FUSE_SCATTER_ADD: tl.constexpr,
    USE_TMA_LOAD: tl.constexpr,
    USE_FAST_ACCUM: tl.constexpr,
    # tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    NUM_CONSUMER_GROUPS: tl.constexpr,
    USE_TMA_LOAD_ON_SCALES: tl.constexpr,
    USE_TMA_STORE: tl.constexpr,
) -> None:
    tl.static_assert(USE_TMA_LOAD, "Always use TMA load with warp specialziation!")
    tl.static_assert(
        not (FUSE_SCATTER_ADD and USE_TMA_STORE),
        "Cannot fuse scatter add with TMA store!",
    )

    tidx = tl.program_id(0)

    dtype = TT_FP8_DTYPE
    TMA_SIZE: tl.constexpr = tl.constexpr(128)
    if USE_TMA_STORE:
        c_desc_ptr = workspace + tidx * TMA_SIZE
    else:
        c_desc_ptr = None

    M_end_offset = 0
    M_end_offset = M_end_offset.to(tl.int64)
    iterated_tiles = 0
    iterated_tiles = iterated_tiles.to(tl.int64)
    for g in tl.range(G):
        # Move across groups
        m_size = tl.load(m_sizes + g, cache_modifier=".ca")

        if m_size > 0:
            M_start_offset = M_end_offset
            M_end_offset = M_start_offset + m_size
            N_start_offset = g.to(tl.int64) * N

            num_m_tiles = tl.cdiv(m_size, BLOCK_SIZE_M)
            tl.static_assert(N % BLOCK_SIZE_N == 0)
            NUM_N_TILES: tl.constexpr = N // BLOCK_SIZE_N
            num_tiles = num_m_tiles * NUM_N_TILES

            if USE_TMA_STORE:
                with tl.async_task([0]):
                    # pyre-ignore
                    tl.extra.cuda.experimental_device_tensormap_create2d(
                        desc_ptr=c_desc_ptr,
                        global_address=c_ptr + M_start_offset * N,
                        load_size=[BLOCK_SIZE_M, BLOCK_SIZE_N],
                        global_size=[m_size, N],
                        element_ty=c_ptr.dtype.element_ty,
                    )
                    # pyre-ignore
                    tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

            # Move across tiles
            next_iterated_tiles = iterated_tiles + num_tiles
            if (tidx >= iterated_tiles) and (tidx < next_iterated_tiles):
                for i in range(tidx, next_iterated_tiles, NUM_SMS):
                    gidx = i - iterated_tiles
                    # Split M first and N second.
                    tile_m_idx = gidx % num_m_tiles
                    tile_n_idx = gidx // num_m_tiles

                    accumulator = tl.zeros(
                        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32
                    )
                    tl.static_assert(K % BLOCK_SIZE_K == 0)

                    m_offset = (M_start_offset + tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                    n_offset = (N_start_offset + tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                    for k_offset in range(0, K, BLOCK_SIZE_K):
                        with tl.async_task([0]):
                            a = tl._experimental_descriptor_load(
                                a_desc_ptr,
                                [m_offset, k_offset],
                                [BLOCK_SIZE_M, BLOCK_SIZE_K],
                                dtype,
                            )
                            b = tl._experimental_descriptor_load(
                                b_desc_ptr,
                                [n_offset, k_offset],
                                [BLOCK_SIZE_N, BLOCK_SIZE_K],
                                dtype,
                            )
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            if USE_FAST_ACCUM:
                                accumulator = tl.dot(a, b.T, accumulator)
                            else:
                                accumulator += tl.dot(a, b.T)

                    if USE_TMA_LOAD_ON_SCALES:
                        with tl.async_task([0]):
                            b_scale = tl._experimental_descriptor_load(
                                b_scale_desc_ptr,
                                [n_offset],
                                [BLOCK_SIZE_N],
                                tl.float32,
                            )

                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(
                                0, BLOCK_SIZE_M
                            )
                            a_scale = tl.load(
                                a_scale_ptr + M_start_offset + offs_am[:, None],
                                mask=offs_am[:, None] < m_size,
                                cache_modifier=".ca",
                            )
                            c = accumulator.to(tl.float32) * a_scale * b_scale[None, :]
                    else:
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(
                                0, BLOCK_SIZE_M
                            )
                            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(
                                0, BLOCK_SIZE_N
                            )
                            a_scale = tl.load(
                                a_scale_ptr + M_start_offset + offs_am[:, None],
                                mask=offs_am[:, None] < m_size,
                                cache_modifier=".ca",
                            )
                            b_scale = tl.load(
                                b_scale_ptr + N_start_offset + offs_bn[None, :],
                                cache_modifier=".ca",
                            )
                            c = accumulator.to(tl.float32) * a_scale * b_scale

                    if USE_TMA_STORE:
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            m_offset = (tile_m_idx * BLOCK_SIZE_M).to(tl.int32)
                            n_offset = (tile_n_idx * BLOCK_SIZE_N).to(tl.int32)
                            tl._experimental_descriptor_store(
                                c_desc_ptr,
                                c.to(c_ptr.dtype.element_ty),
                                [m_offset, n_offset],
                            )
                    elif FUSE_SCATTER_ADD:
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(
                                0, BLOCK_SIZE_M
                            )
                            mask = offs_am < m_size
                            m_offsets = tl.load(
                                scatter_add_indices + M_start_offset + offs_am,
                                mask=mask,
                                cache_modifier=".ca",
                            )
                            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(
                                0, BLOCK_SIZE_N
                            )
                            tl.atomic_add(
                                c_ptr + m_offsets[:, None] * N + offs_bn[None, :],
                                c,
                                mask=mask[:, None],
                                sem="relaxed",
                            )
                    else:
                        with tl.async_task([1, NUM_CONSUMER_GROUPS]):
                            offs_am = tile_m_idx * BLOCK_SIZE_M + tl.arange(
                                0, BLOCK_SIZE_M
                            )
                            offs_bn = tile_n_idx * BLOCK_SIZE_N + tl.arange(
                                0, BLOCK_SIZE_N
                            )
                            tl.store(
                                c_ptr
                                + (M_start_offset + offs_am[:, None]) * N
                                + offs_bn[None, :],
                                c,
                                mask=offs_am[:, None] < m_size,
                                cache_modifier=".cs",
                            )
                    tidx += NUM_SMS

            iterated_tiles += num_tiles


warnings.simplefilter("once")


def _grouped_gemm(
    *,
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    x_scale: Optional[torch.Tensor],
    w_scale: Optional[torch.Tensor],
    use_fast_accum: bool,
    use_warp_specialization: bool,
    output_tensor: Optional[torch.Tensor],
    scatter_add_indices: Optional[torch.Tensor],
) -> torch.Tensor:

    USE_TMA_LOAD = not torch.version.hip
    USE_TMA_STORE = False

    if USE_TMA_LOAD and not HAS_TMA_DESC:
        USE_TMA_LOAD = False
        warnings.warn("TMA load is disabled as there is no TMA descriptor support!")

    if USE_TMA_STORE and not HAS_TMA_DESC:
        USE_TMA_STORE = False
        warnings.warn("TMA store is disabled as there is no TMA descriptor support!")

    # TODO(shikaili): Check the readniess of WS on ROCm side in Meta's Triton.
    if use_warp_specialization and torch.version.hip:
        warnings.warn("Warp specialization is disabled as it is not supported on ROCm.")
        use_warp_specialization = False

    if use_warp_specialization and not _HAS_WS_SUPPORT:
        warnings.warn(
            "Warp specialization is disabled as the Triton build in current environment doesn't have such support. Please build from https://github.com/facebookexperimental/triton/tree/ws-3.2.x to enable it for best performance on Nvidia's SM90 GPUs."
        )
        use_warp_specialization = False

    if use_warp_specialization:
        assert HAS_TMA_DESC
        USE_TMA_STORE = True  # Tuning decision

    G = m_sizes.shape[0]

    assert x.is_contiguous()
    assert w.is_contiguous()
    assert m_sizes.is_contiguous()

    M, K = x.shape
    N = w.shape[0] // G
    assert K == w.shape[1]

    if output_tensor is None:
        FUSE_SCATTER_ADD = False
        assert scatter_add_indices is None
        y = torch.empty((M, N), device=x.device, dtype=torch.bfloat16)
    else:
        FUSE_SCATTER_ADD = True
        assert scatter_add_indices is not None
        assert scatter_add_indices.is_contiguous()
        assert scatter_add_indices.shape == (M,)
        y = output_tensor
    if M == 0 or N == 0:
        return y

    NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count

    desc_helper = None
    desc_x = x
    desc_w = w
    desc_ws = w_scale
    workspace = None

    if USE_TMA_LOAD:
        desc_helper = TmaAutoTuneHelper()
        desc_helper.init_tma_descriptor("x")
        desc_helper.init_tma_descriptor("w")
        desc_x = desc_helper.get_tma_descriptor_kernel_param("x")
        desc_w = desc_helper.get_tma_descriptor_kernel_param("w")
        if use_warp_specialization and w_scale is not None:
            desc_helper.init_tma_descriptor("ws")
            desc_ws = desc_helper.get_tma_descriptor_kernel_param("ws")

    if USE_TMA_STORE:
        workspace = torch.empty(
            NUM_SMS * TmaAutoTuneHelper.TMA_SIZE,
            device=x.device,
            dtype=torch.uint8,
        )

    def grid(META):
        if USE_TMA_LOAD:
            nonlocal desc_helper  # noqa: F824
            desc_helper.fill_2d_tma_descriptor(
                "x",
                x.data_ptr(),
                M,
                K,
                META["BLOCK_SIZE_M"] // META["NUM_CONSUMER_GROUPS"],
                META["BLOCK_SIZE_K"],
                x.element_size(),
            )

            desc_helper.fill_2d_tma_descriptor(
                "w",
                w.data_ptr(),
                N * G,
                K,
                META["BLOCK_SIZE_N"],
                META["BLOCK_SIZE_K"],
                w.element_size(),
            )

            if META.get("USE_TMA_LOAD_ON_SCALES", False):
                desc_helper.fill_1d_tma_descriptor(
                    "ws",
                    w_scale.data_ptr(),
                    N * G,
                    META["BLOCK_SIZE_N"],
                    w_scale.element_size(),
                )

        return (NUM_SMS,)

    M_BUCKET_CAP = 16384
    M_BUCKET = min(triton.next_power_of_2(M), M_BUCKET_CAP)
    if x_scale is not None and w_scale is not None:
        assert x_scale.is_contiguous()
        assert w_scale.is_contiguous()
        fn = (
            _fbgemm_grouped_gemm_fp8_rowwise_ws
            if use_warp_specialization
            else _fbgemm_grouped_gemm_fp8_rowwise
        )
        args = (
            desc_x,
            x_scale,
            desc_w,
            w_scale,
            desc_ws,
            y,
            workspace,
            scatter_add_indices,
            m_sizes,
            G,
            M_BUCKET,
            N,
            K,
            NUM_SMS,
            FUSE_SCATTER_ADD,
            USE_TMA_LOAD,
        )
        if use_warp_specialization:
            args += (use_fast_accum,)
        else:
            args += (USE_TMA_STORE, use_fast_accum)
        fn[grid](*args)
    else:
        assert x_scale is None
        assert w_scale is None
        fn = (
            _fbgemm_grouped_gemm_ws if use_warp_specialization else _fbgemm_grouped_gemm
        )
        args = (
            desc_x,
            desc_w,
            y,
            workspace,
            scatter_add_indices,
            m_sizes,
            G,
            M_BUCKET,
            N,
            K,
            NUM_SMS,
            FUSE_SCATTER_ADD,
            USE_TMA_LOAD,
        )
        if use_warp_specialization:
            args += (use_fast_accum,)
        else:
            args += (USE_TMA_STORE, use_fast_accum)
        fn[grid](*args)

    return y


def grouped_gemm(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    use_fast_accum: bool = True,
    *,
    _use_warp_specialization: bool = True,
    _output_tensor: Optional[torch.Tensor] = None,
    _scatter_add_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _grouped_gemm(
        x=x,
        w=w,
        m_sizes=m_sizes,
        x_scale=None,
        w_scale=None,
        use_fast_accum=use_fast_accum,
        use_warp_specialization=_use_warp_specialization,
        output_tensor=_output_tensor,
        scatter_add_indices=_scatter_add_indices,
    )


def grouped_gemm_fp8_rowwise(
    x: torch.Tensor,
    w: torch.Tensor,
    m_sizes: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    use_fast_accum: bool = True,
    *,
    _use_warp_specialization: bool = True,
    _output_tensor: Optional[torch.Tensor] = None,
    _scatter_add_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _grouped_gemm(
        x=x,
        w=w,
        m_sizes=m_sizes,
        x_scale=x_scale,
        w_scale=w_scale,
        use_fast_accum=use_fast_accum,
        use_warp_specialization=_use_warp_specialization,
        output_tensor=_output_tensor,
        scatter_add_indices=_scatter_add_indices,
    )
