# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
"""Host-side policy and launch state for the SM120 forward kernel.

The generic FA4 interface owns argument normalization, compilation, and
architecture dispatch. This module owns the SM120-specific decisions that
must remain consistent across those phases:

* tile, stage, and warp configuration;
* SplitKV sizing for paged decode;
* single-QK versus N-distributed-QK decode specialization;
* the direct uniform-batch decode specialization;
* cached TVM-FFI launch plans and their temporary workspaces.
"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import cutlass.utils as utils_basic
import torch
from cutlass import Int32

from sglang.kernels.ops.attention.fa4_sm120.flash_fwd import (
    FlashAttentionForwardSm120,
)
from sglang.kernels.ops.attention.fa4_sm120.policy import (
    LOW_HD_DECODE_LONG_MIN_TILES_PER_CTA,
    LOW_HD_DECODE_MAX_SPLITS,
    LOW_HD_DECODE_SHAPES,
    is_low_hd_paged_decode_tile,
    visible_decode_seqlen_k,
)
from sglang.kernels.ops.attention.flash_attn.cute import fa_logging
from sglang.kernels.ops.attention.flash_attn.cute.utils import AuxData

_LAUNCH_PLAN_CAPACITY = 4096
_WORKSPACE_CAPACITY = 64
_WORKSPACE_VIEW_CAPACITY = 512
_EMPTY_AUX_DATA = AuxData(None, None)
_DECODE_REFERENCE_CORE_GHZ = 2.4
_DECODE_POWER_OF_TWO_SPLITS = (1, 2, 4, 8, 16, 32, 64, 128)

# Joint fit of interleaved NCU duration curves from the 110-SM/384-bit and
# 188-SM/512-bit SM120 SKUs. The inputs below are physical quantities rather
# than product names or sequence-length thresholds. This calibration applies
# only to the qualified HD256 M16N64 paged-decode gather kernel.
_DECODE_LATENCY_FIXED_REF_US = 4.31046205978645
_DECODE_LATENCY_PER_KV_TILE_REF_US = 1.3115656095100847
_DECODE_MEMORY_FIXED_REF_US = 4.532105547934724
_DECODE_MEMORY_SOL = 0.9415559971890943
_DECODE_UNDERFILL_TAU_CTAS_PER_CHANNEL = 0.741265436766481
_DECODE_COMBINE_FIXED_US = 3.654860520719416
_DECODE_COMBINE_SPLIT_TO_8_US = 0.015181182195831644
_DECODE_COMBINE_SPLIT_ABOVE_8_US = 0.12157753908563328
_DECODE_COMBINE_OUTPUT_CTA_US = 0.019185022429169515


@dataclass(frozen=True)
class Sm120ForwardConfig:
    tile_m: int
    tile_n: int
    num_stages: int
    num_threads: int

    @property
    def compile_key(self) -> tuple:
        """Return SM120 configuration state baked into generated code."""
        return (
            self.tile_m,
            self.tile_n,
            self.num_stages,
            self.num_threads,
        )


@dataclass(frozen=True)
class Sm120ForwardPlan:
    """Compile- and launch-time decisions for one normalized forward call."""

    num_splits: int
    split_num_n_blocks: int
    direct_uniform_batch: bool
    split_qk_n: bool
    split_kv_blocks_per_cta: int
    transpose_qk_pv: bool
    launch_split_combine_early: bool

    @property
    def compile_key(self) -> tuple:
        """Return only specialization state baked into the generated kernel."""
        return (
            self.direct_uniform_batch,
            self.split_qk_n,
            self.split_kv_blocks_per_cta,
            self.transpose_qk_pv,
        )


@dataclass(frozen=True)
class _VarlenLaunchPlan:
    compiled_fn: Callable
    compile_key: tuple


@dataclass(frozen=True)
class _PagedDecodeLaunchPlan:
    compiled_fn: Callable
    compile_key: tuple
    actual_num_splits: int
    compiled_combine: Optional[Callable]
    partial_dtype: torch.dtype


@dataclass(frozen=True)
class _DecodeHardware:
    num_sms: int
    memory_channels: int
    peak_memory_gbps: float
    core_clock_ghz: float
    is_integrated: bool


@lru_cache(maxsize=None)
def _get_decode_hardware(device: torch.device) -> _DecodeHardware:
    properties = torch.cuda.get_device_properties(device)
    memory_bus_width = properties.memory_bus_width
    is_integrated = bool(getattr(properties, "is_integrated", False))
    # Discrete GPUs report the physical memory clock, while CUDA reports the
    # effective LPDDR data rate on integrated devices. Convert kHz and bus bits
    # to GB/s without doubling the already-effective integrated rate.
    data_rate_multiplier = 1 if is_integrated else 2
    peak_memory_gbps = (
        properties.memory_clock_rate
        * memory_bus_width
        * data_rate_multiplier
        / 8_000_000
    )
    return _DecodeHardware(
        num_sms=properties.multi_processor_count,
        memory_channels=max(1, memory_bus_width // 32),
        peak_memory_gbps=peak_memory_gbps,
        core_clock_ghz=properties.clock_rate / 1_000_000,
        is_integrated=is_integrated,
    )


def _normalize_num_splits(num_splits: int, num_n_blocks: int) -> int:
    if num_splits <= 1 or num_n_blocks <= 1:
        return 1
    requested_splits = min(num_splits, num_n_blocks)
    blocks_per_split = (num_n_blocks + requested_splits - 1) // requested_splits
    return (num_n_blocks + blocks_per_split - 1) // blocks_per_split


def _predict_decode_split_us(
    *,
    num_splits: int,
    num_n_blocks: int,
    total_mblocks: int,
    packed_q_rows: int,
    tile_m: int,
    tile_n: int,
    head_dim: int,
    head_dim_v: int,
    element_size: int,
    hardware: _DecodeHardware,
) -> float:
    num_m_blocks = (packed_q_rows + tile_m - 1) // tile_m
    batch_head_groups = total_mblocks // num_m_blocks
    main_ctas = total_mblocks * num_splits
    kv_tiles_per_cta = (num_n_blocks + num_splits - 1) // num_splits
    clock_scale = _DECODE_REFERENCE_CORE_GHZ / hardware.core_clock_ghz
    latency_main_us = (
        _DECODE_LATENCY_FIXED_REF_US
        + _DECODE_LATENCY_PER_KV_TILE_REF_US * kv_tiles_per_cta
    ) * clock_scale
    ctas_per_channel = main_ctas / hardware.memory_channels
    memory_fill = max(
        1e-6,
        1.0 - math.exp(-ctas_per_channel / _DECODE_UNDERFILL_TAU_CTAS_PER_CHANNEL),
    )
    logical_kv_bytes = (
        batch_head_groups
        * num_n_blocks
        * tile_n
        * (head_dim + head_dim_v)
        * element_size
    )
    theoretical_transfer_us = logical_kv_bytes / hardware.peak_memory_gbps / 1000.0
    memory_main_us = (
        _DECODE_MEMORY_FIXED_REF_US * clock_scale
        + theoretical_transfer_us / _DECODE_MEMORY_SOL / memory_fill
    )
    main_us = max(latency_main_us, memory_main_us)
    if num_splits == 1:
        return main_us
    output_rows = batch_head_groups * packed_q_rows
    combine_ctas = ((output_rows + 7) // 8) * ((head_dim_v + 127) // 128)
    combine_us = (
        _DECODE_COMBINE_FIXED_US
        + _DECODE_COMBINE_SPLIT_TO_8_US * min(num_splits, 8)
        + _DECODE_COMBINE_SPLIT_ABOVE_8_US * max(num_splits - 8, 0)
        + _DECODE_COMBINE_OUTPUT_CTA_US * combine_ctas
    )
    return main_us + combine_us


def _select_decode_num_splits(
    *,
    head_dim: int,
    head_dim_v: int,
    element_size: int,
    packed_q_rows: int,
    tile_m: int,
    tile_n: int,
    total_mblocks: int,
    num_n_blocks: int,
    hardware: _DecodeHardware,
) -> int:
    # At four or fewer N tiles, SplitKV's fixed combine launch costs more than
    # the remaining serialized work. N=5 is the first measured crossover.
    if num_n_blocks <= 4:
        return 1

    max_one_wave_splits = min(
        128,
        num_n_blocks,
        max(1, hardware.num_sms // total_mblocks),
    )
    requested_candidates = (
        *_DECODE_POWER_OF_TWO_SPLITS,
        max_one_wave_splits,
    )
    candidates = {
        _normalize_num_splits(requested, num_n_blocks)
        for requested in requested_candidates
    }
    candidates = {
        splits
        for splits in candidates
        if splits == 1 or total_mblocks * splits <= hardware.num_sms
    }
    if hardware.is_integrated:
        if total_mblocks >= 3 * hardware.memory_channels:
            # Three unsplit CTAs per LPDDR channel already saturate the LPDDR path.
            return 1
        if num_n_blocks >= 64:
            if total_mblocks >= 2 * hardware.memory_channels:
                return 1
            target_main_ctas = (2 * hardware.num_sms + 2) // 3
            if total_mblocks == hardware.memory_channels:
                # One base CTA per channel needs a second wave to cover UMA
                # latency; retain the next balanced power-of-two split.
                max_integrated_splits = 1 << (
                    (hardware.num_sms // total_mblocks - 1).bit_length()
                )
                candidates.add(
                    _normalize_num_splits(max_integrated_splits, num_n_blocks)
                )
            elif total_mblocks < hardware.memory_channels:
                max_integrated_splits = max(
                    1,
                    (target_main_ctas + total_mblocks - 1) // total_mblocks,
                )
            else:
                max_integrated_splits = None
            if max_integrated_splits is not None:
                bounded_candidates = {
                    splits for splits in candidates if splits <= max_integrated_splits
                }
                if bounded_candidates:
                    candidates = bounded_candidates
    # A 160-thread HD256 CTA has enough independent QK/PV work that roughly
    # two thirds of an SM wave saturates both qualified SM120 SKUs. The fitted
    # latency/bandwidth model is useful above that floor, but underestimates
    # the cost of sparse grids at larger batch/head-group counts.
    target_main_ctas = (2 * hardware.num_sms + 2) // 3
    fill_splits = min(
        max_one_wave_splits,
        max(1, (target_main_ctas + total_mblocks - 1) // total_mblocks),
    )
    min_fill_splits = _normalize_num_splits(fill_splits, num_n_blocks)
    filled_candidates = {splits for splits in candidates if splits >= min_fill_splits}
    if filled_candidates:
        candidates = filled_candidates
    return min(
        candidates,
        key=lambda splits: _predict_decode_split_us(
            num_splits=splits,
            num_n_blocks=num_n_blocks,
            total_mblocks=total_mblocks,
            packed_q_rows=packed_q_rows,
            tile_m=tile_m,
            tile_n=tile_n,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            element_size=element_size,
            hardware=hardware,
        ),
    )


def _select_low_hd_decode_num_splits(
    *,
    max_head_dim: int,
    packed_q_rows: int,
    total_mblocks: int,
    num_n_blocks: int,
    hardware: _DecodeHardware,
) -> int:
    """Overschedule the short-M low-HD decode kernel by bounded waves.

    One hardware wave leaves each CTA with too long a serial K loop for this
    one-warp kernel. For short KV, use the fewest splits that approximately
    fill one device-specific SM wave; more waves only add combine work. For
    longer KV, bound the launch by hardware waves and a minimum per-CTA KV
    grain. Once eight-way splitting leaves a long grain, prefer balanced
    power-of-two partitions over device-specific odd split counts.
    """
    if num_n_blocks <= 4 or total_mblocks <= 0:
        return 1
    num_sms = hardware.num_sms
    if total_mblocks >= num_sms and (num_n_blocks <= 8 or num_n_blocks >= 64):
        return 1
    if hardware.is_integrated and total_mblocks <= hardware.memory_channels // 2:
        # Small batches need enough independent CTAs to cover UMA latency.
        # Two thirds of the compact SM array is sufficient; larger grids only
        # add combine work.
        target_main_ctas = (2 * num_sms + 2) // 3
        fill_splits = min(
            32,
            num_n_blocks,
            max(
                1,
                (target_main_ctas + total_mblocks - 1) // total_mblocks,
            ),
        )
        return _normalize_num_splits(fill_splits, num_n_blocks)
    wave_split_budget = max(1, (3 * num_sms) // total_mblocks)
    if num_n_blocks >= LOW_HD_DECODE_MAX_SPLITS * LOW_HD_DECODE_LONG_MIN_TILES_PER_CTA:
        # Long KV can afford the next power-of-two split without making the
        # per-CTA grain too small. Balanced partitions avoid partial waves and
        # sustain memory bandwidth better than device-specific odd counts.
        wave_split_budget = 1 << (wave_split_budget - 1).bit_length()
    if num_n_blocks <= 8:
        one_wave_splits = max(1, (num_sms + total_mblocks - 1) // total_mblocks)
        if hardware.is_integrated:
            one_wave_splits = 1 << (one_wave_splits - 1).bit_length()
        kv_grain_splits = min(LOW_HD_DECODE_MAX_SPLITS, one_wave_splits)
    else:
        blocks_per_cta_floor = (
            8
            if (
                max_head_dim >= 128
                and (hardware.is_integrated or 2 <= packed_q_rows <= 4)
            )
            else 4
        )
        kv_grain_splits = min(
            LOW_HD_DECODE_MAX_SPLITS,
            num_n_blocks // blocks_per_cta_floor,
        )
    return _normalize_num_splits(
        min(wave_split_budget, kv_grain_splits),
        num_n_blocks,
    )


def _tensor_signature(tensor: torch.Tensor) -> tuple:
    return (
        type(tensor),
        tensor.device,
        tensor.dtype,
        tensor.shape,
        tensor.stride(),
        tensor.requires_grad,
    )


def _resolve_causal_local_window(
    causal: bool,
    window_size_left: Optional[int],
    window_size_right: Optional[int],
) -> tuple[bool, Optional[int], Optional[int]]:
    if causal:
        window_size_right = 0
    if (
        window_size_left is not None
        and window_size_right is not None
        and window_size_left + window_size_right < 0
    ):
        window_size_left = None
        window_size_right = None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal = True
            window_size_right = None
        else:
            causal = False
    return causal, window_size_left, window_size_right


class Sm120ForwardPolicy:
    """Pure SM120 configuration, scheduling, and kernel-selection policy."""

    @staticmethod
    def supports_arch(arch: int) -> bool:
        return arch // 10 == 12

    @staticmethod
    def use_graph_capture_split_combine_pdl(
        *,
        is_stream_capturing: bool,
        is_split_kv: bool,
    ) -> bool:
        """Overlap SplitKV combine setup only when both grids are captured."""
        return is_stream_capturing and is_split_kv

    @staticmethod
    @lru_cache(maxsize=1)
    def implementation_token() -> tuple:
        from sglang.kernels.ops.attention.fa4_sm120.flash_fwd_decode import (
            FlashAttentionForwardSm120DecodeTranspose,
        )

        return (
            FlashAttentionForwardSm120,
            FlashAttentionForwardSm120.get_fwd_tile_size,
            Sm120ForwardHost.resolve_plan,
            Sm120ForwardHost.select_num_splits,
            Sm120ForwardHost.select_paged_decode_split_kv_blocks_per_cta,
            Sm120ForwardHost.use_paged_decode_transpose_qk_pv,
            Sm120ForwardHost.make_kernel,
            FlashAttentionForwardSm120DecodeTranspose,
            FlashAttentionForwardSm120DecodeTranspose.paged_tma,
            FlashAttentionForwardSm120DecodeTranspose.query_in_regs,
        )

    @classmethod
    def resolve_plan(
        cls,
        *,
        requested_num_splits: int,
        generic_num_n_blocks: int,
        head_dim: int,
        head_dim_v: int,
        batch_size: int,
        num_head_kv: int,
        paged_kv: bool,
        page_size: Optional[int],
        k: Optional[torch.Tensor],
        v: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        pack_gqa: bool,
        element_size: int,
        packed_q_rows: int,
        tile_m: int,
        tile_n: int,
        num_m_blocks: int,
        total_mblocks: int,
        num_sms: int,
        total_q: int,
        has_cu_seqlens_q: bool,
        has_seqused_q: bool,
        has_seqused_k: bool,
        is_causal: bool,
        is_local: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        has_score_or_mask_mod: bool,
        is_stream_capturing: bool,
        device: torch.device,
        fake_mode: bool,
        generic_heuristic: Callable[[int, int, int, int], int],
    ) -> Sm120ForwardPlan:
        """Resolve all SM120 dataflow decisions behind one host-side boundary."""
        has_compact_q_groups = pack_gqa or packed_q_rows == max_seqlen_q
        split_num_n_blocks = cls.split_num_n_blocks(
            generic_num_n_blocks=generic_num_n_blocks,
            paged_kv=paged_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            is_local=is_local,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            tile_n=tile_n,
        )
        num_splits = cls.select_num_splits(
            requested_num_splits=requested_num_splits,
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            paged_kv=paged_kv,
            max_seqlen_q=max_seqlen_q,
            has_compact_q_groups=has_compact_q_groups,
            element_size=element_size,
            packed_q_rows=packed_q_rows,
            tile_m=tile_m,
            tile_n=tile_n,
            total_mblocks=total_mblocks,
            num_sms=num_sms,
            num_n_blocks=split_num_n_blocks,
            device=device,
            fake_mode=fake_mode,
            generic_heuristic=generic_heuristic,
        )
        is_split_kv = num_splits > 1
        direct_uniform_batch = cls.use_direct_uniform_batch(
            batch_size=batch_size,
            paged_kv=paged_kv,
            has_cu_seqlens_q=has_cu_seqlens_q,
            has_seqused_q=has_seqused_q,
            total_q=total_q,
            max_seqlen_q=max_seqlen_q,
            num_m_blocks=num_m_blocks,
        )
        split_qk_n = cls.use_paged_decode_split_qk_n(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            paged_kv=paged_kv,
            max_seqlen_q=max_seqlen_q,
            has_compact_q_groups=has_compact_q_groups,
            packed_q_rows=packed_q_rows,
            tile_m=tile_m,
            tile_n=tile_n,
            num_n_blocks=generic_num_n_blocks,
            is_causal=is_causal,
            is_local=is_local,
            has_score_or_mask_mod=has_score_or_mask_mod,
        )
        dense_paged_kv = (
            page_size is not None
            and k is not None
            and k.stride(-1) == 1
            and v.stride(-1) == 1
            and k.stride(-2) == head_dim
            and v.stride(-2) == head_dim_v
            and k.stride(1) == num_head_kv * head_dim
            and v.stride(1) == num_head_kv * head_dim_v
            and k.stride(0) == page_size * num_head_kv * head_dim
            and v.stride(0) == page_size * num_head_kv * head_dim_v
        )
        transpose_qk_pv = cls.use_paged_decode_transpose_qk_pv(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            batch_size=batch_size,
            num_head_kv=num_head_kv,
            paged_kv=paged_kv,
            page_size=page_size,
            dense_paged_kv=dense_paged_kv,
            max_seqlen_q=max_seqlen_q,
            pack_gqa=pack_gqa,
            packed_q_rows=packed_q_rows,
            tile_m=tile_m,
            tile_n=tile_n,
            num_splits=num_splits,
            num_n_blocks=split_num_n_blocks,
            is_causal=is_causal,
            is_local=is_local,
            has_score_or_mask_mod=has_score_or_mask_mod,
        )
        if transpose_qk_pv:
            split_qk_n = False
        split_kv_blocks_per_cta = cls.select_paged_decode_split_kv_blocks_per_cta(
            head_dim=head_dim,
            head_dim_v=head_dim_v,
            batch_size=batch_size,
            paged_kv=paged_kv,
            max_seqlen_q=max_seqlen_q,
            has_compact_q_groups=has_compact_q_groups,
            packed_q_rows=packed_q_rows,
            is_split_kv=is_split_kv,
            direct_uniform_batch=direct_uniform_batch,
            has_seqused_k=has_seqused_k,
            split_qk_n=split_qk_n,
            num_splits=num_splits,
            num_n_blocks=split_num_n_blocks,
        )
        return Sm120ForwardPlan(
            num_splits=num_splits,
            split_num_n_blocks=split_num_n_blocks,
            direct_uniform_batch=direct_uniform_batch,
            split_qk_n=split_qk_n,
            split_kv_blocks_per_cta=split_kv_blocks_per_cta,
            transpose_qk_pv=transpose_qk_pv,
            launch_split_combine_early=cls.use_graph_capture_split_combine_pdl(
                is_stream_capturing=is_stream_capturing,
                is_split_kv=is_split_kv,
            ),
        )

    @staticmethod
    def compile_arguments(plan: Sm120ForwardPlan) -> tuple:
        """Extra CuTe arguments required by the selected SM120 kernel ABI."""
        return (Int32(0),)

    @staticmethod
    def runtime_arguments(plan: Sm120ForwardPlan) -> tuple:
        """Extra runtime arguments required by the selected SM120 kernel ABI."""
        return (int(plan.launch_split_combine_early),)

    @staticmethod
    def select_config(
        *,
        head_dim: int,
        head_dim_v: int,
        tile_mn: Optional[tuple[int, int]],
        has_bias: bool,
        total_q_rows: int,
        num_sms: Optional[int],
        num_batch: int,
        seqlen_q: Optional[int],
        seqlen_k: Optional[int],
        num_head_kv: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        is_local: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        pack_gqa: bool,
        paged_kv: bool,
    ) -> Sm120ForwardConfig:
        if has_bias:
            if max(head_dim, head_dim_v) > 128:
                raise ValueError(
                    "SM120 relative bias currently supports head_dim and "
                    "head_dim_v up to 128"
                )
            if tile_mn is not None and tile_mn != (64, 128):
                raise ValueError("SM120 relative bias requires tile_mn=(64, 128)")
            tile_m, tile_n = 64, 128
        elif tile_mn is None:
            tile_m, tile_n = FlashAttentionForwardSm120.get_fwd_tile_size(
                head_dim,
                head_dim_v,
                total_q_rows=total_q_rows,
                num_sms=num_sms,
                num_batch=num_batch,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                num_head_kv=num_head_kv,
                qhead_per_kvhead=qhead_per_kvhead,
                is_causal=is_causal,
                is_local=is_local,
                window_size_left=window_size_left,
                window_size_right=window_size_right,
                pack_gqa=pack_gqa,
                paged_kv=paged_kv,
            )
        else:
            tile_m, tile_n = tile_mn
        return Sm120ForwardConfig(
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=FlashAttentionForwardSm120.get_fwd_num_stages(
                head_dim, head_dim_v, tile_m, tile_n
            ),
            num_threads=FlashAttentionForwardSm120.get_fwd_num_threads(
                head_dim,
                head_dim_v,
                tile_m,
                tile_n,
                paged_kv=paged_kv,
            ),
        )

    @staticmethod
    def select_num_splits(
        *,
        requested_num_splits: int,
        head_dim: int,
        head_dim_v: int,
        paged_kv: bool,
        max_seqlen_q: int,
        has_compact_q_groups: bool,
        element_size: int,
        packed_q_rows: int,
        tile_m: int,
        tile_n: int,
        total_mblocks: int,
        num_sms: int,
        num_n_blocks: int,
        device: torch.device,
        fake_mode: bool,
        generic_heuristic: Callable[[int, int, int, int], int],
    ) -> int:
        num_splits = requested_num_splits
        if num_splits < 1:
            is_hd256_paged_decode = (
                head_dim == 256
                and head_dim_v == 256
                and paged_kv
                and max_seqlen_q == 1
                and has_compact_q_groups
            )
            is_low_hd_paged_decode = is_low_hd_paged_decode_tile(
                head_dim=head_dim,
                head_dim_v=head_dim_v,
                paged_kv=paged_kv,
                seqlen_q=max_seqlen_q,
                tile_m=tile_m,
                tile_n=tile_n,
            )
            if is_low_hd_paged_decode and tile_m == 16:
                num_splits = _select_low_hd_decode_num_splits(
                    max_head_dim=max(head_dim, head_dim_v),
                    packed_q_rows=packed_q_rows,
                    total_mblocks=total_mblocks,
                    num_n_blocks=num_n_blocks,
                    hardware=_get_decode_hardware(device),
                )
            elif is_hd256_paged_decode:
                if fake_mode:
                    max_splits = min(128, 1 << (num_sms.bit_length() - 1))
                    num_splits = generic_heuristic(
                        total_mblocks,
                        num_sms,
                        num_n_blocks,
                        max_splits,
                    )
                else:
                    num_splits = _select_decode_num_splits(
                        head_dim=head_dim,
                        head_dim_v=head_dim_v,
                        element_size=element_size,
                        packed_q_rows=packed_q_rows,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        total_mblocks=total_mblocks,
                        num_n_blocks=num_n_blocks,
                        hardware=_get_decode_hardware(device),
                    )
            else:
                num_splits = generic_heuristic(
                    total_mblocks,
                    num_sms,
                    num_n_blocks,
                    128,
                )

        if num_splits <= 1 or num_n_blocks == 0:
            return 1

        # BlockInfo assigns a uniform ceil-div chunk to every split. Normalize
        # the count so no SM120 CTA owns an empty tail chunk.
        return _normalize_num_splits(num_splits, num_n_blocks)

    @staticmethod
    def split_num_n_blocks(
        *,
        generic_num_n_blocks: int,
        paged_kv: bool,
        max_seqlen_q: int,
        max_seqlen_k: int,
        is_local: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        tile_n: int,
    ) -> int:
        """Return the KV-block count used by the decode SplitKV policy.

        The generic local-attention bound includes a full query tile because
        prefill tiles can span multiple query positions. Packed decode has one
        query position regardless of how many GQA heads occupy the M tile, so
        its exact visible span is just left + current + right.
        """
        if not (paged_kv and max_seqlen_q == 1 and is_local):
            return generic_num_n_blocks
        visible_seqlen_k = visible_decode_seqlen_k(
            max_seqlen_k,
            is_local=True,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
        )
        return max(0, (visible_seqlen_k + tile_n - 1) // tile_n)

    @staticmethod
    def use_direct_uniform_batch(
        *,
        batch_size: int,
        paged_kv: bool,
        has_cu_seqlens_q: bool,
        has_seqused_q: bool,
        total_q: int,
        max_seqlen_q: int,
        num_m_blocks: int,
    ) -> bool:
        """Use arithmetic scheduling for one uniform query row per request.

        This invariant is independent of whether GQA heads are folded into the
        query-row mode.  In particular, unpacked MHA should not pay the generic
        varlen scheduler's prefix-sum walk merely to preserve its ordinary Q
        loader.
        """
        return (
            paged_kv
            and has_cu_seqlens_q
            and not has_seqused_q
            and max_seqlen_q == 1
            and total_q == batch_size * max_seqlen_q
            and num_m_blocks == 1
        )

    @staticmethod
    def use_paged_decode_split_qk_n(
        *,
        head_dim: int,
        head_dim_v: int,
        paged_kv: bool,
        max_seqlen_q: int,
        has_compact_q_groups: bool,
        packed_q_rows: int,
        tile_m: int,
        tile_n: int,
        num_n_blocks: int,
        is_causal: bool,
        is_local: bool,
        has_score_or_mask_mod: bool,
    ) -> bool:
        """Select four-way N-distributed QK for qualified SM120 decode.

        N-distribution removes the single-QK-warp critical path, lowers the
        register footprint, and is faster from the first K/V tile on both
        qualified SM120 SKUs. Both structures use the same threads and
        one-CTA-per-SM shared-memory residency.
        """
        return (
            (head_dim, head_dim_v) == (256, 256)
            and paged_kv
            and max_seqlen_q == 1
            and has_compact_q_groups
            and packed_q_rows <= 16
            and (tile_m, tile_n) == (16, 64)
            and num_n_blocks >= 1
            and (is_causal or is_local)
            and not has_score_or_mask_mod
        )

    @staticmethod
    def select_paged_decode_split_kv_blocks_per_cta(
        *,
        head_dim: int,
        head_dim_v: int,
        batch_size: int,
        paged_kv: bool,
        max_seqlen_q: int,
        has_compact_q_groups: bool,
        packed_q_rows: int,
        is_split_kv: bool,
        direct_uniform_batch: bool,
        has_seqused_k: bool,
        split_qk_n: bool,
        num_splits: int,
        num_n_blocks: int,
    ) -> int:
        """Preserve the longest request's split grain for ragged decode."""
        if not (
            (head_dim, head_dim_v) == (256, 256)
            and batch_size > 1
            and paged_kv
            and max_seqlen_q == 1
            and has_compact_q_groups
            and packed_q_rows <= 16
            and is_split_kv
            and direct_uniform_batch
            and has_seqused_k
            and split_qk_n
            and num_splits > 1
            and num_n_blocks > 0
        ):
            return 0
        return (num_n_blocks + num_splits - 1) // num_splits

    @staticmethod
    def use_paged_decode_transpose_qk_pv(
        *,
        head_dim: int,
        head_dim_v: int,
        batch_size: int,
        num_head_kv: int,
        paged_kv: bool,
        page_size: Optional[int],
        dense_paged_kv: bool,
        max_seqlen_q: int,
        pack_gqa: bool,
        packed_q_rows: int,
        tile_m: int,
        tile_n: int,
        num_splits: int,
        num_n_blocks: int,
        is_causal: bool,
        is_local: bool,
        has_score_or_mask_mod: bool,
    ) -> bool:
        """Select the qualified page-TMA M64N8 decode dataflow.

        Full QK+PV transpose removes M16 padding for two through eight packed
        query rows. Interleaved NCU on the 110-SM/384-bit and 188-SM/512-bit
        SM120 SKUs qualifies up to eight batch/head groups. One or two groups
        amortize the transpose with two KV tiles per CTA; three through eight
        groups require four.
        """
        effective_splits = max(1, num_splits)
        kv_tiles_per_cta = (
            (num_n_blocks + effective_splits - 1) // effective_splits
            if num_n_blocks > 0
            else 0
        )
        batch_head_groups = batch_size * num_head_kv
        min_kv_tiles_per_cta = 2 if batch_head_groups <= 2 else 4
        return (
            (head_dim, head_dim_v) == (256, 256)
            and 1 <= batch_head_groups <= 8
            and paged_kv
            and page_size == tile_n
            and dense_paged_kv
            and max_seqlen_q == 1
            and pack_gqa
            and 2 <= packed_q_rows <= 8
            and (tile_m, tile_n) == (16, 64)
            and kv_tiles_per_cta >= min_kv_tiles_per_cta
            and (is_causal or is_local)
            and not has_score_or_mask_mod
        )

    @staticmethod
    def make_kernel(
        *,
        dtype,
        head_dim: int,
        head_dim_v: int,
        qhead_per_kvhead: int,
        is_causal: bool,
        is_local: bool,
        pack_gqa: bool,
        config: Sm120ForwardConfig,
        paged_kv: bool,
        score_mod: Optional[Callable],
        mask_mod: Optional[Callable],
        has_aux_tensors: bool,
        is_split_kv: bool,
        has_bias: bool,
        bias_block_size: int,
        rel_extent_padded: int,
        plan: Sm120ForwardPlan,
    ) -> FlashAttentionForwardSm120:
        if not FlashAttentionForwardSm120.can_implement(
            dtype,
            head_dim,
            head_dim_v,
            config.tile_m,
            config.tile_n,
            num_stages=config.num_stages,
            num_threads=config.num_threads,
            is_causal=is_causal,
            Q_in_regs=False,
            paged_kv=paged_kv,
        ):
            raise ValueError(
                "The requested FlashAttention forward configuration exceeds "
                "SM120 kernel constraints or shared-memory capacity"
            )
        if has_bias:
            bias_smem_bytes = (
                bias_block_size * config.tile_n * (dtype.width // 8) * config.num_stages
            )
            total_smem_bytes = (
                FlashAttentionForwardSm120._smem_usage_in_bytes(
                    head_dim,
                    head_dim_v,
                    config.tile_m,
                    config.tile_n,
                    config.num_stages,
                    False,
                )
                + bias_smem_bytes
            )
            if total_smem_bytes > utils_basic.get_smem_capacity_in_bytes("sm_120"):
                raise ValueError(
                    "The requested SM120 sheared-bias specialization exceeds "
                    "shared-memory capacity"
                )
        Kernel = FlashAttentionForwardSm120
        if plan.transpose_qk_pv:
            from sglang.kernels.ops.attention.fa4_sm120.flash_fwd_decode import (
                FlashAttentionForwardSm120DecodeTranspose,
            )

            Kernel = FlashAttentionForwardSm120DecodeTranspose
        return Kernel(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            is_causal=is_causal,
            is_local=is_local,
            pack_gqa=pack_gqa,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            num_stages=config.num_stages,
            num_threads=config.num_threads,
            Q_in_regs=False,
            score_mod=score_mod,
            mask_mod=mask_mod,
            has_aux_tensors=has_aux_tensors,
            is_split_kv=is_split_kv,
            has_bias=has_bias,
            bias_block_size=bias_block_size,
            rel_extent_padded=rel_extent_padded,
            direct_uniform_batch=plan.direct_uniform_batch,
            paged_kv=paged_kv,
            split_qk_n=plan.split_qk_n,
            split_kv_blocks_per_cta=plan.split_kv_blocks_per_cta,
        )


class Sm120ForwardHost(Sm120ForwardPolicy):
    """Mutable direct-launch plans and workspaces for the SM120 policy."""

    def __init__(self) -> None:
        self._varlen_plans: OrderedDict[tuple, _VarlenLaunchPlan] = OrderedDict()
        self._paged_plans: OrderedDict[tuple, _PagedDecodeLaunchPlan] = OrderedDict()
        self._paged_plan_tiles: OrderedDict[tuple, tuple[int, int]] = OrderedDict()
        self._paged_workspaces: OrderedDict[
            tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]
        ] = OrderedDict()
        self._paged_workspace_views: OrderedDict[
            tuple[torch.device, int, int, tuple[int, ...], int, torch.dtype],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        ] = OrderedDict()

    def _varlen_key(
        self,
        *,
        arch: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        learnable_sink: Optional[torch.Tensor],
        pack_gqa: bool,
    ) -> tuple:
        sink_signature = (
            None if learnable_sink is None else _tensor_signature(learnable_sink)
        )
        return (
            "basic-varlen",
            arch,
            tuple(_tensor_signature(t) for t in (q, k, v, cu_seqlens_q, cu_seqlens_k)),
            sink_signature,
            max_seqlen_q,
            max_seqlen_k,
            causal,
            window_size_left,
            window_size_right,
            pack_gqa,
            self.implementation_token(),
            fa_logging.get_fa_log_level(),
        )

    def try_varlen(
        self,
        *,
        arch: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: Optional[int],
        max_seqlen_k: Optional[int],
        softmax_scale: Optional[float],
        causal: bool,
        window_size: tuple[Optional[int], Optional[int]],
        learnable_sink: Optional[torch.Tensor],
        pack_gqa: Optional[bool],
        out: Optional[torch.Tensor],
    ) -> Optional[tuple[torch.Tensor, None]]:
        if (
            not self.supports_arch(arch)
            or q.ndim != 3
            or k.ndim != 3
            or k.shape[1] == 0
        ):
            return None
        expected_out_shape = (*q.shape[:-1], v.shape[-1])
        if out is not None and (
            out.shape != expected_out_shape
            or out.dtype != q.dtype
            or out.device != q.device
            or not out.is_contiguous()
        ):
            return None
        actual_pack_gqa = q.shape[1] // k.shape[1] > 1 if pack_gqa is None else pack_gqa
        actual_max_seqlen_q = q.shape[0] if max_seqlen_q is None else max_seqlen_q
        actual_max_seqlen_k = k.shape[0] if max_seqlen_k is None else max_seqlen_k
        causal, window_size_left, window_size_right = _resolve_causal_local_window(
            causal,
            window_size[0],
            window_size[1],
        )
        key = self._varlen_key(
            arch=arch,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=actual_max_seqlen_q,
            max_seqlen_k=actual_max_seqlen_k,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            learnable_sink=learnable_sink,
            pack_gqa=actual_pack_gqa,
        )
        plan = self._varlen_plans.get(key)
        if plan is None:
            return None
        self._varlen_plans.move_to_end(key)
        if out is None:
            out = torch.empty(
                expected_out_shape,
                dtype=q.dtype,
                device=q.device,
            )
        scale = 1.0 / math.sqrt(q.shape[-1]) if softmax_scale is None else softmax_scale
        plan.compiled_fn(
            q,
            k,
            v,
            out,
            None,
            scale,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            None,
            window_size_left,
            window_size_right,
            learnable_sink,
            None,
            _EMPTY_AUX_DATA,
            None,
            0,
        )
        return out, None

    def register_varlen(
        self,
        *,
        arch: int,
        compiled_fn: Callable,
        compile_key: tuple,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        learnable_sink: Optional[torch.Tensor],
        pack_gqa: bool,
    ) -> None:
        key = self._varlen_key(
            arch=arch,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            learnable_sink=learnable_sink,
            pack_gqa=pack_gqa,
        )
        self._varlen_plans[key] = _VarlenLaunchPlan(compiled_fn, compile_key)
        self._varlen_plans.move_to_end(key)
        while len(self._varlen_plans) > _LAUNCH_PLAN_CAPACITY:
            self._varlen_plans.popitem(last=False)

    def _paged_base_key(
        self,
        *,
        arch: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        page_table: torch.Tensor,
        max_seqlen_q: int,
        causal: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        learnable_sink: Optional[torch.Tensor],
        pack_gqa: bool,
        requested_num_splits: int,
    ) -> tuple:
        sink_signature = (
            None if learnable_sink is None else _tensor_signature(learnable_sink)
        )
        return (
            "paged-forward",
            arch,
            tuple(
                _tensor_signature(t)
                for t in (q, k, v, cu_seqlens_q, seqused_k, page_table)
            ),
            sink_signature,
            max_seqlen_q,
            causal,
            window_size_left,
            window_size_right,
            pack_gqa,
            requested_num_splits,
            self.implementation_token(),
            fa_logging.get_fa_log_level(),
        )

    @staticmethod
    def _paged_selection_key(
        base_key: tuple,
        tile_m: int,
        tile_n: int,
        max_seqlen_k: int,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
    ) -> tuple:
        is_local = window_size_left is not None or window_size_right is not None
        if is_local:
            left = max_seqlen_k if window_size_left is None else window_size_left
            right = max_seqlen_k if window_size_right is None else window_size_right
            seqlen_k_loaded = max(
                0,
                min(max_seqlen_k, left + 1 + right),
            )
        else:
            seqlen_k_loaded = max_seqlen_k
        num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
        return (*base_key, (tile_m, tile_n, num_n_blocks))

    def _paged_workspace(
        self,
        plan: _PagedDecodeLaunchPlan,
        q: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        stream_key = torch.cuda.current_stream(q.device).cuda_stream
        key = (q.device, stream_key)
        view_key = (
            q.device,
            stream_key,
            plan.actual_num_splits,
            tuple(q.shape[:-1]),
            v.shape[-1],
            plan.partial_dtype,
        )
        cached_view = self._paged_workspace_views.get(view_key)
        if cached_view is not None:
            self._paged_workspace_views.move_to_end(view_key)
            return cached_view
        out_numel = plan.actual_num_splits * math.prod(q.shape[:-1]) * v.shape[-1]
        lse_numel = plan.actual_num_splits * q.shape[-2] * q.shape[0]
        workspace = self._paged_workspaces.get(key)
        if workspace is not None:
            self._paged_workspaces.move_to_end(key)
        if (
            workspace is None
            or workspace[0].numel() < out_numel
            or workspace[1].numel() < lse_numel
            or workspace[0].dtype != plan.partial_dtype
        ):
            self._drop_workspace_views(key)
            workspace = (
                torch.empty(out_numel, dtype=plan.partial_dtype, device=q.device),
                torch.empty(lse_numel, dtype=torch.float32, device=q.device),
            )
            self._cache_workspace(key, workspace)
        out_partial = workspace[0][:out_numel].view(
            plan.actual_num_splits,
            *q.shape[:-1],
            v.shape[-1],
        )
        lse_partial = workspace[1][:lse_numel].view(
            plan.actual_num_splits,
            q.shape[-2],
            q.shape[0],
        )
        result = (out_partial, lse_partial, lse_partial.transpose(-1, -2))
        self._cache_workspace_view(view_key, result)
        return result

    @staticmethod
    def _supports_cached_paged_decode(
        *,
        head_dim: int,
        head_dim_v: int,
        effective_q_rows: int,
    ) -> bool:
        """Qualify shapes covered by the SM12x paged-decode launch cache."""
        head_dims = (head_dim, head_dim_v)
        return (
            head_dims in LOW_HD_DECODE_SHAPES or head_dims == (256, 256)
        ) and effective_q_rows <= 16

    def try_paged_decode(
        self,
        *,
        arch: int,
        q: Optional[torch.Tensor],
        k: Optional[torch.Tensor],
        v: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor],
        cu_seqlens_k: Optional[torch.Tensor],
        seqused_q: Optional[torch.Tensor],
        seqused_k: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        max_seqlen_q: Optional[int],
        max_seqlen_k: Optional[int],
        softmax_scale: Optional[float],
        causal: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        learnable_sink: Optional[torch.Tensor],
        requested_num_splits: int,
        pack_gqa: Optional[bool],
        out: Optional[torch.Tensor],
    ) -> Optional[tuple[torch.Tensor, None]]:
        if (
            not self.supports_arch(arch)
            or q is None
            or k is None
            or q.ndim != 3
            or k.ndim != 4
            or cu_seqlens_q is None
            or cu_seqlens_k is not None
            or seqused_q is not None
            or seqused_k is None
            or page_table is None
            or torch.cuda.is_current_stream_capturing()
            or any(t.requires_grad for t in (q, k, v))
            or (learnable_sink is not None and learnable_sink.requires_grad)
        ):
            return None
        expected_out_shape = (*q.shape[:-1], v.shape[-1])
        if out is not None and (
            out.shape != expected_out_shape
            or out.dtype != q.dtype
            or out.device != q.device
            or not out.is_contiguous()
        ):
            return None
        actual_pack_gqa = (
            q.shape[1] // k.shape[-2] > 1 if pack_gqa is None else pack_gqa
        )
        actual_max_seqlen_q = q.shape[0] if max_seqlen_q is None else max_seqlen_q
        actual_max_seqlen_k = (
            k.shape[0] * k.shape[1] if max_seqlen_k is None else max_seqlen_k
        )
        qhead_per_kvhead = q.shape[1] // k.shape[-2]
        effective_q_rows = actual_max_seqlen_q * (
            qhead_per_kvhead if actual_pack_gqa else 1
        )
        if not self._supports_cached_paged_decode(
            head_dim=q.shape[-1],
            head_dim_v=v.shape[-1],
            effective_q_rows=effective_q_rows,
        ):
            return None
        causal, window_size_left, window_size_right = _resolve_causal_local_window(
            causal,
            window_size_left,
            window_size_right,
        )
        base_key = self._paged_base_key(
            arch=arch,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            page_table=page_table,
            max_seqlen_q=actual_max_seqlen_q,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            learnable_sink=learnable_sink,
            pack_gqa=actual_pack_gqa,
            requested_num_splits=requested_num_splits,
        )
        tile_mn = self._paged_plan_tiles.get(base_key)
        if tile_mn is None:
            return None
        self._paged_plan_tiles.move_to_end(base_key)
        selection_key = self._paged_selection_key(
            base_key,
            *tile_mn,
            actual_max_seqlen_k,
            window_size_left,
            window_size_right,
        )
        plan = self._paged_plans.get(selection_key)
        if plan is None:
            return None
        self._paged_plans.move_to_end(selection_key)
        if out is None:
            out = torch.empty(
                expected_out_shape,
                dtype=q.dtype,
                device=q.device,
            )
        scale = 1.0 / math.sqrt(q.shape[-1]) if softmax_scale is None else softmax_scale
        kernel_out = out
        kernel_lse = None
        lse_partial_transposed = None
        if plan.actual_num_splits > 1:
            if plan.compiled_combine is None:
                return None
            kernel_out, kernel_lse, lse_partial_transposed = self._paged_workspace(
                plan, q, v
            )
        plan.compiled_fn(
            q,
            k,
            v,
            kernel_out,
            kernel_lse,
            scale,
            cu_seqlens_q,
            None,
            None,
            seqused_k,
            page_table,
            window_size_left,
            window_size_right,
            learnable_sink,
            None,
            _EMPTY_AUX_DATA,
            None,
            0,
        )
        if plan.actual_num_splits > 1:
            plan.compiled_combine(
                kernel_out,
                lse_partial_transposed,
                out,
                None,
                cu_seqlens_q,
                None,
                None,
                None,
                None,
            )
        return out, None

    def register_paged_decode(
        self,
        *,
        arch: int,
        compiled_fn: Callable,
        compile_key: tuple,
        compiled_combine: Optional[Callable],
        actual_num_splits: int,
        tile_m: int,
        tile_n: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        seqused_k: torch.Tensor,
        page_table: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        causal: bool,
        window_size_left: Optional[int],
        window_size_right: Optional[int],
        learnable_sink: Optional[torch.Tensor],
        pack_gqa: bool,
        requested_num_splits: int,
        out_partial: Optional[torch.Tensor],
        lse_partial: Optional[torch.Tensor],
    ) -> None:
        qhead_per_kvhead = q.shape[1] // k.shape[-2]
        effective_q_rows = max_seqlen_q * (qhead_per_kvhead if pack_gqa else 1)
        if not self._supports_cached_paged_decode(
            head_dim=q.shape[-1],
            head_dim_v=v.shape[-1],
            effective_q_rows=effective_q_rows,
        ):
            return
        base_key = self._paged_base_key(
            arch=arch,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            page_table=page_table,
            max_seqlen_q=max_seqlen_q,
            causal=causal,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            learnable_sink=learnable_sink,
            pack_gqa=pack_gqa,
            requested_num_splits=requested_num_splits,
        )
        selection_key = self._paged_selection_key(
            base_key,
            tile_m,
            tile_n,
            max_seqlen_k,
            window_size_left,
            window_size_right,
        )
        self._paged_plan_tiles[base_key] = (tile_m, tile_n)
        self._paged_plan_tiles.move_to_end(base_key)
        while len(self._paged_plan_tiles) > _LAUNCH_PLAN_CAPACITY:
            stale_base_key, _ = self._paged_plan_tiles.popitem(last=False)
            stale_selection_keys = [
                key for key in self._paged_plans if key[:-1] == stale_base_key
            ]
            for key in stale_selection_keys:
                del self._paged_plans[key]
        self._paged_plans[selection_key] = _PagedDecodeLaunchPlan(
            compiled_fn=compiled_fn,
            compile_key=compile_key,
            actual_num_splits=actual_num_splits,
            compiled_combine=compiled_combine,
            partial_dtype=(out_partial.dtype if out_partial is not None else q.dtype),
        )
        self._paged_plans.move_to_end(selection_key)
        while len(self._paged_plans) > _LAUNCH_PLAN_CAPACITY:
            self._paged_plans.popitem(last=False)

        if out_partial is None or lse_partial is None:
            return
        stream_key = torch.cuda.current_stream(q.device).cuda_stream
        workspace_key = (q.device, stream_key)
        current_workspace = self._paged_workspaces.get(workspace_key)
        if current_workspace is not None:
            self._paged_workspaces.move_to_end(workspace_key)
        if (
            current_workspace is None
            or current_workspace[0].numel() < out_partial.numel()
            or current_workspace[1].numel() < lse_partial.numel()
            or current_workspace[0].dtype != out_partial.dtype
        ):
            self._drop_workspace_views(workspace_key)
            self._cache_workspace(
                workspace_key,
                (
                    out_partial.view(-1),
                    lse_partial.view(-1),
                ),
            )
        view_key = (
            q.device,
            stream_key,
            actual_num_splits,
            tuple(q.shape[:-1]),
            v.shape[-1],
            out_partial.dtype,
        )
        workspace = self._paged_workspaces[workspace_key]
        out_partial_view = workspace[0][: out_partial.numel()].view(out_partial.shape)
        lse_partial_view = workspace[1][: lse_partial.numel()].view(lse_partial.shape)
        self._cache_workspace_view(
            view_key,
            (
                out_partial_view,
                lse_partial_view,
                lse_partial_view.transpose(-1, -2),
            ),
        )

    def _cache_workspace_view(
        self,
        key: tuple,
        value: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        self._paged_workspace_views[key] = value
        self._paged_workspace_views.move_to_end(key)
        while len(self._paged_workspace_views) > _WORKSPACE_VIEW_CAPACITY:
            self._paged_workspace_views.popitem(last=False)

    def _cache_workspace(
        self,
        key: tuple[torch.device, int],
        value: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        self._paged_workspaces[key] = value
        self._paged_workspaces.move_to_end(key)
        while len(self._paged_workspaces) > _WORKSPACE_CAPACITY:
            stale_key, _ = self._paged_workspaces.popitem(last=False)
            self._drop_workspace_views(stale_key)

    def _drop_workspace_views(self, workspace_key: tuple[torch.device, int]) -> None:
        stale_keys = [
            key for key in self._paged_workspace_views if key[:2] == workspace_key
        ]
        for key in stale_keys:
            del self._paged_workspace_views[key]

    def clear_launch_plans(self) -> None:
        self._varlen_plans.clear()
        self._paged_plans.clear()
        self._paged_plan_tiles.clear()
        self._paged_workspaces.clear()
        self._paged_workspace_views.clear()


sm120_forward_host = Sm120ForwardHost()
