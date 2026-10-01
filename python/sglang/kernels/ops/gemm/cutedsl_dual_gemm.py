"""Blackwell dual GEMM and SwiGLU fusion.

The operator computes::

    gate = gemm(x, gate_weight)
    up = gemm(x, up_weight)
    activation = silu(gate) * up

BF16 and FP16 inputs return ``activation`` in the input dtype. FP8 inputs add
static or dynamic, per-tensor or per-token quantization and return
``activation_fp8, activation_scale``.

``gate_up_weight`` stores ``[gate; up]`` along dimension zero.  The kernel is
specialized for small-batch transformer decode. Each CTA owns 64 or 128
intermediate features and up to sixteen tokens, and issues both tcgen05 MMAs
from one TMA-loaded input tile. Dynamic FP8 quantization reduces the activation
maximum at three levels:

* lanes reduce within each warp;
* warp leaders reduce within the CTA;
* CTA leaders atomically reduce into token-wide or tensor-wide global maxima.

The launch is bounded to its guaranteed one- or two-CTA-per-SM residency.  That
makes its release/acquire grid barrier safe: after the global maximum, every CTA
quantizes its own activation slice in parallel.  This is a one-launch
implementation; it neither recomputes the projections nor launches a separate
quantization kernel.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import utils
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import blackwell_helpers as sm100_utils

from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.srt.utils import is_sm100_supported
from sglang.srt.utils.common import direct_register_custom_op

_THREADS = 256
_FP8_MAX = 448.0
_NARROW_CTA_SMEM_BYTES = 112 * 1024
MAX_DUAL_GEMM_DECODE_TOKENS = 16
MAX_DUAL_GEMM_INTERMEDIATE_SIZE = 18944


class DualGemmQuantMode(IntEnum):
    STATIC_PER_TENSOR = 0
    STATIC_PER_TOKEN = 1
    DYNAMIC_PER_TENSOR = 2
    DYNAMIC_PER_TOKEN = 3
    UNQUANT = 4

    @property
    def is_quantized(self) -> bool:
        return self is not self.UNQUANT

    @property
    def is_dynamic(self) -> bool:
        return self in (self.DYNAMIC_PER_TENSOR, self.DYNAMIC_PER_TOKEN)

    @property
    def is_per_token(self) -> bool:
        return self in (self.STATIC_PER_TOKEN, self.DYNAMIC_PER_TOKEN)


# (CTA features, MMA token columns, K tile, pipeline stages, use two-CTA MMA).
# The two-CTA variants pair adjacent feature CTAs into one cluster MMA.  SM100
# requires at least 16 token columns for a two-CTA MMA; small-batch decode
# therefore predicates the unused columns.
_FP8_TACTICS: tuple[tuple[int, int, int, int, bool], ...] = (
    # K=128, six stages.
    (128, 8, 128, 6, False),  # 0: one wave for Llama 3 8B
    (64, 8, 128, 6, False),  # 1: two resident CTAs/SM for wider intermediates
    (64, 16, 128, 6, True),  # 2: paired 2-CTA MMA, two feature waves
    (128, 16, 128, 6, True),  # 3: paired 2-CTA MMA, one feature wave
    # K=256, three stages: same buffered K depth and SMEM footprint.
    (128, 8, 256, 3, False),  # 4
    (64, 8, 256, 3, False),  # 5
    (64, 16, 256, 3, True),  # 6
    (128, 16, 256, 3, True),  # 7
)

# Two-byte inputs use K=128 and three stages to retain the same shared-memory
# footprint and CTA residency as FP8 K=256/stage-3 tactics.
_FLOAT16_TACTICS: tuple[tuple[int, int, int, int, bool], ...] = (
    (128, 8, 128, 3, False),  # 0
    (64, 8, 128, 3, False),  # 1
    (64, 16, 128, 3, True),  # 2
    (128, 16, 128, 3, True),  # 3
)

_COMPILED_DUAL_GEMM: dict[tuple[object, ...], object] = {}


def _resolve_tactic(
    tactic: int, quantize_output: bool
) -> tuple[int, int, int, int, bool]:
    tactics = _FP8_TACTICS if quantize_output else _FLOAT16_TACTICS
    if tactic < 0 or tactic >= len(tactics):
        raise ValueError(f"dual GEMM tactic {tactic} out of range [0, {len(tactics)})")
    return tactics[tactic]


def _pick_fp8_tactic(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    multiprocessor_count: int,
    quant_mode: DualGemmQuantMode,
) -> int:
    """Choose an FP8 tactic measured on B200 small-batch decode shapes."""
    tactic = _pick_fp8_tactic_base(
        num_tokens,
        hidden_size,
        intermediate_size,
        multiprocessor_count,
        quant_mode,
    )
    if num_tokens > 8:
        # Tactics 0, 1, 4, and 5 expose only eight MMA token columns. Preserve
        # the selected K tile and feature width while promoting to a 16-column
        # tactic. Dynamic quantization promotes narrow tactics to the wide form
        # because two-CTA MMA halves guaranteed CTA residency.
        if tactic in (0, 4):
            return tactic + 3
        if tactic in (1, 5):
            return tactic + (2 if quant_mode.is_dynamic else 1)
    return tactic


def _pick_fp8_tactic_base(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    multiprocessor_count: int,
    quant_mode: DualGemmQuantMode,
) -> int:

    wide_feature_tiles = cute.ceil_div(intermediate_size, 128)

    # For grids below half a wave, paired 64-feature CTAs make the best use of
    # otherwise idle SMs. This wins for every batch size and quantization mode.
    if wide_feature_tiles * 2 <= multiprocessor_count:
        return 2

    # Between half and three quarters of a wave, a 128-feature K=128 tile wins.
    # BS9-16 needs the 16-column two-CTA form; dynamic quantization favors the
    # wide variant because its grid barrier must remain resident.
    if wide_feature_tiles * 4 <= multiprocessor_count * 3:
        if num_tokens > 8:
            return 3 if quant_mode.is_dynamic else 2
        return 0

    # At roughly three quarters of a wave, the hidden reduction determines
    # whether K=128 or K=256 repays its pipeline overhead. The measured K=256
    # regime begins at H=2048, with exceptions at H=3072 and H=3584.
    if wide_feature_tiles < multiprocessor_count:
        use_k256 = hidden_size == 2048 or hidden_size >= 4096
        if not use_k256:
            if num_tokens > 8:
                if quant_mode.is_dynamic:
                    if hidden_size == 3584:
                        return 5 if quant_mode.is_per_token else 1
                    return 3 if quant_mode.is_per_token else 1
                return 5 if hidden_size == 3584 else 2
            if not quant_mode.is_dynamic:
                if hidden_size == 3584 and num_tokens in (1, 4):
                    return 1
                return 0
            if quant_mode.is_dynamic and hidden_size % 1024:
                return 1
            return 0

        if hidden_size == 2048 and num_tokens > 8:
            if not quant_mode.is_dynamic:
                return 2
            return 7 if quant_mode.is_per_token else 1

        if not quant_mode.is_dynamic:
            if hidden_size >= 5120 and num_tokens in (1, 4):
                return 4
            return 6 if hidden_size >= 4096 else 4

        if hidden_size >= 8192:
            if num_tokens in (1, 2):
                return 4
            if num_tokens == 4:
                return 7
            if num_tokens > 8 and quant_mode.is_per_token:
                return 7
            return 5
        if hidden_size >= 5120:
            return 5
        if num_tokens > 8:
            return 7
        if hidden_size == 4096 and num_tokens in (2, 4):
            return 5
        return 4

    # A full wave of 128-feature CTAs is also two full resident waves of the
    # narrow tactic. Qwen-sized and wider projections consistently favor K=256
    # with 64 features per CTA. The paired form only helps BS16 at H=4096.
    if (
        wide_feature_tiles == multiprocessor_count
        and num_tokens > 8
        and hidden_size == 4096
    ):
        return 7 if quant_mode.is_dynamic else 6
    return 5


def _pick_float16_tactic(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    multiprocessor_count: int,
    input_dtype: torch.dtype,
) -> int:
    """Choose a measured BF16/FP16 small-batch tactic."""
    tactic = _pick_float16_tactic_base(
        num_tokens,
        hidden_size,
        intermediate_size,
        multiprocessor_count,
        input_dtype,
    )
    if num_tokens > 8:
        # Tactics 0 and 1 expose only eight MMA token columns. Preserve the
        # selected feature width while promoting to the corresponding 16-column
        # paired tactic.
        return {0: 3, 1: 2}.get(tactic, tactic)
    return tactic


def _pick_float16_tactic_base(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    multiprocessor_count: int,
    input_dtype: torch.dtype,
) -> int:

    wide_feature_tiles = cute.ceil_div(intermediate_size, 128)

    # The paired tactic wins decisively for very small projections. A narrow
    # one-CTA band follows before the paired tactic wins again up to 3/4 wave.
    if wide_feature_tiles * 4 <= multiprocessor_count:
        return 2
    if wide_feature_tiles * 5 <= multiprocessor_count * 2:
        return 1
    if wide_feature_tiles * 4 <= multiprocessor_count * 3:
        return 2

    if wide_feature_tiles < multiprocessor_count:
        if hidden_size >= 8192:
            if input_dtype == torch.float16 and num_tokens > 8:
                return 0
            return 2
        if hidden_size == 4096 and input_dtype == torch.bfloat16:
            return 2
        return 0

    # Once the output grid reaches a full wave, two resident 64-feature CTAs
    # generally hide the long weight stream best. FP16 at exactly one wave and
    # power-of-two hidden sizes is the repeatable exception.
    if (
        wide_feature_tiles * 4 < multiprocessor_count * 5
        and input_dtype == torch.float16
        and hidden_size % 4096 == 0
    ):
        return 2
    if (
        wide_feature_tiles == multiprocessor_count
        and input_dtype == torch.bfloat16
        and hidden_size == 4096
        and num_tokens in (2, 4)
    ):
        return 2
    if hidden_size == 5120 and num_tokens == 4 and input_dtype == torch.float16:
        return 2
    if hidden_size >= 8192 and num_tokens == 8:
        return 2
    return 1


class BlackwellDualGemmKernel:
    """Persistent-style warp-specialized dual projection for SM10x."""

    def __init__(
        self,
        num_tokens: int,
        intermediate_size: int,
        element_type,
        quantize_output: bool,
        quant_mode: DualGemmQuantMode,
        cta_features: int,
        cta_tokens: int,
        cta_reduction: int,
        stages: int,
        use_2cta: bool,
    ) -> None:
        self.num_tokens = num_tokens
        self.intermediate_size = intermediate_size
        self.quantize_output = quantize_output
        self.quant_mode = quant_mode
        self.dynamic_quant = quantize_output and quant_mode.is_dynamic
        self.per_token_quant = quantize_output and quant_mode.is_per_token
        self.cta_m = cta_features
        self.cta_n = cta_tokens
        self.cta_k = cta_reduction
        self.stages = stages
        self.use_2cta = use_2cta
        self.threads = _THREADS
        self.feature_tiles = cute.ceil_div(intermediate_size, self.cta_m)
        self.element_type = element_type
        self.accumulator_type = cutlass.Float32
        self.activation_type = (
            cutlass.BFloat16 if quantize_output else self.element_type
        )
        if use_2cta:
            self.cluster_shape = (2, 1, 1)
            self.cta_group = tcgen05.CtaGroup.TWO
            self.tma_op = cute_ext.OperationTypeEnum.SM100_TMA_LOAD_2SM
            self.mma_tiler_mn = (self.cta_m * 2, self.cta_n)
        else:
            self.cluster_shape = (1, 1, 1)
            self.cta_group = tcgen05.CtaGroup.ONE
            self.tma_op = cute_ext.OperationTypeEnum.SM90_TMA_LOAD
            self.mma_tiler_mn = (self.cta_m, self.cta_n)
        self.resident_ctas_per_sm = 2 if cta_features == 64 and not use_2cta else 1
        self.dynamic_smem_bytes = (
            _NARROW_CTA_SMEM_BYTES
            if cta_features == 64
            else utils.get_smem_capacity_in_bytes("sm_100")
        )

    def __repr__(self) -> str:
        quantization = (
            "dynamic"
            if self.dynamic_quant
            else "static"
            if self.quantize_output
            else "unquantized"
        )
        granularity = (
            f"_{'token' if self.per_token_quant else 'tensor'}"
            if self.quantize_output
            else ""
        )
        return (
            "BlackwellDualGemmKernel"
            f"_t{self.num_tokens}_i{self.intermediate_size}"
            f"_m{self.cta_m}_n{self.cta_n}_k{self.cta_k}"
            f"_s{self.stages}_2cta{int(self.use_2cta)}_{quantization}"
            f"{granularity}"
        )

    @cute.experimental.jit
    def make_tiled_mma(self, operand_a: cute.Tensor, operand_b: cute.Tensor):
        return sm100_utils.make_trivial_tiled_mma(
            self.element_type,
            self.element_type,
            utils.LayoutEnum.from_tensor(operand_a).mma_major_mode(),
            utils.LayoutEnum.from_tensor(operand_b).mma_major_mode(),
            self.accumulator_type,
            self.cta_group,
            self.mma_tiler_mn,
        )

    def _make_storage_layouts(self, tiled_mma: cute.TiledMma):
        tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        shared_weight_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, tiler, self.element_type, self.stages
        )
        shared_input_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, tiler, self.element_type, self.stages
        )
        accumulator_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma,
            self.mma_tiler_mn,
            acc_stage=2,
        )
        return shared_weight_layout, shared_input_layout, accumulator_layout

    @cute.experimental.jit
    def weight_dma_warp(
        self,
        full,
        empty,
        leader_rank: cutlass.Int32,
        gate_tile: cute.Tensor,
        up_tile: cute.Tensor,
        shared_gate: cute.Tensor,
        shared_up: cute.Tensor,
        gate_map: cute.Layout,
        up_map: cute.Layout,
        reduction_tiles: cutlass.Int32,
    ):
        """Warp 0: issue gate and up TMA loads into independent stage buffers."""
        empty_phase = cutlass.Int32(1)
        transaction_bytes = cute.size_in_bytes(
            shared_gate.element_type,
            cute.slice_(shared_gate.layout, (None, None, None, 0)),
        ) + cute.size_in_bytes(
            shared_up.element_type,
            cute.slice_(shared_up.layout, (None, None, None, 0)),
        )
        for reduction_tile in cutlass.range(reduction_tiles, unroll=1):
            stage = reduction_tile % self.stages
            cute.arch.mbarrier_wait(empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full + stage,
                    transaction_bytes,
                    peer_cta_rank_in_cluster=(leader_rank if self.use_2cta else None),
                )
            cute_ext.tma_load(
                gate_tile[None, None, reduction_tile],
                shared_gate[None, None, None, stage],
                (full + stage).value,
                cta_v_map=gate_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            cute_ext.tma_load(
                up_tile[None, None, reduction_tile],
                shared_up[None, None, None, stage],
                (full + stage).value,
                cta_v_map=up_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == self.stages - 1:
                empty_phase ^= 1
        # Drain the final MMA commits before the DMA warp can reach the CTA
        # and cluster exit barriers.  This is mandatory for peer-directed
        # 2-CTA commits, whose arrival can otherwise target a retired CTA.
        for reduction_tile in cutlass.range_constexpr(self.stages):
            stage = (reduction_tile + reduction_tiles) % self.stages
            cute.arch.mbarrier_wait(empty + stage, empty_phase)
            if stage == self.stages - 1:
                empty_phase ^= 1

    @cute.experimental.jit
    def input_dma_warp(
        self,
        full,
        empty,
        leader_rank: cutlass.Int32,
        input_tile: cute.Tensor,
        shared_input: cute.Tensor,
        input_map: cute.Layout,
        reduction_tiles: cutlass.Int32,
    ):
        """Warp 1: TMA-load X once for both projection MMAs."""
        empty_phase = cutlass.Int32(1)
        transaction_bytes = cute.size_in_bytes(
            shared_input.element_type,
            cute.slice_(shared_input.layout, (None, None, None, 0)),
        )
        for reduction_tile in cutlass.range(reduction_tiles, unroll=1):
            stage = reduction_tile % self.stages
            cute.arch.mbarrier_wait(empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    full + stage,
                    transaction_bytes,
                    peer_cta_rank_in_cluster=(leader_rank if self.use_2cta else None),
                )
            cute_ext.tma_load(
                input_tile[None, None, reduction_tile],
                shared_input[None, None, None, stage],
                (full + stage).value,
                cta_v_map=input_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == self.stages - 1:
                empty_phase ^= 1
        for reduction_tile in cutlass.range_constexpr(self.stages):
            stage = (reduction_tile + reduction_tiles) % self.stages
            cute.arch.mbarrier_wait(empty + stage, empty_phase)
            if stage == self.stages - 1:
                empty_phase ^= 1

    @cute.experimental.jit
    def mma_warp(
        self,
        is_leader: cutlass.Boolean,
        full,
        empty,
        ready,
        tmem_ready,
        tmem_slot,
        accumulator_layout: cutlass.Constexpr,
        tiled_mma: cute.TiledMma,
        shared_gate: cute.Tensor,
        shared_up: cute.Tensor,
        shared_input: cute.Tensor,
        reduction_tiles: cutlass.Int32,
    ):
        """Warp 2: issue the gate and up tcgen05 MMAs into two TMEM stages."""
        cute.arch.alloc_tmem(256, tmem_slot, is_two_cta=self.use_2cta)
        cute.arch.mbarrier_arrive(tmem_ready)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=self.use_2cta)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.accumulator_type, 16, tmem_slot)
        tmem = cute.make_tensor(tmem_ptr, accumulator_layout)
        gate_accumulator = tmem[None, None, None, 0]
        up_accumulator = tmem[None, None, None, 1]
        if is_leader:
            mma_atom = cute.make_mma_atom(tiled_mma.op)
            instruction_k = cute.size(tiled_mma.shape_mnk, mode=[2])
            instructions_per_tile = self.cta_k // instruction_k
            commit_mask = 0b11 if self.use_2cta else None
            full_phase = cutlass.Int32(0)
            for reduction_tile in cutlass.range(reduction_tiles, unroll=1):
                stage = reduction_tile % self.stages
                cute.arch.mbarrier_wait(full + stage, full_phase)
                for k_block in cutlass.range_constexpr(instructions_per_tile):
                    input_fragment = cute.append_ones(
                        shared_input[None, None, k_block, stage], up_to_rank=3
                    )
                    mma_atom.set(
                        tcgen05.Field.ACCUMULATE,
                        reduction_tile != 0 or k_block != 0,
                    )
                    gate_fragment = cute.append_ones(
                        shared_gate[None, None, k_block, stage], up_to_rank=3
                    )
                    cute_ext.dot(
                        mma_atom, gate_fragment, input_fragment, gate_accumulator
                    )
                    mma_atom.set(
                        tcgen05.Field.ACCUMULATE,
                        reduction_tile != 0 or k_block != 0,
                    )
                    up_fragment = cute.append_ones(
                        shared_up[None, None, k_block, stage], up_to_rank=3
                    )
                    cute_ext.dot(mma_atom, up_fragment, input_fragment, up_accumulator)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage, commit_mask, cta_group=self.cta_group)
                if stage == self.stages - 1:
                    full_phase ^= 1
            with cute.arch.elect_one():
                tcgen05.commit(ready, commit_mask, cta_group=self.cta_group)
        # Phase 1: the MMA warp and four epilogue warps contribute 32 + 128
        # arrivals.  Do not free cluster-coherent TMEM until every async load
        # has been fenced by the epilogue warps.
        cute.arch.mbarrier_arrive(tmem_ready)
        cute.arch.mbarrier_wait(tmem_ready, 1)
        cute.arch.dealloc_tmem(tmem_ptr, 256, is_two_cta=self.use_2cta)

    @cute.experimental.jit
    def _copy_accumulator(
        self,
        accumulator: cute.Tensor,
        destination: cute.Tensor,
        tiled_mma: cute.TiledMma,
        epilogue_thread: cutlass.Int32,
    ):
        destination_layout = utils.LayoutEnum.from_tensor(destination)
        copy_atom = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k),
            destination_layout,
            self.accumulator_type,
            self.accumulator_type,
            (self.cta_m, self.cta_n),
            self.use_2cta,
        )
        tiled_copy = tcgen05.make_tmem_copy(copy_atom, accumulator)
        destination_divided = cute.flat_divide(destination, (self.cta_m, self.cta_n))
        register_layout = cute_ext.make_t2r_rmem_layout(
            tiled_copy, destination_divided, epilogue_thread
        )
        registers = cute_ext.allocate(
            self.accumulator_type,
            cute.AddressSpace.rmem,
            register_layout,
            alignment=32,
        )
        thread_copy = tiled_copy.get_slice(epilogue_thread)
        cute_ext.partition_and_copy(thread_copy, accumulator, registers)
        cute.arch.fence_view_async_tmem_load()
        cute_ext.partition_and_copy(
            thread_copy, registers, destination_divided[None, None, 0, 0]
        )

    @cute.experimental.jit
    def tmem_store_warps(
        self,
        ready,
        tmem_ready,
        tmem_slot,
        accumulator_layout: cutlass.Constexpr,
        gate_tile: cute.Tensor,
        up_tile: cute.Tensor,
        tiled_mma: cute.TiledMma,
        epilogue_thread: cutlass.Int32,
    ):
        """Warps 4-7: move both TMEM accumulators to shared memory."""
        cute.arch.mbarrier_arrive(tmem_ready)
        cute.arch.mbarrier_wait(tmem_ready, 0)
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.accumulator_type, 16, tmem_slot)
        tmem = cute.make_tensor(tmem_ptr, accumulator_layout)
        gate_accumulator = tmem[((None, None), 0, 0, 0)]
        up_accumulator = tmem[((None, None), 0, 0, 1)]
        cute.arch.mbarrier_wait(ready, 0)
        self._copy_accumulator(gate_accumulator, gate_tile, tiled_mma, epilogue_thread)
        self._copy_accumulator(up_accumulator, up_tile, tiled_mma, epilogue_thread)
        cute.arch.mbarrier_arrive(tmem_ready)

    @cute.experimental.jit
    def silu_and_mul(
        self,
        gate: cutlass.Float32,
        up: cutlass.Float32,
    ):
        """Apply the SwiGLU activation and round to the activation dtype."""
        return (
            gate
            * cute.arch.rcp_approx(
                cutlass.Float32(1.0) + cute.math.exp(-gate, fastmath=True)
            )
            * up
        ).to(self.activation_type)

    @cute.experimental.jit
    def activation_epilogue(
        self,
        warp: cutlass.Int32,
        tid: cutlass.Int32,
        lane: cutlass.Int32,
        feature_tile: cutlass.Int32,
        gate_tile: cute.Tensor,
        up_tile: cute.Tensor,
        output: cute.Tensor,
        x_scale: cute.Tensor,
        gate_up_weight_scale: cute.Tensor,
        output_scale: cute.Tensor,
        global_amax: cute.Tensor,
        supplied_scale: cute.Tensor,
        completion_counter: cute.Tensor,
        shared_scale: cute.Tensor,
    ):
        """All warps: apply SwiGLU, then store or quantize the result."""
        cute.arch.sync_threads()

        if cutlass.const_expr(not self.dynamic_quant and self.num_tokens <= 8):
            static_feature = feature_tile * self.cta_m + tid
            if tid < self.cta_m and static_feature < self.intermediate_size:
                for static_token in cutlass.range_constexpr(self.num_tokens):
                    static_gate = gate_tile[tid, static_token].to(cutlass.Float32)
                    static_up = up_tile[tid, static_token].to(cutlass.Float32)

                    if cutlass.const_expr(self.quantize_output):
                        static_input_scale_idx = (
                            static_token if x_scale.shape[0] != 1 else cutlass.Int32(0)
                        )
                        static_input_scale = x_scale[static_input_scale_idx].to(
                            cutlass.Float32
                        )
                        if cutlass.const_expr(gate_up_weight_scale.shape[0] == 1):
                            static_gate_scale = gate_up_weight_scale[0].to(
                                cutlass.Float32
                            )
                            static_up_scale = static_gate_scale
                        else:
                            static_gate_scale = gate_up_weight_scale[static_feature].to(
                                cutlass.Float32
                            )
                            static_up_scale = gate_up_weight_scale[
                                static_feature + self.intermediate_size
                            ].to(cutlass.Float32)
                        static_gate *= static_input_scale * static_gate_scale
                        static_up *= static_input_scale * static_up_scale

                    static_gate = static_gate.to(self.activation_type).to(
                        cutlass.Float32
                    )
                    static_up = static_up.to(self.activation_type).to(cutlass.Float32)
                    static_activation = self.silu_and_mul(static_gate, static_up)

                    if cutlass.const_expr(not self.quantize_output):
                        output[static_feature, static_token, 0] = static_activation
                    else:
                        static_scale_idx = (
                            static_token if self.per_token_quant else cutlass.Int32(0)
                        )
                        static_quantized = (
                            static_activation.to(cutlass.Float32)
                            / supplied_scale[static_scale_idx]
                        )
                        if static_quantized > cutlass.Float32(_FP8_MAX):
                            static_quantized = cutlass.Float32(_FP8_MAX)
                        if static_quantized < cutlass.Float32(-_FP8_MAX):
                            static_quantized = cutlass.Float32(-_FP8_MAX)
                        output[static_feature, static_token, 0] = static_quantized.to(
                            self.element_type
                        )

            if cutlass.const_expr(self.quantize_output):
                if feature_tile == 0 and warp == 0:
                    if cutlass.const_expr(self.per_token_quant):
                        if lane < self.num_tokens:
                            output_scale[lane] = supplied_scale[lane]
                    elif lane == 0:
                        output_scale[0] = supplied_scale[0]
            return

        num_warps = self.threads // 32
        token_rounds = cute.ceil_div(self.num_tokens, num_warps)
        feature_groups = self.cta_m // 32
        activations = cute_ext.allocate(
            self.activation_type,
            cute.AddressSpace.rmem,
            cute.make_layout((token_rounds, feature_groups)),
            alignment=16,
        )

        # Assign one token to each warp. For BS9-16, every warp processes a
        # second token. Lanes walk the CTA's feature rows, so each token needs
        # only a warp reduction rather than a serialized CTA-wide reduction.
        for token_round in cutlass.range_constexpr(token_rounds):
            token = warp + token_round * num_warps
            if token < self.num_tokens:
                token_amax = cutlass.Float32(0.0)
                for feature_group in cutlass.range_constexpr(feature_groups):
                    local_feature = lane + feature_group * 32
                    feature = feature_tile * self.cta_m + local_feature
                    gate_value = gate_tile[local_feature, token].to(cutlass.Float32)
                    up_value = up_tile[local_feature, token].to(cutlass.Float32)

                    if cutlass.const_expr(self.quantize_output):
                        input_scale_idx = (
                            token if x_scale.shape[0] != 1 else cutlass.Int32(0)
                        )
                        input_scale = x_scale[input_scale_idx].to(cutlass.Float32)

                        if cutlass.const_expr(gate_up_weight_scale.shape[0] == 1):
                            gate_scale = gate_up_weight_scale[0].to(cutlass.Float32)
                            up_scale = gate_scale
                        else:
                            gate_scale = gate_up_weight_scale[feature].to(
                                cutlass.Float32
                            )
                            up_scale = gate_up_weight_scale[
                                feature + self.intermediate_size
                            ].to(cutlass.Float32)

                        gate_value *= input_scale * gate_scale
                        up_value *= input_scale * up_scale

                    # Match the unfused operator chain: projection outputs
                    # round before SwiGLU consumes them.
                    gate_value = gate_value.to(self.activation_type).to(cutlass.Float32)
                    up_value = up_value.to(self.activation_type).to(cutlass.Float32)
                    activation = self.silu_and_mul(gate_value, up_value)
                    activations[token_round, feature_group] = activation

                    if cutlass.const_expr(self.dynamic_quant):
                        abs_activation = cute.math.absf(activation.to(cutlass.Float32))
                        if abs_activation > token_amax:
                            token_amax = abs_activation
                    elif cutlass.const_expr(not self.quantize_output):
                        output[feature, token, 0] = activation
                    else:
                        scale_idx = token if self.per_token_quant else cutlass.Int32(0)
                        scale = supplied_scale[scale_idx].to(cutlass.Float32)
                        quantized = activation.to(cutlass.Float32) / scale
                        if quantized > cutlass.Float32(_FP8_MAX):
                            quantized = cutlass.Float32(_FP8_MAX)
                        if quantized < cutlass.Float32(-_FP8_MAX):
                            quantized = cutlass.Float32(-_FP8_MAX)
                        output[feature, token, 0] = quantized.to(self.element_type)

                if cutlass.const_expr(self.dynamic_quant):
                    token_amax = cute.arch.warp_reduction_max(token_amax)
                    if lane == 0:
                        shared_scale[token] = token_amax

        if cutlass.const_expr(self.quantize_output and self.dynamic_quant):
            cute.arch.sync_threads()

            if cutlass.const_expr(self.feature_tiles == 1):
                if cutlass.const_expr(self.per_token_quant):
                    if warp == 0 and lane < self.num_tokens:
                        scale = shared_scale[lane] / cutlass.Float32(_FP8_MAX)
                        if scale == cutlass.Float32(0.0):
                            scale = cutlass.Float32(1.0)
                        shared_scale[lane] = scale
                        output_scale[lane] = scale
                else:
                    if warp == 0:
                        amax = (
                            shared_scale[lane]
                            if lane < self.num_tokens
                            else cutlass.Float32(0.0)
                        )
                        amax = cute.arch.warp_reduction_max(amax)
                        if lane == 0:
                            scale = amax / cutlass.Float32(_FP8_MAX)
                            if scale == cutlass.Float32(0.0):
                                scale = cutlass.Float32(1.0)
                            shared_scale[0] = scale
                            output_scale[0] = scale
                cute.arch.sync_threads()
            else:
                # Level 3: publish one CTA maximum per output scale, then use
                # one counter as the release/acquire grid barrier.
                if warp == 0:
                    if cutlass.const_expr(self.per_token_quant):
                        if lane < self.num_tokens:
                            cute.arch.atomic_fmax(
                                global_amax.iterator + lane,
                                shared_scale[lane],
                                sign_bit=False,
                                sem="acq_rel",
                                scope="gpu",
                            )
                    else:
                        cta_amax = (
                            shared_scale[lane]
                            if lane < self.num_tokens
                            else cutlass.Float32(0.0)
                        )
                        cta_amax = cute.arch.warp_reduction_max(cta_amax)
                        if lane == 0:
                            cute.arch.atomic_fmax(
                                global_amax.iterator,
                                cta_amax,
                                sign_bit=False,
                                sem="acq_rel",
                                scope="gpu",
                            )
                # Publish every lane's atomic before the CTA leader signals
                # completion to other resident CTAs.
                cute.arch.sync_threads()

                if tid == 0:
                    cute.arch.fence_acq_rel_gpu()
                    cute.arch.atomic_add(
                        completion_counter.iterator,
                        cutlass.Int32(1),
                        sem="acq_rel",
                        scope="gpu",
                    )

                    # The host bounds the grid to guaranteed residency, so
                    # this cannot wait on a CTA that has not been scheduled.
                    completed = cutlass.Int32(0)
                    while completed < self.feature_tiles:
                        completed = cute.arch.atomic_add(
                            completion_counter.iterator,
                            cutlass.Int32(0),
                            sem="acquire",
                            scope="gpu",
                        )

                    if cutlass.const_expr(self.per_token_quant):
                        for token in cutlass.range_constexpr(self.num_tokens):
                            scale = global_amax[token].to(
                                cutlass.Float32
                            ) / cutlass.Float32(_FP8_MAX)
                            if scale == cutlass.Float32(0.0):
                                scale = cutlass.Float32(1.0)
                            shared_scale[token] = scale
                            if feature_tile == 0:
                                output_scale[token] = scale
                    else:
                        scale = global_amax[0].to(cutlass.Float32) / cutlass.Float32(
                            _FP8_MAX
                        )
                        if scale == cutlass.Float32(0.0):
                            scale = cutlass.Float32(1.0)
                        shared_scale[0] = scale
                        if feature_tile == 0:
                            output_scale[0] = scale
                cute.arch.sync_threads()

            for token_round in cutlass.range_constexpr(token_rounds):
                token = warp + token_round * num_warps
                if token < self.num_tokens:
                    scale_idx = token if self.per_token_quant else cutlass.Int32(0)
                    for feature_group in cutlass.range_constexpr(feature_groups):
                        local_feature = lane + feature_group * 32
                        feature = feature_tile * self.cta_m + local_feature
                        quantized = (
                            activations[token_round, feature_group].to(cutlass.Float32)
                            / shared_scale[scale_idx]
                        )
                        if quantized > cutlass.Float32(_FP8_MAX):
                            quantized = cutlass.Float32(_FP8_MAX)
                        if quantized < cutlass.Float32(-_FP8_MAX):
                            quantized = cutlass.Float32(-_FP8_MAX)
                        output[feature, token, 0] = quantized.to(self.element_type)
        elif cutlass.const_expr(self.quantize_output):
            if feature_tile == 0 and warp == 0:
                if cutlass.const_expr(self.per_token_quant):
                    if lane < self.num_tokens:
                        output_scale[lane] = supplied_scale[lane]
                elif lane == 0:
                    output_scale[0] = supplied_scale[0]

    @cute.experimental.jit
    def __call__(
        self,
        x: cute.Tensor,
        gate: cute.Tensor,
        up: cute.Tensor,
        output: cute.Tensor,
        x_scale: cute.Tensor,
        gate_up_weight_scale: cute.Tensor,
        output_scale: cute.Tensor,
        global_amax: cute.Tensor,
        supplied_scale: cute.Tensor,
        completion_counter: cute.Tensor,
        stream: cuda.CUstream,
    ):
        launch = self.kernel(
            x,
            gate,
            up,
            output,
            x_scale,
            gate_up_weight_scale,
            output_scale,
            global_amax,
            supplied_scale,
            completion_counter,
        )

        if cutlass.const_expr(self.use_2cta):
            launch.launch(
                grid=cute.round_up((self.feature_tiles, 1, 1), self.cluster_shape),
                block=(self.threads, 1, 1),
                cluster=self.cluster_shape,
                smem=cute.Int64(self.dynamic_smem_bytes),
                stream=stream,
            )
        else:
            launch.launch(
                grid=(self.feature_tiles, 1, 1),
                block=(self.threads, 1, 1),
                max_number_threads=(self.threads, 1, 1),
                min_blocks_per_mp=self.resident_ctas_per_sm,
                smem=cute.Int64(self.dynamic_smem_bytes),
                stream=stream,
            )

    @cute.experimental.kernel
    def kernel(
        self,
        x: cute.Tensor,
        gate: cute.Tensor,
        up: cute.Tensor,
        output: cute.Tensor,
        x_scale: cute.Tensor,
        gate_up_weight_scale: cute.Tensor,
        output_scale: cute.Tensor,
        global_amax: cute.Tensor,
        supplied_scale: cute.Tensor,
        completion_counter: cute.Tensor,
    ):
        tid, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        feature_tile, _, _ = cute.arch.block_idx()
        leader_rank = cutlass.Int32(0)

        if cutlass.const_expr(self.use_2cta):
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            is_leader = cta_rank == leader_rank
        else:
            cta_rank = leader_rank
            is_leader = cutlass.Boolean(True)

        reduction_size = x.shape[1]
        reduction_tiles = cute.ceil_div(reduction_size, self.cta_k)

        tiled_mma = self.make_tiled_mma(gate, x)
        tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)

        shared_weight_layout, shared_input_layout, accumulator_layout = (
            self._make_storage_layouts(tiled_mma)
        )
        shared_gate = cute_ext.allocate(
            self.element_type,
            cute.AddressSpace.smem,
            shared_weight_layout,
            alignment=1024,
        )
        shared_up = cute_ext.allocate(
            self.element_type,
            cute.AddressSpace.smem,
            shared_weight_layout,
            alignment=1024,
        )
        shared_input = cute_ext.allocate(
            self.element_type,
            cute.AddressSpace.smem,
            shared_input_layout,
            alignment=1024,
        )
        accumulator_tile_layout = cute.make_layout(
            (self.cta_m, self.cta_n), stride=(1, self.cta_m)
        )
        gate_tile = cute_ext.allocate(
            self.accumulator_type,
            cute.AddressSpace.smem,
            accumulator_tile_layout,
            alignment=128,
        )
        up_tile = cute_ext.allocate(
            self.accumulator_type,
            cute.AddressSpace.smem,
            accumulator_tile_layout,
            alignment=128,
        )
        shared_scale = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(self.num_tokens),
            alignment=16,
        )
        full_storage = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.stages),
            alignment=8,
        )
        empty_storage = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.stages),
            alignment=8,
        )
        ready_storage = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        tmem_slot = cute_ext.allocate(
            cutlass.Int32, cute.AddressSpace.smem, cute.make_layout(1), alignment=4
        )
        tmem_ready = cute_ext.allocate(
            cutlass.Int64, cute.AddressSpace.smem, cute.make_layout(1), alignment=8
        )
        full, empty = full_storage.iterator, empty_storage.iterator
        ready = ready_storage.iterator

        if tid == 0:
            for stage in cutlass.range_constexpr(self.stages):
                # Weight and input DMA warps each arrive once.  The weight
                # arrival accounts for both gate and up transaction bytes.
                cute.arch.mbarrier_init(full + stage, 4 if self.use_2cta else 2)
                cute.arch.mbarrier_init(empty + stage, 1)
            cute.arch.mbarrier_init(ready, 1)
            cute.arch.mbarrier_init(tmem_ready.iterator, 32 + 128)

        cute.arch.mbarrier_init_fence()

        if cutlass.const_expr(self.use_2cta):
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.sync_threads()

        gate_map = cute_ext.get_cta_v_map_ab(gate, tiler, tiled_mma, "A")
        up_map = cute_ext.get_cta_v_map_ab(up, tiler, tiled_mma, "A")
        input_map = cute_ext.get_cta_v_map_ab(x, tiler, tiled_mma, "B")
        global_gate = cute.local_tile(
            gate, (self.cta_m, self.cta_k), (feature_tile, None, 0)
        )
        global_up = cute.local_tile(
            up, (self.cta_m, self.cta_k), (feature_tile, None, 0)
        )

        if cutlass.const_expr(self.use_2cta):
            global_input = cute.local_tile(
                x, (self.cta_n // 2, self.cta_k), (cta_rank, None, 0)
            )
        else:
            global_input = cute.local_tile(x, (self.cta_n, self.cta_k), (0, None, 0))

        if cutlass.const_expr(self.use_2cta):
            cute.arch.cluster_wait()

        if warp == 0:
            self.weight_dma_warp(
                full,
                empty,
                leader_rank,
                global_gate,
                global_up,
                shared_gate,
                shared_up,
                gate_map,
                up_map,
                reduction_tiles,
            )
        elif warp == 1:
            self.input_dma_warp(
                full,
                empty,
                leader_rank,
                global_input,
                shared_input,
                input_map,
                reduction_tiles,
            )
        elif warp == 2:
            self.mma_warp(
                is_leader,
                full,
                empty,
                ready,
                tmem_ready.iterator,
                tmem_slot.iterator,
                accumulator_layout,
                tiled_mma,
                shared_gate,
                shared_up,
                shared_input,
                reduction_tiles,
            )
        elif warp >= 4:
            self.tmem_store_warps(
                ready,
                tmem_ready.iterator,
                tmem_slot.iterator,
                accumulator_layout,
                gate_tile,
                up_tile,
                tiled_mma,
                tid - 128,
            )

        self.activation_epilogue(
            warp,
            tid,
            lane,
            feature_tile,
            gate_tile,
            up_tile,
            output,
            x_scale,
            gate_up_weight_scale,
            output_scale,
            global_amax,
            supplied_scale,
            completion_counter,
            shared_scale,
        )

        if cutlass.const_expr(self.use_2cta):
            # Keep each peer alive until all peer-directed TMA arrivals,
            # multicast MMA commits, and cluster-coherent TMEM frees retire.
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()


@cute.experimental.jit
def _dual_gemm_sm100_wrapper(
    kernel: cutlass.Constexpr,
    x: cute.Tensor,
    gate: cute.Tensor,
    up: cute.Tensor,
    output: cute.Tensor,
    x_scale: cute.Tensor,
    gate_up_weight_scale: cute.Tensor,
    output_scale: cute.Tensor,
    global_amax: cute.Tensor,
    supplied_scale: cute.Tensor,
    completion_counter: cute.Tensor,
    stream: cuda.CUstream,
):
    # PyTorch tensors are wrapped as (L, rows, reduction).  The kernel uses
    # TGV's (rows, reduction, L) convention so output features are contiguous.
    x = cute.make_tensor(x.iterator, cute.select(x.layout, mode=[1, 2, 0]))
    gate = cute.make_tensor(gate.iterator, cute.select(gate.layout, mode=[1, 2, 0]))
    up = cute.make_tensor(up.iterator, cute.select(up.layout, mode=[1, 2, 0]))
    output = cute.make_tensor(
        output.iterator, cute.select(output.layout, mode=[2, 1, 0])
    )

    kernel(
        x,
        gate,
        up,
        output,
        x_scale,
        gate_up_weight_scale,
        output_scale,
        global_amax,
        supplied_scale,
        completion_counter,
        stream,
    )


def _cute_tensor(tensor: torch.Tensor):
    return from_dlpack(tensor.detach(), assumed_align=16)


def _cute_tensor_dynamic(tensor: torch.Tensor):
    return from_dlpack(tensor.detach(), assumed_align=32).mark_layout_dynamic(
        leading_dim=tensor.ndim - 1
    )


def _validate_inputs(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    x_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    output_scale: Optional[torch.Tensor],
) -> tuple[int, int, int]:
    if x.device.type != "cuda" or gate_up_weight.device != x.device:
        raise ValueError("x and gate_up_weight must be on the same CUDA device")
    if x.dtype != torch.float8_e4m3fn or gate_up_weight.dtype != x.dtype:
        raise ValueError("x and gate_up_weight must use float8_e4m3fn")
    if x.ndim != 2 or gate_up_weight.ndim != 2:
        raise ValueError("x and gate_up_weight must be 2-D")
    if x.stride(1) != 1 or gate_up_weight.stride(1) != 1:
        raise ValueError("x and gate_up_weight must be contiguous along K")
    num_tokens, hidden_size = x.shape
    packed_intermediate_size, weight_hidden_size = gate_up_weight.shape
    if not 1 <= num_tokens <= MAX_DUAL_GEMM_DECODE_TOKENS:
        raise ValueError(
            f"the SM100 dual GEMM supports 1 to {MAX_DUAL_GEMM_DECODE_TOKENS} tokens"
        )
    if packed_intermediate_size % 2:
        raise ValueError("gate_up_weight.shape[0] must be even")
    intermediate_size = packed_intermediate_size // 2
    if weight_hidden_size != hidden_size:
        raise ValueError("gate_up_weight.shape[1] must equal x.shape[1]")
    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError("hidden_size and intermediate_size must be multiples of 128")
    for scale, valid_sizes, name in (
        (x_scale, (1, num_tokens), "x_scale"),
        (
            gate_up_weight_scale,
            (1, packed_intermediate_size),
            "gate_up_weight_scale",
        ),
    ):
        if scale.device != x.device or scale.dtype != torch.float32:
            raise ValueError(f"{name} must be float32 on x.device")
        if scale.numel() not in valid_sizes:
            raise ValueError(f"{name} must contain one of {valid_sizes} values")
    if output_scale is not None:
        if output_scale.device != x.device or output_scale.dtype != torch.float32:
            raise ValueError("output_scale must be float32 on x.device")
        if output_scale.numel() not in (1, num_tokens):
            raise ValueError(
                "output_scale must contain one scale or one scale per token"
            )
    return num_tokens, hidden_size, intermediate_size


def _validate_float16_inputs(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
) -> tuple[int, int, int]:
    if x.device.type != "cuda" or gate_up_weight.device != x.device:
        raise ValueError("x and gate_up_weight must be on the same CUDA device")
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("x must use bfloat16 or float16")
    if gate_up_weight.dtype != x.dtype:
        raise ValueError("x and gate_up_weight must have the same dtype")
    if x.ndim != 2 or gate_up_weight.ndim != 2:
        raise ValueError("x and gate_up_weight must be 2-D")
    if x.stride(1) != 1 or gate_up_weight.stride(1) != 1:
        raise ValueError("x and gate_up_weight must be contiguous along K")

    num_tokens, hidden_size = x.shape
    packed_intermediate_size, weight_hidden_size = gate_up_weight.shape
    if not 1 <= num_tokens <= MAX_DUAL_GEMM_DECODE_TOKENS:
        raise ValueError(
            f"the SM100 dual GEMM supports 1 to {MAX_DUAL_GEMM_DECODE_TOKENS} tokens"
        )
    if packed_intermediate_size % 2:
        raise ValueError("gate_up_weight.shape[0] must be even")

    intermediate_size = packed_intermediate_size // 2
    if weight_hidden_size != hidden_size:
        raise ValueError("gate_up_weight.shape[1] must equal x.shape[1]")
    if hidden_size % 128 or intermediate_size % 128:
        raise ValueError("hidden_size and intermediate_size must be multiples of 128")
    return num_tokens, hidden_size, intermediate_size


def _dual_gemm_swiglu_fp8_run(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    x_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    output_scale: Optional[torch.Tensor] = None,
    quant_mode: int = int(DualGemmQuantMode.DYNAMIC_PER_TOKEN),
    tactic: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not is_sm100_supported():
        raise RuntimeError("CuTe DSL dual GEMM requires an SM10x GPU")

    num_tokens, hidden_size, intermediate_size = _validate_inputs(
        x, gate_up_weight, x_scale, gate_up_weight_scale, output_scale
    )
    quant_mode = DualGemmQuantMode(quant_mode)
    if not quant_mode.is_quantized:
        raise ValueError("UNQUANT is not valid for the FP8 dual GEMM")
    dynamic_quant = quant_mode.is_dynamic
    per_token_quant = quant_mode.is_per_token
    if dynamic_quant != (output_scale is None):
        expected = "None" if dynamic_quant else "a scale tensor"
        raise ValueError(f"output_scale must be {expected} for {quant_mode.name}")
    if not dynamic_quant:
        expected_scales = num_tokens if per_token_quant else 1
        if output_scale.numel() != expected_scales:
            raise ValueError(
                f"{quant_mode.name} requires {expected_scales} output scale value(s)"
            )
    multiprocessor_count = torch.cuda.get_device_properties(
        x.device
    ).multi_processor_count

    if tactic < 0:
        tactic = _pick_fp8_tactic(
            num_tokens,
            hidden_size,
            intermediate_size,
            multiprocessor_count,
            quant_mode,
        )

    cta_features, cta_tokens, cta_reduction, stages, use_2cta = _resolve_tactic(
        tactic, True
    )
    if num_tokens > cta_tokens:
        raise ValueError(
            f"dual GEMM tactic {tactic} has {cta_tokens} token columns, "
            f"but the input contains {num_tokens} tokens"
        )
    resident_ctas_per_sm = 2 if cta_features == 64 and not use_2cta else 1
    feature_tiles = intermediate_size // cta_features

    if dynamic_quant and feature_tiles > resident_ctas_per_sm * multiprocessor_count:
        raise ValueError(
            "dynamic dual GEMM grid exceeds its guaranteed residency: "
            f"got {feature_tiles} CTAs for {multiprocessor_count} SMs at "
            f"{resident_ctas_per_sm} CTAs/SM"
        )

    x_scale = x_scale.reshape(-1)
    gate_up_weight_scale = gate_up_weight_scale.reshape(-1)
    quantized = torch.empty(
        (num_tokens, intermediate_size), dtype=torch.float8_e4m3fn, device=x.device
    )
    scale_count = num_tokens if per_token_quant else 1
    result_scale_shape = (num_tokens, 1) if per_token_quant and num_tokens > 1 else (1,)

    if dynamic_quant:
        # With one token, dynamic per-token and per-tensor scaling are
        # numerically equivalent. Return a 1-D scalar scale so downstream FP8
        # GEMMs can use their fused per-tensor scaling path.
        result_scale = torch.empty(
            result_scale_shape, dtype=torch.float32, device=x.device
        )
        if feature_tiles == 1:
            global_amax = result_scale.reshape(-1)
            completion_counter = torch.empty((1,), dtype=torch.int32, device=x.device)
        else:
            global_amax = torch.zeros(
                (scale_count,), dtype=torch.float32, device=x.device
            )
            completion_counter = torch.zeros((1,), dtype=torch.int32, device=x.device)
        supplied_scale = x_scale
    else:
        result_scale = torch.empty(
            result_scale_shape, dtype=torch.float32, device=x.device
        )
        global_amax = result_scale.reshape(-1)
        completion_counter = torch.empty((1,), dtype=torch.int32, device=x.device)
        supplied_scale = output_scale.reshape(-1)

    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    x_3d = x.unsqueeze(0)
    gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)
    gate_3d = gate_weight.unsqueeze(0)
    up_3d = up_weight.unsqueeze(0)
    output_3d = quantized.unsqueeze(0)

    key = (
        "sm100",
        quant_mode,
        x.device,
        hidden_size,
        intermediate_size,
        num_tokens,
        per_token_quant,
        tactic,
        tuple(x_scale.shape),
        tuple(gate_up_weight_scale.shape),
    )

    compiled = _COMPILED_DUAL_GEMM.get(key)
    arguments = (
        _cute_tensor_dynamic(x_3d),
        _cute_tensor_dynamic(gate_3d),
        _cute_tensor_dynamic(up_3d),
        _cute_tensor_dynamic(output_3d),
        _cute_tensor(x_scale),
        _cute_tensor(gate_up_weight_scale),
        _cute_tensor(result_scale.reshape(-1)),
        _cute_tensor(global_amax),
        _cute_tensor(supplied_scale),
        _cute_tensor(completion_counter),
        stream,
    )

    if compiled is None:
        kernel = BlackwellDualGemmKernel(
            num_tokens=num_tokens,
            intermediate_size=intermediate_size,
            element_type=cutlass.Float8E4M3FN,
            quantize_output=True,
            quant_mode=quant_mode,
            cta_features=cta_features,
            cta_tokens=cta_tokens,
            cta_reduction=cta_reduction,
            stages=stages,
            use_2cta=use_2cta,
        )
        compiled = cute_ext.compile(_dual_gemm_sm100_wrapper, kernel, *arguments)
        _COMPILED_DUAL_GEMM[key] = compiled

    compiled(*arguments)

    return quantized, result_scale


def _dual_gemm_swiglu_run(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    tactic: int = -1,
) -> torch.Tensor:
    if not is_sm100_supported():
        raise RuntimeError("CuTe DSL dual GEMM requires an SM10x GPU")

    num_tokens, hidden_size, intermediate_size = _validate_float16_inputs(
        x, gate_up_weight
    )
    multiprocessor_count = torch.cuda.get_device_properties(
        x.device
    ).multi_processor_count
    if tactic < 0:
        tactic = _pick_float16_tactic(
            num_tokens,
            hidden_size,
            intermediate_size,
            multiprocessor_count,
            x.dtype,
        )

    cta_features, cta_tokens, cta_reduction, stages, use_2cta = _resolve_tactic(
        tactic, False
    )
    if num_tokens > cta_tokens:
        raise ValueError(
            f"dual GEMM tactic {tactic} has {cta_tokens} token columns, "
            f"but the input contains {num_tokens} tokens"
        )
    element_type = cutlass.BFloat16 if x.dtype == torch.bfloat16 else cutlass.Float16
    output = torch.empty(
        (num_tokens, intermediate_size), dtype=x.dtype, device=x.device
    )
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    x_3d = x.unsqueeze(0)
    gate_weight, up_weight = gate_up_weight.chunk(2, dim=0)
    gate_3d = gate_weight.unsqueeze(0)
    up_3d = up_weight.unsqueeze(0)
    output_3d = output.unsqueeze(0)

    key = (
        "sm100",
        "unquantized",
        x.dtype,
        x.device,
        hidden_size,
        intermediate_size,
        num_tokens,
        tactic,
    )
    output_cute = _cute_tensor_dynamic(output_3d)
    arguments = (
        _cute_tensor_dynamic(x_3d),
        _cute_tensor_dynamic(gate_3d),
        _cute_tensor_dynamic(up_3d),
        output_cute,
        output_cute,
        output_cute,
        output_cute,
        output_cute,
        output_cute,
        output_cute,
        stream,
    )

    compiled = _COMPILED_DUAL_GEMM.get(key)
    if compiled is None:
        kernel = BlackwellDualGemmKernel(
            num_tokens=num_tokens,
            intermediate_size=intermediate_size,
            element_type=element_type,
            quantize_output=False,
            quant_mode=DualGemmQuantMode.UNQUANT,
            cta_features=cta_features,
            cta_tokens=cta_tokens,
            cta_reduction=cta_reduction,
            stages=stages,
            use_2cta=use_2cta,
        )
        compiled = cute_ext.compile(_dual_gemm_sm100_wrapper, kernel, *arguments)
        _COMPILED_DUAL_GEMM[key] = compiled

    compiled(*arguments)
    return output


def _dual_gemm_swiglu_fake(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    tactic: int = -1,
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], gate_up_weight.shape[0] // 2),
        dtype=x.dtype,
        device=x.device,
    )


def _dual_gemm_swiglu_fp8_fake(
    x,
    gate_up_weight,
    x_scale,
    gate_up_weight_scale,
    output_scale=None,
    quant_mode=int(DualGemmQuantMode.DYNAMIC_PER_TOKEN),
    tactic=-1,
):
    intermediate_size = gate_up_weight.shape[0] // 2
    per_token_quant = DualGemmQuantMode(quant_mode).is_per_token
    scale_shape = (x.shape[0], 1) if per_token_quant and x.shape[0] > 1 else (1,)

    return (
        torch.empty(
            (x.shape[0], intermediate_size),
            dtype=torch.float8_e4m3fn,
            device=x.device,
        ),
        torch.empty(scale_shape, dtype=torch.float32, device=x.device),
    )


direct_register_custom_op(
    op_name="cutedsl_dual_gemm_swiglu_fp8",
    op_func=_dual_gemm_swiglu_fp8_run,
    mutates_args=[],
    fake_impl=_dual_gemm_swiglu_fp8_fake,
)


direct_register_custom_op(
    op_name="cutedsl_dual_gemm_swiglu",
    op_func=_dual_gemm_swiglu_run,
    mutates_args=[],
    fake_impl=_dual_gemm_swiglu_fake,
)


def can_use_dual_gemm(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
) -> bool:
    """Return whether dimensions fit the small-batch dual GEMM contract.

    The modeling layer owns quantization-policy gating; this predicate checks
    both the SM100 requirement and the fused kernel's shape contract.
    """
    return (
        1 <= num_tokens <= MAX_DUAL_GEMM_DECODE_TOKENS
        and hidden_size > 0
        and 0 < intermediate_size <= MAX_DUAL_GEMM_INTERMEDIATE_SIZE
        and hidden_size % 128 == 0
        and intermediate_size % 128 == 0
        and is_sm100_supported()
    )


@debug_kernel_api
def dual_gemm_swiglu(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
) -> torch.Tensor:
    """Run BF16/FP16 gate/up projections followed by SwiGLU."""
    return _dual_gemm_swiglu_with_tactic(x, gate_up_weight, -1)


def _dual_gemm_swiglu_with_tactic(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    tactic: int,
) -> torch.Tensor:
    """Testing/tuning entry point; production callers use the auto picker."""
    return torch.ops.sglang.cutedsl_dual_gemm_swiglu(x, gate_up_weight, tactic)


@debug_kernel_api
def dual_gemm_swiglu_fp8(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    x_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    output_scale: Optional[torch.Tensor] = None,
    quant_mode: DualGemmQuantMode = DualGemmQuantMode.DYNAMIC_PER_TOKEN,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FP8 gate/up projections, SwiGLU, and static or dynamic FP8 quant.

    ``quant_mode`` independently selects static versus dynamic and per-tensor
    versus per-token scaling. Static modes require ``output_scale``; dynamic
    modes require it to be ``None``. A single-token result always uses shape
    ``(1,)`` so downstream FP8 GEMMs can consume it as a per-tensor scale.
    Shape eligibility is intentionally the responsibility of the caller.
    """
    return _dual_gemm_swiglu_fp8_with_tactic(
        x,
        gate_up_weight,
        x_scale,
        gate_up_weight_scale,
        output_scale,
        quant_mode,
        -1,
    )


def _dual_gemm_swiglu_fp8_with_tactic(
    x: torch.Tensor,
    gate_up_weight: torch.Tensor,
    x_scale: torch.Tensor,
    gate_up_weight_scale: torch.Tensor,
    output_scale: Optional[torch.Tensor],
    quant_mode: DualGemmQuantMode,
    tactic: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Testing/tuning entry point; production callers use the auto picker."""
    return torch.ops.sglang.cutedsl_dual_gemm_swiglu_fp8(
        x,
        gate_up_weight,
        x_scale,
        gate_up_weight_scale,
        output_scale,
        int(quant_mode),
        tactic,
    )
