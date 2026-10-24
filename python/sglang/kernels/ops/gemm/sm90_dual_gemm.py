"""SM90 TMA/WGMMA dual projection with a fused, rounded SwiGLU epilogue.

Weights are stored as [gate; up], as in the Blackwell dual GEMM. Each CTA
computes 64 features for up to 16 decode tokens. A producer warpgroup
loads gate, up, and one shared X tile into a multistage TMA pipeline. A second
warpgroup issues both WGMMA projections and stores SwiGLU directly from two
FP32 register accumulators. No projection intermediate is written to GMEM.

PipelineTmaAsync coordinates the TMA transactions and stage ownership;
one committed WGMMA group stays in flight while the next stage is consumed.
The final wait precedes releasing the last stage and reading accumulators.
FP8, bias, LoRA, and automatic model dispatch are outside this operator.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from sglang.srt.utils.common import direct_register_custom_op

MAX_SM90_DUAL_GEMM_TOKENS = 16
# (features, reduction, stages). Exposed for offline performance selection;
# benchmark all tactics before admitting a shape to the model fast path.
# Consecutive one-based IDs, local to the Hopper backend.
SM90_DUAL_GEMM_TACTICS = {
    1: (64, 128, 3),
    2: (64, 64, 5),
}
_COMPILED = {}


def _pick_sm90_tactic(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    multiprocessor_count: int,
    input_dtype: torch.dtype,
) -> int:
    """Select a shape heuristic calibrated on H100 PCIe measurements."""
    # The K=64/stage-5 pipeline won at K=2048..4096 and I=3584/7168,
    # while K=128/stage-3 won for wider grids and K=8192. Use the short
    # K tile when the 64-feature grid fits in one wave of SMs. This extends
    # the measured trend to nearby shapes; it is not a measured optimum
    # for every shape in the region. Other SM counts remain uncalibrated.
    if (
        multiprocessor_count != 114
        or input_dtype not in (torch.bfloat16, torch.float16)
        or not 1 <= num_tokens <= MAX_SM90_DUAL_GEMM_TOKENS
        or intermediate_size <= 0
    ):
        return 1
    feature_tiles = (intermediate_size + 63) // 64
    if 2048 <= hidden_size <= 4096 and feature_tiles <= multiprocessor_count:
        return 2
    return 1


class SM90DualGemmKernel:
    """One TMA producer warpgroup and one WGMMA consumer warpgroup per CTA."""

    def __init__(self, num_tokens: int, tactic: int):
        self.features, self.reduction, self.stages = SM90_DUAL_GEMM_TACTICS[tactic]
        self.tokens = 8 if num_tokens <= 8 else 16

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        gate: cute.Tensor,
        up: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        dtype = x.element_type
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            dtype,
            dtype,
            warpgroup.OperandMajorMode.K,
            warpgroup.OperandMajorMode.K,
            cutlass.Float32,
            (1, 1, 1),
            tiler_mn=(64, self.tokens),
        )
        atom = warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                utils.LayoutEnum.ROW_MAJOR, dtype, self.reduction
            ),
            dtype,
        )
        weight_layout = cute.tile_to_shape(
            atom, (self.features, self.reduction, self.stages), order=(0, 1, 2)
        )
        input_layout = cute.tile_to_shape(
            atom, (self.tokens, self.reduction, self.stages), order=(0, 1, 2)
        )
        gate_atom, gate_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            gate,
            cute.slice_(weight_layout, (None, None, 0)),
            (self.features, self.reduction),
        )
        up_atom, up_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            up,
            cute.slice_(weight_layout, (None, None, 0)),
            (self.features, self.reduction),
        )
        x_atom, x_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            x,
            cute.slice_(input_layout, (None, None, 0)),
            (self.tokens, self.reduction),
        )

        @cute.struct
        class SharedStorage:
            barriers: cute.struct.MemRange[cutlass.Int64, self.stages * 2]
            gate: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(weight_layout)], 1024
            ]
            up: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(weight_layout)], 1024
            ]
            x: cute.struct.Align[
                cute.struct.MemRange[dtype, cute.cosize(input_layout)], 1024
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            gate_atom,
            gate_tma,
            up_atom,
            up_tma,
            x_atom,
            x_tma,
            output,
            tiled_mma,
            weight_layout,
            input_layout,
        ).launch(
            grid=(cute.ceil_div(output.shape[1], self.features), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gate_atom: cute.CopyAtom,
        gate_tma: cute.Tensor,
        up_atom: cute.CopyAtom,
        up_tma: cute.Tensor,
        x_atom: cute.CopyAtom,
        x_tma: cute.Tensor,
        output: cute.Tensor,
        tiled_mma: cute.TiledMma,
        weight_layout: cute.ComposedLayout,
        input_layout: cute.ComposedLayout,
    ):
        tid, _, _ = cute.arch.thread_idx()
        feature_tile, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // 32)
        wg = cute.arch.make_warp_uniform(tid // 128)
        if warp == 0:
            cpasync.prefetch_descriptor(gate_atom)
            cpasync.prefetch_descriptor(up_atom)
            cpasync.prefetch_descriptor(x_atom)

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        sg = storage.gate.get_tensor(weight_layout.outer, swizzle=weight_layout.inner)
        su = storage.up.get_tensor(weight_layout.outer, swizzle=weight_layout.inner)
        sx = storage.x.get_tensor(input_layout.outer, swizzle=input_layout.inner)
        tx_bytes = (2 * self.features + self.tokens) * self.reduction * 2
        mainloop = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.barriers.data_ptr(),
            num_stages=self.stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            # consumer_release elects one lane per consumer warp: four arrivals.
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 4),
            tx_count=tx_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        gg = cute.local_tile(
            gate_tma, (self.features, self.reduction), (feature_tile, None)
        )
        gu = cute.local_tile(
            up_tma, (self.features, self.reduction), (feature_tile, None)
        )
        gx = cute.local_tile(x_tma, (self.tokens, self.reduction), (0, None))
        tsg, tgg = cpasync.tma_partition(
            gate_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sg, 0, 2),
            cute.group_modes(gg, 0, 2),
        )
        tsu, tgu = cpasync.tma_partition(
            up_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(su, 0, 2),
            cute.group_modes(gu, 0, 2),
        )
        tsx, tgx = cpasync.tma_partition(
            x_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sx, 0, 2),
            cute.group_modes(gx, 0, 2),
        )
        k_tiles = cute.size(gg, mode=[2])
        pipeline_init_wait(cluster_shape_mn=(1, 1))

        if wg == 0:
            cute.arch.warpgroup_reg_dealloc(40)
            if warp == 0:
                producer = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.stages
                )
                for k in range(k_tiles):
                    mainloop.producer_acquire(producer)
                    barrier = mainloop.producer_get_barrier(producer)
                    cute.copy(
                        gate_atom,
                        tgg[None, k],
                        tsg[None, producer.index],
                        tma_bar_ptr=barrier,
                    )
                    cute.copy(
                        up_atom,
                        tgu[None, k],
                        tsu[None, producer.index],
                        tma_bar_ptr=barrier,
                    )
                    # The same X stage feeds both projections.
                    cute.copy(
                        x_atom,
                        tgx[None, k],
                        tsx[None, producer.index],
                        tma_bar_ptr=barrier,
                    )
                    mainloop.producer_commit(producer)
                    producer.advance()
                mainloop.producer_tail(producer)
        else:
            cute.arch.warpgroup_reg_alloc(128)
            # Descriptors are warpgroup-uniform; output coordinates are per lane.
            mma_wg = tiled_mma.get_slice(0)
            ag = tiled_mma.make_fragment_A(mma_wg.partition_A(sg))
            au = tiled_mma.make_fragment_A(mma_wg.partition_A(su))
            bx = tiled_mma.make_fragment_B(mma_wg.partition_B(sx))
            coordinates = tiled_mma.get_slice(tid - 128).partition_C(
                cute.make_identity_tensor((self.features, self.tokens))
            )
            gate_acc = cute.make_rmem_tensor(coordinates.shape, cutlass.Float32)
            up_acc = cute.make_rmem_tensor(coordinates.shape, cutlass.Float32)
            gate_acc.fill(0.0)
            up_acc.fill(0.0)
            tiled_mma.set(warpgroup.Field.ACCUMULATE, True)
            read = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.stages
            )
            release = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.stages
            )
            warpgroup.fence()
            for k in range(k_tiles):
                mainloop.consumer_wait(read)
                for kb in cutlass.range_constexpr(cute.size(ag, mode=[2])):
                    coord = (None, None, kb, read.index)
                    cute.gemm(tiled_mma, gate_acc, ag[coord], bx[coord], gate_acc)
                    cute.gemm(tiled_mma, up_acc, au[coord], bx[coord], up_acc)
                warpgroup.commit_group()
                if k > 0:
                    # Both projections share one group; stage k-1 is reusable
                    # only when that entire group, including up, has completed.
                    warpgroup.wait_group(1)
                    mainloop.consumer_release(release)
                    release.advance()
                read.advance()
            warpgroup.wait_group(0)
            mainloop.consumer_release(release)
            warpgroup.fence()

            for i in cutlass.range_constexpr(cute.size(gate_acc)):
                feature = feature_tile * self.features + coordinates[i][0]
                token = coordinates[i][1]
                if feature < output.shape[1] and token < output.shape[0]:
                    # Preserve the unfused GEMM storage boundary before SwiGLU,
                    # matching the Blackwell BF16/FP16 operator semantics.
                    g = gate_acc[i].to(output.element_type).to(cutlass.Float32)
                    u = up_acc[i].to(output.element_type).to(cutlass.Float32)
                    y = (
                        g
                        * cute.arch.rcp_approx(
                            cutlass.Float32(1.0) + cute.math.exp(-g, fastmath=True)
                        )
                        * u
                    )
                    output[token, feature] = y.to(output.element_type)


def _validate(x, weight):
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("x and gate_up_weight must be matrices")
    if x.device.type != "cuda" or weight.device != x.device:
        raise ValueError("inputs must be on the same CUDA device")
    if torch.cuda.get_device_capability(x.device) != (9, 0):
        raise ValueError("Hopper DualGEMM requires SM90")
    if x.dtype not in (torch.float16, torch.bfloat16) or weight.dtype != x.dtype:
        raise ValueError("inputs must have the same FP16 or BF16 dtype")
    if x.requires_grad or weight.requires_grad:
        raise ValueError("Hopper DualGEMM is inference-only")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("inputs must be contiguous")
    if x.data_ptr() % 16 or weight.data_ptr() % 16:
        raise ValueError("TMA inputs must be 16-byte aligned")
    m, k = x.shape
    if not 0 <= m <= MAX_SM90_DUAL_GEMM_TOKENS:
        raise ValueError("Hopper DualGEMM supports at most 16 decode tokens")
    if k <= 0 or k % 64 or weight.shape[1] != k:
        raise ValueError("K must match and be a positive multiple of 64")
    if weight.shape[0] == 0 or weight.shape[0] % 2:
        raise ValueError("weight must contain equal nonempty gate/up rows")


def _tma_tensor(tensor: torch.Tensor):
    # A static singleton row is canonicalized to stride zero and removed from
    # the TMA map. Keep the row extent dynamic so M=1 (or I=1 for weights)
    # still uses a 2D tile with hardware zero-fill for the remaining rows.
    # K stays static, preserving the reduction-loop specialization.
    return from_dlpack(tensor.detach(), assumed_align=16).mark_compact_shape_dynamic(
        mode=0, stride_order=(0, 1)
    )


def _run(
    x: torch.Tensor, gate_up_weight: torch.Tensor, tactic: int = -1
) -> torch.Tensor:
    _validate(x, gate_up_weight)
    m, k = x.shape
    n = gate_up_weight.shape[0] // 2
    if tactic == -1:
        tactic = _pick_sm90_tactic(
            m,
            k,
            n,
            torch.cuda.get_device_properties(x.device).multi_processor_count,
            x.dtype,
        )
    if tactic not in SM90_DUAL_GEMM_TACTICS:
        raise ValueError("invalid Hopper DualGEMM tactic")
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if m == 0:
        return out
    with torch.cuda.device(x.device):
        stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
        arguments = (
            *(_tma_tensor(t) for t in (x, *gate_up_weight.chunk(2, dim=0))),
            from_dlpack(out.detach(), assumed_align=16),
            stream,
        )
        key = (x.device, x.dtype, m, n, k, tactic)
        compiled = _COMPILED.get(key)
        if compiled is None:
            compiled = cute.compile(
                SM90DualGemmKernel(m, tactic),
                *arguments,
                options="--gpu-arch=sm_90a",
            )
            _COMPILED[key] = compiled
        compiled(*arguments)
    return out


def _fake(
    x: torch.Tensor, gate_up_weight: torch.Tensor, tactic: int = -1
) -> torch.Tensor:
    return torch.empty(
        (x.shape[0], gate_up_weight.shape[0] // 2), device=x.device, dtype=x.dtype
    )


direct_register_custom_op(
    op_name="cutedsl_sm90_dual_gemm_swiglu",
    op_func=_run,
    mutates_args=[],
    fake_impl=_fake,
)


def sm90_dual_gemm_swiglu(
    x: torch.Tensor, gate_up_weight: torch.Tensor, tactic: int = -1
) -> torch.Tensor:
    """FP16/BF16 decode DualGEMM; -1 selects automatically, 1/2 override."""
    return torch.ops.sglang.cutedsl_sm90_dual_gemm_swiglu(x, gate_up_weight, tactic)
