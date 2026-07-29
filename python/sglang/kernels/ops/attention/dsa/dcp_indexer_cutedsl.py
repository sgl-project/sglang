# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""CUDA candidate packing and exact Top-K for owner-sharded DCP Indexer.

The selector's packed key makes the cutoff deterministic: higher scores win,
then lower global token IDs. The selected IDs are emitted with CTA atomics, so
their array order is deliberately unspecified.
"""

from functools import cache

import cutlass
import torch
import triton
import triton.language as tl
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Uint32, Uint64, cute
from quack.compile_utils import make_fake_tensor

from sglang.kernels.ops.attention.cute_utils import recast_val


def stable_topk_from_gathered_candidates_cutedsl(
    gathered: torch.Tensor,
    topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the exact stable-key Top-K set in unspecified array order."""
    if out is None:
        out = torch.empty(
            (gathered.shape[0], topk),
            dtype=torch.int32,
            device=gathered.device,
        )
    StableTopKFromGatheredCandidatesKernel.compile(topk, gathered.shape[1])(
        gathered, out
    )
    return out


def stable_topk_from_rank_major_candidates_cutedsl(
    rank_major_candidates: torch.Tensor,
    topk: int,
    out: torch.Tensor,
) -> None:
    """Select the exact Top-K set directly from rank-major peer mappings.

    The output order is unspecified; the score/ID key only defines membership
    at the Top-K cutoff.
    """
    if rank_major_candidates.ndim != 4 or rank_major_candidates.shape[-1] != 2:
        raise ValueError(
            "rank-major candidates must have shape "
            "[world, rows, local_candidates, 2]"
        )
    world_size = rank_major_candidates.shape[0]
    local_candidates = rank_major_candidates.shape[2]
    StableTopKFromRankMajorCandidatesKernel.compile(
        topk,
        world_size,
        local_candidates,
    )(rank_major_candidates, out)


def pack_dcp_topk_candidates_cutedsl(
    logits: torch.Tensor,
    topk_indices: torch.Tensor,
    packed: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    row_starts: torch.Tensor | None,
) -> None:
    topk = topk_indices.shape[1]
    grid = (topk_indices.shape[0], triton.cdiv(topk, 512))
    row_starts_arg = row_starts if row_starts is not None else topk_indices
    _pack_dcp_topk_candidates_triton_kernel[grid](
        logits,
        topk_indices,
        packed,
        row_starts_arg,
        logits.stride(0),
        logits.stride(1),
        topk_indices.stride(0),
        topk_indices.stride(1),
        packed.stride(0),
        packed.stride(1),
        packed.stride(2),
        logits.shape[1],
        dcp_rank=dcp_rank,
        dcp_world_size=dcp_world_size,
        has_row_starts=row_starts is not None,
        topk=topk,
        block_size=512,
        num_warps=8,
    )


@triton.jit
def _pack_dcp_topk_candidates_triton_kernel(
    logits,
    topk_indices,
    packed,
    row_starts,
    logits_stride0: tl.constexpr,
    logits_stride1: tl.constexpr,
    topk_stride0: tl.constexpr,
    topk_stride1: tl.constexpr,
    packed_stride0: tl.constexpr,
    packed_stride1: tl.constexpr,
    packed_stride2: tl.constexpr,
    num_cols,
    dcp_rank: tl.constexpr,
    dcp_world_size: tl.constexpr,
    has_row_starts: tl.constexpr,
    topk: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    columns = tile * block_size + tl.arange(0, block_size)
    mask = columns < topk

    local_idx = tl.load(
        topk_indices + row * topk_stride0 + columns * topk_stride1,
        mask=mask,
        other=-1,
    )
    valid = local_idx >= 0
    safe_local_idx = tl.maximum(local_idx, 0)

    row_start = 0
    if has_row_starts:
        row_start = tl.load(row_starts + row)
    score_column = safe_local_idx + row_start
    score_column = tl.minimum(score_column, tl.maximum(num_cols - 1, 0))
    score = tl.load(
        logits + row * logits_stride0 + score_column * logits_stride1,
        mask=mask & valid,
        other=-float("inf"),
    )

    global_id = safe_local_idx * dcp_world_size + dcp_rank
    global_id = tl.where(valid, global_id, -1).to(tl.int32)

    packed_base = packed + row * packed_stride0 + columns * packed_stride1
    tl.store(packed_base, score, mask=mask)
    tl.store(
        packed_base + packed_stride2,
        global_id.to(tl.float32, bitcast=True),
        mask=mask,
    )


@cute.jit
def _warp_scan_inclusive_i32(value: Int32, lane: Int32) -> Int32:
    for i in cutlass.range_constexpr(cute.arch.WARP_SIZE.bit_length() - 1):
        offset = 1 << i
        partial = cute.arch.shuffle_sync_up(value, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            value += partial
    return value


@cute.jit
def _block_scan_inclusive_i32(
    value: Int32,
    lane: Int32,
    warp_id: Int32,
    warp_scratch: cute.Tensor,
    warps_per_block: int,
) -> Int32:
    prefix = _warp_scan_inclusive_i32(value, lane)
    if lane == Int32(cute.arch.WARP_SIZE - 1):
        warp_scratch[0, warp_id] = prefix
    cute.arch.sync_threads()

    if warp_id == Int32(0):
        warp_total = Int32(0)
        if lane < Int32(warps_per_block):
            warp_total = warp_scratch[0, lane]
        warp_prefix = _warp_scan_inclusive_i32(warp_total, lane)
        if lane < Int32(warps_per_block):
            warp_scratch[0, lane] = warp_prefix - warp_total
    cute.arch.sync_threads()

    return prefix + warp_scratch[0, warp_id]


class StableTopKFromGatheredCandidatesKernel:
    tb_size = 512
    hist_bins = 2048
    radix_bits = (hist_bins - 1).bit_length()
    assert hist_bins == 1 << radix_bits
    key_bits = Uint64.width
    radix_passes = (key_bits + radix_bits - 1) // radix_bits
    final_radix_bits = key_bits - radix_bits * (radix_passes - 1)
    hist_chunks = (hist_bins + tb_size - 1) // tb_size
    warps_per_block = tb_size // cute.arch.WARP_SIZE

    def __init__(self, topk: int, num_candidates: int):
        assert num_candidates % self.tb_size == 0, (
            "stable DCP Top-K requires the candidate count to be a multiple "
            f"of {self.tb_size}, got {num_candidates}"
        )
        self.topk = topk
        self.keys_per_thread = num_candidates // self.tb_size

        @cute.struct
        class SharedStorage:
            hist: cute.struct.MemRange[Int32, self.hist_bins]
            committed_count: cute.struct.MemRange[Int32, 1]
            running_count: cute.struct.MemRange[Int32, 1]
            threshold_bin: cute.struct.MemRange[Int32, 1]
            threshold_found: cute.struct.MemRange[Int32, 1]
            include_threshold_bin: cute.struct.MemRange[Int32, 1]
            prefix_s: cute.struct.Align[cute.struct.MemRange[Uint64, 1], 8]
            warp_totals: cute.struct.MemRange[Int32, self.warps_per_block]

        self.shared_storage = SharedStorage

    @cute.jit
    def __call__(
        self,
        gathered: cute.Tensor,
        out: cute.Tensor,
        stream: CUstream,
    ):
        grid = (gathered.shape[0], 1, 1)
        self.kernel(gathered, out).launch(
            grid=grid,
            block=(self.tb_size, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _stable_key(self, score: Float32, token_id: Int32) -> Uint64:
        bits = recast_val(score, Uint32)
        mask = Uint32(0x80000000)
        if (bits & Uint32(0x80000000)) != Uint32(0):
            mask = Uint32(0xFFFFFFFF)
        score_key = Uint64(bits ^ mask) << Uint64(32)
        id_key = Uint64(~Uint32(token_id))
        key = score_key | id_key
        if token_id < Int32(0):
            key = Uint64(0)
        return key

    @cute.jit
    def _prefix_matches(
        self,
        key: Uint64,
        prefix: Uint64,
        prefix_bits: Int32,
    ):
        matches = prefix_bits == Int32(0)
        if prefix_bits != Int32(0):
            shift = Int32(self.key_bits) - prefix_bits
            matches = (key >> Uint64(shift)) == (prefix >> Uint64(shift))
        return matches

    @cute.jit
    def _radix_pass(
        self,
        keys: cute.Tensor,
        output: cute.Tensor,
        storage,
        thread_id: Int32,
        step: Int32,
        bits: int,
        is_final_pass: bool,
    ):
        histogram = storage.hist.get_tensor(cute.make_layout((self.hist_bins,)))
        committed_count = storage.committed_count.data_ptr()
        running_count_ptr = storage.running_count.data_ptr()
        threshold_bin_ptr = storage.threshold_bin.data_ptr()
        threshold_found_ptr = storage.threshold_found.data_ptr()
        include_threshold_ptr = storage.include_threshold_bin.data_ptr()
        prefix_ptr = storage.prefix_s.data_ptr()
        warp_totals = storage.warp_totals.get_tensor(
            cute.make_layout((1, self.warps_per_block))
        )

        prefix_bits = step * Int32(self.radix_bits)
        num_bins = 1 << bits
        block_scan_iterations = (num_bins + self.tb_size - 1) // self.tb_size
        shift = Int32(self.key_bits) - prefix_bits - Int32(bits)
        bin_mask = Uint64(num_bins - 1)
        prefix = prefix_ptr.load()

        for chunk in cutlass.range_constexpr(self.hist_chunks):
            histogram[thread_id + Int32(chunk * self.tb_size)] = Int32(0)
        if thread_id == Int32(0):
            running_count_ptr.store(committed_count.load())
            include_threshold_ptr.store(Int32(0))
            threshold_found_ptr.store(Int32(0))
        cute.arch.sync_threads()

        for key_index in cutlass.range_constexpr(self.keys_per_thread):
            key = keys[key_index]
            if self._prefix_matches(key, prefix, prefix_bits):
                bin_index = Int32((key >> Uint64(shift)) & bin_mask)
                cute.arch.atomic_add(
                    histogram.iterator + bin_index,
                    Int32(1),
                    sem="relaxed",
                    scope="cta",
                )
        cute.arch.sync_threads()

        lane = cute.arch.lane_idx()
        warp_id = cute.arch.warp_idx()
        iteration = Int32(0)
        threshold_found = threshold_found_ptr.load()
        while threshold_found == Int32(0) and iteration < Int32(block_scan_iterations):
            bin_index = Int32(num_bins - 1) - (
                iteration * Int32(self.tb_size) + thread_id
            )
            count = histogram[bin_index]
            chunk_inclusive = _block_scan_inclusive_i32(
                count,
                lane,
                warp_id,
                warp_totals,
                self.warps_per_block,
            )
            running_count = running_count_ptr.load()
            prior_in_scan_slice = chunk_inclusive - count
            remaining = Int32(self.topk) - running_count - prior_in_scan_slice
            if count > Int32(0) and remaining > Int32(0) and remaining <= count:
                threshold_bin_ptr.store(bin_index)
                if count <= remaining or cutlass.const_expr(is_final_pass):
                    include_threshold_ptr.store(Int32(1))
                threshold_found_ptr.store(Int32(1))
            cute.arch.sync_threads()
            if thread_id == Int32(self.tb_size - 1):
                running_count_ptr.store(running_count + chunk_inclusive)
            cute.arch.sync_threads()

            threshold_found = threshold_found_ptr.load()
            iteration += Int32(1)

        threshold = threshold_bin_ptr.load()
        should_include_threshold = include_threshold_ptr.load() != Int32(0)
        for key_index in cutlass.range_constexpr(self.keys_per_thread):
            key = keys[key_index]
            if self._prefix_matches(key, prefix, prefix_bits):
                bin_index = Int32((key >> Uint64(shift)) & bin_mask)
                selected = bin_index > threshold
                if should_include_threshold:
                    selected = selected or bin_index == threshold
                if selected:
                    destination = cute.arch.atomic_add(
                        committed_count,
                        Int32(1),
                        sem="relaxed",
                        scope="cta",
                    )
                    if destination < Int32(self.topk):
                        output[destination] = recast_val(~Uint32(key), Int32)
        cute.arch.sync_threads()

        pass_finished = include_threshold_ptr.load()
        if thread_id == Int32(0) and pass_finished == Int32(0):
            prefix_ptr.store(prefix | (Uint64(threshold) << Uint64(shift)))
        cute.arch.sync_threads()
        return pass_finished

    @cute.kernel
    def kernel(
        self,
        input_tensor: cute.Tensor,
        output: cute.Tensor,
    ):
        row, _, _ = cute.arch.block_idx()
        thread_id, _, _ = cute.arch.thread_idx()
        input_row = input_tensor[row, None, None]
        output_row = output[row, None]
        keys = cute.make_rmem_tensor((self.keys_per_thread,), Uint64)

        shared_allocator = cutlass.utils.SmemAllocator()
        storage = shared_allocator.allocate(self.shared_storage, 8)
        committed_count = storage.committed_count.data_ptr()
        prefix_ptr = storage.prefix_s.data_ptr()
        for index in range(thread_id, self.topk, self.tb_size):
            output_row[index] = Int32(-1)

        for key_index in cutlass.range_constexpr(self.keys_per_thread):
            column = thread_id + Int32(key_index * self.tb_size)
            score = Float32(input_row[column, 0])
            token_id = recast_val(input_row[column, 1], Int32)
            keys[key_index] = self._stable_key(score, token_id)

        if thread_id == Int32(0):
            committed_count.store(Int32(0))
            prefix_ptr.store(Uint64(0))
        cute.arch.sync_threads()

        step = Int32(0)
        finished = Int32(0)
        while finished == Int32(0) and step < Int32(self.radix_passes - 1):
            finished = self._radix_pass(
                keys,
                output_row,
                storage,
                thread_id,
                step,
                self.radix_bits,
                False,
            )
            step += Int32(1)

        if finished == Int32(0):
            self._radix_pass(
                keys,
                output_row,
                storage,
                thread_id,
                Int32(self.radix_passes - 1),
                self.final_radix_bits,
                True,
            )

    @cache
    @staticmethod
    def compile(topk: int, num_candidates: int):
        num_rows = cute.sym_int()

        gathered = cute.runtime.make_fake_tensor(
            Float32,
            (num_rows, num_candidates, 2),
            stride=(cute.sym_int64(divisibility=2), 2, 1),
            assumed_align=8,
        )
        output = make_fake_tensor(Int32, (num_rows, topk), divisibility=1)

        kernel = StableTopKFromGatheredCandidatesKernel(topk, num_candidates)
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            gathered,
            output,
            stream,
            options="--enable-tvm-ffi",
        )


class StableTopKFromRankMajorCandidatesKernel(StableTopKFromGatheredCandidatesKernel):
    """Exact stable-key Top-K set over VMM owner segments."""

    def __init__(self, topk: int, world_size: int, local_candidates: int):
        super().__init__(topk, world_size * local_candidates)
        self.world_size = world_size
        self.local_candidates = local_candidates

    @cute.jit
    def __call__(
        self,
        rank_major_candidates: cute.Tensor,
        output: cute.Tensor,
        stream: CUstream,
    ):
        grid = (rank_major_candidates.shape[1], 1, 1)
        self.kernel(rank_major_candidates, output).launch(
            grid=grid,
            block=(self.tb_size, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        input_tensor: cute.Tensor,
        output: cute.Tensor,
    ):
        row, _, _ = cute.arch.block_idx()
        thread_id, _, _ = cute.arch.thread_idx()
        output_row = output[row, None]
        keys = cute.make_rmem_tensor((self.keys_per_thread,), Uint64)

        shared_allocator = cutlass.utils.SmemAllocator()
        storage = shared_allocator.allocate(self.shared_storage, 8)
        committed_count = storage.committed_count.data_ptr()
        prefix_ptr = storage.prefix_s.data_ptr()
        for index in range(thread_id, self.topk, self.tb_size):
            output_row[index] = Int32(-1)

        for key_index in cutlass.range_constexpr(self.keys_per_thread):
            column = thread_id + Int32(key_index * self.tb_size)
            owner = column // Int32(self.local_candidates)
            local_column = column - owner * Int32(self.local_candidates)
            score = Float32(input_tensor[owner, row, local_column, 0])
            token_id = recast_val(input_tensor[owner, row, local_column, 1], Int32)
            keys[key_index] = self._stable_key(score, token_id)

        if thread_id == Int32(0):
            committed_count.store(Int32(0))
            prefix_ptr.store(Uint64(0))
        cute.arch.sync_threads()

        step = Int32(0)
        finished = Int32(0)
        while finished == Int32(0) and step < Int32(self.radix_passes - 1):
            finished = self._radix_pass(
                keys,
                output_row,
                storage,
                thread_id,
                step,
                self.radix_bits,
                False,
            )
            step += Int32(1)

        if finished == Int32(0):
            self._radix_pass(
                keys,
                output_row,
                storage,
                thread_id,
                Int32(self.radix_passes - 1),
                self.final_radix_bits,
                True,
            )

    @cache
    @staticmethod
    def compile(topk: int, world_size: int, local_candidates: int):
        num_rows = cute.sym_int()
        rank_stride = cute.sym_int64(divisibility=2)

        rank_major = cute.runtime.make_fake_tensor(
            Float32,
            (world_size, num_rows, local_candidates, 2),
            stride=(rank_stride, local_candidates * 2, 2, 1),
            assumed_align=8,
        )
        output = make_fake_tensor(Int32, (num_rows, topk), divisibility=1)

        kernel = StableTopKFromRankMajorCandidatesKernel(
            topk,
            world_size,
            local_candidates,
        )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        return cute.compile(
            kernel,
            rank_major,
            output,
            stream,
            options="--enable-tvm-ffi",
        )
