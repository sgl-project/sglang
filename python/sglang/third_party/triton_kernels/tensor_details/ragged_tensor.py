from dataclasses import dataclass
import triton
import torch
import triton.language as tl
from ..target_info import is_hip

# ---------------------------------------------------------------------------- #
# metadata
# ---------------------------------------------------------------------------- #


@dataclass
class RaggedTensorMetadata:
    """
    Example:
    `slice_sizes`= [15 17 0 127]
    `slice_offs`= [0 15 32 32 332]
    `block_offs_data` = {
        16: [0 1 3 3 11]
        32: [0 1 2 2 6]
        64: [0 1 2 2 4]
        128: [0 1 2 2 3]
    }
    `block_schedule_data` = {
        16:  [(0, 0) (0, 1) (0, 3) (1, 3) (2, 3) ... (7, 3) -1 ... -1]
        32:  [(0, 0) (0, 1) (0, 3) (1, 3) (2, 3) (3, 3) -1 ...     -1]
        64:  [(0, 0) (0, 1) (0, 3) (1, 3) (2, 3) -1 ...            -1]
        128: [(0, 0) (0, 1) (0, 3) (1, 3) -1 ...                   -1]
    }
    """
    # slice_sizes[i] is the number of elements in slice i along the ragged dimension
    slice_sizes: torch.Tensor
    # slice_offs = [0] + cumsum(slice_sizes)
    # i.e., slice_offs[i] is the offset of the first element in slice `i`
    slice_offs: torch.Tensor
    # block_offs_data[k] = [0] + cumsum(ceil_div(slice_sizes, 16 * k))
    # i.e., `block_offs_data[k][i]` is the offset of the first block of
    # `16*k`` token for batch `i` in a `bath_sizes`-shaped ragged tensor
    block_offs_data: torch.Tensor
    # let `num_blocks[k] = block_offs_data[k, 1:] - block_offs_data[k, :-1]
    # block_schedule_data[k] = cat(*[[(batch, blk) for blk in range(blks)] for batch, blks in enumerate(num_blocks)])
    # i.e., if the schedule of batch `i` is [(i, 0), (i, 1), ..., (i, num_blocks[k][i] - 1)]
    # then `block_schedule_data[k]` is the concatenation of the schedules for all batches
    # NOTE 1: `block_schedule_data[k][j]` is a packed 32-bit integer
    # NOTE 2: because the size of `block_schedule_data[k]` is data-dependent, we pad it with -1s
    # up to an user-provided upper bound
    block_schedule_data: torch.Tensor

    def __post_init__(self):
        assert self.block_offs_data.shape[0] == len(RaggedTensorMetadata.block_sizes())
        assert self.block_schedule_data.shape[0] == len(RaggedTensorMetadata.block_sizes())
        assert self.block_offs_data.dtype == torch.int32
        assert self.block_schedule_data.dtype == torch.int32
        if self.slice_sizes is not None:
            assert self.slice_sizes.dtype == torch.int32
        if self.slice_offs is not None:
            assert self.slice_offs.dtype == torch.int32

    def block_offs(self, block_size):
        return self.block_offs_data[RaggedTensorMetadata.block_sizes().index(block_size)]

    def block_schedule(self, block_size):
        return self.block_schedule_data[RaggedTensorMetadata.block_sizes().index(block_size)]

    @staticmethod
    def max_n_tiles(n_slices, n_total_rows):
        if n_total_rows <= n_slices:
            return n_total_rows
        return n_slices - 1 - ((n_slices - n_total_rows - 1) // min(RaggedTensorMetadata.block_sizes()))

    @staticmethod
    def block_sizes_log2():
        return range(4, 9) if is_hip() else range(4, 8)

    @staticmethod
    def block_sizes():
        return [2**x for x in RaggedTensorMetadata.block_sizes_log2()]


# utilities
# --------------------------------------------------------- #


def exact_div(x, y):
    assert x % y == 0
    return x // y


def empty_aligned(shape, dtype, device, pad_size):
    cdiv = lambda x, y: (x + y - 1) // y
    pad = lambda x: cdiv(x, pad_size) * pad_size
    ret = torch.empty((*shape[:-1], pad(shape[-1])), dtype=dtype, device=device)
    ret_slices = (*[slice(None)] * (len(shape) - 1), slice(0, shape[-1]))
    return ret[ret_slices], ret.numel()


# ============================================================================ #
# make_ragged_tensor_metadata
# ============================================================================ #

# optimized implementation
# ---------------------------------------------------------------------------- #


@triton.jit
def _cdiv_pow2(n, log2_k):
    # ceil_div(n, 2**log2_k)
    return (n + ((1 << log2_k) - 1)) >> log2_k


@triton.jit
def _ragged_tensor_metadata_memset(SliceSizes, n_slices, BlockOffs, slice_offs_stride_m, BlockSchedule,
                                   first_block_size_log2, SIZES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    if pid <= SIZES:
        BlockOffs += pid * slice_offs_stride_m
        BlockOffsPtrs = BlockOffs + tl.arange(0, BLOCK)
        block_size_log2 = tl.where(pid == 0, 0, pid + first_block_size_log2 - 1)
        # total number of blocks in slice processed as the loop iterates
        n_blocks_tot = tl.zeros([BLOCK], dtype=BlockOffs.dtype.element_ty)
        for i in range(0, n_slices + 1, BLOCK):
            # load slice sizes
            offs = tl.arange(0, BLOCK) + i
            mask = offs < n_slices
            slice_sizes = tl.load(SliceSizes + offs, mask=mask, other=0)
            # number of blocks in the slices loaded
            n_blocks = _cdiv_pow2(slice_sizes, block_size_log2)
            # start index of the blocks for the slices loaded
            block_starts = tl.cumsum(n_blocks, 0) + n_blocks_tot
            n_blocks_tot += tl.sum(n_blocks, 0)
            tl.store(BlockOffsPtrs, block_starts - n_blocks)
            BlockOffsPtrs += BLOCK
    else:
        # initialize block schedule to -1
        pid -= (SIZES + 1)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        tl.store(BlockSchedule + offs, 0xffffffff)


@triton.jit
def _ragged_tensor_metadata_compute(SliceSizes,  #
                                    BlockOffs, block_offs_stride_m,  #
                                    BlockSchedule, block_schedule_stride_m,  #
                                    first_block_size_log2,  #
                                    SIZES: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    slice_id = pid // SIZES
    block_size_id = pid % SIZES
    # offset pointers
    BlockOffs += block_size_id * block_offs_stride_m
    BlockSchedule += block_size_id * block_schedule_stride_m
    # load slice sizes
    slice_sizes = tl.load(SliceSizes + slice_id)
    # number of blocks in the slices loaded
    block_size_log2 = first_block_size_log2 + block_size_id
    n_blocks = _cdiv_pow2(slice_sizes, block_size_log2)
    # compute block schedule
    block_off = tl.load(BlockOffs + slice_id)
    BlockSchedule += block_off
    for block_off in range(0, n_blocks, BLOCK):
        block_offs = block_off + tl.arange(0, BLOCK)
        data = (block_offs << 16) + slice_id
        tl.store(BlockSchedule + block_offs, data, mask=block_offs < n_blocks)


def make_ragged_tensor_metadata(slice_sizes, n_total_rows):
    assert slice_sizes.ndim == 1
    n_slices = slice_sizes.shape[0]
    block_sizes_log2 = RaggedTensorMetadata.block_sizes_log2()
    block_size_num = len(block_sizes_log2)
    MEMSET_BLOCK = 512
    dtype = torch.int32
    device = slice_sizes.device
    max_n_blocks = RaggedTensorMetadata.max_n_tiles(n_slices, n_total_rows)
    slice_offs_combined, _ = empty_aligned((block_size_num + 1, n_slices + 1), dtype, device, MEMSET_BLOCK)
    block_schedule_data, n_memset_elts = empty_aligned((block_size_num, max_n_blocks), dtype, device, MEMSET_BLOCK)
    slice_offs, block_offs_data = slice_offs_combined[0], slice_offs_combined[1:]
    n_memset_blocks = exact_div(n_memset_elts, MEMSET_BLOCK)

    _ragged_tensor_metadata_memset[(slice_offs_combined.shape[0] + n_memset_blocks, )](
        slice_sizes, n_slices,  #
        slice_offs_combined, slice_offs_combined.stride(0),  #
        block_schedule_data,  #
        block_sizes_log2[0], SIZES=len(block_sizes_log2), BLOCK=MEMSET_BLOCK,  # optimization parameters
        num_warps=4)

    _ragged_tensor_metadata_compute[(block_size_num * n_slices, )](
        slice_sizes, block_offs_data, block_offs_data.stride(0), block_schedule_data,
        block_schedule_data.stride(0),  # outputs
        block_sizes_log2[0], SIZES=len(block_sizes_log2), BLOCK=512,  # optimization parameters
        num_warps=4)

    return RaggedTensorMetadata(slice_sizes, slice_offs, block_offs_data, block_schedule_data)


# reference implementation
# ---------------------------------------------------------------------------- #


def make_ragged_tensor_metadata_torch(slice_sizes, n_total_rows):
    assert slice_sizes.ndim == 1
    n_slices = slice_sizes.shape[0]
    max_n_blocks = RaggedTensorMetadata.max_n_tiles(n_slices, n_total_rows)
    # offset for each experts
    device = slice_sizes.device
    slice_offs = torch.cumsum(slice_sizes, dim=0)
    slice_offs = torch.cat((torch.zeros(1, device=device), slice_offs))
    slice_offs = slice_offs.int()
    # fill up tile offset/infos for each block
    col = torch.arange(max_n_blocks, device=device)
    slice_vals = torch.arange(n_slices, device=device)[:, None]

    def _build_schedule(block_off, n_blocks):
        total_tiles = int(block_off[-1].item())
        out = -torch.ones(max_n_blocks, dtype=torch.int32, device=device)
        if total_tiles == 0:
            return out
        tmp = -torch.ones(total_tiles, dtype=torch.int32, device=device)
        map_idxs = block_off[:-1, None] + col[None, :]
        mask = col[None, :] < n_blocks[:, None]
        tmp.index_put_((map_idxs[mask], ), (slice_vals + (col << 16)[None, :]).int()[mask])
        take = min(max_n_blocks, total_tiles)
        out[:take] = tmp[:take]
        return out

    block_offs = dict()
    block_pid_map = dict()
    for block_size in RaggedTensorMetadata.block_sizes():
        n_tiles = (slice_sizes + block_size - 1) // block_size
        block = torch.cumsum(n_tiles, dim=0)
        block = torch.cat((torch.zeros(1, device=device), block)).int()
        block_offs[block_size] = block
        block_pid_map[block_size] = _build_schedule(block, n_tiles)
    block_offs = torch.stack(list(block_offs.values()))
    block_pid_map = torch.stack(list(block_pid_map.values()))
    return RaggedTensorMetadata(slice_sizes, slice_offs, block_offs, block_pid_map)


# ============================================================================ #
# remap_ragged_tensor_metadata
# ============================================================================ #

# optimized implementation
# ---------------------------------------------------------------------------- #


@triton.jit
def _generic_compaction(Out, compute_vals_and_cond_fn, compute_vals_and_cond_fn_args, sentinel, N, BLOCK: tl.constexpr):
    curr_sum = 0
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        vals, conds = compute_vals_and_cond_fn(*compute_vals_and_cond_fn_args, offs)
        # compute values
        exc_cumsum = curr_sum + tl.cumsum(conds, 0) - conds
        active_flags = conds.to(tl.int1)
        rev_arange = N - start - 1 - tl.arange(0, BLOCK)
        write_indx = exc_cumsum + tl.where(active_flags, 0, rev_arange)
        out = tl.where(active_flags, vals, sentinel)
        # store
        tl.store(Out + write_indx, out, mask=offs < N)
        # update running sum
        curr_sum += tl.sum(conds, 0)
    return curr_sum


@triton.jit
def _compact_from_slice_map(Vals, SliceMap, n_slices, offs):
    slice_ids = offs
    mask = slice_ids < n_slices
    conds = (tl.load(SliceMap + slice_ids, mask=mask, other=-1) != -1).to(tl.int32)
    vals = tl.load(Vals + offs, mask=mask)
    return vals, conds


@triton.jit
def _compact_block_schedule(BlockSchedule, SliceMap, n_blocks, offs):
    block_id = tl.load(BlockSchedule + offs, mask=offs < n_blocks, other=-1)
    block_id = block_id.to(tl.uint32, bitcast=True)
    slice_id = block_id & 0x0000FFFF
    mask = slice_id != 65535
    conds = (tl.load(SliceMap + slice_id, mask=mask, other=-1) != -1).to(tl.int32)
    block_id = block_id.to(tl.int32, bitcast=True)
    conds = conds.to(tl.int32, bitcast=True)
    new_slice_id = tl.load(SliceMap + slice_id, mask=mask)
    pid_mask = tl.full([
        1,
    ], 0xFFFF0000, dtype=tl.uint32)
    new_block_id = ((block_id & pid_mask) | new_slice_id).to(tl.int32, bitcast=True)
    return new_block_id, conds


@triton.jit
def _remap_ragged_tensor_metadata(BatchSizesOut, BatchSizesInp,  #
                                  BatchOffsOut, BatchOffsInp,  #
                                  BlockOffsOut, block_offs_out_stride_m,  #
                                  BlockOffsInp, block_offs_in_stride_m,  #
                                  BlockScheduleOut, block_schedule_out_stride_m,  #
                                  BlockScheduleInp, block_schedule_in_stride_m,  #
                                  SliceMap,  #
                                  n_slices, n_blocks,  #
                                  BLOCK: tl.constexpr  #
                                  ):
    pid_m = tl.program_id(0)
    # number of valid slices

    # offset pointers
    BlockOffsOut += pid_m * block_offs_out_stride_m
    BlockOffsInp += pid_m * block_offs_in_stride_m
    BlockScheduleOut += pid_m * block_schedule_out_stride_m
    BlockScheduleInp += pid_m * block_schedule_in_stride_m
    # compute batch sizes for this slice by compacting input batch sizes
    _generic_compaction(BatchSizesOut, _compact_from_slice_map,  #
                        (BatchSizesInp, SliceMap, n_slices), -1, n_slices,  #
                        BLOCK=BLOCK)
    # compute batch offsets for this slice by compacting input batch offsets
    _generic_compaction(BatchOffsOut, _compact_from_slice_map,  #
                        (BatchOffsInp, SliceMap, n_slices), -1, n_slices + 1,  #
                        BLOCK=BLOCK)
    # compute block offsets
    n_compacted_blocks = _generic_compaction(BlockOffsOut, _compact_from_slice_map,  #
                                             (BlockOffsInp, SliceMap, n_slices), -1, n_slices + 1,  #
                                             BLOCK=BLOCK)
    # compute block schedule
    n_total_blocks = _generic_compaction(BlockScheduleOut, _compact_block_schedule,  #
                                         (BlockScheduleInp, SliceMap, n_blocks), -1, n_blocks,  #
                                         BLOCK=BLOCK)
    # Record the total number of tiles in the trailing slot
    tl.store(BlockOffsOut + n_compacted_blocks, n_total_blocks)


def remap_ragged_tensor_metadata(src_ragged_tensor_metadata: RaggedTensorMetadata,
                                 slice_map: torch.Tensor) -> RaggedTensorMetadata:
    """
    Let `src` be a ragged tensor, and `src_slices`/`src_ragged_tensor_metadata` be its slices/metadata.

    This function returns the metadata of `dst`, i.e. the ragged tensor s.t.:
    dst_slices = [`src_slices[slice_id]` if `slice_id != -1` for slice_id in `slice_map`]
    """
    assert slice_map.ndim == 1
    assert slice_map.shape[0] == src_ragged_tensor_metadata.slice_sizes.shape[0]
    slice_sizes = torch.empty_like(src_ragged_tensor_metadata.slice_sizes)
    slice_offs = torch.empty_like(src_ragged_tensor_metadata.slice_offs)
    block_offs_data = torch.empty_like(src_ragged_tensor_metadata.block_offs_data)
    block_schedule_data = torch.empty_like(src_ragged_tensor_metadata.block_schedule_data)

    _remap_ragged_tensor_metadata[(block_offs_data.shape[0], )](
        slice_sizes,  #
        src_ragged_tensor_metadata.slice_sizes,  #
        slice_offs,  #
        src_ragged_tensor_metadata.slice_offs,  #
        block_offs_data,
        block_offs_data.stride(0),  #
        src_ragged_tensor_metadata.block_offs_data,
        src_ragged_tensor_metadata.block_offs_data.stride(0),  #
        block_schedule_data,
        block_schedule_data.stride(0),  #
        src_ragged_tensor_metadata.block_schedule_data,
        src_ragged_tensor_metadata.block_schedule_data.stride(0),  #
        slice_map,  #
        len(slice_sizes),
        block_schedule_data.shape[-1],
        BLOCK=128,
    )
    return RaggedTensorMetadata(slice_sizes, slice_offs, block_offs_data, block_schedule_data)


# reference implementation
# ---------------------------------------------------------------------------- #


def remap_ragged_tensor_metadata_torch(ragged_tensor_metadata, slice_map):
    """
    reference implementation of `remap_ragged_tensor_metadata`
    """

    def compact(vals, conds, sentinel):
        assert conds.shape == vals.shape
        keep = conds.nonzero().flatten()
        sentinels = torch.full(((conds == 0).sum().item(), ), sentinel, dtype=vals.dtype, device=vals.device)
        return torch.cat((vals[keep], sentinels))

    def make_mask(block_pid_map):
        slice_id = (block_pid_map & 0x0000FFFF)
        valid_id = slice_id != 65535
        valid_slice_id = slice_id[valid_id]
        mask = torch.zeros_like(slice_id)
        mask[valid_id] = (slice_map[valid_slice_id] != -1).to(torch.int32)
        return mask

    def map_slice_id(block_pid_map):
        slice_id = (block_pid_map & 0x0000FFFF)
        valid_id = slice_id != 65535
        slice_id[valid_id] = slice_map[slice_id[valid_id]]
        return (block_pid_map & 0xFFFF0000) | slice_id

    n_slices = len(ragged_tensor_metadata.slice_sizes)
    n_block_sizes = ragged_tensor_metadata.block_offs_data.shape[0]
    slice_global = torch.arange(n_slices, device=ragged_tensor_metadata.slice_sizes.device)
    slice_local = slice_map[slice_global] != -1
    slice_mask = torch.cat((slice_local, torch.zeros((1, ), dtype=torch.bool, device=slice_local.device)))
    slice_sizes = compact(ragged_tensor_metadata.slice_sizes, slice_mask[:-1], -1)
    slice_offs = compact(ragged_tensor_metadata.slice_offs, slice_mask, -1)
    block_offs_data = []
    block_schedule_data = []
    for i in range(n_block_sizes):
        block_offs = compact(ragged_tensor_metadata.block_offs_data[i, :], slice_mask, -1)
        block_schedule = ragged_tensor_metadata.block_schedule_data[i, :]
        block_schedule = map_slice_id(compact(block_schedule, make_mask(block_schedule), -1))
        # replace the first -1 in `block_offs` with the number of valid blocks
        indx = (block_offs == -1).nonzero()[0].item()
        block_offs[indx] = (block_schedule != -1).sum().item()
        # update block_offs/block_schedules/
        block_offs_data += [block_offs]
        block_schedule_data += [block_schedule]
    return RaggedTensorMetadata(slice_sizes, slice_offs, torch.stack(block_offs_data, dim=0),
                                torch.stack(block_schedule_data, dim=0))
