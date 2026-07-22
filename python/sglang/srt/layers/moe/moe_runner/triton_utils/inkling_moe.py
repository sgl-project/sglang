from functools import partial

import helion
import helion.language as hl
import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.layers.moe.moe_runner.triton_utils.helion_utils import (
    get_model_depths,
    helion_aot_autotune,
)

DEFAULT_BLOCK_SIZE = 4096
BLOCK_SIZE_M = 128


def silu_and_mul_key(
    gateup_output: torch.Tensor,
    topk_weights: torch.Tensor | None,
    out_dtype: object | None = None,
):
    # Keep this stable across inputs for Helion AOT autotune.
    del out_dtype
    return gateup_output.shape[1], (gateup_output.dtype,), (topk_weights is not None,)


def silu_and_mul_inputs(sizes: list[int]):
    # Used only for Helion autotune input generation.
    inputs = []
    numel = 2**30
    with torch.device("cuda"):
        for size in sizes:
            x = torch.randn(numel // size, 2 * size, dtype=torch.bfloat16)
            inputs.append((x, None, None))
            inputs.append((x, torch.randn(numel // size, dtype=torch.bfloat16), None))
    return inputs


@helion_aot_autotune(
    "silu_and_mul_interleaved",
    kernel_key=silu_and_mul_key,
    primary_inputs=partial(silu_and_mul_inputs, sizes=[512, 2048, 48 * 96, 6144, 8192]),
    secondary_inputs=partial(
        silu_and_mul_inputs,
        sizes=[512]
        + [i * 96 for i in get_model_depths()]
        + list(range(1024, 8192 + 1, 1024)),
    ),
)
@helion.kernel(static_shapes=False)
def _silu_and_mul_helion_interleaved_kernel(
    gateup_output,
    topk_weights: torch.Tensor | None = None,
    out_dtype: hl.constexpr | None = None,
):
    """
    Interleaved version of silu_and_mul using Helion kernel.
    Input format: [gate[0], up[0], gate[1], up[1], ...]
    This matches the interleaved w13 weight format.
    """
    batch_size, hidden_size = gateup_output.shape
    hidden_size = hl.specialize(hidden_size)
    assert hidden_size % 2 == 0, f"{hidden_size=}"

    half_hidden_size = hidden_size // 2
    down_input = gateup_output.new_empty(
        batch_size, half_hidden_size, dtype=out_dtype or gateup_output.dtype
    )
    for batch_tile, hidden_tile in hl.tile([batch_size, half_hidden_size]):
        gate_output = gateup_output[batch_tile, 2 * hidden_tile.index].to(torch.float32)
        up_output = gateup_output[batch_tile, 2 * hidden_tile.index + 1].to(
            torch.float32
        )
        silu_mul_output = gate_output * torch.sigmoid(gate_output) * up_output
        if topk_weights is not None:
            weight_scale = topk_weights[batch_tile, None].to(torch.float32)
            silu_mul_output = silu_mul_output * weight_scale
        down_input[batch_tile, hidden_tile] = silu_mul_output
    return down_input


@helion_aot_autotune(
    "silu_and_mul",
    kernel_key=silu_and_mul_key,
    primary_inputs=partial(silu_and_mul_inputs, sizes=[512, 2048, 48 * 96, 6144, 8192]),
    secondary_inputs=partial(
        silu_and_mul_inputs,
        sizes=[512]
        + [i * 96 for i in get_model_depths()]
        + list(range(1024, 8192 + 1, 1024)),
    ),
)
@helion.kernel(static_shapes=False)
def _silu_and_mul_helion_non_interleaved_kernel(
    gateup_output,
    topk_weights: torch.Tensor | None = None,
    out_dtype: hl.constexpr | None = None,
):
    """
    Non-interleaved version of silu_and_mul using Helion kernel.
    Input format: [gate[0], gate[1], ..., gate[N-1], up[0], up[1], ..., up[N-1]]
    """
    batch_size, hidden_size = gateup_output.shape
    hidden_size = hl.specialize(hidden_size)
    assert hidden_size % 2 == 0, f"{hidden_size=}"

    half_hidden_size = hidden_size // 2
    down_input = gateup_output.new_empty(
        batch_size, half_hidden_size, dtype=out_dtype or gateup_output.dtype
    )
    for batch_tile, hidden_tile in hl.tile([batch_size, half_hidden_size]):
        gate_output = gateup_output[batch_tile, hidden_tile.index].to(torch.float32)
        up_output = gateup_output[batch_tile, hidden_tile.index + half_hidden_size].to(
            torch.float32
        )
        silu_mul_output = gate_output * torch.sigmoid(gate_output) * up_output
        if topk_weights is not None:
            weight_scale = topk_weights[batch_tile, None].to(torch.float32)
            silu_mul_output = silu_mul_output * weight_scale
        down_input[batch_tile, hidden_tile] = silu_mul_output
    return down_input


def silu_and_mul_helion(
    gateup_output: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    use_interleaved: bool = True,
) -> torch.Tensor:
    """
    Unified silu_and_mul function using Helion kernel.
    Supports both interleaved and non-interleaved input formats.

    Args:
        gateup_output: Input tensor of shape (batch_size, hidden_size)
        topk_weights: Optional topk weights tensor
        out_dtype: Optional output dtype
        use_interleaved: If True, expects interleaved format [gate[0], up[0], gate[1], up[1], ...]
                        If False, expects non-interleaved format [gate[0], ..., gate[N-1], up[0], ..., up[N-1]]

    Returns:
        Output tensor of shape (batch_size, hidden_size // 2)
    """
    if use_interleaved:
        return _silu_and_mul_helion_interleaved_kernel(
            gateup_output, topk_weights, out_dtype
        )
    else:
        return _silu_and_mul_helion_non_interleaved_kernel(
            gateup_output, topk_weights, out_dtype
        )


# ---------------------------------------------------------------------------
# Triton silu_and_mul
# Used by InklingBatchDenseMLP._swiglu because the helion kernel above produces
# NaN for small shared-expert batches in EP+DP configs.
# ---------------------------------------------------------------------------


@triton.jit
def _silu_and_mul_triton_kernel(
    gateup_out_ptr,
    topk_weights_ptr,
    down_inp_ptr,
    M_ptr,
    N: tl.constexpr,
    TOPK_WEIGHTS: tl.constexpr,
    GRID_SIZE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    EVEN_N: tl.constexpr,
    INT64_INDEX: tl.constexpr,
    USE_PDL: tl.constexpr = False,
):
    start_pid = tl.program_id(0)
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
    if isinstance(M_ptr, tl.tensor) and M_ptr.dtype.is_ptr():
        M = tl.load(M_ptr)
    else:
        M = M_ptr
    if INT64_INDEX:
        start_pid = start_pid.to(tl.int64)
        M = M.to(tl.int64)

    NUM_BLOCKS_N: tl.constexpr = tl.cdiv(N, BLOCK_SIZE_N)
    num_blocks_mn = tl.cdiv(M, BLOCK_SIZE_M) * NUM_BLOCKS_N

    for pid in tl.range(start_pid, num_blocks_mn, GRID_SIZE, num_stages=NUM_STAGES):
        pid_m = pid // NUM_BLOCKS_N
        pid_n = pid % NUM_BLOCKS_N

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        mask_m = offs_m < M
        mask_n = offs_n < N

        mask_offs_2n = pid_n * BLOCK_SIZE_N + tl.arange(0, 2 * BLOCK_SIZE_N) // 2
        tl.static_assert(BLOCK_SIZE_N % 8 == 0, f"{BLOCK_SIZE_N=}")
        mask_2n = mask_offs_2n < N
        mask_2n = tl.max_constancy(mask_2n, [16])

        offs_2n = pid_n * 2 * BLOCK_SIZE_N + tl.arange(0, 2 * BLOCK_SIZE_N)
        offs_m2n = offs_m[:, None] * N * 2 + offs_2n[None, :]

        if EVEN_N or pid_n * BLOCK_SIZE_N + BLOCK_SIZE_N <= N:
            gateup_out = tl.load(
                gateup_out_ptr + offs_m2n, mask=mask_m[:, None], other=0.0
            )
        else:
            mask_m2n = mask_m[:, None] & mask_2n[None, :]
            gateup_out = tl.load(gateup_out_ptr + offs_m2n, mask=mask_m2n, other=0.0)

        gate_out, up_out = tl.split(
            tl.reshape(gateup_out, (BLOCK_SIZE_M, BLOCK_SIZE_N, 2))
        )
        gate_out = gate_out.to(tl.float32)
        up_out = up_out.to(tl.float32)

        gate_out = gate_out * tl.sigmoid(gate_out)
        down_inp = gate_out * up_out
        if TOPK_WEIGHTS:
            weight_scale = tl.load(topk_weights_ptr + offs_m, mask=mask_m).to(
                tl.float32
            )
            down_inp = down_inp * weight_scale[:, None]

        mask_mn = mask_m[:, None] if EVEN_N else mask_m[:, None] & mask_n[None, :]
        offs_mn = offs_m[:, None] * N + offs_n[None, :]
        tl.store(down_inp_ptr + offs_mn, down_inp, mask=mask_mn)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()


def silu_and_mul_triton(
    gateup_output: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """SiLU-and-mul for interleaved gate/up layout using a Triton kernel.

    Adapted from ``inkling_kernels.activation.silu_and_mul_fwd`` (without MXFP).
    """
    assert (
        gateup_output.is_contiguous()
    ), f"{gateup_output.shape=} {gateup_output.stride()=}"
    assert gateup_output.ndim == 2, f"{gateup_output.shape=}"
    if topk_weights is not None:
        assert (
            topk_weights.is_contiguous()
        ), f"{topk_weights.shape=} {topk_weights.stride()=}"
        assert topk_weights.ndim == 1, f"{topk_weights.shape=}"

    M = gateup_output.shape[0]
    N = gateup_output.shape[1] // 2

    dtype = out_dtype or gateup_output.dtype
    down_input = torch.empty((M, N), device=gateup_output.device, dtype=dtype)

    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = min(128, triton.next_power_of_2(N))
    NUM_STAGES = 2
    max_grid_size = triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N)
    # Use ~512 SMs worth of blocks, capped to actual work
    num_sms = torch.cuda.get_device_properties(
        gateup_output.device
    ).multi_processor_count
    grid_size = min(num_sms * 4, max_grid_size)

    _silu_and_mul_triton_kernel[(grid_size,)](
        gateup_out_ptr=gateup_output,
        topk_weights_ptr=topk_weights,
        down_inp_ptr=down_input,
        M_ptr=M,
        N=N,
        TOPK_WEIGHTS=topk_weights is not None,
        GRID_SIZE=grid_size,
        NUM_STAGES=NUM_STAGES,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        EVEN_N=N % BLOCK_SIZE_N == 0,
        INT64_INDEX=gateup_output.nbytes >= 2**31,
        **({"USE_PDL": True, "launch_pdl": True} if is_arch_support_pdl() else {}),
    )

    return down_input


@triton.jit
def _compute_expert_offsets_kernel(ReorderTopkIds, ExpertOffsets, num_toks):
    expert = tl.program_id(0)
    if expert == 0:
        # Specially have pid 0 write the 0 value to index 0
        tl.store(ExpertOffsets, 0)
    low = 0
    high = num_toks - 1
    target_location = -1
    while low <= high:
        mid = (low + high) // 2

        if tl.load(ReorderTopkIds + mid) > expert:
            high = mid - 1
        else:
            low = mid + 1
            target_location = mid
    tl.store(ExpertOffsets + expert + 1, target_location + 1)


@triton.jit
def _compute_src2dst_kernel(ReorderIds, Src2Dst, num_toks, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    dst_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = dst_id < num_toks
    src_id = tl.load(ReorderIds + dst_id, mask=mask)
    tl.store(Src2Dst + src_id, dst_id, mask=mask)


def get_src2dst(reorder_ids: torch.Tensor):
    num_tokens = reorder_ids.numel()
    src2dst = torch.empty(num_tokens, device=reorder_ids.device, dtype=torch.int32)
    _compute_src2dst_kernel[(triton.cdiv(num_tokens, DEFAULT_BLOCK_SIZE),)](
        reorder_ids, src2dst, num_tokens, BLOCK_SIZE=DEFAULT_BLOCK_SIZE
    )
    return src2dst


@triton.jit
def _compute_num_tokens_per_expert_from_offs_kernel(
    expert_token_offs_ptr,  # [E + 1] input
    num_tokens_per_expert_ptr,  # [E] output
):
    expert_id = tl.program_id(0)
    expert_start_off = tl.load(expert_token_offs_ptr + expert_id)
    expert_end_off = tl.load(expert_token_offs_ptr + expert_id + 1)
    num_expert_tokens = expert_end_off - expert_start_off
    tl.store(num_tokens_per_expert_ptr + expert_id, num_expert_tokens)


def _get_max_num_blocks(
    num_routed_tokens: int, block_sizes: list[int], num_experts: int
):
    return triton.cdiv(num_routed_tokens, min(block_sizes)) + num_experts - 1


@triton.jit
def _memset_block_metadata_kernel(
    num_tokens_per_expert_ptr,
    expert_block_offs_ptr,
    expert_block_schedule_ptr,
    max_num_blocks,
    E,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_MEMSET: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)

    if pid == 0:
        curr_sum = tl.zeros((), dtype=tl.int32)
        for off in range(0, E, BLOCK_SIZE_MEMSET):
            offs = off + tl.arange(0, BLOCK_SIZE_MEMSET)
            mask = offs < E
            num_tokens_per_expert = tl.load(
                num_tokens_per_expert_ptr + offs, mask=mask, other=0
            )
            num_blocks_per_expert = tl.cdiv(num_tokens_per_expert, BLOCK_SIZE_M)
            block_offs = (
                tl.cumsum(num_blocks_per_expert, 0) - num_blocks_per_expert + curr_sum
            )
            curr_sum += tl.sum(num_blocks_per_expert, 0).to(tl.int32)
            tl.store(expert_block_offs_ptr + offs, block_offs, mask=mask)
        tl.store(expert_block_offs_ptr + E, curr_sum)
    else:
        pid = pid - 1
        offs = pid * BLOCK_SIZE_MEMSET + tl.arange(0, BLOCK_SIZE_MEMSET)
        mask = offs < max_num_blocks
        tl.store(expert_block_schedule_ptr + offs, -1, mask=mask)


@triton.jit
def _compute_block_metadata_kernel(
    num_tokens_per_expert_ptr,
    expert_block_offs_ptr,
    expert_block_schedule_ptr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)

    expert_id = pid
    num_expert_tokens = tl.load(num_tokens_per_expert_ptr + expert_id)
    num_expert_blocks = tl.cdiv(num_expert_tokens, BLOCK_SIZE_M)

    expert_block_off = tl.load(expert_block_offs_ptr + expert_id)
    expert_block_schedule_ptr += expert_block_off
    for block_off in range(0, num_expert_blocks, BLOCK_SIZE):
        block_offs = block_off + tl.arange(0, BLOCK_SIZE)
        data = (block_offs << 16) + expert_id
        mask = block_offs < num_expert_blocks
        tl.store(expert_block_schedule_ptr + block_offs, data, mask=mask)


def compute_expert_block_metadata(
    num_tokens_per_expert: torch.Tensor,
    num_routed_tokens: int,
    *,
    block_size_m: int = BLOCK_SIZE_M,
):
    assert num_tokens_per_expert.ndim == 1, f"{num_tokens_per_expert.shape=}"
    assert (
        num_tokens_per_expert.is_contiguous()
    ), f"{num_tokens_per_expert.shape=} {num_tokens_per_expert.stride()=}"

    num_experts = num_tokens_per_expert.numel()
    max_num_blocks = _get_max_num_blocks(num_routed_tokens, [block_size_m], num_experts)
    expert_block_offs = torch.empty(
        (num_experts + 1,), dtype=torch.int32, device=num_tokens_per_expert.device
    )
    expert_block_schedule = torch.empty(
        (max_num_blocks,), dtype=torch.int32, device=num_tokens_per_expert.device
    )

    block_size_memset = 512
    int64_index = (
        expert_block_offs.nbytes >= 2**31 or expert_block_schedule.nbytes >= 2**31
    )
    grid = (1 + triton.cdiv(expert_block_schedule.numel(), block_size_memset),)
    _memset_block_metadata_kernel[grid](
        num_tokens_per_expert_ptr=num_tokens_per_expert,
        expert_block_offs_ptr=expert_block_offs,
        expert_block_schedule_ptr=expert_block_schedule,
        max_num_blocks=max_num_blocks,
        E=num_experts,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_MEMSET=block_size_memset,
        INT64_INDEX=int64_index,
    )

    _compute_block_metadata_kernel[(num_experts,)](
        num_tokens_per_expert_ptr=num_tokens_per_expert,
        expert_block_offs_ptr=expert_block_offs,
        expert_block_schedule_ptr=expert_block_schedule,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE=block_size_memset,
        INT64_INDEX=int64_index,
    )

    return expert_block_offs, expert_block_schedule


SMALL_M_BLOCK_SIZE_M = 16
# Use smaller blocks for sparse decode workloads.
GROUPED_GEMM_SMALL_M_MAX = 6144
# Single-CTA fused preprocess capability bound (in-register tl.sort of n
# packed keys); correctness-tested to this size.
FUSED_PREPROCESS_MAX_TOKENS = 2048
# Keep larger workloads on the general sort path.
FUSED_PREPROCESS_WIN_TOKENS = 1024


@triton.jit
def _fused_moe_preprocess_kernel(
    topk_ids_ptr,  # [n] int32, unsorted
    reorder_topk_ids_ptr,  # [n] int32 out, sorted
    src2dst_ptr,  # [n] int32 out
    num_tokens_per_expert_ptr,  # [E] int32 out
    expert_token_offs_ptr,  # [E+1] int32 out
    expert_block_offs_ptr,  # [E+1] int32 out
    expert_block_schedule_ptr,  # [max_num_blocks] int32 out, -1 padded
    n,
    max_num_blocks,
    E: tl.constexpr,
    BLOCK_N: tl.constexpr,  # pow2 >= n, <= 4096 (12-bit position pack)
    BLOCK_E1: tl.constexpr,  # pow2 >= E+1
    BLOCK_SCHED: tl.constexpr,  # pow2 >= max_num_blocks
    BLOCK_SIZE_M: tl.constexpr,
):
    """Single-CTA replacement for the whole grouped-gemm preprocess at decode
    sizes: int16 cast + torch.sort(stable) + src2dst + expert offsets + counts
    + block memset/schedule (~10 launches -> 1).

    Stable sort: one in-register tl.sort of (id << 12 | position) packed keys
    -- position ties reproduce torch.sort(stable=True) exactly. Offsets come
    from a vectorized binary search on the sorted ids (same contract as
    _compute_expert_offsets_kernel); the block schedule from a binary search
    over the block-offset cumsum.
    """
    offs = tl.arange(0, BLOCK_N)
    mask = offs < n
    ids = tl.load(topk_ids_ptr + offs, mask=mask, other=E)  # pads sort last
    skey = tl.sort((ids << 12) | offs)
    s_ids = skey >> 12
    s_src = skey & 0xFFF
    tl.store(reorder_topk_ids_ptr + offs, s_ids, mask=mask)
    tl.store(src2dst_ptr + s_src, offs, mask=mask)
    tl.debug_barrier()  # publish sorted ids for the binary searches below

    # expert_token_offs[e] = first sorted index with id >= e (e in 0..E)
    e_offs = tl.arange(0, BLOCK_E1)
    e_mask = e_offs < E + 1
    low = tl.zeros([BLOCK_E1], dtype=tl.int32)
    high = tl.full([BLOCK_E1], n, dtype=tl.int32)
    for _ in tl.static_range(12):  # n <= 4096
        mid = (low + high) // 2
        v = tl.load(reorder_topk_ids_ptr + mid, mask=e_mask & (mid < n), other=E + 1)
        go_right = v < e_offs
        low = tl.where(go_right, mid + 1, low)
        high = tl.where(go_right, high, mid)
    tl.store(expert_token_offs_ptr + e_offs, low, mask=e_mask)
    tl.debug_barrier()

    counts = tl.load(
        expert_token_offs_ptr + e_offs + 1, mask=e_offs < E, other=0
    ) - tl.load(expert_token_offs_ptr + e_offs, mask=e_offs < E, other=0)
    tl.store(num_tokens_per_expert_ptr + e_offs, counts, mask=e_offs < E)

    num_blocks = tl.cdiv(counts, BLOCK_SIZE_M)
    block_offs_excl = tl.cumsum(num_blocks, 0) - num_blocks
    tl.store(expert_block_offs_ptr + e_offs, block_offs_excl, mask=e_offs < E)
    total_blocks = tl.sum(num_blocks, 0)
    tl.store(expert_block_offs_ptr + E, total_blocks)
    tl.debug_barrier()

    # schedule[s] = ((s - block_offs[e]) << 16) | e, e = last expert with
    # block_offs[e] <= s; -1 for padding slots
    s_offs = tl.arange(0, BLOCK_SCHED)
    s_mask = s_offs < max_num_blocks
    lo = tl.zeros([BLOCK_SCHED], dtype=tl.int32)
    hi = tl.full([BLOCK_SCHED], E, dtype=tl.int32)
    for _ in tl.static_range(9):  # E = 256
        mid = (lo + hi + 1) // 2
        v = tl.load(expert_block_offs_ptr + mid, mask=s_mask, other=0)
        go_left = v > s_offs
        hi = tl.where(go_left, mid - 1, hi)
        lo = tl.where(go_left, lo, mid)
    e_of_s = lo
    base = tl.load(expert_block_offs_ptr + e_of_s, mask=s_mask, other=0)
    data = ((s_offs - base) << 16) + e_of_s
    data = tl.where(s_offs < total_blocks, data, -1)
    tl.store(expert_block_schedule_ptr + s_offs, data, mask=s_mask)


def fused_moe_preprocess(topk_ids_flat: torch.Tensor, num_experts: int):
    """One-launch preprocess for n <= FUSED_PREPROCESS_MAX_TOKENS routed rows.

    Returns (src2dst, num_tokens_per_expert, expert_token_offs,
    expert_block_offs, expert_block_schedule, reorder_topk_ids) --
    bit-identical to the torch.sort-based path, with reorder_topk_ids in int32
    and the block schedule built for SMALL_M_BLOCK_SIZE_M.
    """
    n = topk_ids_flat.numel()
    assert 0 < n <= FUSED_PREPROCESS_MAX_TOKENS, f"{n=}"
    assert topk_ids_flat.is_contiguous()
    device = topk_ids_flat.device
    block_size_m = SMALL_M_BLOCK_SIZE_M
    max_num_blocks = _get_max_num_blocks(n, [block_size_m], num_experts)
    reorder_topk_ids = torch.empty(n, device=device, dtype=torch.int32)
    src2dst = torch.empty(n, device=device, dtype=torch.int32)
    num_tokens_per_expert = torch.empty(num_experts, device=device, dtype=torch.int32)
    expert_token_offs = torch.empty(num_experts + 1, device=device, dtype=torch.int32)
    expert_block_offs = torch.empty(num_experts + 1, device=device, dtype=torch.int32)
    expert_block_schedule = torch.empty(
        max_num_blocks, device=device, dtype=torch.int32
    )
    _fused_moe_preprocess_kernel[(1,)](
        topk_ids_flat.int(),
        reorder_topk_ids,
        src2dst,
        num_tokens_per_expert,
        expert_token_offs,
        expert_block_offs,
        expert_block_schedule,
        n,
        max_num_blocks,
        E=num_experts,
        BLOCK_N=max(triton.next_power_of_2(n), 16),
        BLOCK_E1=triton.next_power_of_2(num_experts + 1),
        BLOCK_SCHED=triton.next_power_of_2(max_num_blocks),
        BLOCK_SIZE_M=block_size_m,
        num_warps=4,
    )
    return (
        src2dst,
        num_tokens_per_expert,
        expert_token_offs,
        expert_block_offs,
        expert_block_schedule,
        reorder_topk_ids,
    )


def select_grouped_gemm_block_m(num_routed_tokens: int) -> int:
    return (
        SMALL_M_BLOCK_SIZE_M
        if num_routed_tokens <= GROUPED_GEMM_SMALL_M_MAX
        else BLOCK_SIZE_M
    )


def compute_grouped_gemm_metadata(
    sorted_topk_ids: torch.Tensor, num_experts: int, *, block_size_m: int = BLOCK_SIZE_M
):
    num_routed_tokens = sorted_topk_ids.numel()
    device = sorted_topk_ids.device
    num_tokens_per_expert = torch.empty(num_experts, device=device, dtype=torch.int32)
    expert_token_offs = torch.empty(num_experts + 1, device=device, dtype=torch.int32)
    _compute_expert_offsets_kernel[(num_experts,)](
        sorted_topk_ids, expert_token_offs, num_routed_tokens
    )
    _compute_num_tokens_per_expert_from_offs_kernel[(num_experts,)](
        expert_token_offs_ptr=expert_token_offs,
        num_tokens_per_expert_ptr=num_tokens_per_expert,
    )
    expert_block_offs, expert_block_schedule = compute_expert_block_metadata(
        num_tokens_per_expert, num_routed_tokens, block_size_m=block_size_m
    )
    return (
        num_tokens_per_expert,
        expert_token_offs,
        expert_block_offs,
        expert_block_schedule,
    )


@triton.jit
def _pre_reorder_kernel(
    input_ptr,
    gateup_input_ptr,
    src2dst_ptr,
    TOP_K: tl.constexpr,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    src_idx = tl.program_id(0).to(tl.int64)
    src2dst_ptr = src2dst_ptr + src_idx * TOP_K

    src_ptr = input_ptr + src_idx * hidden_size
    for idx in tl.static_range(TOP_K):
        dst_idx = tl.load(src2dst_ptr + idx).to(tl.int64)
        dst_ptr = gateup_input_ptr + dst_idx * hidden_size
        for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
            offset = start_offset + tl.arange(0, BLOCK_SIZE)
            mask = offset < hidden_size
            in_data = tl.load(src_ptr + offset, mask=mask).to(tl.float32)
            tl.store(dst_ptr + offset, in_data, mask=mask)


def pre_reorder(input: torch.Tensor, src2dst: torch.Tensor, topk: int):
    assert input.is_contiguous()
    assert src2dst.is_contiguous()

    batch_size, hidden_size = input.shape
    (batch_size_expanded,) = src2dst.shape

    output = torch.empty(
        (batch_size_expanded, hidden_size),
        device=input.device,
        dtype=input.dtype,
    )

    _pre_reorder_kernel[(batch_size,)](
        input,
        output,
        src2dst,
        topk,
        hidden_size,
        BLOCK_SIZE=DEFAULT_BLOCK_SIZE,
    )

    return output


@triton.jit
def _post_reorder_kernel(
    down_output_ptr,
    output_ptr,
    src2dst_ptr,
    topk_weights_ptr,
    TOP_K: tl.constexpr,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    src_idx = tl.program_id(0).to(tl.int64)
    src2dst_ptr = src2dst_ptr + src_idx * TOP_K
    topk_weights_ptr = topk_weights_ptr + src_idx * TOP_K

    store_ptr = output_ptr + src_idx * hidden_size
    for start_offset in tl.range(0, hidden_size, BLOCK_SIZE):
        offset = start_offset + tl.arange(0, BLOCK_SIZE)
        mask = offset < hidden_size

        sum_vec = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        for idx in tl.static_range(TOP_K):
            dst_idx = tl.load(src2dst_ptr + idx).to(tl.int64)
            weight_scale = tl.load(topk_weights_ptr + idx).to(tl.float32)
            load_ptr = down_output_ptr + dst_idx * hidden_size
            in_data = tl.load(load_ptr + offset, mask=mask)
            sum_vec += in_data * weight_scale
        tl.store(store_ptr + offset, sum_vec, mask=mask)


def post_reorder(
    down_output: torch.Tensor, src2dst: torch.Tensor, topk_weights: torch.Tensor
):
    assert down_output.is_contiguous()
    assert src2dst.is_contiguous()
    assert topk_weights.is_contiguous()

    # Constants
    batch_size = topk_weights.shape[0]
    hidden_size = down_output.shape[1]
    TOP_K = topk_weights.shape[1]

    # Output tensor
    output = torch.empty(
        batch_size, hidden_size, device=down_output.device, dtype=down_output.dtype
    )

    # Launch kernel
    grid = (batch_size,)
    _post_reorder_kernel[grid](
        down_output,
        output,
        src2dst,
        topk_weights,
        TOP_K=TOP_K,
        hidden_size=hidden_size,
        BLOCK_SIZE=DEFAULT_BLOCK_SIZE,
    )
    return output


@triton.jit
def _compute_expert_attrs(
    pid_m,
    NumTokensPerExpert,  # [E]
    ExpertTokenOffs,  # [E + 1]
    ExpertBlockSchedule,  # [max_num_blocks]
    BLOCK_SIZE_M: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    expert_data = tl.load(ExpertBlockSchedule + pid_m)
    expert_id = expert_data & 0xFFFF
    block_id = expert_data >> 16
    expert_num_tokens = tl.load(NumTokensPerExpert + expert_id)
    token_start_m = tl.load(ExpertTokenOffs + expert_id)
    if INT64_INDEX:
        expert_id = expert_id.to(tl.int64)
        block_id = block_id.to(tl.int64)
        token_start_m = token_start_m.to(tl.int64)
    block_start = token_start_m + block_id * BLOCK_SIZE_M
    block_end = tl.minimum(
        block_start + BLOCK_SIZE_M, token_start_m + expert_num_tokens
    )
    return expert_id, block_start, block_end


@triton.jit
def _grouped_gemm_kernel(
    A,
    B,
    C,
    NumTokensPerExpert,
    ExpertTokenOffs,
    ExpertBlockOffs,
    ExpertBlockSchedule,
    a_stride_0: tl.constexpr,
    b_stride_0: tl.constexpr,
    b_stride_1: tl.constexpr,
    c_stride_0: tl.constexpr,
    E: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    grid_m,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    KN_MASK: tl.constexpr,
    INT64_INDEX: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = grid_m
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
        num_pid_m = num_pid_m.to(tl.int64)
        num_pid_n = num_pid_n.to(tl.int64)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_blocks_m = tl.load(ExpertBlockOffs + E)
    if pid_m >= num_blocks_m:
        return
    expert_id, block_start, block_end = _compute_expert_attrs(
        pid_m,
        NumTokensPerExpert,
        ExpertTokenOffs,
        ExpertBlockSchedule,
        BLOCK_SIZE_M,
        INT64_INDEX,
    )

    offs_am = tl.arange(0, BLOCK_SIZE_M)
    offs_bn = tl.arange(0, BLOCK_SIZE_N)
    offs_am = tl.where(block_start + offs_am < block_end, offs_am, 0)
    offs_am = tl.max_contiguous(tl.multiple_of(offs_am, BLOCK_SIZE_M), BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # [M, K] tile
    a_ptr = A + (block_start + offs_am[:, None]) * a_stride_0 + offs_k[None, :]
    # [K, N] tile
    b_ptr = B + (
        (expert_id * b_stride_0)
        + (pid_n * BLOCK_SIZE_N + offs_bn[None, :]) * b_stride_1
        + offs_k[:, None]
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        if not KN_MASK:
            a_tile = tl.load(a_ptr)
            b_tile = tl.load(b_ptr)
        else:
            n_mask = pid_n * BLOCK_SIZE_N + offs_bn < N
            k_mask = k * BLOCK_SIZE_K + offs_k < K
            a_tile = tl.load(a_ptr, mask=k_mask[None, :], other=0.0)
            b_tile = tl.load(b_ptr, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        accumulator = tl.dot(a_tile, b_tile, acc=accumulator)
        a_ptr += BLOCK_SIZE_K
        b_ptr += BLOCK_SIZE_K

    accumulator = accumulator.to(C.dtype.element_ty)

    offs_cm = block_start + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptr = C + offs_cm[:, None] * c_stride_0 + offs_cn[None, :]
    c_mask = offs_cm[:, None] < block_end
    if KN_MASK:
        n_mask = offs_cn < N
        c_mask = c_mask & n_mask[None, :]
    tl.store(c_ptr, accumulator, mask=c_mask)


def grouped_gemm_triton(
    a: torch.Tensor,
    b: torch.Tensor,
    num_experts: int,
    num_tokens_per_expert: torch.Tensor,  # [E]
    expert_token_offs: torch.Tensor,  # [E + 1]
    expert_block_offs: torch.Tensor,  # [E + 1]
    expert_block_schedule: torch.Tensor,  # [max_num_blocks]
    block_size_m: int = BLOCK_SIZE_M,  # must match the schedule's build value
) -> torch.Tensor:
    assert a.is_contiguous(), f"{a.shape=} {a.stride()=}"
    assert b.is_contiguous(), f"{b.shape=} {b.stride()=}"

    M, K = a.shape
    E, N, K_ = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    assert K == K_, f"{a.shape=} {b.shape=}"
    assert num_experts == E, f"{num_experts=} {b.shape=}"

    # Sparse decode uses smaller row blocks; prefill uses the standard size.
    # This must match the size used to build expert_block_schedule.
    if block_size_m == SMALL_M_BLOCK_SIZE_M:
        config = {
            "BLOCK_SIZE_M": SMALL_M_BLOCK_SIZE_M,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
            "num_warps": 4,
            "num_stages": 4,
        }
    else:
        assert block_size_m == BLOCK_SIZE_M, f"{block_size_m=}"
        config = {
            "BLOCK_SIZE_M": BLOCK_SIZE_M,
            "BLOCK_SIZE_N": 256 if a.dtype != torch.float32 else 64,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
            "num_warps": 8,
            "num_stages": 3,
        }
    # Set grid_m to the max number of M blocks and skip padding-only blocks
    # in the kernel based on expert_block_offs[-1]
    grid_m = expert_block_schedule.numel()

    def grid(META: dict[str, int]):
        return (grid_m * triton.cdiv(N, META["BLOCK_SIZE_N"]),)

    with torch.profiler.record_function(
        f"grouped_gemm_[M:{M},K:{K},E:{num_experts},N:{N}]"
    ):
        _grouped_gemm_kernel[grid](
            A=a,
            B=b,
            C=c,
            NumTokensPerExpert=num_tokens_per_expert,
            ExpertTokenOffs=expert_token_offs,
            ExpertBlockOffs=expert_block_offs,
            ExpertBlockSchedule=expert_block_schedule,
            a_stride_0=a.stride(0),
            b_stride_0=b.stride(0),
            b_stride_1=b.stride(1),
            c_stride_0=c.stride(0),
            E=num_experts,
            N=N,
            K=K,
            grid_m=grid_m,
            KN_MASK=K % config["BLOCK_SIZE_K"] != 0 or N % config["BLOCK_SIZE_N"] != 0,
            INT64_INDEX=a.nbytes >= 2**31 or b.nbytes >= 2**31 or c.nbytes >= 2**31,
            **config,
        )

    return c
