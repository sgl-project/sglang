"""Fused decode-phase indexer input preparation (Triton-Ascend).

Fuses the 13 small torch ops issued before ``npu_lightning_indexer`` in the
decode path (weights GEMM + pool_seqlens + pool_block_tables) into 2 Triton
kernels:

  Kernel 1 ``_fused_decode_prepare_indexer_gemm_kernel`` (pure Cube, grid=num_aicore)
      w_bf16 = (x @ W.T) * (n_heads**-0.5 * softmax_scale) -> bf16
      Batch-grouped strided assignment: BLOCK_B batches packed into the M dim
      (avoids inefficient M=1 vector-matrix GEMM).

  Kernel 2 ``_fused_decode_prepare_indexer_vv_kernel`` (pure Vector, grid=num_vectorcore)
      Section A: pool_seqlens = seqlens // POOL_SIZE (per-batch split).
      Section B: pool_block_tables = block_tables[:, ::POOL_SIZE]. Loads
      [N, NUM_BLOCKS] contiguously into UB as 1D tiles, then extracts the
      strided columns via reshape + split (avoids the 2x (Index+Transpose)
      materialization of ``build_pooled_page_table_64`` + ``.contiguous()``).

``actual_seq_lengths_q = arange(1, N+1)`` stays outside the operator.
"""

import torch
import torch.nn.functional as F
import torch_npu
import triton
import triton.language as tl

# ============================================================================
# auxiliary
# ============================================================================


def get_device_properties():
    """Query Ascend NPU device properties."""
    device = torch.npu.current_device()
    device_properties = triton.runtime.driver.active.utils.get_device_properties(
        device
    )
    num_aicore = device_properties.get("num_aicore", -1)
    num_vectorcore = device_properties.get("num_vectorcore", -1)
    assert num_aicore > 0 and num_vectorcore > 0, "Failed to detect device properties."
    return num_aicore, num_vectorcore


# ============================================================================
# kernel 1: pure Cube GEMM for w_bf16
# ============================================================================


@triton.jit
def _fused_decode_prepare_indexer_gemm_kernel(
    x_ptr,  # bf16* [N, HIDDEN_SIZE]
    w_ptr,  # fp32* [N_HEADS, HIDDEN_SIZE] (ReplicatedLinear weight)
    w_bf16_ptr,  # bf16* [N, N_HEADS]
    N,  # decode batch size (n_real)
    combined_scale,  # n_heads_pow * softmax_scale, inlined at compile time
    HIDDEN_SIZE: tl.constexpr,
    N_HEADS: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)

    # 算子类型: 纯 Cube (仅 tl.dot), 分核 num_aicore
    # 分核: 沿 batch 分组跨步分配, BLOCK_B 个 batch 打包进 M 维 (避免 M=1 的 vector-matrix 低效)
    # N_HEADS=32 极小且固定, BLOCK_N=N_HEADS 整块加载, 无 N 维 mask
    # 权重 w_tile [N_HEADS, BLOCK_K] 不依赖 batch, 每个 K 循环只加载一次、被同组所有 batch 复用
    NUM_B_BLOCKS = (N + BLOCK_B - 1) // BLOCK_B
    for block_idx in range(pid, NUM_B_BLOCKS, NUM_CORES):
        rows = block_idx * BLOCK_B + tl.arange(0, BLOCK_B)
        rows64 = rows.to(tl.int64)
        mask_b = rows < N

        # a_block: x[rows, k0:k0+BLOCK_K] bf16, M 维尾块越界保护
        a_block = tl.make_block_ptr(
            base=x_ptr,
            shape=(N, HIDDEN_SIZE),
            strides=(HIDDEN_SIZE, 1),
            offsets=(block_idx * BLOCK_B, 0),
            block_shape=(BLOCK_B, BLOCK_K),
            order=(1, 0),
        )
        # w_block: W[n, k0:k0+BLOCK_K] fp32, 全 batch 共享, 仅沿 K 维 advance
        w_block = tl.make_block_ptr(
            base=w_ptr,
            shape=(N_HEADS, HIDDEN_SIZE),
            strides=(HIDDEN_SIZE, 1),
            offsets=(0, 0),
            block_shape=(N_HEADS, BLOCK_K),
            order=(1, 0),
        )

        acc = tl.zeros((BLOCK_B, N_HEADS), dtype=tl.float32)
        for k0 in range(0, HIDDEN_SIZE, BLOCK_K):
            # A tile: x[rows, k0:k0+BLOCK_K] bf16 -> fp32
            a = tl.load(a_block, boundary_check=(0,)).to(tl.float32)
            # B tile: W[n, k0:k0+BLOCK_K] -> [BLOCK_K, N_HEADS] fp32
            w_k = tl.load(w_block)  # [N_HEADS, BLOCK_K]
            b = tl.trans(w_k)  # [BLOCK_K, N_HEADS]
            acc = tl.dot(a, b, acc)
            a_block = tl.advance(a_block, (0, BLOCK_K))
            w_block = tl.advance(w_block, (0, BLOCK_K))

        w = acc * combined_scale
        w_bf16 = w.to(tl.bfloat16)
        w_out_ptrs = w_bf16_ptr + rows64[:, None] * N_HEADS + tl.arange(0, N_HEADS)[
            None, :
        ]
        tl.store(w_out_ptrs, w_bf16, mask=mask_b[:, None])


# ============================================================================
# kernel 2: pure Vector for pool_seqlens + pool_block_tables
# ============================================================================


@triton.jit
def _fused_decode_prepare_indexer_vv_kernel(
    seqlens_ptr,  # int32* [N]
    block_tables_ptr,  # int32* [N, NUM_BLOCKS]
    pool_seqlens_ptr,  # int32* [N]
    pool_block_tables_ptr,  # int32* [N, N_POOL]
    N,  # decode batch size (n_real)
    NUM_BLOCKS: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    BLOCK_M_A: tl.constexpr,
    BLOCK_M_B: tl.constexpr,
    NUM_CORES: tl.constexpr,
):
    pid = tl.program_id(0)
    N_POOL: tl.constexpr = NUM_BLOCKS // POOL_SIZE
    P_TOTAL: tl.constexpr = BLOCK_M_B // POOL_SIZE

    # ===================== 段 A: pool_seqlens =====================
    NUM_SEQ_BLOCKS = (N + BLOCK_M_A - 1) // BLOCK_M_A
    for block_idx in range(pid, NUM_SEQ_BLOCKS, NUM_CORES):
        rows = block_idx * BLOCK_M_A + tl.arange(0, BLOCK_M_A)
        mask = rows < N
        s = tl.load(seqlens_ptr + rows, mask=mask)
        pool_s = s // POOL_SIZE
        tl.store(pool_seqlens_ptr + rows, pool_s, mask=mask)

    # ===================== 段 B: pool_block_tables =====================
    # 把 [N, NUM_BLOCKS] 视为一整行 (N*NUM_BLOCKS 个元素), 按 BLOCK_M_B 分块.
    # 每块连续 load 到 UB, 再 reshape+split 抽取 ::POOL_SIZE (等价 block_tables[:, ::POOL_SIZE]).
    # 注: tl.gather 不支持 int32, tl.sum 对 int32 归约会出错, 故用纯布局操作 tl.split 抽取.
    NUM_TILES = (N * NUM_BLOCKS + BLOCK_M_B - 1) // BLOCK_M_B

    for tile_idx in range(pid, NUM_TILES, NUM_CORES):
        # 1) 1D 连续 load BLOCK_M_B 个元素, 可合并访存
        offs_in = tile_idx * BLOCK_M_B + tl.arange(0, BLOCK_M_B)
        mask_in = offs_in < N * NUM_BLOCKS
        block = tl.load(block_tables_ptr + offs_in, mask=mask_in, other=0)  # [BLOCK_M_B]

        # 2) reshape + split 抽取 ::4 (POOL_SIZE=4 固定: 两次 split 取偶数半)
        block = tl.reshape(block, [P_TOTAL, 2, 2])
        block, _ = tl.split(block)  # [P_TOTAL, 2], 每组 2 个取第 0 个 (::2)
        block, _ = tl.split(block)  # [P_TOTAL], 再 ::2 得 ::4
        pt = block  # [P_TOTAL]

        # 3) 1D 连续写回
        offs_out = tile_idx * P_TOTAL + tl.arange(0, P_TOTAL)
        mask_out = offs_out < N * N_POOL
        tl.store(pool_block_tables_ptr + offs_out, pt, mask=mask_out)


# ============================================================================
# wrapper
# ============================================================================


def fused_decode_prepare_indexer(
    x,
    weights_proj_weight,
    seqlens_32,
    block_tables,
    pool_size=4,
    n_heads_pow=32**-0.5,
    softmax_scale=128**-0.5,
):
    """Fuse decode indexer input preparation into 2 Triton kernels.

    Args:
        x: [N, HIDDEN_SIZE] bf16, hidden state
        weights_proj_weight: [N_HEADS, HIDDEN_SIZE] fp32, ReplicatedLinear weight
        seqlens_32: [N] int32, per-request real KV length
        block_tables: [N, NUM_BLOCKS] int32, 64-page page table (contiguous)
        pool_size: int, pooling size (fixed 4)
        n_heads_pow: float, n_heads**-0.5
        softmax_scale: float, head_dim**-0.5
    Returns:
        w_bf16: [N, N_HEADS] bf16
        pool_seqlens: [N] int32
        pool_block_tables: [N, N_POOL] int32
    """
    # 1. shape constraints
    N, HIDDEN_SIZE = x.shape
    N_HEADS, W_HIDDEN = weights_proj_weight.shape
    assert HIDDEN_SIZE == W_HIDDEN, (
        f"x.shape[-1]={HIDDEN_SIZE} != weights_proj_weight.shape[-1]={W_HIDDEN}"
    )
    assert weights_proj_weight.dtype == torch.float32, (
        f"weights_proj_weight must be fp32, got {weights_proj_weight.dtype}"
    )
    assert seqlens_32.shape == (N,), f"seqlens_32 shape {seqlens_32.shape} != ({N},)"
    assert block_tables.shape[0] == N, (
        f"block_tables.shape[0]={block_tables.shape[0]} != N={N}"
    )
    NUM_BLOCKS = block_tables.shape[1]
    # The split-based ::pool extraction below needs a pool_size-multiple
    # width. Short decode sequences (e.g. a 1-block warmup request) can leave
    # the table narrower than pool_size; pad with dummy pages — entries past
    # pool_seqlens (= seqlens_32 // pool_size) are never consumed downstream,
    # and padded width == ceil-div of the real width, matching the torch-side
    # build_pooled_page_table_64 slicing.
    padded_blocks = -(-NUM_BLOCKS // pool_size) * pool_size or pool_size
    if padded_blocks != NUM_BLOCKS:
        block_tables = F.pad(block_tables, (0, padded_blocks - NUM_BLOCKS))
    NUM_BLOCKS = padded_blocks
    N_POOL = NUM_BLOCKS // pool_size

    # 2. hyper-params (fixed internally, not exposed)
    block_b = 32
    block_k = 256
    BLOCK_M_A = 128
    BLOCK_M_B = 4096
    # GEMM (Cube) 用 num_aicore, VV (纯 Vector) 用 num_vectorcore
    num_aicore, num_vectorcore = get_device_properties()

    assert HIDDEN_SIZE % block_k == 0, (
        f"HIDDEN_SIZE={HIDDEN_SIZE} not divisible by block_k={block_k}"
    )
    assert N_HEADS <= 32, f"N_HEADS={N_HEADS} unsupported"
    assert BLOCK_M_B % pool_size == 0, (
        f"BLOCK_M_B={BLOCK_M_B} not divisible by pool_size={pool_size}"
    )
    assert pool_size == 4, f"pool_size={pool_size} must be 4 (split 抽取硬编码为 ::4)"

    # 3. output tensors
    w_bf16 = torch.empty((N, N_HEADS), dtype=torch.bfloat16, device=x.device)
    pool_seqlens = torch.empty((N,), dtype=torch.int32, device=x.device)
    pool_block_tables = torch.empty((N, N_POOL), dtype=torch.int32, device=x.device)

    # 4. combined_scale (inlined at compile time)
    combined_scale = float(n_heads_pow * softmax_scale)

    # 5. kernel 1 (pure Cube GEMM), grid=num_aicore
    _fused_decode_prepare_indexer_gemm_kernel[(num_aicore,)](
        x,
        weights_proj_weight,
        w_bf16,
        N,
        combined_scale,
        HIDDEN_SIZE=HIDDEN_SIZE,
        N_HEADS=N_HEADS,
        BLOCK_B=block_b,
        BLOCK_K=block_k,
        NUM_CORES=num_aicore,
    )

    # 6. kernel 2 (pure Vector: pool_seqlens + pool_block_tables), grid=num_vectorcore
    #    两 kernel 间无需显式同步 (HBM store→load 弱依赖, NPU 默认保序)
    _fused_decode_prepare_indexer_vv_kernel[(num_vectorcore,)](
        seqlens_32,
        block_tables,
        pool_seqlens,
        pool_block_tables,
        N,
        NUM_BLOCKS=NUM_BLOCKS,
        POOL_SIZE=pool_size,
        BLOCK_M_A=BLOCK_M_A,
        BLOCK_M_B=BLOCK_M_B,
        NUM_CORES=num_vectorcore,
    )

    return w_bf16, pool_seqlens, pool_block_tables
