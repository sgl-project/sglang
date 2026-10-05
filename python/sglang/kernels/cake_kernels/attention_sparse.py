"""Cake sparse / block-sparse attention and indexer kernels via FlashInfer.

FlashInfer entries (all at FlashInfer ``46340689a5ab``):

* ``flashinfer.sparse.BlockSparseAttentionWrapper(workspace, backend="cake")``
  (``flashinfer.cake_vsa.plan_cake_vsa`` / ``run_cake_vsa``; SM100 / SM103;
  cubins built with nvcc from ``csrc/cake_vsa``): block-sparse attention with
  square blocks ``R == C in {64, 128}``, ``M, N % R == 0``, head_dim 64 / 96 /
  128, FP16 / BF16 ``q [M, Hq, D]``, ``k / v [N, Hkv, D]`` contiguous. Masks
  through ``block_mask`` bool ``[Hq | Hkv, M/R, N/C]``, CSR ``indptr`` /
  ``indices`` or ``q2k_indices`` int32 ``[Hq, M/R, topk]`` (+ ``q2k_num``),
  ``kv_block_lens [N/C]`` (blk64 only). Block 64 => BF16 D128 native heads
  only; D64 / D96 => native-head BF16; BF16 GQA requires ``Hq == 8``, group
  2 / 4 / 8 and identical masks per KV group; every row >= 1 block; compact
  profiles <= 64 blocks; longseq (``N >= 16384``, ``Hq == 8``) fixed topk <=
  192; ultrasparse exactly 6 blocks. ``return_lse`` is unsupported on D64 /
  D96, BF16 GQA, ultrasparse and longseq. FP8 scale tensors are rejected.
* ``flashinfer.sparse.VariableBlockSparseAttentionWrapper(workspace,
  backend="cake" | "cake_cute")`` (``flashinfer.cake_vsa_sm90``, JIT
  ``flashinfer.jit.cake_vsa_sm90``; SM90 only): BF16 HND ``q / k / v
  (H, S, 128)``, ``Hq == Hkv``, noncausal, 64-token query and KV blocks
  (``block_row_sz`` / ``block_col_sz``), 1..64 selected KV blocks per row
  (ragged allowed), finite ``sm_scale`` (0 / negative allowed), no LSE / PDL /
  pos-enc / soft-cap; ``out`` must not overlap Q/K/V. ``"cake_cute"`` runs the
  same plans through the CuTe DSL engine (needs ``nvidia-cutlass-dsl``).
* ``flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64.bsa_attn_sm100_blk64_fwd(
  backend="cake", q_scale=, k_scale=, v_scale=)`` ->
  ``bsa_sage_sm100_cake.bsa_attn_sm100_sage_fwd_cake`` (JIT
  ``flashinfer.jit.cake_sage_block_sparse_attention``; SM100 / SM103):
  Sage-FP8 block-sparse attention. BSHD e4m3 ``q (B, Sq, H, 128)``, ``k / v
  (B, Sk, Hkv, 128)`` contiguous, ``H % Hkv == 0``; int32 ``q2k_block_index
  (B, H, ceil(Sq/64), capacity)``; ``block_sparse_num <= capacity``; FP32
  ``q_scale (B, H, Sq)``, ``k_scale (B, Hkv, ceil(Sk/16))``, ``v_scale
  (Hkv, 128) | (B, Hkv, 128)``; optional int32 ``block_sizes`` (rank 1 / 2 /
  3) and ``q2k_block_nums (B, H, ceil(Sq/64))`` (0 -> zero rows, ``-inf``
  LSE); out BF16 BSHD, FP32 ``lse (B, H, Sq)``. ``kv_splits`` / ``use_clc``
  are ignored. Quantize with ``bsa_sage_sm100_cake.sage_fp8_quantize_sm100``
  (forwarded by ``sglang.kernels.cake_kernels.quantization``).
* ``flashinfer.cute_dsl.sparse.bsa_attn_sm120.bsa_attn_sm120_blk64_sage_fwd(
  backend="cake")`` (JIT ``flashinfer.jit.cake_sage_block_sparse_attention``;
  exact cc 12.0): INT8 BHSD ``q_int8 / k_int8 [B, H, S, 128]`` (MHA), FP8
  e4m3 HDS ``v_fp8 [B, H, 128, padded_Sk]``, FP32 scales, int32
  ``q2k_block_index [B, H, ceil(Sq/64), cap]``, ``block_sparse_num`` in
  ``[0, cap]``, optional ``q2k_block_nums``; caller-owned BF16 BHSD ``out``
  (required); ``tma_descriptor_workspace`` accepted but unused; non-causal,
  no LSE. ``uniform_block_count`` / ``contiguous_block_indices`` are unchecked
  caller guarantees.
* ``flashinfer.msa_ops.prepare_msa_nvfp4_sparse_decode`` (impl
  ``flashinfer.experimental.msa_nvfp4_decode.cake_backend``; SM100 / SM103 /
  SM107): NVFP4 paged-KV MiniMax Sparse Attention decode. BF16 ``q [B *
  seqlen_q, Hq, 128]``, uint8 ``k / v [pages, Hkv, 128, 64]`` (E2M1 packed
  page-pool views), ``k_scale / v_scale [pages, Hkv, 128, 8]`` uint8 or e4m3
  (V scale swizzled), int32 ``q2k_indices [Hkv, B * seqlen_q, 16]`` ascending
  with ``-1`` padding, int32 ``page_table [B, max_pages]``, int32
  ``seqused_k [B]``; page 128, head_dim 128, ``seqlen_q`` 1..32 (causal,
  right-aligned), 1..16 Q heads per KV head, positive global scales; BF16
  out, natural-log FP32 lse; uint8 workspace >=
  ``msa_nvfp4_decode_workspace_size`` when the plan splits.
* ``flashinfer.dsa_indexer.dsa_indexer_topk`` and
  ``flashinfer.experimental.cake_dsa_indexer.cake_backend.prepare_dsa_indexer_topk``
  (SM100 / SM103 / SM107): exact DSA indexer top-k. BF16 ``q [T, 32, 128]``
  contiguous, BF16 ``k [Tkv, 128]`` (row stride a multiple of 8 elements,
  strided ``packed[:, :128]`` views accepted), FP32 ``w [T, 32]``, int32
  device ``cu_seqlens_q / cu_seqlens_k [S + 1]``, optional int64
  ``q_causal_offsets [S]``, ``1 <= top_k <= 4096``; returns int32 ``indices
  [T, top_k]`` ascending (``-1`` padding) and FP32 ``scores`` (``-inf``
  padding). Workspace sized by ``dsa_indexer_topk_workspace_size``.
* ``flashinfer.dense_mqa.prepare_dense_mqa_logits`` (impl
  ``flashinfer.experimental.deepgemm_dense_mqa``; exact sm_100a 148-SM /
  sm_103a 152-SM parts): DeepSeek-V3.2 lightning-indexer logits. 32 heads,
  D=128; FP4: uint8 / int8 ``q [Q, 32, 64]``, ``kv [K, 64]``, UE8M0 scales
  ``q_scales [Q, 32, 4]``, ``kv_scales [K, 4]``; FP8: e4m3 ``q [Q_storage >=
  max(4, Q), 32, 128]``, ``kv [K, 128]``, FP32 ``kv_scales [K]``; FP32
  ``weights [Q(_storage), 32]``; int32 ``starts / ends [Q]``; ``K % 256 ==
  0``; FP32 ``output [Q, align(K + 256, 8)]`` with ``-inf`` outside the
  window. Exported routes: ``Q in {1, 16, 128} x K in {4096, 32768, 131072}``
  plus ``Q=16, K=1048576``.
* ``flashinfer.dense_mqa.fp8_mqa_logits`` (same package; DeepGEMM
  ``fp8_mqa_logits`` signature): e4m3 ``q [Q, H, 128]`` with ``H`` in the
  catalog's head set (32; 64 once exported), ``kv = (e4m3 [K, 128], f32
  scales [K])``, f32 ``weights [Q, H]``, int32 ``ks / ke [Q]``,
  ``clean_logits`` accepted (no effect: every cell is written), ``max_seqlen_k``
  must be 0 -> f32 ``[Q, K]`` view. The host helper
  ``dense_route_available(H, Q, K)`` tells whether the shipped catalog serves
  the point (today ``K % 256 == 0``, ``Q <= max_queries()``).
* ``flashinfer.paged_mqa.get_paged_mqa_logits_metadata`` /
  ``fp8_paged_mqa_logits`` / ``prepare_paged_mqa_logits`` (impl
  ``flashinfer.experimental.deepgemm_dense_mqa.paged_mqa``; DeepGEMM paged
  signatures): int32 ``context_lens [B, next_n]`` (2-D), ``block_kv`` 64,
  ``num_sms`` -> int32 ``[num_sms + 1, 2]`` metadata; e4m3 ``q [B, next_n, H,
  128]``, uint8 fused ``kv_cache [pages, 64, 1, 132]``, f32 ``weights [B *
  next_n, H]``, int32 ``block_table [B, S]`` (unit column stride),
  ``max_context_len``, ``clean_logits=False`` only -> f32 ``[B * next_n,
  max_context_len]`` view (DeepGEMM mask semantics). Any batch size.
  ``paged_route_available(H, 64, next_n)`` consults the shipped catalog.
* ``flashinfer.sparse_mqa.prepare_sparse_mqa_metadata`` /
  ``prepare_sparse_mqa_logits`` (impl
  ``flashinfer.experimental.deepgemm_sparse_mqa``; same parts): sorted
  dup-padded int32 ``sparse_indices [Q, capacity]`` (2048 blocks of 8 KV
  tokens, 64-token pages), opaque uint8 metadata + int32 workspace (counters
  self-restoring), then packed E2M1 ``q [Q, 32, 64]`` / e4m3 ``q [Q, 32,
  128]`` with int32 ``sf_q [Q, 32]`` and BF16 ``weights [Q, 32]`` against
  contiguous or paged KV; BF16 ``output [Q, capacity * 8]``.

CUDA graphs: the SM100 VSA wrapper plans outside capture and ``run`` launches
one route (``return_lse`` as documented); the SM90 wrapper's ``plan`` uploads
tile metadata and must not replace a plan a captured graph still uses
(``check_replan``), ``run`` is exactly one launch after an event wait and
capture needs a preallocated ``out``. The Sage SM100 route keeps a per-device
384-byte TMA workspace; the SM120 Sage route has every buffer caller-owned.
MSA NVFP4 / DSA indexer / dense and sparse MQA prepared runners launch
without allocation or host sync (the kernels self-reset their counters; do not
share a workspace region between live runners); the one-shot
``dsa_indexer_topk`` allocates outputs and workspace through the caching
allocator.

Not supported here (keep the existing SGLang path): FP8 scale tensors in the
SM100 VSA wrapper; causal / pos-enc / soft-cap / LSE on the SM90 wrapper;
BF16 (non-Sage) SM120 block-sparse attention (no Cake branch); MSA routes
other than NVFP4 page 128 / top-16; indexer head counts other than 32.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from sglang.kernels.cake_kernels.attention_common import (
    SM90,
    SM100,
    SM103,
    SM107,
    SM120,
    archs_in,
    device_capability,
    device_sm_count,
    flashinfer_module_available,
)

if TYPE_CHECKING:
    import torch

FI_VSA_MODULE = "flashinfer.cake_vsa"
FI_SPARSE_MODULE = "flashinfer.sparse"
FI_VSA_SM90_MODULE = "flashinfer.cake_vsa_sm90"
FI_VSA_SM90_JIT_MODULE = "flashinfer.jit.cake_vsa_sm90"
FI_VSA_SM90_CUTE_MODULE = "flashinfer.experimental.cake_vsa_sm90_cute"
FI_BSA_SM100_MODULE = "flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64"
FI_SAGE_SM100_MODULE = "flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake"
FI_SAGE_JIT_MODULE = "flashinfer.jit.cake_sage_block_sparse_attention"
FI_BSA_SM120_MODULE = "flashinfer.cute_dsl.sparse.bsa_attn_sm120"
FI_MSA_MODULE = "flashinfer.msa_ops"
FI_MSA_BACKEND_MODULE = "flashinfer.experimental.msa_nvfp4_decode.cake_backend"
FI_DSA_INDEXER_MODULE = "flashinfer.dsa_indexer"
FI_DSA_INDEXER_BACKEND_MODULE = "flashinfer.experimental.cake_dsa_indexer.cake_backend"
FI_DENSE_MQA_MODULE = "flashinfer.dense_mqa"
FI_DENSE_MQA_BACKEND_MODULE = "flashinfer.experimental.deepgemm_dense_mqa.dense_mqa"
FI_PAGED_MQA_MODULE = "flashinfer.paged_mqa"
FI_PAGED_MQA_BACKEND_MODULE = "flashinfer.experimental.deepgemm_dense_mqa.paged_mqa"
FI_SPARSE_MQA_MODULE = "flashinfer.sparse_mqa"
FI_SPARSE_MQA_BACKEND_MODULE = "flashinfer.experimental.deepgemm_sparse_mqa.sparse_mqa"

VSA_ARCHS = (SM100, SM103)
VSA_SM90_ARCHS = (SM90,)
SAGE_SM100_ARCHS = (SM100, SM103)
SAGE_SM120_ARCHS = (SM120,)
MSA_ARCHS = (SM100, SM103, SM107)
DSA_INDEXER_ARCHS = (SM100, SM103, SM107)
MQA_ARCHS = (SM100, SM103)
MQA_SM_COUNTS = {SM100: 148, SM103: 152}

VSA_BLOCK_SIZES = (64, 128)
VSA_HEAD_DIMS = (64, 96, 128)
VSA_SM90_HEAD_DIM = 128
VSA_SM90_BLOCK = 64
VSA_SM90_MAX_SELECTED = 64
SAGE_HEAD_DIM = 128
SAGE_BLOCK = 64
SAGE_K_SCALE_GROUP = 16
MSA_HEAD_DIM = 128
MSA_PAGE_SIZE = 128
MSA_TOPK = 16
MSA_MAX_SEQLEN_Q = 32
MSA_MAX_GROUP_SIZE = 16
DSA_INDEXER_NUM_HEADS = 32
DSA_INDEXER_HEAD_DIM = 128
DSA_INDEXER_MAX_TOP_K = 4096
DENSE_MQA_HEADS = 32
DENSE_MQA_K_MULTIPLE = 256
SPARSE_MQA_HEADS = 32
# DeepGEMM-signature indexer entries (DeepSeek-V3.2 engine contract).
DSA_MQA_HEADS = (32, 64)
DSA_MQA_HEAD_DIM = 128
DSA_MQA_PAGE = 64
DSA_MQA_FUSED_ROW_BYTES = DSA_MQA_HEAD_DIM + 4  # FP8 row + one FP32 scale


# --------------------------------------------------------------------------
# SM100 / SM103 Cake VSA (BlockSparseAttentionWrapper backend="cake")
# --------------------------------------------------------------------------


def supports_block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> bool:
    """Coarse admission check for the SM100 Cake VSA; never raises.

    ``plan_cake_vsa`` performs the exact per-profile admission (block counts,
    GQA mask identity, longseq / ultrasparse shapes) and raises for masks no
    generated profile serves.
    """
    try:
        import torch

        if not archs_in(VSA_ARCHS, q, k, v):
            return False
        if not flashinfer_module_available(FI_VSA_MODULE, FI_SPARSE_MODULE):
            return False
        if q.dtype not in (torch.float16, torch.bfloat16):
            return False
        if k.dtype != q.dtype or v.dtype != q.dtype:
            return False
        if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
            return False
        if block_size not in VSA_BLOCK_SIZES or head_dim not in VSA_HEAD_DIMS:
            return False
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            return False
        m, n = int(q.shape[0]), int(k.shape[0])
        if m % block_size or n % block_size:
            return False
        if tuple(q.shape[1:]) != (num_qo_heads, head_dim):
            return False
        if tuple(k.shape[1:]) != (num_kv_heads, head_dim):
            return False
        if tuple(v.shape) != tuple(k.shape):
            return False
        if num_qo_heads % num_kv_heads:
            return False
        group = num_qo_heads // num_kv_heads
        if group != 1 and (num_qo_heads != 8 or group not in (2, 4, 8)):
            return False
        if block_size == 64 and (head_dim != 128 or q.dtype != torch.bfloat16):
            return False
        if head_dim != 128 and q.dtype != torch.bfloat16:
            return False
        return True
    except Exception:
        return False


def create_block_sparse_attention_wrapper(float_workspace_buffer: torch.Tensor):
    """``flashinfer.sparse.BlockSparseAttentionWrapper(workspace, backend="cake")``.

    Call ``wrapper.plan(indptr, indices, M, N, R, C, num_qo_heads,
    num_kv_heads, head_dim, ..., block_mask=, kv_block_lens=, q2k_indices=,
    q2k_num=, q_data_type=, kv_data_type=)`` outside capture, then
    ``wrapper.run(q, k, v, out=, lse=, return_lse=)``.
    """
    from flashinfer.sparse import BlockSparseAttentionWrapper

    return BlockSparseAttentionWrapper(float_workspace_buffer, backend="cake")


def plan_vsa(
    indptr: Optional[torch.Tensor],
    indices: Optional[torch.Tensor],
    block_mask: Optional[torch.Tensor],
    kv_block_lens: Optional[torch.Tensor],
    q2k_indices: Optional[torch.Tensor],
    q2k_num: Optional[torch.Tensor],
    *,
    M: int,
    N: int,
    R: int,
    C: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    q_data_type: torch.dtype,
    sm_scale: Optional[float],
    device: torch.device,
):
    """Forward of ``flashinfer.cake_vsa.plan_cake_vsa`` (returns the plan dict)."""
    from flashinfer.cake_vsa import plan_cake_vsa

    return plan_cake_vsa(
        indptr,
        indices,
        block_mask,
        kv_block_lens,
        q2k_indices,
        q2k_num,
        M=M,
        N=N,
        R=R,
        C=C,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_data_type=q_data_type,
        sm_scale=sm_scale,
        device=device,
    )


def run_vsa(
    plan,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
):
    """Forward of ``flashinfer.cake_vsa.run_cake_vsa(..., backend="cake")``."""
    from flashinfer.cake_vsa import run_cake_vsa

    return run_cake_vsa(
        plan, q, k, v, out=out, lse=lse, return_lse=return_lse, backend="cake"
    )


# --------------------------------------------------------------------------
# SM90 Cake VSA (VariableBlockSparseAttentionWrapper backend="cake"/"cake_cute")
# --------------------------------------------------------------------------


def supports_variable_block_sparse_attention_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    engine: str = "cuda",
) -> bool:
    """Admission check for the Hopper Cake VSA wrapper; never raises."""
    try:
        import torch

        modules = [FI_VSA_SM90_MODULE, FI_VSA_SM90_JIT_MODULE, FI_SPARSE_MODULE]
        if engine == "cute":
            modules.append(FI_VSA_SM90_CUTE_MODULE)
        elif engine != "cuda":
            return False
        return (
            archs_in(VSA_SM90_ARCHS, q, k, v)
            and flashinfer_module_available(*modules)
            and q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and v.dtype == torch.bfloat16
            and q.ndim == 3
            and k.ndim == 3
            and v.ndim == 3
            and int(q.shape[0]) == int(k.shape[0]) == int(v.shape[0])
            and int(q.shape[2]) == VSA_SM90_HEAD_DIM
            and int(k.shape[2]) == VSA_SM90_HEAD_DIM
            and int(v.shape[2]) == VSA_SM90_HEAD_DIM
            and int(q.shape[1]) % VSA_SM90_BLOCK == 0
            and int(k.shape[1]) % VSA_SM90_BLOCK == 0
            and int(k.shape[1]) == int(v.shape[1])
        )
    except Exception:
        return False


def create_variable_block_sparse_attention_wrapper_sm90(
    float_workspace_buffer: torch.Tensor, *, engine: str = "cuda"
):
    """``VariableBlockSparseAttentionWrapper(workspace, backend="cake"|"cake_cute")``.

    ``engine="cuda"`` selects ``backend="cake"``; ``engine="cute"`` selects
    ``backend="cake_cute"``. Call ``wrapper.plan(block_mask_map, block_row_sz,
    block_col_sz, num_qo_heads, num_kv_heads, head_dim, sm_scale=,
    q_data_type=, kv_data_type=)`` outside capture, then
    ``wrapper.run(q, k, v, out=)`` (one launch; graph-capturable with a fixed
    plan and preallocated ``out``).
    """
    from flashinfer.sparse import VariableBlockSparseAttentionWrapper

    if engine not in ("cuda", "cute"):
        raise ValueError(f"engine must be 'cuda' or 'cute', got {engine!r}")
    backend = "cake" if engine == "cuda" else "cake_cute"
    return VariableBlockSparseAttentionWrapper(float_workspace_buffer, backend=backend)


# --------------------------------------------------------------------------
# Sage-FP8 block-sparse attention, SM100 / SM103 (bsa_attn_sm100_blk64_fwd)
# --------------------------------------------------------------------------


def supports_bsa_attn_sm100_blk64_sage_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    *,
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
) -> bool:
    """Admission check mirroring the Cake Sage SM100 contract; never raises."""
    try:
        import torch

        if q_scale is None or k_scale is None or v_scale is None:
            return False
        if not archs_in(SAGE_SM100_ARCHS, q, k, v, q2k_block_index, q_scale):
            return False
        if not flashinfer_module_available(
            FI_BSA_SM100_MODULE, FI_SAGE_SM100_MODULE, FI_SAGE_JIT_MODULE
        ):
            return False
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q2k_block_index.ndim != 4:
            return False
        batch, sq, heads, dim = (int(s) for s in q.shape)
        sk, kv_heads = int(k.shape[1]), int(k.shape[2])
        q_blocks = -(-sq // SAGE_BLOCK)
        k_groups = -(-sk // SAGE_K_SCALE_GROUP)
        return (
            q.dtype == torch.float8_e4m3fn
            and k.dtype == torch.float8_e4m3fn
            and v.dtype == torch.float8_e4m3fn
            and q.is_contiguous()
            and k.is_contiguous()
            and v.is_contiguous()
            and dim == SAGE_HEAD_DIM
            and tuple(k.shape) == (batch, sk, kv_heads, SAGE_HEAD_DIM)
            and tuple(v.shape) == tuple(k.shape)
            and kv_heads > 0
            and heads % kv_heads == 0
            and q2k_block_index.dtype == torch.int32
            and tuple(q2k_block_index.shape[:3]) == (batch, heads, q_blocks)
            and q_scale.dtype == torch.float32
            and tuple(q_scale.shape) == (batch, heads, sq)
            and k_scale.dtype == torch.float32
            and tuple(k_scale.shape) == (batch, kv_heads, k_groups)
            and v_scale.dtype == torch.float32
            and tuple(v_scale.shape)
            in ((kv_heads, SAGE_HEAD_DIM), (batch, kv_heads, SAGE_HEAD_DIM))
        )
    except Exception:
        return False


def bsa_attn_sm100_blk64_sage_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    *,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Forward to ``bsa_attn_sm100_blk64_fwd(..., backend="cake")``.

    Returns ``(out, lse)`` with ``lse`` ``None`` unless ``return_lse``.
    """
    from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import (
        bsa_attn_sm100_blk64_fwd,
    )

    return bsa_attn_sm100_blk64_fwd(
        q,
        k,
        v,
        q2k_block_index,
        block_sparse_num,
        block_sizes=block_sizes,
        q2k_block_nums=q2k_block_nums,
        softmax_scale=softmax_scale,
        return_lse=return_lse,
        out=out,
        lse=lse,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        backend="cake",
    )


# --------------------------------------------------------------------------
# Sage block-sparse attention, SM120 (bsa_attn_sm120_blk64_sage_fwd)
# --------------------------------------------------------------------------


def supports_bsa_attn_sm120_blk64_sage_fwd(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    q2k_block_index: torch.Tensor,
    *,
    out: Optional[torch.Tensor],
) -> bool:
    """Admission check mirroring the Cake SM120 Sage contract; never raises."""
    try:
        import torch

        if out is None:
            return False
        if not archs_in(SAGE_SM120_ARCHS, q_int8, k_int8, v_fp8, q2k_block_index, out):
            return False
        if not flashinfer_module_available(FI_BSA_SM120_MODULE, FI_SAGE_JIT_MODULE):
            return False
        if q_int8.ndim != 4 or k_int8.ndim != 4 or v_fp8.ndim != 4:
            return False
        batch, heads, sq, dim = (int(s) for s in q_int8.shape)
        sk = int(k_int8.shape[2])
        return (
            q_int8.dtype == torch.int8
            and k_int8.dtype == torch.int8
            and v_fp8.dtype == torch.float8_e4m3fn
            and q_int8.is_contiguous()
            and k_int8.is_contiguous()
            and v_fp8.is_contiguous()
            and dim == SAGE_HEAD_DIM
            and tuple(k_int8.shape) == (batch, heads, sk, SAGE_HEAD_DIM)
            and tuple(v_fp8.shape[:3]) == (batch, heads, SAGE_HEAD_DIM)
            and int(v_fp8.shape[3]) >= sk
            and q2k_block_index.dtype == torch.int32
            and q2k_block_index.ndim == 4
            and tuple(q2k_block_index.shape[:3]) == (batch, heads, -(-sq // SAGE_BLOCK))
            and out.dtype == torch.bfloat16
            and tuple(out.shape) == (batch, heads, sq, SAGE_HEAD_DIM)
        )
    except Exception:
        return False


def bsa_attn_sm120_blk64_sage_fwd(
    q_int8: torch.Tensor,
    k_int8: torch.Tensor,
    v_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q2k_block_index: torch.Tensor,
    block_sparse_num: int,
    block_sizes: Optional[torch.Tensor] = None,
    q2k_block_nums: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    *,
    out: Optional[torch.Tensor] = None,
    tma_descriptor_workspace: Optional[torch.Tensor] = None,
    uniform_block_count: bool = False,
    contiguous_block_indices: bool = False,
) -> torch.Tensor:
    """Forward to ``bsa_attn_sm120_blk64_sage_fwd(..., backend="cake")``."""
    from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_sage_fwd

    return bsa_attn_sm120_blk64_sage_fwd(
        q_int8,
        k_int8,
        v_fp8,
        q_scale,
        k_scale,
        v_scale,
        q2k_block_index,
        block_sparse_num,
        block_sizes=block_sizes,
        q2k_block_nums=q2k_block_nums,
        softmax_scale=softmax_scale,
        out=out,
        tma_descriptor_workspace=tma_descriptor_workspace,
        uniform_block_count=uniform_block_count,
        contiguous_block_indices=contiguous_block_indices,
        backend="cake",
    )


# --------------------------------------------------------------------------
# MSA NVFP4 paged sparse decode (prepared runner)
# --------------------------------------------------------------------------


def supports_msa_nvfp4_sparse_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    seqlen_q: int = 1,
) -> bool:
    """Admission check mirroring ``validate_msa_nvfp4_decode_inputs``."""
    try:
        import torch

        if not archs_in(MSA_ARCHS, q, k, v, q2k_indices, page_table, seqused_k):
            return False
        if not flashinfer_module_available(FI_MSA_MODULE, FI_MSA_BACKEND_MODULE):
            return False
        if q.ndim != 3 or k.ndim != 4 or v.ndim != 4 or q2k_indices.ndim != 3:
            return False
        total_q, num_q_heads, head_dim = (int(s) for s in q.shape)
        num_kv_heads = int(k.shape[1])
        batch = int(seqused_k.numel())
        return (
            q.dtype == torch.bfloat16
            and k.dtype == torch.uint8
            and v.dtype == torch.uint8
            and k_scale.dtype in (torch.uint8, torch.float8_e4m3fn)
            and v_scale.dtype in (torch.uint8, torch.float8_e4m3fn)
            and q2k_indices.dtype == torch.int32
            and page_table.dtype == torch.int32
            and seqused_k.dtype == torch.int32
            and head_dim == MSA_HEAD_DIM
            and tuple(k.shape[1:]) == (num_kv_heads, MSA_PAGE_SIZE, MSA_HEAD_DIM // 2)
            and tuple(v.shape) == tuple(k.shape)
            and tuple(k_scale.shape)
            == (int(k.shape[0]), num_kv_heads, MSA_PAGE_SIZE, MSA_HEAD_DIM // 16)
            and tuple(v_scale.shape) == tuple(k_scale.shape)
            and num_kv_heads > 0
            and num_q_heads % num_kv_heads == 0
            and 1 <= num_q_heads // num_kv_heads <= MSA_MAX_GROUP_SIZE
            and 1 <= seqlen_q <= MSA_MAX_SEQLEN_Q
            and page_table.ndim == 2
            and int(page_table.shape[0]) == batch
            and total_q == batch * seqlen_q
            and tuple(q2k_indices.shape) == (num_kv_heads, total_q, MSA_TOPK)
        )
    except Exception:
        return False


def prepare_msa_nvfp4_sparse_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    k_global_scale: float,
    v_global_scale: float,
    workspace_buffer: Optional[torch.Tensor] = None,
    seqlen_q: int = 1,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
):
    """Forward to FlashInfer; returns an ``MSANvfp4DecodeRunner``.

    ``runner.launch()`` (or ``runner()``) returns the BF16 output; the bound
    ``lse`` receives natural-log LSE. Allocation-free, graph-capturable.
    """
    from flashinfer.msa_ops import prepare_msa_nvfp4_sparse_decode

    return prepare_msa_nvfp4_sparse_decode(
        q,
        k,
        v,
        q2k_indices,
        k_scale=k_scale,
        v_scale=v_scale,
        page_table=page_table,
        seqused_k=seqused_k,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        workspace_buffer=workspace_buffer,
        seqlen_q=seqlen_q,
        softmax_scale=softmax_scale,
        out=out,
        lse=lse,
        backend="cake",
    )


def msa_nvfp4_decode_workspace_size(
    batch: int, num_kv_heads: int, device, *, seqlen_q: int = 1
) -> int:
    """Host-only forward of ``msa_nvfp4_decode_workspace_size``."""
    from flashinfer.experimental.msa_nvfp4_decode.cake_backend import (
        msa_nvfp4_decode_workspace_size,
    )

    return msa_nvfp4_decode_workspace_size(
        batch, num_kv_heads, device, seqlen_q=seqlen_q
    )


# --------------------------------------------------------------------------
# DSA indexer top-k (one-shot and prepared)
# --------------------------------------------------------------------------


def supports_dsa_indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int = 2048,
) -> bool:
    """Admission check mirroring ``validate_dsa_indexer_inputs``; never raises."""
    try:
        import torch

        if not archs_in(DSA_INDEXER_ARCHS, q, k, w, cu_seqlens_q, cu_seqlens_k):
            return False
        if not flashinfer_module_available(
            FI_DSA_INDEXER_MODULE, FI_DSA_INDEXER_BACKEND_MODULE
        ):
            return False
        return (
            q.dtype == torch.bfloat16
            and k.dtype == torch.bfloat16
            and w.dtype == torch.float32
            and cu_seqlens_q.dtype == torch.int32
            and cu_seqlens_k.dtype == torch.int32
            and q.ndim == 3
            and tuple(q.shape[1:]) == (DSA_INDEXER_NUM_HEADS, DSA_INDEXER_HEAD_DIM)
            and q.is_contiguous()
            and k.ndim == 2
            and int(k.shape[1]) == DSA_INDEXER_HEAD_DIM
            and int(k.stride(1)) == 1
            and int(k.stride(0)) % 8 == 0
            and tuple(w.shape) == (int(q.shape[0]), DSA_INDEXER_NUM_HEADS)
            and cu_seqlens_q.ndim == 1
            and tuple(cu_seqlens_k.shape) == tuple(cu_seqlens_q.shape)
            and int(cu_seqlens_q.numel()) >= 2
            and 1 <= top_k <= DSA_INDEXER_MAX_TOP_K
        )
    except Exception:
        return False


def dsa_indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int = 2048,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward to ``flashinfer.dsa_indexer.dsa_indexer_topk(backend="cake")``.

    Returns ``(indices int32 [T, top_k], scores f32 [T, top_k])``.
    """
    from flashinfer.dsa_indexer import dsa_indexer_topk

    return dsa_indexer_topk(
        q,
        k,
        w,
        cu_seqlens_q,
        cu_seqlens_k,
        top_k=top_k,
        softmax_scale=softmax_scale,
        q_causal_offsets=q_causal_offsets,
        ratio=ratio,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        workspace_buffer=workspace_buffer,
        indices=indices,
        scores=scores,
        backend="cake",
    )


def prepare_dsa_indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int = 2048,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
):
    """Forward to the backend ``prepare_dsa_indexer_topk``; returns the runner.

    ``runner.run()`` returns ``(indices, scores)`` and allocates nothing when
    ``workspace_buffer`` / ``indices`` / ``scores`` are caller-owned.
    """
    from flashinfer.experimental.cake_dsa_indexer.cake_backend import (
        prepare_dsa_indexer_topk,
    )

    return prepare_dsa_indexer_topk(
        q,
        k,
        w,
        cu_seqlens_q,
        cu_seqlens_k,
        top_k=top_k,
        softmax_scale=softmax_scale,
        q_causal_offsets=q_causal_offsets,
        ratio=ratio,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        workspace_buffer=workspace_buffer,
        indices=indices,
        scores=scores,
        backend="cake",
    )


def dsa_indexer_topk_workspace_size(top_k: int = 2048, device=None) -> int:
    """Host-only forward of ``dsa_indexer_topk_workspace_size``."""
    from flashinfer.dsa_indexer import dsa_indexer_topk_workspace_size

    return dsa_indexer_topk_workspace_size(top_k, device)


# --------------------------------------------------------------------------
# DeepGEMM-family dense / sparse MQA logits (prepared plans)
# --------------------------------------------------------------------------


def _mqa_device_ok(*tensors) -> bool:
    if not archs_in(MQA_ARCHS, *tensors):
        return False
    index = tensors[0].device.index
    return device_sm_count(index) == MQA_SM_COUNTS.get(device_capability(index))


def supports_dense_mqa_logits(
    precision: str,
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
) -> bool:
    """Admission check for the dense MQA logits plan; never raises."""
    try:
        import torch

        if not _mqa_device_ok(q, kv, weights, starts, ends):
            return False
        if not flashinfer_module_available(
            FI_DENSE_MQA_MODULE, FI_DENSE_MQA_BACKEND_MODULE
        ):
            return False
        if q.ndim != 3 or kv.ndim != 2 or int(q.shape[1]) != DENSE_MQA_HEADS:
            return False
        keys = int(kv.shape[0])
        if keys % DENSE_MQA_K_MULTIPLE:
            return False
        if precision == "fp4":
            dtype_ok = q.dtype in (torch.uint8, torch.int8) and kv.dtype == q.dtype
            dims_ok = int(q.shape[2]) == 64 and int(kv.shape[1]) == 64
        elif precision == "fp8":
            dtype_ok = q.dtype == torch.float8_e4m3fn and kv.dtype == q.dtype
            dims_ok = int(q.shape[2]) == 128 and int(kv.shape[1]) == 128
        else:
            return False
        return (
            dtype_ok
            and dims_ok
            and weights.dtype == torch.float32
            and starts.dtype == torch.int32
            and ends.dtype == torch.int32
            and starts.ndim == 1
            and tuple(ends.shape) == tuple(starts.shape)
            and int(starts.numel()) <= int(q.shape[0])
            and int(weights.shape[1]) == DENSE_MQA_HEADS
        )
    except Exception:
        return False


def prepare_dense_mqa_logits(
    precision: str,
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    q_scales: Optional[torch.Tensor] = None,
    kv_scales: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
):
    """Forward to ``flashinfer.dense_mqa.prepare_dense_mqa_logits``.

    Returns a ``DenseMqaPlan``; ``plan.run()`` submits the metadata program and
    the fused logits kernel, ``plan.logical_output`` views ``[Q, K]``.
    """
    from flashinfer.dense_mqa import prepare_dense_mqa_logits

    return prepare_dense_mqa_logits(
        precision,
        q,
        kv,
        weights,
        starts,
        ends,
        q_scales=q_scales,
        kv_scales=kv_scales,
        output=output,
        metadata=metadata,
    )


def supports_sparse_mqa_logits(
    q: torch.Tensor, sf_q: torch.Tensor, kv: torch.Tensor, weights: torch.Tensor
) -> bool:
    """Coarse admission check for the sparse MQA plans; never raises."""
    try:
        import torch

        if not _mqa_device_ok(q, sf_q, kv, weights):
            return False
        if not flashinfer_module_available(
            FI_SPARSE_MQA_MODULE, FI_SPARSE_MQA_BACKEND_MODULE
        ):
            return False
        if q.ndim != 3 or int(q.shape[1]) != SPARSE_MQA_HEADS:
            return False
        packed_fp4 = q.dtype in (torch.uint8, torch.int8) and int(q.shape[2]) == 64
        fp8 = q.dtype == torch.float8_e4m3fn and int(q.shape[2]) == 128
        return (
            (packed_fp4 or fp8)
            and sf_q.dtype == torch.int32
            and tuple(sf_q.shape) == (int(q.shape[0]), SPARSE_MQA_HEADS)
            and weights.dtype == torch.bfloat16
            and tuple(weights.shape) == (int(q.shape[0]), SPARSE_MQA_HEADS)
            and kv.ndim == 2
        )
    except Exception:
        return False


def prepare_sparse_mqa_metadata(
    sparse_indices: torch.Tensor,
    *,
    fmt: str = "mxfp4",
    sparse_block_kv: int = 8,
    page_kv: int = 64,
    use_unaligned_ks: bool = False,
    starts: Optional[torch.Tensor] = None,
    ends: Optional[torch.Tensor] = None,
    num_kv_tokens: int = 0,
    context_lens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    request_indices: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
):
    """Forward to ``flashinfer.sparse_mqa.prepare_sparse_mqa_metadata``."""
    from flashinfer.sparse_mqa import prepare_sparse_mqa_metadata

    return prepare_sparse_mqa_metadata(
        sparse_indices,
        fmt=fmt,
        sparse_block_kv=sparse_block_kv,
        page_kv=page_kv,
        use_unaligned_ks=use_unaligned_ks,
        starts=starts,
        ends=ends,
        num_kv_tokens=num_kv_tokens,
        context_lens=context_lens,
        block_table=block_table,
        request_indices=request_indices,
        metadata=metadata,
        workspace=workspace,
    )


def prepare_sparse_mqa_logits(
    q: torch.Tensor,
    sf_q: torch.Tensor,
    kv: torch.Tensor,
    sf_kv: Optional[torch.Tensor],
    weights: torch.Tensor,
    metadata_plan,
    *,
    output: Optional[torch.Tensor] = None,
):
    """Forward to ``flashinfer.sparse_mqa.prepare_sparse_mqa_logits``.

    Returns a ``SparseMqaPlan``; ``plan.run()`` submits the metadata + logits
    pipeline without allocation.
    """
    from flashinfer.sparse_mqa import prepare_sparse_mqa_logits

    return prepare_sparse_mqa_logits(
        q, sf_q, kv, sf_kv, weights, metadata_plan, output=output
    )


# --------------------------------------------------------------------------
# DeepGEMM-signature indexer logits (DeepSeek-V3.2 engine contract)
# --------------------------------------------------------------------------


def _dsa_mqa_device_ok(*tensors) -> bool:
    """Exact sm_100a / sm_103a parts; the FlashInfer runtime builds the programs
    for the device's SM count (or the caller's CTA budget), so no SM-count pin."""
    return archs_in(MQA_ARCHS, *tensors)


def _fp8_mqa_logits_runtime():
    from flashinfer.experimental.deepgemm_dense_mqa import dense_mqa as runtime

    return runtime


def _fp8_paged_mqa_logits_runtime():
    from flashinfer.experimental.deepgemm_dense_mqa import paged_mqa as runtime

    return runtime


def supports_fp8_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
) -> bool:
    """Admission for ``fp8_mqa_logits`` on the exact engine tensors; never raises.

    Beyond the dtype / layout contract, the shipped FlashInfer catalog is
    consulted (``dense_route_available`` with this device's architecture) so
    only points with exported programs admitted on this architecture are taken
    (``policy.dense_admission`` is per arch for the 64-head family).
    """
    try:
        import torch

        if not _dsa_mqa_device_ok(q, kv, kv_scales, weights, ks, ke):
            return False
        if not flashinfer_module_available(
            FI_DENSE_MQA_MODULE, FI_DENSE_MQA_BACKEND_MODULE
        ):
            return False
        if q.ndim != 3 or kv.ndim != 2:
            return False
        queries, heads, head_dim = (int(v) for v in q.shape)
        keys = int(kv.shape[0])
        if (
            heads not in DSA_MQA_HEADS
            or head_dim != DSA_MQA_HEAD_DIM
            or int(kv.shape[1]) != DSA_MQA_HEAD_DIM
            or queries < 1
            or keys < 1
        ):
            return False
        if not (
            q.dtype == torch.float8_e4m3fn
            and kv.dtype == torch.float8_e4m3fn
            and kv_scales.dtype == torch.float32
            and tuple(kv_scales.shape) == (keys,)
            and weights.dtype == torch.float32
            and tuple(weights.shape) == (queries, heads)
            and ks.dtype == torch.int32
            and ke.dtype == torch.int32
            and tuple(ks.shape) == (queries,)
            and tuple(ke.shape) == (queries,)
            and q.is_contiguous()
            and kv.is_contiguous()
            and kv_scales.is_contiguous()
            and weights.is_contiguous()
            and ks.is_contiguous()
            and ke.is_contiguous()
        ):
            return False
        runtime = _fp8_mqa_logits_runtime()
        if not hasattr(runtime, "dense_route_available"):
            return False
        # Dense admission is decided per architecture too (policy.dense_admission: a 64-head tier
        # withheld on one architecture keeps its record but is not served there).
        return bool(
            runtime.dense_route_available(
                heads, queries, keys, arch=runtime.device_arch(q.device)
            )
        )
    except Exception:
        return False


def fp8_mqa_logits(
    q: torch.Tensor,
    kv: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    clean_logits: bool = False,
    max_seqlen_k: int = 0,
    *,
    sm_count: Optional[int] = None,
) -> torch.Tensor:
    """Forward to ``flashinfer.dense_mqa.fp8_mqa_logits`` (DeepGEMM signature).

    Returns the f32 ``[Q, K]`` logits view; ``sm_count`` is the CTA budget the
    engine would hand DeepGEMM (``deep_gemm.get_num_sms()``).
    """
    from flashinfer.dense_mqa import fp8_mqa_logits

    return fp8_mqa_logits(
        q,
        kv,
        weights,
        ks,
        ke,
        clean_logits=clean_logits,
        max_seqlen_k=max_seqlen_k,
        sm_count=sm_count,
    )


def supports_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_context_len: Optional[int] = None,
) -> bool:
    """Admission for the paged entries on the exact engine tensors; never raises.

    ``q`` e4m3 ``[B, next_n, H, 128]`` (``H`` in {32, 64}), fused uint8
    ``kv_cache [pages, 64, 1, 132]`` (page 64), f32 ``weights [B * next_n, H]``,
    int32 2-D ``context_lens [B, next_n]``, int32 ``block_table [B, S]`` with
    unit column stride; the shipped catalog must carry the ``(H, 64, next_n)``
    program (``paged_route_available``) and admit it on this device's
    architecture for this call's batch (``context_lens.shape[0]``) and
    ``max_context_len`` (the catalog's per-architecture
    ``policy.paged.admission`` rules; a withheld (arch, route, batch, context)
    falls back to stock DeepGEMM). Without ``max_context_len`` the block table's
    capacity ``S * 64`` is the conservative stand-in.
    """
    try:
        import torch

        if not _dsa_mqa_device_ok(q, kv_cache, weights, context_lens, block_table):
            return False
        if not flashinfer_module_available(
            FI_PAGED_MQA_MODULE, FI_PAGED_MQA_BACKEND_MODULE
        ):
            return False
        if q.ndim != 4 or kv_cache.ndim != 4 or context_lens.ndim != 2:
            return False
        batch, next_n, heads, head_dim = (int(v) for v in q.shape)
        if (
            heads not in DSA_MQA_HEADS
            or head_dim != DSA_MQA_HEAD_DIM
            or batch < 1
            or next_n < 1
        ):
            return False
        page, kv_heads, row_bytes = (int(v) for v in kv_cache.shape[1:])
        if (
            page != DSA_MQA_PAGE
            or kv_heads != 1
            or row_bytes != DSA_MQA_FUSED_ROW_BYTES
        ):
            return False
        if not (
            q.dtype == torch.float8_e4m3fn
            and q.is_contiguous()
            and kv_cache.dtype == torch.uint8
            and int(kv_cache.stride(3)) == 1
            and int(kv_cache.stride(2)) == DSA_MQA_FUSED_ROW_BYTES
            and int(kv_cache.stride(1)) == DSA_MQA_FUSED_ROW_BYTES
            and weights.dtype == torch.float32
            and tuple(weights.shape) == (batch * next_n, heads)
            and weights.is_contiguous()
            and context_lens.dtype == torch.int32
            and tuple(context_lens.shape) == (batch, next_n)
            and context_lens.is_contiguous()
            and block_table.dtype == torch.int32
            and block_table.ndim == 2
            and int(block_table.shape[0]) == batch
            and int(block_table.stride(1)) == 1
        ):
            return False
        runtime = _fp8_paged_mqa_logits_runtime()
        if not hasattr(runtime, "paged_route_available"):
            return False
        arch = _fp8_mqa_logits_runtime().device_arch(q.device)
        if max_context_len is None:
            max_context_len = int(block_table.shape[1]) * page
        # batch = requests of this call (context_lens.shape[0], the engine's per-call chunk);
        # the catalog's admission rules are (arch, route, batch, max_context_len).
        return bool(
            runtime.paged_route_available(
                heads,
                page,
                next_n,
                arch=arch,
                batch=int(context_lens.shape[0]),
                max_context_len=int(max_context_len),
            )
        )
    except Exception:
        return False


def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor,
    block_kv: int,
    num_sms: int,
    indices: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward to ``flashinfer.paged_mqa.get_paged_mqa_logits_metadata``.

    Returns int32 ``[num_sms + 1, 2]`` walk bounds for ``fp8_paged_mqa_logits``
    (the Cake metadata program; not interchangeable with DeepGEMM's buffer).
    """
    from flashinfer.paged_mqa import get_paged_mqa_logits_metadata

    return get_paged_mqa_logits_metadata(
        context_lens, block_kv, num_sms, indices=indices, out=out
    )


def fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: torch.Tensor,
    max_context_len: int,
    clean_logits: bool = False,
    indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Forward to ``flashinfer.paged_mqa.fp8_paged_mqa_logits`` (DeepGEMM signature).

    Returns the f32 ``[B * next_n, max_context_len]`` logits view.
    """
    from flashinfer.paged_mqa import fp8_paged_mqa_logits

    return fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits=clean_logits,
        indices=indices,
    )


def prepare_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_context_len: int,
    *,
    schedule_meta: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    sm_count: Optional[int] = None,
):
    """Forward to ``flashinfer.paged_mqa.prepare_paged_mqa_logits``.

    Returns a ``PagedMqaPlan``; ``plan.run()`` submits the metadata and logits
    programs without allocation (CUDA-graph replay with changed contents).
    """
    from flashinfer.paged_mqa import prepare_paged_mqa_logits

    return prepare_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        max_context_len,
        schedule_meta=schedule_meta,
        output=output,
        sm_count=sm_count,
    )
