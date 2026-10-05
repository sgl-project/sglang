"""Cake (FlashInfer) backends for the ``attention`` operator group.

Metadata-only at import time: ``KernelSpec`` targets point at the lazy
adapters in :mod:`sglang.kernels.cake_kernels.attention_fmha`,
:mod:`~sglang.kernels.cake_kernels.attention_mla`,
:mod:`~sglang.kernels.cake_kernels.attention_sparse` and
:mod:`~sglang.kernels.cake_kernels.attention_misc`, which import FlashInfer
only when a kernel is actually called. Linear attention (KDA / GDN) and Mamba
live in :mod:`sglang.kernels.ops.attention.cake_linear`.

Every registration below has an explicit ``cake_<name>`` entry point; callers
gate on the matching ``supports_<name>`` of the adapter module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Tuple, Union

from sglang.kernels.registry import register_kernel
from sglang.kernels.selector import get_kernel
from sglang.kernels.spec import (
    CapabilityRequirement,
    FormatSignature,
    KernelBackend,
    KernelSpec,
)

if TYPE_CHECKING:
    import torch

_PKG = "sglang.kernels.cake_kernels."
_FMHA = _PKG + "attention_fmha:"
_MLA = _PKG + "attention_mla:"
_MLA_DSV41 = _PKG + "attention_mla_sm120_dsv41:"
_SPARSE = _PKG + "attention_sparse:"
_MISC = _PKG + "attention_misc:"

# Exact sm_100a / sm_103a (compute capability 10.0 / 10.3).
_BLACKWELL_DC = frozenset({CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 3))})
# sm_100a / sm_103a plus the sm_107a (10.7) programs some products ship.
_BLACKWELL_DC_107 = frozenset(
    {CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 7))}
)
_SM103_ONLY = frozenset({CapabilityRequirement.cuda(min_sm=(10, 3), max_sm=(10, 3))})
_SM90_ONLY = frozenset({CapabilityRequirement.cuda(min_sm=(9, 0), max_sm=(9, 0))})
_SM110_ONLY = frozenset({CapabilityRequirement.cuda(min_sm=(11, 0), max_sm=(11, 0))})
_SM120_ONLY = frozenset({CapabilityRequirement.cuda(min_sm=(12, 0), max_sm=(12, 0))})
_SM120_121 = frozenset({CapabilityRequirement.cuda(min_sm=(12, 0), max_sm=(12, 1))})
_BLACKWELL_DC_OR_SM120 = frozenset(
    {
        CapabilityRequirement.cuda(min_sm=(10, 0), max_sm=(10, 3)),
        CapabilityRequirement.cuda(min_sm=(12, 0), max_sm=(12, 1)),
    }
)


def _reg(
    name: str,
    target: str,
    capabilities,
    *,
    dtypes: Tuple[str, ...],
    contract: str,
    description: str,
    in_place: bool = False,
) -> None:
    register_kernel(
        KernelSpec(
            op=f"attention.{name}",
            backend=KernelBackend.FLASHINFER,
            target=target,
            capabilities=capabilities,
            format_signature=FormatSignature(
                supported_dtypes=dtypes, in_place=in_place, description=contract
            ),
            description=description,
        )
    )


def _k(name: str):
    return get_kernel(f"attention.{name}", KernelBackend.FLASHINFER)


# --------------------------------------------------------------------------
# Dense FMHA / GQA / DCP / SM110 (attention_fmha)
# --------------------------------------------------------------------------

_reg(
    "fmha_batch_decode_with_kv_cache",
    _FMHA + "batch_decode_with_kv_cache",
    _BLACKWELL_DC,
    dtypes=("bfloat16", "float16", "float8_e4m3fn"),
    contract=(
        "trtllm paged-decode ABI: Q [B*q_len, Hq, D] (D in 64/128/256/512), "
        "paged K/V 4-D HND (NHD for the fp16 route), page 16/32/64, group "
        "ratio 1..16, q_len 1 or MTP 3..8; FP8 Q + NVFP4 KV via kv_cache_sf"
    ),
    description=(
        "Cake FMHA paged decode distributed by FlashInfer "
        "(trtllm_batch_decode_with_kv_cache backend='cake')."
    ),
)
_reg(
    "fmha_batch_context_with_kv_cache",
    _FMHA + "batch_context_with_kv_cache",
    _BLACKWELL_DC,
    dtypes=("bfloat16", "float16", "float8_e4m3fn"),
    contract=(
        "trtllm paged-context ABI: ragged Q [T, Hq, D] (D 128/256) with "
        "cum_seq_lens_q/kv, paged K/V HND (NHD hd256 routes), page 16..1024 "
        "pow2, window_left == -1, host scalar bmm scales"
    ),
    description=(
        "Cake FMHA paged context distributed by FlashInfer "
        "(trtllm_batch_context_with_kv_cache backend='cake')."
    ),
)
_reg(
    "dcp_spec_decode",
    _FMHA + "dcp_spec_decode",
    _BLACKWELL_DC_107,
    dtypes=("bfloat16", "float8_e4m3fn"),
    contract=(
        "BF16 Q [B*q_len, Hq, D]; HND K/V BF16 page 16 D128 or e4m3 page 64 "
        "D128/D256; cp_world in 1/2/4/8 with causal_seqlens_kv_global[B]; "
        "returns (out BF16, lse f32 base-2 [tokens, Hq])"
    ),
    description=(
        "Cake DCP speculative decode distributed by FlashInfer "
        "(trtllm_batch_decode_with_kv_cache with causal_seqlens_kv_global, "
        "backend='cake'; cc 10.0/10.3/10.7)."
    ),
)
_reg(
    "prepare_balanced_batch_decode_with_kv_cache",
    _FMHA + "prepare_balanced_batch_decode_with_kv_cache",
    _BLACKWELL_DC,
    dtypes=("bfloat16",),
    contract=(
        "BF16 Q [B*q_len, 8*Hkv, 128], K/V tuple [pages, Hkv, 16, 128] HND, "
        "int32 block_tables/seq_lens, q_len 1..8, B <= 1024, uint8 workspace "
        ">= balanced_gqa_decode_workspace_size -> runner; launch() -> out"
    ),
    description=(
        "Cake on-device load-balanced BF16 paged GQA decode distributed by "
        "FlashInfer (prepare_balanced_batch_decode_with_kv_cache)."
    ),
)
_reg(
    "sm110_gqa_decode",
    _FMHA + "sm110_gqa_decode",
    _SM110_ONLY,
    dtypes=("float16",),
    contract=(
        "FP16 q [B, 32, 128], kv [B, 2, 8, capacity, 128], int32 "
        "sequence_lengths[B] in [1, capacity] -> out FP16 [B, 32, 128]"
    ),
    description="Thor (SM110) GQA decode distributed by FlashInfer (one-shot).",
)
_reg(
    "prepare_sm110_gqa_decode",
    _FMHA + "prepare_sm110_gqa_decode",
    _SM110_ONLY,
    dtypes=("float16",),
    contract=(
        "inputs dict {Q [B,32,128], KV [B,2,8,C,128], O [B,32,128], "
        "sequence_lengths int32 [B]}, num_splits in {None,1,2,4,8,10,16} -> "
        "prepared dict owning the split workspace"
    ),
    description="Thor (SM110) GQA decode route selection + workspace (prepare).",
)
_reg(
    "launch_sm110_gqa_decode_prepared",
    _FMHA + "launch_sm110_gqa_decode_prepared",
    _SM110_ONLY,
    dtypes=("float16",),
    contract="prepared dict from prepare_sm110_gqa_decode -> O FP16 [B, 32, 128]",
    description="Thor (SM110) GQA decode launch of a prepared route.",
)
_reg(
    "sm110_xqa_prepare",
    _FMHA + "sm110_xqa_prepare",
    _SM110_ONLY,
    dtypes=("float16", "float8_e4m3fn"),
    contract=(
        "FP16 q; decode D128 (q [B,Hq,128], kv [B,2,Hkv,C,128], Hq/Hkv in "
        "4/8/16) or tree D512 (packed q + q_cu_seq_lens + mask, FP16/e4m3 KV, "
        "optional page_size=128 paging) -> PreparedAttention.run()"
    ),
    description="Thor (SM110) XQA decode/tree attention prepare (FlashInfer).",
)
_reg(
    "sm110_xqa_attention",
    _FMHA + "sm110_xqa_attention",
    _SM110_ONLY,
    dtypes=("float16", "float8_e4m3fn"),
    contract="prepare(...).run() with the sm110_xqa_prepare contract",
    description="Thor (SM110) XQA one-shot attention (FlashInfer).",
)

# --------------------------------------------------------------------------
# MLA (attention_mla)
# --------------------------------------------------------------------------

_reg(
    "trtllm_batch_decode_sparse_mla_dsv4",
    _MLA + "trtllm_batch_decode_sparse_mla_dsv4",
    _BLACKWELL_DC_OR_SM120,
    dtypes=("bfloat16", "float8_e4m3fn", "uint8"),
    contract=(
        "SM100/103 (kv_cache_format='fp8'): BF16/FP8 query [B,Q,H,512] or "
        "ragged [sum_q,H,512], dense 512-wide SWA + compressed pools, int32 "
        "sparse tables (combined [T,topk] or [T,128] + extra [T,topk_c]), "
        "seq_lens[B]; SM120/121 (kv_cache_format='nvfp4'): BF16 q [T,H,512], "
        "packed NVFP4 uint8 cache (384 B/token), int32 indices/lengths"
    ),
    description=(
        "Cake DeepSeek-V4 sparse MLA decode distributed by FlashInfer "
        "(trtllm_batch_decode_sparse_mla_dsv4 backend='cake')."
    ),
)
_reg(
    "create_sparse_mla_sm120_wrapper",
    _MLA + "create_sparse_mla_sm120_wrapper",
    _SM120_121,
    dtypes=("bfloat16", "uint8"),
    contract=(
        "SparseMLASm120Wrapper(backend='cake', kv_cache_format='nvfp4'); "
        "run(q, kv_cache, indices, output, sm_scale, ...) per FlashInfer"
    ),
    description=(
        "Cake SM120/121 NVFP4 DSv4 sparse-MLA wrapper distributed by FlashInfer."
    ),
)
_reg(
    "sparse_mla_sm120_dsv4_nvfp4_decode",
    _MLA + "sparse_mla_sm120_dsv4_nvfp4_decode",
    _SM120_121,
    dtypes=("bfloat16", "uint8"),
    contract=(
        "BF16 q [T,H,512], NVFP4 uint8 pages, int32 indices [T,topk], BF16 "
        "output [T,H,512], f32 out_lse [T,H] (base 2 x lse_scale); mid_out/"
        "mid_lse required when num_splits > 1 -> plan dict"
    ),
    description="Cake SM120/121 NVFP4 DSv4 sparse-MLA decode kernel (FlashInfer).",
)
_reg(
    "sparse_mla_sm120_dsv4_nvfp4_prefill",
    _MLA + "sparse_mla_sm120_dsv4_nvfp4_prefill",
    _SM120_121,
    dtypes=("bfloat16", "uint8"),
    contract=(
        "as the decode entry minus mid_*; H multiple of 16 in 16..128, "
        "topk + extra_topk <= 1024; one launch, no scratch -> plan dict"
    ),
    description="Cake SM120/121 NVFP4 DSv4 sparse-MLA prefill kernel (FlashInfer).",
)
_reg(
    "trtllm_batch_decode_with_kv_cache_mla",
    _MLA + "trtllm_batch_decode_with_kv_cache_mla",
    _BLACKWELL_DC,
    dtypes=("bfloat16", "float8_e4m3fn"),
    contract=(
        "BF16/FP8 query [B,Q,H,D] or compact [T,H,D] + cum_seq_lens_q, paged "
        "KV of the same dtype (page 32/64), int32 block_tables/seq_lens; "
        "generated (nope,lora,rope,H) tuples -> Blackwell MLA programs "
        "(148/152-SM parts), other tuples -> Kimi-K3 FP8 route (D=576, page 64)"
    ),
    description=(
        "Cake paged MLA decode distributed by FlashInfer "
        "(trtllm_batch_decode_with_kv_cache_mla backend='cake')."
    ),
)
_reg(
    "kimi_k3_mla_fp8_paged_attention",
    _MLA + "kimi_k3_mla_fp8_paged_attention",
    _BLACKWELL_DC,
    dtypes=("float8_e4m3fn",),
    contract=(
        "FP8 query [rows,H,576] (dense or packed var-Q with cum_seq_lens_q), "
        "FP8 kv_cache [pages,64,576], int32 block_tables/seq_lens, BF16 out "
        "[rows,H,512], caller workspace >= workspace_bytes(rows_max, num_split)"
    ),
    description="Cake Kimi-K3 FP8 paged MLA attention (one-shot) via FlashInfer.",
)
_reg(
    "prepare_kimi_k3_mla_fp8_paged_attention",
    _MLA + "prepare_kimi_k3_mla_fp8_paged_attention",
    _BLACKWELL_DC,
    dtypes=("float8_e4m3fn",),
    contract=(
        "KimiK3MlaFp8PagedAttention(query=, kv_cache=, block_tables=, "
        "seq_lens=, out=, workspace_buffer=, bmm1_scale=, ...) -> launch()"
    ),
    description="Cake Kimi-K3 FP8 paged MLA attention (prepared) via FlashInfer.",
)
_reg(
    "prepare_nvfp4_batch_decode_with_kv_cache_mla",
    _MLA + "prepare_nvfp4_batch_decode_with_kv_cache_mla",
    _BLACKWELL_DC,
    dtypes=("uint8", "float8_e4m3fn"),
    contract=(
        "uint8 query [B*q_len,H,256] + query_scale [B*q_len,H,32], uint8 "
        "kv_cache [pages,64,256] + kv_scale [pages,64,32], int32 block_tables/"
        "seq_lens, sm_scale, optional f32 sinks[H], B*q_len*H >= 128 -> runner "
        "-> out BF16 [B*q_len,H,512] (+ natural-log f32 lse)"
    ),
    description="Cake NVFP4 DeepSeek-V4 MLA decode (prepared) via FlashInfer.",
)
_reg(
    "mla_varq_dcp_decode",
    _MLA + "mla_varq_dcp_decode",
    _BLACKWELL_DC,
    dtypes=("bfloat16", "float8_e4m3fn"),
    contract=(
        "compact query [total_q,H,576] (H <= 128) + cum_seq_lens_q/max_q_len, "
        "paged KV [pages,page,576] page 32/64/128, int32 block_tables/seq_lens, "
        "cyclic DCP (cp_world, cp_rank, causal_seqlens_kv_global) -> (out BF16 "
        "[total_q,H,512], lse f32 natural log)"
    ),
    description="Cake variable-q MLA decode with DCP (one-shot) via FlashInfer.",
)
_reg(
    "prepare_mla_varq_dcp_decode",
    _MLA + "prepare_mla_varq_dcp_decode",
    _BLACKWELL_DC,
    dtypes=("bfloat16", "float8_e4m3fn"),
    contract="as mla_varq_dcp_decode -> CakeMLAVarQDcpDecodeRunner (launch() -> (out, lse))",
    description="Cake variable-q MLA decode with DCP (prepared) via FlashInfer.",
)
_reg(
    "concat_mla_k",
    _MLA + "concat_mla_k",
    _BLACKWELL_DC,
    dtypes=("bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"),
    in_place=True,
    contract=(
        "k [T,128,192] (strides (24576,192,1) or (32768,256,1)) <- k_nope "
        "[T,128,128] | k_rope [T,1,64] broadcast; proven stride profiles only"
    ),
    description=(
        "Cake in-place MLA K assembly distributed by FlashInfer "
        "(concat_mla_k backend='cake')."
    ),
)

# SM120/121 DeepSeek-V4.1 mixed cache (attention_mla_sm120_dsv41; post-baseline,
# FlashInfer PR #5983 at main e4f94f948)
_reg(
    "sparse_mla_sm120_dsv41_mixed_decode",
    _MLA_DSV41 + "sparse_mla_sm120_dsv41_mixed_decode",
    _SM120_121,
    dtypes=("bfloat16", "uint8", "float32", "int32"),
    in_place=True,
    contract=(
        "BF16 q [T,H,512]; 528 B/token FP8+UE8M0 main cache and optional 288 "
        "B/token V41_FP4 extra cache (uint8 pages, 2-D/3-D/HND/NHD views); "
        "int32 indices/lengths; BF16 output [T,H,512], f32 out_lse [T,H] "
        "(base 2 x lse_scale); mid_out/mid_lse required when num_splits > 1; "
        "compute_precision bf16 (default) or fp8 -> plan dict"
    ),
    description=(
        "Cake SM120/121 DeepSeek-V4.1 mixed-cache sparse-MLA decode kernel "
        "distributed by FlashInfer (kv_cache_format='fp8_dsv41_fp4_ca')."
    ),
)
_reg(
    "create_sparse_mla_sm120_dsv41_mixed_wrapper",
    _MLA_DSV41 + "create_sparse_mla_sm120_dsv41_mixed_wrapper",
    _SM120_121,
    dtypes=("bfloat16", "uint8"),
    contract=(
        "SparseMLASm120Wrapper(backend='cake', kv_cache_format='fp8', "
        "kv_scale_format='ue8m0_g32', extra_kv_fp4=True); decode-only "
        "run(q, kv_cache, indices, output, sm_scale, ...) per FlashInfer"
    ),
    description=(
        "Cake SM120/121 DeepSeek-V4.1 mixed-cache sparse-MLA wrapper "
        "distributed by FlashInfer."
    ),
)
_reg(
    "dsv41_fp8_quantize_pack_sparse_mla_cache",
    _MLA_DSV41 + "dsv41_fp8_quantize_pack_sparse_mla_cache",
    _SM120_121,
    dtypes=("bfloat16", "float16", "uint8"),
    contract=(
        "BF16/FP16 latent [pages,page_size,512] (optional singleton head axis) "
        "-> uint8 [pages,1,page_size,528] (HND) or [pages,page_size,1,528] "
        "(NHD); E4M3 values + per-page UE8M0 group-32 scale footer"
    ),
    description=(
        "DeepSeek-V4.1 FP8 main-cache full-page writer for the Cake SM120/121 "
        "mixed-cache sparse MLA (FlashInfer PR #5983)."
    ),
)
_reg(
    "dsv41_fp8_quantize_append_sparse_mla_cache",
    _MLA_DSV41 + "dsv41_fp8_quantize_append_sparse_mla_cache",
    _SM120_121,
    dtypes=("bfloat16", "float16", "uint8", "int32", "int64"),
    in_place=True,
    contract=(
        "BF16/FP16 rows [N,512], int32/int64 slot_mapping[N] "
        "(page*page_size+entry; negative = padding), uint8 528 B/token cache "
        "(2-D/3-D/HND/NHD view) written in place"
    ),
    description=(
        "DeepSeek-V4.1 FP8 main-cache slot append for the Cake SM120/121 "
        "mixed-cache sparse MLA (FlashInfer PR #5983)."
    ),
)

# --------------------------------------------------------------------------
# Sparse / block-sparse / indexer (attention_sparse)
# --------------------------------------------------------------------------

_reg(
    "create_block_sparse_attention_wrapper",
    _SPARSE + "create_block_sparse_attention_wrapper",
    _BLACKWELL_DC,
    dtypes=("bfloat16", "float16"),
    contract=(
        "BlockSparseAttentionWrapper(workspace, backend='cake'): plan(block "
        "R==C in 64/128, head_dim 64/96/128, block_mask | CSR | q2k_indices) "
        "then run(q [M,Hq,D], k/v [N,Hkv,D], return_lse=)"
    ),
    description="Cake VSA block-sparse attention wrapper (SM100/103) via FlashInfer.",
)
_reg(
    "create_variable_block_sparse_attention_wrapper_sm90",
    _SPARSE + "create_variable_block_sparse_attention_wrapper_sm90",
    _SM90_ONLY,
    dtypes=("bfloat16",),
    contract=(
        "VariableBlockSparseAttentionWrapper(workspace, backend='cake'|"
        "'cake_cute'): BF16 HND q/k/v (H,S,128), Hq==Hkv, noncausal, 64-token "
        "blocks, 1..64 selected KV blocks per row; plan() then run(q,k,v,out=)"
    ),
    description="Cake VSA block-sparse attention wrapper (SM90) via FlashInfer.",
)
_reg(
    "bsa_attn_sm100_blk64_sage_fwd",
    _SPARSE + "bsa_attn_sm100_blk64_sage_fwd",
    _BLACKWELL_DC,
    dtypes=("float8_e4m3fn",),
    contract=(
        "e4m3 BSHD q (B,Sq,H,128), k/v (B,Sk,Hkv,128), int32 q2k_block_index "
        "(B,H,ceil(Sq/64),cap), f32 q_scale (B,H,Sq), k_scale (B,Hkv,"
        "ceil(Sk/16)), v_scale (Hkv,128)|(B,Hkv,128) -> (out BF16 BSHD, lse)"
    ),
    description=(
        "Cake Sage-FP8 block-sparse attention (SM100/103) via FlashInfer "
        "(bsa_attn_sm100_blk64_fwd backend='cake')."
    ),
)
_reg(
    "bsa_attn_sm120_blk64_sage_fwd",
    _SPARSE + "bsa_attn_sm120_blk64_sage_fwd",
    _SM120_ONLY,
    dtypes=("int8", "float8_e4m3fn"),
    contract=(
        "INT8 BHSD q/k [B,H,S,128], e4m3 HDS v [B,H,128,padded_Sk], f32 "
        "scales, int32 q2k_block_index [B,H,ceil(Sq/64),cap], caller-owned "
        "BF16 BHSD out; noncausal, no LSE"
    ),
    description=(
        "Cake Sage block-sparse attention (SM120) via FlashInfer "
        "(bsa_attn_sm120_blk64_sage_fwd backend='cake')."
    ),
)
_reg(
    "prepare_msa_nvfp4_sparse_decode",
    _SPARSE + "prepare_msa_nvfp4_sparse_decode",
    _BLACKWELL_DC_107,
    dtypes=("bfloat16", "uint8", "float8_e4m3fn"),
    contract=(
        "BF16 q [B*sq,Hq,128], uint8 k/v [pages,Hkv,128,64] E2M1 pools + "
        "[pages,Hkv,128,8] e4m3 scales, int32 q2k_indices [Hkv,B*sq,16], "
        "page_table [B,max_pages], seqused_k [B], sq 1..32, group 1..16 -> "
        "runner (launch() -> out BF16; lse natural log)"
    ),
    description=(
        "Cake NVFP4 paged MiniMax sparse-attention decode (prepared) via "
        "FlashInfer (prepare_msa_nvfp4_sparse_decode)."
    ),
)
_reg(
    "dsa_indexer_topk",
    _SPARSE + "dsa_indexer_topk",
    _BLACKWELL_DC_107,
    dtypes=("bfloat16", "float32"),
    contract=(
        "BF16 q [T,32,128], BF16 k [Tkv,128] (row stride % 8 == 0), f32 w "
        "[T,32], int32 cu_seqlens_q/k [S+1], 1 <= top_k <= 4096 -> (indices "
        "int32 [T,top_k] ascending -1 padded, scores f32 -inf padded)"
    ),
    description="Cake DSA indexer exact top-k (one-shot) via FlashInfer.",
)
_reg(
    "prepare_dsa_indexer_topk",
    _SPARSE + "prepare_dsa_indexer_topk",
    _BLACKWELL_DC_107,
    dtypes=("bfloat16", "float32"),
    contract="as dsa_indexer_topk with caller-owned workspace/outputs -> runner.run()",
    description="Cake DSA indexer exact top-k (prepared) via FlashInfer.",
)
_reg(
    "prepare_dense_mqa_logits",
    _SPARSE + "prepare_dense_mqa_logits",
    _BLACKWELL_DC,
    dtypes=("uint8", "float8_e4m3fn", "float32"),
    contract=(
        "precision 'fp4' (uint8 q [Q,32,64], kv [K,64], UE8M0 scales) or 'fp8' "
        "(e4m3 q [>=max(4,Q),32,128], kv [K,128], f32 kv_scales [K]); f32 "
        "weights [Q,32], int32 starts/ends [Q], K % 256 == 0 -> plan.run() -> "
        "f32 logits [Q, align(K+256, 8)]; exported Q/K routes only; 148/152 SMs"
    ),
    description="DeepGEMM-family dense MQA indexer logits (prepared) via FlashInfer.",
)
_reg(
    "fp8_mqa_logits",
    _SPARSE + "fp8_mqa_logits",
    _BLACKWELL_DC,
    dtypes=("float8_e4m3fn", "float32", "int32"),
    contract=(
        "DeepGEMM fp8_mqa_logits signature: e4m3 q [Q,H,128] (H in 32/64), kv = "
        "(e4m3 [K,128], f32 scales [K]), f32 weights [Q,H], int32 ks/ke [Q], "
        "clean_logits (no effect), max_seqlen_k == 0, sm_count= CTA budget -> "
        "f32 logits view [Q,K]; shipped routes per dense_route_available(H,Q,K)"
    ),
    description=(
        "DeepSeek-V3.2 ragged indexer logits with the DeepGEMM signature "
        "(one-shot) via FlashInfer."
    ),
)
_reg(
    "get_paged_mqa_logits_metadata",
    _SPARSE + "get_paged_mqa_logits_metadata",
    _BLACKWELL_DC,
    dtypes=("int32",),
    contract=(
        "DeepGEMM signature: int32 context_lens [B,next_n] (2-D), block_kv 64, "
        "num_sms -> zero int32 [num_sms+1, 2] placeholder for fp8_paged_mqa_logits "
        "(no kernel launch; the Cake program derives its schedule in-kernel)"
    ),
    description="DeepSeek-V3.2 paged indexer schedule placeholder (no launch) via FlashInfer.",
)
_reg(
    "fp8_paged_mqa_logits",
    _SPARSE + "fp8_paged_mqa_logits",
    _BLACKWELL_DC,
    dtypes=("float8_e4m3fn", "uint8", "float32", "int32"),
    contract=(
        "DeepGEMM signature: e4m3 q [B,next_n,H,128] (H in 32/64), uint8 fused "
        "kv_cache [pages,64,1,132], f32 weights [B*next_n,H], int32 context_lens "
        "[B,next_n], int32 block_table [B,S] (unit column stride), schedule_meta, "
        "max_context_len, clean_logits=False -> f32 logits view "
        "[B*next_n,max_context_len]; any batch size"
    ),
    description=(
        "DeepSeek-V3.2 paged indexer logits with the DeepGEMM signature "
        "(one-shot) via FlashInfer."
    ),
)
_reg(
    "prepare_paged_mqa_logits",
    _SPARSE + "prepare_paged_mqa_logits",
    _BLACKWELL_DC,
    dtypes=("float8_e4m3fn", "uint8", "float32", "int32"),
    contract=(
        "as fp8_paged_mqa_logits with caller-owned schedule_meta/output -> "
        "plan.run() (one logits launch, no allocation; graph replay)"
    ),
    description="DeepSeek-V3.2 paged indexer logits (prepared) via FlashInfer.",
)
_reg(
    "prepare_sparse_mqa_metadata",
    _SPARSE + "prepare_sparse_mqa_metadata",
    _BLACKWELL_DC,
    dtypes=("int32",),
    contract=(
        "sorted dup-padded int32 sparse_indices [Q, capacity] (+ optional "
        "starts/ends, block_table, request_indices), fmt mxfp4|mxfp8, 8-token "
        "blocks, 64-token pages -> metadata plan (run() rebuilds metadata)"
    ),
    description="DeepGEMM-family sparse MQA metadata plan via FlashInfer.",
)
_reg(
    "prepare_sparse_mqa_logits",
    _SPARSE + "prepare_sparse_mqa_logits",
    _BLACKWELL_DC,
    dtypes=("uint8", "float8_e4m3fn", "bfloat16"),
    contract=(
        "packed E2M1 q [Q,32,64] or e4m3 q [Q,32,128], int32 sf_q [Q,32], "
        "BF16 weights [Q,32], contiguous or paged KV, metadata plan -> "
        "plan.run() -> BF16 output [Q, capacity*8]"
    ),
    description="DeepGEMM-family sparse MQA logits plan via FlashInfer.",
)

# --------------------------------------------------------------------------
# AttnRes / MiniMax-H3 varlen / NVFP4 attention (attention_misc)
# --------------------------------------------------------------------------

_reg(
    "kimi_k3_attn_res",
    _MISC + "kimi_k3_attn_res",
    _BLACKWELL_DC,
    in_place=True,
    dtypes=("bfloat16",),
    contract=(
        "BF16 prefix [M,7168] (updated in place with delta), delta [M,7168] | "
        "None, blocks [M,B<=8,7168], weights [7168], out [M,7168]; "
        "num_blocks K in 0..8; registered (M, K) cells only"
    ),
    description="Cake Kimi-K3 AttnRes (one-shot) via FlashInfer.",
)
_reg(
    "prepare_kimi_k3_attn_res",
    _MISC + "prepare_kimi_k3_attn_res",
    _BLACKWELL_DC,
    in_place=True,
    dtypes=("bfloat16",),
    contract="as kimi_k3_attn_res -> KimiK3AttnResRunner (launch() -> out)",
    description="Cake Kimi-K3 AttnRes (prepared) via FlashInfer.",
)
_reg(
    "minimax_h3_varlen_attention",
    _MISC + "minimax_h3_varlen_attention",
    _BLACKWELL_DC,
    dtypes=("bfloat16",),
    contract=(
        "BF16 THD q/k/v [T,H,128], int32 cu_seqlens [B+1], noncausal Hq==Hkv "
        "-> out BF16 [T,H,128]"
    ),
    description="Cake MiniMax-H3 packed-varlen attention (one-shot) via FlashInfer.",
)
_reg(
    "prepare_minimax_h3_varlen_attention",
    _MISC + "prepare_minimax_h3_varlen_attention",
    _BLACKWELL_DC,
    dtypes=("bfloat16",),
    contract="as minimax_h3_varlen_attention -> runner (launch() -> out)",
    description="Cake MiniMax-H3 packed-varlen attention (prepared) via FlashInfer.",
)
_reg(
    "minimax_h3_varlen_nvfp4_attention",
    _MISC + "minimax_h3_varlen_nvfp4_attention",
    _BLACKWELL_DC,
    dtypes=("bfloat16",),
    contract=(
        "BF16 THD q/k/v [T,H,128] + int32 cu_seqlens; pv_mode 'fp8' (NVFP4 QK, "
        "e4m3 PV) or 'fp4' -> out BF16 [T,H,128] (atol 1.0 / rtol 0.1 vs FP32)"
    ),
    description="Cake MiniMax-H3 NVFP4 varlen attention (one-shot) via FlashInfer.",
)
_reg(
    "prepare_minimax_h3_varlen_nvfp4_attention",
    _MISC + "prepare_minimax_h3_varlen_nvfp4_attention",
    _BLACKWELL_DC,
    dtypes=("bfloat16",),
    contract=(
        "as minimax_h3_varlen_nvfp4_attention (+ optional workspace dict) -> "
        "runner (quantize() / attention() / launch())"
    ),
    description="Cake MiniMax-H3 NVFP4 varlen attention (prepared) via FlashInfer.",
)
_reg(
    "prepare_nvfp4_attention",
    _MISC + "prepare_nvfp4_attention",
    _SM103_ONLY,
    dtypes=("bfloat16",),
    contract=(
        "BF16 contiguous q/k/v/out [B,H,S,128] identical shapes, S % 512 == 0, "
        "noncausal, fixed 1/sqrt(128) scale -> runner (launch() -> out)"
    ),
    description="Cake NVFP4 dense attention (SM103, prepared) via FlashInfer.",
)


# --------------------------------------------------------------------------
# Explicit entry points
# --------------------------------------------------------------------------


def cake_fmha_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    **kwargs,
):
    """Explicit Cake FMHA decode; keyword arguments as the adapter / FlashInfer."""
    return _k("fmha_batch_decode_with_kv_cache")(
        query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len, **kwargs
    )


def cake_fmha_batch_context_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    batch_size: int,
    cum_seq_lens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    **kwargs,
):
    """Explicit Cake FMHA context; keyword arguments as the adapter / FlashInfer."""
    return _k("fmha_batch_context_with_kv_cache")(
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_q_len,
        max_kv_len,
        bmm1_scale,
        bmm2_scale,
        batch_size,
        cum_seq_lens_q,
        cum_seq_lens_kv,
        **kwargs,
    )


def cake_dcp_spec_decode(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    causal_seqlens_kv_global: torch.Tensor,
    *,
    bmm1_scale: float = 1.0,
    bmm2_scale: float = 1.0,
    cp_world: int = 1,
    cp_rank: int = 0,
    q_len_per_req: int = 1,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = True,
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor] = None,
    kv_layout: str = "HND",
):
    """Explicit Cake DCP speculative decode entry point."""
    return _k("dcp_spec_decode")(
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_seq_len,
        causal_seqlens_kv_global,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        cp_world=cp_world,
        cp_rank=cp_rank,
        q_len_per_req=q_len_per_req,
        out=out,
        lse=lse,
        return_lse=return_lse,
        multi_ctas_kv_counter_buffer=multi_ctas_kv_counter_buffer,
        kv_layout=kv_layout,
    )


def cake_prepare_balanced_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    sm_scale: Optional[float] = None,
    q_len_per_req: int = 1,
    out: Optional[torch.Tensor] = None,
    kv_layout: str = "HND",
):
    """Explicit Cake balanced GQA decode prepare; returns the runner."""
    return _k("prepare_balanced_batch_decode_with_kv_cache")(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        workspace_buffer,
        sm_scale=sm_scale,
        q_len_per_req=q_len_per_req,
        out=out,
        kv_layout=kv_layout,
    )


def cake_sm110_gqa_decode(
    q: torch.Tensor,
    kv: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    q_scale: float = 1.0,
) -> torch.Tensor:
    """Explicit Thor GQA decode entry point."""
    return _k("sm110_gqa_decode")(q, kv, sequence_lengths, out=out, q_scale=q_scale)


def cake_prepare_sm110_gqa_decode(
    inputs: Dict[str, Any], num_splits: Optional[int] = None
) -> Dict[str, Any]:
    """Explicit Thor GQA decode prepare entry point."""
    return _k("prepare_sm110_gqa_decode")(inputs, num_splits=num_splits)


def cake_launch_sm110_gqa_decode_prepared(prepared: Dict[str, Any]) -> torch.Tensor:
    """Explicit Thor GQA decode prepared-launch entry point."""
    return _k("launch_sm110_gqa_decode_prepared")(prepared)


def cake_sm110_xqa_prepare(
    q: torch.Tensor, kv: torch.Tensor, sequence_lengths: torch.Tensor, **kwargs
):
    """Explicit Thor XQA prepare; keyword arguments as the adapter / FlashInfer."""
    return _k("sm110_xqa_prepare")(q, kv, sequence_lengths, **kwargs)


def cake_sm110_xqa_attention(
    q: torch.Tensor, kv: torch.Tensor, sequence_lengths: torch.Tensor, **kwargs
) -> torch.Tensor:
    """Explicit Thor XQA one-shot attention."""
    return _k("sm110_xqa_attention")(q, kv, sequence_lengths, **kwargs)


def cake_trtllm_batch_decode_sparse_mla_dsv4(
    query: torch.Tensor,
    swa_kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    **kwargs,
):
    """Explicit Cake DSv4 sparse MLA decode; keyword arguments as the adapter."""
    return _k("trtllm_batch_decode_sparse_mla_dsv4")(
        query, swa_kv_cache, workspace_buffer, **kwargs
    )


def cake_create_sparse_mla_sm120_wrapper(
    max_num_tokens: Optional[int] = None,
    max_num_heads: Optional[int] = None,
    *,
    device=None,
):
    """Explicit Cake SM120 NVFP4 sparse-MLA wrapper factory."""
    return _k("create_sparse_mla_sm120_wrapper")(
        max_num_tokens, max_num_heads, device=device
    )


def cake_sparse_mla_sm120_dsv4_nvfp4_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    **kwargs,
) -> Dict[str, int]:
    """Explicit Cake SM120 NVFP4 DSv4 decode kernel entry point."""
    return _k("sparse_mla_sm120_dsv4_nvfp4_decode")(
        q, kv_cache, indices, output, out_lse, sm_scale, **kwargs
    )


def cake_sparse_mla_sm120_dsv4_nvfp4_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    **kwargs,
) -> Dict[str, int]:
    """Explicit Cake SM120 NVFP4 DSv4 prefill kernel entry point."""
    return _k("sparse_mla_sm120_dsv4_nvfp4_prefill")(
        q, kv_cache, indices, output, out_lse, sm_scale, **kwargs
    )


def cake_sparse_mla_sm120_dsv41_mixed_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    output: torch.Tensor,
    out_lse: torch.Tensor,
    sm_scale: float,
    **kwargs,
) -> Dict[str, int]:
    """Explicit Cake SM120/121 DSv4.1 mixed-cache decode kernel entry point."""
    return _k("sparse_mla_sm120_dsv41_mixed_decode")(
        q, kv_cache, indices, output, out_lse, sm_scale, **kwargs
    )


def cake_create_sparse_mla_sm120_dsv41_mixed_wrapper(
    max_num_tokens: Optional[int] = None,
    max_num_heads: Optional[int] = None,
    *,
    compute_precision: str = "default",
    device=None,
):
    """Explicit Cake SM120/121 DSv4.1 mixed-cache sparse-MLA wrapper factory."""
    return _k("create_sparse_mla_sm120_dsv41_mixed_wrapper")(
        max_num_tokens,
        max_num_heads,
        compute_precision=compute_precision,
        device=device,
    )


def cake_dsv41_fp8_quantize_pack_sparse_mla_cache(
    latent_kv: torch.Tensor, *, kv_layout: str = "HND"
) -> torch.Tensor:
    """Explicit DSv4.1 FP8 main-cache full-page pack (SM120/121)."""
    return _k("dsv41_fp8_quantize_pack_sparse_mla_cache")(
        latent_kv, kv_layout=kv_layout
    )


def cake_dsv41_fp8_quantize_append_sparse_mla_cache(
    latent_kv: torch.Tensor, slot_mapping: torch.Tensor, cache: torch.Tensor
) -> None:
    """Explicit DSv4.1 FP8 main-cache slot append (SM120/121); writes in place."""
    _k("dsv41_fp8_quantize_append_sparse_mla_cache")(latent_kv, slot_mapping, cache)


def cake_trtllm_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    max_seq_len: int,
    **kwargs,
):
    """Explicit Cake paged MLA decode; keyword arguments as the adapter."""
    return _k("trtllm_batch_decode_with_kv_cache_mla")(
        query,
        kv_cache,
        workspace_buffer,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        block_tables,
        seq_lens,
        max_seq_len,
        **kwargs,
    )


def cake_kimi_k3_mla_fp8_paged_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    cum_seq_lens_q: Optional[torch.Tensor] = None,
    max_q_len: Optional[int] = None,
    max_seq_len: Optional[int] = None,
) -> torch.Tensor:
    """Explicit Cake Kimi-K3 FP8 MLA attention (one-shot)."""
    return _k("kimi_k3_mla_fp8_paged_attention")(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        out,
        workspace_buffer,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        max_seq_len=max_seq_len,
    )


def cake_prepare_kimi_k3_mla_fp8_paged_attention(**kwargs):
    """Explicit Cake Kimi-K3 FP8 MLA attention (prepared launcher)."""
    return _k("prepare_kimi_k3_mla_fp8_paged_attention")(**kwargs)


def cake_prepare_nvfp4_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    query_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    sm_scale: float,
    sinks: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    seq_lens_cpu: Optional[torch.Tensor] = None,
):
    """Explicit Cake NVFP4 MLA decode prepare; returns the runner."""
    return _k("prepare_nvfp4_batch_decode_with_kv_cache_mla")(
        query,
        query_scale,
        kv_cache,
        kv_scale,
        block_tables,
        seq_lens,
        workspace_buffer,
        sm_scale=sm_scale,
        sinks=sinks,
        out=out,
        lse=lse,
        return_lse=return_lse,
        seq_lens_cpu=seq_lens_cpu,
    )


def cake_mla_varq_dcp_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    softmax_scale: float,
    *,
    cum_seq_lens_q: torch.Tensor,
    max_q_len: int,
    enable_dcp: bool = False,
    cp_world: int = 1,
    cp_rank: int = 0,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    return_lse: bool = True,
):
    """Explicit Cake variable-q MLA decode (one-shot)."""
    return _k("mla_varq_dcp_decode")(
        query,
        kv_cache,
        workspace_buffer,
        block_tables,
        seq_lens,
        max_seq_len,
        softmax_scale,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=max_q_len,
        enable_dcp=enable_dcp,
        cp_world=cp_world,
        cp_rank=cp_rank,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
        out=out,
        lse=lse,
        return_lse=return_lse,
    )


def cake_prepare_mla_varq_dcp_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    max_q_len: int,
    *,
    max_seq_len: int,
    softmax_scale: float,
    workspace_buffer: torch.Tensor,
    causal_seqlens_kv_global: Optional[torch.Tensor] = None,
    cp_world: int = 1,
    cp_rank: int = 0,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
):
    """Explicit Cake variable-q MLA decode prepare; returns the runner."""
    return _k("prepare_mla_varq_dcp_decode")(
        query,
        kv_cache,
        page_table,
        seq_lens,
        cum_seq_lens_q,
        max_q_len,
        max_seq_len=max_seq_len,
        softmax_scale=softmax_scale,
        workspace_buffer=workspace_buffer,
        causal_seqlens_kv_global=causal_seqlens_kv_global,
        cp_world=cp_world,
        cp_rank=cp_rank,
        out=out,
        lse=lse,
    )


def cake_concat_mla_k(
    k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor
) -> None:
    """Explicit Cake in-place MLA K assembly."""
    _k("concat_mla_k")(k, k_nope, k_rope)


def cake_create_block_sparse_attention_wrapper(float_workspace_buffer: torch.Tensor):
    """Explicit Cake SM100/103 VSA wrapper factory."""
    return _k("create_block_sparse_attention_wrapper")(float_workspace_buffer)


def cake_create_variable_block_sparse_attention_wrapper_sm90(
    float_workspace_buffer: torch.Tensor, *, engine: str = "cuda"
):
    """Explicit Cake SM90 VSA wrapper factory (``engine`` 'cuda' or 'cute')."""
    return _k("create_variable_block_sparse_attention_wrapper_sm90")(
        float_workspace_buffer, engine=engine
    )


def cake_bsa_attn_sm100_blk64_sage_fwd(
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
    """Explicit Cake Sage-FP8 block-sparse attention (SM100/103)."""
    return _k("bsa_attn_sm100_blk64_sage_fwd")(
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
    )


def cake_bsa_attn_sm120_blk64_sage_fwd(
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
    """Explicit Cake Sage block-sparse attention (SM120)."""
    return _k("bsa_attn_sm120_blk64_sage_fwd")(
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
    )


def cake_prepare_msa_nvfp4_sparse_decode(
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
    """Explicit Cake MSA NVFP4 sparse decode prepare; returns the runner."""
    return _k("prepare_msa_nvfp4_sparse_decode")(
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
    )


def cake_dsa_indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Explicit Cake DSA indexer top-k (one-shot); keyword args as the adapter."""
    return _k("dsa_indexer_topk")(q, k, w, cu_seqlens_q, cu_seqlens_k, **kwargs)


def cake_prepare_dsa_indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    **kwargs,
):
    """Explicit Cake DSA indexer top-k prepare; returns the runner."""
    return _k("prepare_dsa_indexer_topk")(q, k, w, cu_seqlens_q, cu_seqlens_k, **kwargs)


def cake_prepare_dense_mqa_logits(
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
    """Explicit dense MQA logits prepare; returns the plan."""
    return _k("prepare_dense_mqa_logits")(
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


def cake_fp8_mqa_logits(
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
    """Explicit DeepGEMM-signature ragged indexer logits (one-shot); f32 [Q, K] view."""
    return _k("fp8_mqa_logits")(
        q,
        kv,
        weights,
        ks,
        ke,
        clean_logits=clean_logits,
        max_seqlen_k=max_seqlen_k,
        sm_count=sm_count,
    )


def cake_get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor,
    block_kv: int,
    num_sms: int,
    indices: Optional[torch.Tensor] = None,
    *,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Explicit DeepGEMM-signature paged schedule metadata; int32 [num_sms + 1, 2]."""
    return _k("get_paged_mqa_logits_metadata")(
        context_lens, block_kv, num_sms, indices=indices, out=out
    )


def cake_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    schedule_meta: Optional[torch.Tensor],
    max_context_len: int,
    clean_logits: bool = False,
    indices: Optional[torch.Tensor] = None,
    *,
    sm_count: Optional[int] = None,
) -> torch.Tensor:
    """Explicit DeepGEMM-signature paged indexer logits (one-shot, one launch); f32 [B * next_n, max_len] view.

    ``schedule_meta`` is a signature-parity placeholder (``None`` allowed); ``sm_count`` fixes the CTA
    budget when it is ``None``.
    """
    return _k("fp8_paged_mqa_logits")(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        schedule_meta,
        max_context_len,
        clean_logits=clean_logits,
        indices=indices,
        sm_count=sm_count,
    )


def cake_prepare_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_table: torch.Tensor,
    max_context_len: int,
    **kwargs,
):
    """Explicit paged indexer logits prepare; returns the plan (keyword args as the adapter)."""
    return _k("prepare_paged_mqa_logits")(
        q, kv_cache, weights, context_lens, block_table, max_context_len, **kwargs
    )


def cake_prepare_sparse_mqa_metadata(sparse_indices: torch.Tensor, **kwargs):
    """Explicit sparse MQA metadata prepare; keyword args as the adapter."""
    return _k("prepare_sparse_mqa_metadata")(sparse_indices, **kwargs)


def cake_prepare_sparse_mqa_logits(
    q: torch.Tensor,
    sf_q: torch.Tensor,
    kv: torch.Tensor,
    sf_kv: Optional[torch.Tensor],
    weights: torch.Tensor,
    metadata_plan,
    *,
    output: Optional[torch.Tensor] = None,
):
    """Explicit sparse MQA logits prepare; returns the plan."""
    return _k("prepare_sparse_mqa_logits")(
        q, sf_q, kv, sf_kv, weights, metadata_plan, output=output
    )


def cake_kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-5,
    output_norm_eps: float = 1e-5,
    enable_pdl: bool = False,
) -> torch.Tensor:
    """Explicit Cake Kimi-K3 AttnRes (one-shot)."""
    return _k("kimi_k3_attn_res")(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        num_blocks=num_blocks,
        block_write_idx=block_write_idx,
        eps=eps,
        output_norm_eps=output_norm_eps,
        enable_pdl=enable_pdl,
    )


def cake_prepare_kimi_k3_attn_res(
    prefix: torch.Tensor,
    delta: Optional[torch.Tensor],
    blocks: torch.Tensor,
    norm_weight: torch.Tensor,
    qk_weight: torch.Tensor,
    output_norm_weight: Optional[torch.Tensor],
    out: torch.Tensor,
    *,
    num_blocks: int,
    block_write_idx: int = -1,
    eps: float = 1e-5,
    output_norm_eps: float = 1e-5,
    enable_pdl: bool = False,
):
    """Explicit Cake Kimi-K3 AttnRes prepare; returns the runner."""
    return _k("prepare_kimi_k3_attn_res")(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        out,
        num_blocks=num_blocks,
        block_write_idx=block_write_idx,
        eps=eps,
        output_norm_eps=output_norm_eps,
        enable_pdl=enable_pdl,
    )


def cake_minimax_h3_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Explicit Cake MiniMax-H3 varlen attention (one-shot)."""
    return _k("minimax_h3_varlen_attention")(
        query,
        key,
        value,
        cu_seqlens,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
    )


def cake_prepare_minimax_h3_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
):
    """Explicit Cake MiniMax-H3 varlen attention prepare; returns the runner."""
    return _k("prepare_minimax_h3_varlen_attention")(
        query,
        key,
        value,
        cu_seqlens,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
    )


def cake_minimax_h3_varlen_nvfp4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    pv_mode: str = "fp8",
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Explicit Cake MiniMax-H3 NVFP4 varlen attention (one-shot)."""
    return _k("minimax_h3_varlen_nvfp4_attention")(
        query,
        key,
        value,
        cu_seqlens,
        pv_mode=pv_mode,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
    )


def cake_prepare_minimax_h3_varlen_nvfp4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    pv_mode: str = "fp8",
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    workspace: Optional[Dict[str, torch.Tensor]] = None,
):
    """Explicit Cake MiniMax-H3 NVFP4 varlen attention prepare; returns the runner."""
    return _k("prepare_minimax_h3_varlen_nvfp4_attention")(
        query,
        key,
        value,
        cu_seqlens,
        pv_mode=pv_mode,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
        workspace=workspace,
    )


def cake_prepare_nvfp4_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    *,
    causal: bool = False,
):
    """Explicit Cake NVFP4 attention prepare (SM103); returns the runner."""
    return _k("prepare_nvfp4_attention")(q, k, v, out, causal=causal)


__all__ = [
    "cake_fmha_batch_decode_with_kv_cache",
    "cake_fmha_batch_context_with_kv_cache",
    "cake_dcp_spec_decode",
    "cake_prepare_balanced_batch_decode_with_kv_cache",
    "cake_sm110_gqa_decode",
    "cake_prepare_sm110_gqa_decode",
    "cake_launch_sm110_gqa_decode_prepared",
    "cake_sm110_xqa_prepare",
    "cake_sm110_xqa_attention",
    "cake_trtllm_batch_decode_sparse_mla_dsv4",
    "cake_create_sparse_mla_sm120_wrapper",
    "cake_sparse_mla_sm120_dsv4_nvfp4_decode",
    "cake_sparse_mla_sm120_dsv4_nvfp4_prefill",
    "cake_sparse_mla_sm120_dsv41_mixed_decode",
    "cake_create_sparse_mla_sm120_dsv41_mixed_wrapper",
    "cake_dsv41_fp8_quantize_pack_sparse_mla_cache",
    "cake_dsv41_fp8_quantize_append_sparse_mla_cache",
    "cake_trtllm_batch_decode_with_kv_cache_mla",
    "cake_kimi_k3_mla_fp8_paged_attention",
    "cake_prepare_kimi_k3_mla_fp8_paged_attention",
    "cake_prepare_nvfp4_batch_decode_with_kv_cache_mla",
    "cake_mla_varq_dcp_decode",
    "cake_prepare_mla_varq_dcp_decode",
    "cake_concat_mla_k",
    "cake_create_block_sparse_attention_wrapper",
    "cake_create_variable_block_sparse_attention_wrapper_sm90",
    "cake_bsa_attn_sm100_blk64_sage_fwd",
    "cake_bsa_attn_sm120_blk64_sage_fwd",
    "cake_prepare_msa_nvfp4_sparse_decode",
    "cake_dsa_indexer_topk",
    "cake_prepare_dsa_indexer_topk",
    "cake_prepare_dense_mqa_logits",
    "cake_prepare_sparse_mqa_metadata",
    "cake_prepare_sparse_mqa_logits",
    "cake_kimi_k3_attn_res",
    "cake_prepare_kimi_k3_attn_res",
    "cake_minimax_h3_varlen_attention",
    "cake_prepare_minimax_h3_varlen_attention",
    "cake_minimax_h3_varlen_nvfp4_attention",
    "cake_prepare_minimax_h3_varlen_nvfp4_attention",
    "cake_prepare_nvfp4_attention",
]
