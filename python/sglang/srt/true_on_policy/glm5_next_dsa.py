"""GLM-5.3-Flash (glm5_next) DSA under TOP: one indexer, selection and sparse-attention program for prefill,
decode and the trainer.

- Head gate: x (BF16) times the BF16 weights_proj on the BI DeepGEMM with FP32 output, scaled by
  n_heads**-0.5, then by the query's UE8M0 scale and head_dim**-0.5 (serving keeps the weight in FP32 holding
  BF16 values, so the products are those of its FP32 GEMM; only the reduction is pinned).
- Pooled keys: each complete group of KPOOL consecutive keys is compressed by serving's two-pass softmax kernel
  (``_kpool_softmax_rotate_write_cache_kernel``), Hadamard-rotated and quantized to FP8 with a UE8M0 scale.
  Decode compresses with serving's decode kernel, which runs the same arithmetic; the extend plan's
  online-softmax kernel is replaced, and extends must start on a group boundary. A speculative verify window
  compresses the groups it completes with that two-pass arithmetic too (``kpool_write_tail_and_maybe_compress``),
  and each of its rows reads the pooled keys and tail of its own position, as the decode of that position.
- Index logits: DeepGEMM's FP8 MQA logits over the pooled keys (ragged in prefill and the trainer, paged in
  decode).
- Selection: FlashInfer's deterministic top-k (ties to the smaller group) of GROUP_TOPK groups among the
  complete ones, in ascending group order, expanded to KPOOL tokens each, then the incomplete group's tokens
  (always selected). Rows with at most GROUP_TOPK complete groups select every group, so a row selects tokens
  [0, t] in ascending order up to INDEX_TOPK + its tail, -1 after.
- Attention: the TileLang sparse MLA kernel over the BF16 latent (NoPE, 512 wide) with all heads in one block
  per token, the selection padded with -1 to ATTENTION_WIDTH columns; no split over the selected rows.
"""

import torch

KPOOL = 4
INDEX_TOPK = 2048
GROUP_TOPK = INDEX_TOPK // KPOOL
# Serving's IndexerKPool.topk_output_width.
SELECTION_WIDTH = INDEX_TOPK + KPOOL - 1
# TileLang reads the selection in 64-index blocks; the tail-extended width is padded to the next block.
ATTENTION_WIDTH = (SELECTION_WIDTH + 63) // 64 * 64
INDEX_HEAD_DIM = 128
INDEX_HEADS = 32
LATENT_DIM = 512
HEADS = 64
SCALE_FMT = "ue8m0"
# IndexerKPool's k_norm: LayerNorm(head_dim, dtype=float32) at its default eps.
INDEX_KEY_NORM_EPS = 1e-6
# Serving's index cache page (DeepGEMM paged MQA logits needs 64 rows per page); logical pages hold KPOOL of them.
INDEX_PAGE = 64
# One trainer selection block's FP32 logits [rows, groups] stay within this many bytes.
_SELECT_LOGITS_BYTES = 1 << 30

_SERVING_ENVS = {"SGLANG_DSA_FUSE_TOPK": False}


class _GroupLayout:
    """The index-cache geometry the compression kernel asserts on when it only returns its result."""

    index_page_size = INDEX_PAGE
    index_head_dim = INDEX_HEAD_DIM


def validate_glm5_next_dsa(config, server_args) -> None:
    """Serving admission for the glm5_next DSA layers: checkpoint shape, backends, cache and features."""
    from sglang.srt.configs.model_config import dsa_layer_skips_topk
    from sglang.srt.true_on_policy.glm import validate_envs

    for name, value in (
        ("qk_rope_head_dim", 0),
        ("qk_nope_head_dim", 256),
        ("kv_lora_rank", LATENT_DIM),
        ("num_attention_heads", HEADS),
        ("index_head_dim", INDEX_HEAD_DIM),
        ("index_n_heads", INDEX_HEADS),
        ("index_topk", INDEX_TOPK),
        ("index_kpool", KPOOL),
        ("index_kpool_compress", True),
        ("index_kpool_always_select_tail", True),
    ):
        if getattr(config, name, None) != value:
            raise NotImplementedError(f"[top] GLM-5.3 DSA requires {name}={value}")
    # The checkpoint carries a default rope entry (no scaling); a scaled rope would change the softmax scale.
    rope_type = (config.rope_scaling or {}).get("rope_type") or (
        config.rope_scaling or {}
    ).get("type")
    if rope_type not in (None, "default"):
        raise NotImplementedError(
            "[top] GLM-5.3 DSA requires no rope_scaling (the softmax scale is qk_head_dim**-0.5)"
        )
    for layer in range(config.num_hidden_layers):
        if not config.is_kda_layer(layer) and dsa_layer_skips_topk(config, layer):
            raise NotImplementedError(
                f"[top] GLM-5.3 DSA layer {layer} reuses another layer's selection; every DSA layer selects its own"
            )
    if (
        # One sparse MLA for prefill and decode, the TileLang kernel the trainer calls (glm5_next_dsa.sparse_attention).
        server_args.dsa_prefill_backend != "tilelang"
        or server_args.dsa_decode_backend != "tilelang"
    ):
        raise NotImplementedError(
            "[top] GLM-5.3 DSA requires the TileLang sparse MLA on prefill and decode"
        )
    if server_args.kv_cache_dtype not in ("bf16", "bfloat16"):
        raise NotImplementedError("[top] GLM-5.3 DSA requires a BF16 latent KV cache")
    if server_args.page_size != INDEX_PAGE * KPOOL:
        raise NotImplementedError(
            f"[top] GLM-5.3 DSA requires {INDEX_PAGE * KPOOL}-token pages (64-row index pages)"
        )
    chunk = server_args.chunked_prefill_size
    if chunk is not None and chunk > 0 and chunk % INDEX_PAGE:
        raise NotImplementedError(
            "[top] GLM-5.3 DSA requires a chunked prefill size on the 64-token grid "
            "(every extend starts on a pooled-key group)"
        )
    if server_args.enable_hisparse:
        raise NotImplementedError(
            "[top] GLM-5.3 DSA does not admit hisparse (it changes which KV the indexer sees)"
        )
    if server_args.dcp_size != 1:
        raise NotImplementedError(
            "[top] GLM-5.3 DSA does not admit decode context parallelism"
        )
    validate_envs(_SERVING_ENVS)


def validate_device(device) -> None:
    if torch.cuda.get_device_capability(device)[0] != 9:
        raise NotImplementedError(
            "[top] GLM-5.3 DSA is qualified on SM90 only (DeepGEMM paged MQA logits and TileLang sparse MLA)"
        )


def head_gate(x, weight, q_scale, *, n_heads, softmax_scale):
    """Index head weights [rows, heads, 1] in FP32 for the MQA logits."""
    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise NotImplementedError("[top] GLM-5.3 head gate requires 2D BF16 rows")
    if weight.dtype not in (torch.bfloat16, torch.float32):
        raise NotImplementedError(
            "[top] GLM-5.3 head gate requires a BF16 or FP32 weights_proj"
        )
    if x.requires_grad or weight.requires_grad:
        raise NotImplementedError("[top] GLM-5.3 head gate is forward-only")
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
        _matmul_persistent_top,
    )

    # Serving casts the BF16 checkpoint weight to FP32 at load; the cast back is exact.
    gate = _matmul_persistent_top(
        x.contiguous(), weight.to(torch.bfloat16).t(), out_dtype=torch.float32
    )
    gate = gate * n_heads**-0.5
    return gate.unsqueeze(-1) * q_scale * softmax_scale


def index_key_norm(key, weight, bias):
    """The indexer's k_norm as serving's FP32-parameter LayerNorm runs it on BF16 rows (FlashInfer)."""
    import flashinfer

    if key.dtype != torch.bfloat16 or key.ndim != 2:
        raise NotImplementedError("[top] GLM-5.3 index keys must be 2D BF16 rows")
    return flashinfer.norm.layernorm(
        key.contiguous(), weight.float(), bias.float(), INDEX_KEY_NORM_EPS
    )


def quantize_index_query(query):
    """Serving's indexer query program: Hadamard rotation, then per-head FP8 with UE8M0 scales."""
    from sglang.kernels.ops.quantization.act_quant_triton import act_quant
    from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation

    return act_quant(rotate_activation(query.contiguous()), INDEX_HEAD_DIM, SCALE_FMT)


def compress_groups(slot_k, slot_score, ape):
    """FP8 pooled keys [groups, 128] and their scales for complete groups [groups, KPOOL, 128]."""
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
        kpool_softmax_rotate_write_cache,
    )

    buf = torch.empty(
        (1, INDEX_PAGE * (INDEX_HEAD_DIM + 4)), dtype=torch.uint8, device=slot_k.device
    )
    loc = torch.zeros((slot_k.shape[0],), dtype=torch.int64, device=slot_k.device)
    return kpool_softmax_rotate_write_cache(
        pool=_GroupLayout,
        buf=buf,
        slot_k=slot_k,
        slot_score=slot_score,
        ape=ape,
        loc=loc,
        round_scale=True,
        return_compressed=True,
        write_cache=False,
    )


def _group_rows(source, chunk_src):
    rows = chunk_src.to(torch.int64)[:, None] + torch.arange(
        KPOOL, device=source.device, dtype=torch.int64
    )
    return source[rows]


def require_group_aligned_extends(extend_seq_lens_cpu, seq_lens_cpu) -> None:
    for extend_len, seq_len in zip(extend_seq_lens_cpu, seq_lens_cpu):
        if (int(seq_len) - int(extend_len)) % KPOOL:
            raise NotImplementedError(
                "[top] GLM-5.3 DSA extends must start on a pooled-key group boundary"
            )


def write_extend_groups(
    *, pool, buf, key, score, ape, writes, extend_seq_lens_cpu, seq_lens_cpu
):
    """The extend plan's complete groups, compressed by the two-pass kernel decode runs, written to the cache."""
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
        kpool_softmax_rotate_write_cache,
    )

    require_group_aligned_extends(extend_seq_lens_cpu, seq_lens_cpu)
    kpool_softmax_rotate_write_cache(
        pool=pool,
        buf=buf,
        slot_k=_group_rows(key, writes.chunk_src),
        slot_score=_group_rows(score, writes.chunk_src),
        ape=ape,
        loc=writes.write_loc.to(torch.int64),
        round_scale=True,
    )


def select_tokens(logits, pool_lens, seq_lens, *, row_starts=None, out_rows=None):
    """Logical token ids [rows, SELECTION_WIDTH] (request-relative, ascending, -1 after) from pooled logits.

    ``logits[r, row_starts[r] + g]`` scores group g of row r's request; ``pool_lens`` counts its complete groups
    and ``seq_lens`` its tokens up to and including the row.
    """
    if logits.dtype != torch.float32 or logits.ndim != 2:
        raise NotImplementedError("[top] GLM-5.3 selection requires 2D FP32 logits")
    from sglang.srt.layers.attention.dsa.kpool_fp8_index import (
        append_kpool_tail_to_topk,
        expand_pooled_groups_to_topk,
    )
    from sglang.srt.true_on_policy.dsa.ordered_topk import ordered_dsa_topk

    rows = logits.shape[0]
    pool_lens = pool_lens[:rows].to(torch.int32)
    seq_lens = seq_lens[:rows].to(torch.int32)
    if row_starts is not None:
        row_starts = row_starts[:rows].to(torch.int32).contiguous()
    groups = ordered_dsa_topk(logits, pool_lens, GROUP_TOPK, row_starts)
    tokens = expand_pooled_groups_to_topk(groups, groups >= 0, INDEX_TOPK, KPOOL)
    out = append_kpool_tail_to_topk(tokens, seq_lens, pool_lens, KPOOL)
    if out_rows is None or out_rows == rows:
        return out
    padded = out.new_full((out_rows, SELECTION_WIDTH), -1)
    padded[:rows] = out
    return padded


def select_serving_tokens(
    logits,
    pool_lens,
    seq_lens,
    *,
    row_starts,
    out_rows,
    page_table,
    page_table_row_index,
    topk_offsets,
):
    """IndexerKPool's selection under TOP: request-relative ids, mapped to cache slots by the backend after."""
    if (
        page_table is not None
        or page_table_row_index is not None
        or topk_offsets is not None
    ):
        raise NotImplementedError(
            "[top] GLM-5.3 selection maps logical ids after the ordered top-k; the fused top-k transform "
            "(SGLANG_DSA_FUSE_TOPK) is not admitted"
        )
    if seq_lens is None:
        raise NotImplementedError(
            "[top] GLM-5.3 selection needs each row's token count for its tail"
        )
    return select_tokens(
        logits, pool_lens, seq_lens, row_starts=row_starts, out_rows=out_rows
    )


def sparse_attention(q, kv, indices, scale):
    """[rows, HEADS, LATENT_DIM] BF16: q [rows, HEADS, LATENT_DIM] attends to the kv rows [n, LATENT_DIM] its
    indices [rows, <= ATTENTION_WIDTH] name (-1 unselected), as serving's TileLang backend calls the kernel."""
    if (
        q.dtype != torch.bfloat16
        or kv.dtype != torch.bfloat16
        or q.ndim != 3
        or q.shape[1:] != (HEADS, LATENT_DIM)
        or kv.shape[-1] != LATENT_DIM
        or indices.dtype != torch.int32
        or indices.shape[0] != q.shape[0]
        or indices.shape[1] > ATTENTION_WIDTH
    ):
        raise NotImplementedError(
            "[top] GLM-5.3 sparse MLA requires BF16 [rows, 64, 512] queries over a 512-wide BF16 latent and "
            f"int32 selections of at most {ATTENTION_WIDTH} columns"
        )
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import tilelang_sparse_fwd

    if indices.shape[1] < ATTENTION_WIDTH:
        indices = torch.nn.functional.pad(
            indices, (0, ATTENTION_WIDTH - indices.shape[1]), value=-1
        )
    return tilelang_sparse_fwd(
        q=q.contiguous(),
        kv=kv.reshape(-1, 1, LATENT_DIM),
        indices=indices.contiguous().unsqueeze(1),
        sm_scale=scale,
        d_v=LATENT_DIM,
    ).squeeze(0)


def _select_block_rows(num_groups, rows):
    block = _SELECT_LOGITS_BYTES // (max(num_groups, 1) * 4) // 64 * 64
    return max(64, min(rows, block))


def trainer_selection(*, query, key, gate_score, hidden, weights_proj, ape, cu_seqlens):
    """The trainer's selection over packed whole samples: [rows, SELECTION_WIDTH] int32 packed-row ids, -1 after.

    query [rows, INDEX_HEADS, 128] is wq_b's BF16 output (before rotation), key [rows, 128] the BF16 normed
    index key, gate_score [rows, 128] the BF16 compression gate projection, hidden [rows, hidden] the layer
    input; cu_seqlens [samples + 1] bounds the samples.
    """
    from deep_gemm import fp8_mqa_logits

    device = query.device
    rows = query.shape[0]
    q_fp8, q_scale = quantize_index_query(query)
    weights = head_gate(
        hidden,
        weights_proj,
        q_scale,
        n_heads=INDEX_HEADS,
        softmax_scale=INDEX_HEAD_DIM**-0.5,
    ).squeeze(-1)

    cu = cu_seqlens.to(torch.int64)
    lens = cu[1:] - cu[:-1]
    groups_per_sample = lens // KPOOL
    pool_cu = torch.zeros_like(cu)
    pool_cu[1:] = torch.cumsum(groups_per_sample, 0)
    num_groups = int(pool_cu[-1])
    sample_of_group = torch.repeat_interleave(
        torch.arange(lens.shape[0], device=device), groups_per_sample
    )
    group_src = (
        cu[sample_of_group]
        + (torch.arange(num_groups, device=device) - pool_cu[sample_of_group]) * KPOOL
    )
    pooled = compress_groups(
        _group_rows(key, group_src), _group_rows(gate_score, group_src), ape
    )

    token = torch.arange(rows, device=device)
    sample = torch.searchsorted(cu, token, right=True) - 1
    local = token - cu[sample]
    seq_lens = (local + 1).to(torch.int32)
    pool_lens = torch.div(seq_lens, KPOOL, rounding_mode="floor").to(torch.int32)
    starts = pool_cu[sample].to(torch.int32)
    out = torch.empty((rows, SELECTION_WIDTH), dtype=torch.int32, device=device)
    block = _select_block_rows(num_groups, rows)
    for begin in range(0, rows, block):
        r = slice(begin, min(begin + block, rows))
        if num_groups == 0:
            logits = torch.empty(
                (r.stop - r.start, 0), dtype=torch.float32, device=device
            )
        else:
            logits = fp8_mqa_logits(
                q_fp8[r].contiguous(),
                pooled,
                weights[r].contiguous(),
                starts[r].contiguous(),
                (starts[r] + pool_lens[r]).contiguous(),
                clean_logits=True,
            )
        out[r] = select_tokens(logits, pool_lens[r], seq_lens[r], row_starts=starts[r])
    return torch.where(out >= 0, out + cu[sample][:, None].to(torch.int32), out)
