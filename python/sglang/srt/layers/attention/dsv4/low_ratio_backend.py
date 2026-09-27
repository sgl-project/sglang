"""DeepSeek V4.1 low compress ratio (1 / 2) sources shared by the DSV4 attention
backends: compression, index-K publishing, indexer top-k and their metadata helpers."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, List, Optional, Tuple, TypeVar

import torch
import torch.nn.functional as F

from sglang.kernels.ops.attention.dsv4 import (
    topk_transform_paged,
    topk_transform_paged_v2,
    topk_transform_ragged_v2,
)
from sglang.kernels.ops.attention.dsv4.sm90_fp4_indexer import fp4_index_logits_decode
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import dsa_use_prefill_cp
from sglang.srt.layers.attention.dsv4.dsv41_sparse import (
    _rope_fq4,
    token_req_indices,
)
from sglang.srt.layers.attention.dsv4.indexer import select_candidate_blocks
from sglang.srt.layers.cp.interleave import interleave_rows_per_request
from sglang.srt.layers.cp.utils import cp_materialize_global_token_order
from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import ceil_align

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

PAGE_INDEX_ALIGNED_SIZE = 64


T = TypeVar("T", bound=Optional[torch.Tensor])


@functools.lru_cache(maxsize=None)
def _is_sm100_or_newer() -> bool:
    """The DeepGEMM fp8_fp4 mqa-logits kernels need SM100/SM120; Hopper takes the torch indexer."""
    return torch.cuda.get_device_capability()[0] >= 10


def _pad_last_dim(x: T, multiples_of: int = PAGE_INDEX_ALIGNED_SIZE) -> T:
    if x is None:
        return None
    curr_size = x.shape[-1]
    target_size = ceil_align(curr_size, multiples_of)
    return F.pad(x, pad=(0, target_size - curr_size), mode="constant", value=-1)


def _expand_index_page_table(
    page_table: torch.Tensor,
    *,
    full_page_size: int,
    compress_ratio: int,
    index_page_size: int,
) -> torch.Tensor:
    """Expand the FULL page table into the block table of a low-ratio indexer-K
    pool, which pages at `index_page_size` slots rather than a FULL page.

    The kernel resolves compressed slot j through
    page_table[b, j // index_page_size] * index_page_size + j % index_page_size,
    which with this expansion is the c1/c2 KV pool slot of the same position.
    [bs, n_full_pages] -> [bs, n_full_pages * blocks_per_page], int32.
    """
    slots_per_page = full_page_size // compress_ratio
    assert slots_per_page % index_page_size == 0, (
        f"{full_page_size = } / {compress_ratio = } must be a multiple of "
        f"{index_page_size = }"
    )
    blocks_per_page = slots_per_page // index_page_size
    if blocks_per_page == 1:
        return page_table
    bs, n = page_table.shape
    base = page_table.to(torch.int64) * blocks_per_page
    offsets = torch.arange(blocks_per_page, device=page_table.device, dtype=torch.int64)
    expanded = base.unsqueeze(-1) + offsets  # [bs, n, blocks_per_page]
    return expanded.reshape(bs, n * blocks_per_page).to(torch.int32)


def _fp4_paged_mqa_logits(
    q_fp4: Tuple[torch.Tensor, torch.Tensor],
    k_cache: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    deep_gemm_metadata,
    max_seq_len: int,
) -> torch.Tensor:
    """DeepGEMM paged fp4 logits for the low-ratio indexer. No hadamard: the
    reference does not apply one."""
    from deep_gemm import fp8_fp4_paged_mqa_logits as fn

    sl = seq_lens.to(torch.int32)
    if sl.dim() == 1:
        sl = sl.unsqueeze(-1)
    return fn(
        q_fp4,
        k_cache,
        weights,
        sl,
        page_table,
        deep_gemm_metadata,
        max_seq_len,
        False,
    )


def two_level_decode_logits(
    logits: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    is_candidate_source: bool,
    uses_candidates: bool,
    topk_blocks: int,
    block_size: int,
    published: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Apply candidate-block filtering and return logits plus an optional published mask.

    Mask columns past each sequence length to -inf before selection: the paged
    logits kernel leaves that tail uninitialized, and an all -inf block means
    unreachable. This path is graph-captured and must not synchronize with the host.
    Work scales with allocated page-table capacity, not live sequence length.
    """
    from sglang.srt.model_executor.runner_utils.capture_mode import (
        get_capture_dsa_variant,
    )

    if get_capture_dsa_variant() in (
        "candidate_all",
        "candidate_c2_all",
        "candidate_unfiltered",
    ):
        # All requests fit the candidate budget; paged top-k masks their tails.
        return logits, None
    if not (is_candidate_source or uses_candidates):
        return logits, None

    lens_col = seq_lens if seq_lens.dim() > 1 else seq_lens.unsqueeze(-1)
    reachable = torch.arange(logits.shape[-1], device=logits.device) < lens_col
    logits = logits.float().masked_fill(~reachable, -torch.inf)

    if is_candidate_source:
        # The source scores over every reachable position itself and only publishes,
        # which is what the reference does.
        return logits, select_candidate_blocks(
            logits, lens_col, topk_blocks=topk_blocks, block_size=block_size
        )

    assert torch.is_tensor(published) and published.shape[0] == logits.shape[0], (
        "candidate mask missing for decode"
    )
    return logits.masked_fill(~published[:, : logits.shape[-1]], -torch.inf), None


# Arbitrary cap on one bf16 [rows, heads, lc] score chunk; transients run ~3x this.
_TORCH_INDEXER_SCORE_BUDGET_BYTES = 1 << 30


def mask_topk_scores(
    scores: torch.Tensor,
    indices: torch.Tensor,
    offsets: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Keep masked indexer scores out of attention even when top-k underfills."""
    columns = indices.to(torch.int64)
    if offsets is not None:
        columns = columns - offsets[:, None]
    selected_scores = scores.gather(1, columns.clamp(0, scores.shape[1] - 1))
    valid = (
        (columns >= 0) & (columns < scores.shape[1]) & (selected_scores > -torch.inf)
    )
    return indices.masked_fill(~valid, -1)


@functools.cache
def _has_dense_fp4_indexer() -> bool:
    if not torch.cuda.is_available() or torch.version.cuda is None:
        return False
    try:
        import deep_gemm
    except ImportError:
        return False
    return hasattr(deep_gemm, "fp8_fp4_mqa_logits")


def _dense_fp4_mqa_logits(
    q_fp4: Tuple[torch.Tensor, torch.Tensor],
    kv_fp4: Tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    ks: torch.Tensor,
    ke: torch.Tensor,
    max_seqlen_k: int,
) -> torch.Tensor:
    from deep_gemm import fp8_fp4_mqa_logits as fn

    # q (int8 [T, H, 64], int32 [T, H]) x kv (int8 [L, 64], int32 [L]) -> fp32
    # [T, max_seqlen_k]; row t's column j is k[ks_t + j], valid for j < ke_t - ks_t
    # (the rest is garbage: the kernel rejects clean_logits).
    return fn(q_fp4, kv_fp4, weights, ks, ke, False, max_seqlen_k)


def _low_ratio_source_projections(layer, x, q_lora, positions, bufs):
    """Projections of a ratio-1/2 source layer, run as an eager break on the
    live rows into static buffers: the compressor's kv / score and the indexer's
    query and head weights. These GEMMs pick their algorithm by M, so at the
    bucket size their rows differ from eager; everything downstream is
    row-independent. Padded rows are zeroed."""
    from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
        get_tc_piecewise_forward_context,
    )

    real = get_tc_piecewise_forward_context().forward_batch.num_token_non_padded_cpu
    if real is None:
        real = x.shape[0]

    def put(name, value):
        buf = bufs[name]
        buf[:real].copy_(value)
        buf[real:].zero_()

    if real == 0:
        # An idle DP-attention rank replays on fabricated rows with no live
        # token; a zero-row GEMM is a launch error, so only zero the buffers.
        for buf in bufs.values():
            buf.zero_()
        return

    if layer.compressor is not None:
        kv, score = layer.compressor.project(x[:real])
        put("kv", kv)
        if score is not None:
            put("score", score)
    if layer.indexer is not None:
        indexer = layer.indexer
        put("q", indexer.queries(q_lora[:real], layer.freqs_cis[positions[:real]]))
        put("w", indexer.head_weights(x[:real]))


def _bcg_low_ratio_source_projections(*args):
    from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.breakable_cuda_graph import (
        eager_on_graph,
    )

    global _bcg_low_ratio_source_projections_fn
    if _bcg_low_ratio_source_projections_fn is None:
        _bcg_low_ratio_source_projections_fn = eager_on_graph(True)(
            _low_ratio_source_projections
        )
    return _bcg_low_ratio_source_projections_fn(*args)


_bcg_low_ratio_source_projections_fn = None


def _as_int_list(values) -> Optional[List[int]]:
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            return None
        values = values.tolist()
    return [int(v) for v in values]


def _low_ratio_compression_metadata(
    compress_ratio: int, seq_lens_casual: torch.Tensor, raw_out_loc: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ratio 1/2 counterpart of the triton c4/c128 metadata; a completed group's
    latent lives at raw_out_loc // ratio, -1 for a token that completes none."""
    num_write_tokens = raw_out_loc.shape[0]
    completes_group = seq_lens_casual[:num_write_tokens] % compress_ratio == 0
    out_loc = torch.where(
        completes_group, raw_out_loc.to(torch.int64) // compress_ratio, -1
    )
    topk_lengths_clamp1 = (seq_lens_casual // compress_ratio).clamp_min(1)
    return out_loc, topk_lengths_clamp1.to(torch.int32)


def _low_ratio_sparse_buffers(
    topk_lengths_clamp1: torch.Tensor, topk: int, is_prefill: bool
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Fixed-shape top-k buffers the index_source layers fill in place. The
    extra_topk_length the kernel reads comes from positions, not from the indexer."""
    sparse_topk_lengths = torch.clamp(topk_lengths_clamp1, max=topk)
    page_indices = _pad_last_dim(
        torch.full(
            (topk_lengths_clamp1.size(0), topk),
            -1,
            dtype=torch.int32,
            device=topk_lengths_clamp1.device,
        )
    )
    raw_indices = torch.empty_like(page_indices) if is_prefill else None
    return sparse_topk_lengths, page_indices, raw_indices


class LowRatioBackendMixin:
    @property
    def low_ratio_prefill_graph(self) -> bool:
        """The ratio-1/2 sources can run inside the prefill CUDA graph (the
        DeepGEMM paged indexer, so Blackwell CUDA only)."""
        return (
            bool(self.low_ratios) and _has_dense_fp4_indexer() and _is_sm100_or_newer()
        )

    def forward_low_ratio_sources(
        self,
        *,
        layer,
        x,
        q_lora,
        positions,
        forward_batch: ForwardBatch,
        run_compressor: bool = True,
        run_indexer: bool = True,
    ) -> None:
        """Runs on every ratio 1/2 layer before its attention; the metadata and
        latents it writes are what the layers after it attend through."""
        if forward_batch.forward_mode.is_idle():
            return
        if forward_batch.encoder_swa_replay:
            run_compressor = False
        if dsa_use_prefill_cp(forward_batch) and forward_batch.forward_mode.is_extend():
            self._forward_low_ratio_sources_cp(
                layer=layer,
                x=x,
                q_lora=q_lora,
                positions=positions,
                forward_batch=forward_batch,
                run_compressor=run_compressor,
                run_indexer=run_indexer,
            )
            return
        meta = self.forward_metadata
        # only the HIP metadata hoists the per-token inputs
        hoisted_req = getattr(meta, "low_ratio_req_indices", None)
        hoisted_pos = getattr(meta, "low_ratio_pos_i64", None)
        if (
            hoisted_req is not None
            and hoisted_pos is not None
            and hoisted_pos.shape[0] == positions.shape[0]
        ):
            # Bucket-sized under the prefill graph; an eager break sees the
            # live rows only and falls through.
            req, pos = hoisted_req, hoisted_pos
        else:
            req = token_req_indices(forward_batch, num_tokens=positions.shape[0])
            pos = positions.to(torch.int64)
        if (
            forward_batch.forward_mode.is_extend()
            and self._low_ratio_in_prefill_graph()
        ):
            bufs = self._source_projection_buffers(x.shape[0], layer.compress_ratio)
            _bcg_low_ratio_source_projections(layer, x, q_lora, pos, bufs)
            if run_compressor and layer.compressor is not None:
                self._low_ratio_compress_torch(
                    layer, x, req, pos, projected=(bufs["kv"], bufs.get("score"))
                )
            if run_indexer and layer.indexer is not None:
                self._low_ratio_index_topk_prefill_graph(
                    layer, pos, bufs["q"], bufs["w"]
                )
            return
        if run_compressor and layer.compressor is not None:
            self._low_ratio_compress(layer, x, req, pos, forward_batch)
        if run_indexer and layer.indexer is not None:
            self._low_ratio_index_topk(layer, x, q_lora, req, pos, forward_batch)

    def _forward_low_ratio_sources_cp(
        self, *, layer, x, q_lora, positions, forward_batch, run_compressor, run_indexer
    ) -> None:
        """Every rank writes the whole prompt's compressed state and scores its own rows."""
        cp_meta = forward_batch.attn_cp_metadata
        total = int(cp_meta.total_seq_lens)
        tail = self.forward_metadata.late_layer_tail
        if tail is not None:
            q_lens_cpu = tail.local_lens_cpu
            req_global, pos_global = tail.req_global, tail.pos_global
        else:
            q_lens_cpu = interleave_rows_per_request(
                _as_int_list(forward_batch.extend_seq_lens_cpu),
                get_parallel().attn_cp_rank,
                get_parallel().attn_cp_size,
            )
            req_global = token_req_indices(forward_batch, num_tokens=total)
            pos_global = forward_batch.positions[:total].to(torch.int64)
        num_local = sum(q_lens_cpu)
        if run_compressor and layer.compressor is not None:
            x_global = cp_materialize_global_token_order(
                x.contiguous(), forward_batch, torch.cuda.current_stream()
            )[:total]
            self._low_ratio_compress_torch(layer, x_global, req_global, pos_global)
        if run_indexer and layer.indexer is not None:
            self._low_ratio_index_topk_dense(
                layer,
                x[:num_local],
                q_lora[:num_local],
                positions[:num_local].to(torch.int64),
                forward_batch,
                torch.tensor(q_lens_cpu, dtype=torch.int32, device=x.device),
                q_lens_cpu,
            )

    def _low_ratio_compress(self, layer, x, req, pos, forward_batch) -> None:
        if forward_batch.forward_mode.is_decode():
            self._low_ratio_compress_decode(layer, x, req, pos)
        else:
            self._low_ratio_compress_torch(layer, x, req, pos)

    def _low_ratio_in_prefill_graph(self) -> bool:
        from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
            is_in_breakable_cuda_graph,
        )

        return self.low_ratio_prefill_graph and is_in_breakable_cuda_graph()

    def _low_ratio_compress_decode(self, layer, x, req, pos) -> None:
        # decided at load time with the projection layout, see `fused_low_ratio_compress_supported`
        if layer.compressor.use_fused_compress:
            self._low_ratio_compress_decode_fused(layer, x, req, pos)
            return
        if layer.compress_ratio == 1:
            core = self.forward_metadata.core_metadata
            kv, _ = layer.compressor.project(x)
            slots = torch.where(
                core.c1_out_loc >= 0, core.c1_out_loc, torch.zeros_like(core.c1_out_loc)
            )
            self._low_ratio_write_group(
                layer,
                kv,
                slots,
                pos,
                fuse_index_store=(
                    x.is_cuda
                    and torch.version.cuda is not None
                    and _is_sm100_or_newer()
                ),
            )
            return
        if not (x.is_cuda and torch.version.cuda):
            self._low_ratio_compress_torch(layer, x, req, pos)
            return

        from sglang.kernels.ops.attention.dsv4.pair_pool_decode import pair_pool_decode

        core = self.forward_metadata.core_metadata
        state = self.token_to_kv_pool.get_attention_compress_states(layer.layer_id)
        kv, score = layer.compressor.project(x)
        pooled, group_pos, slots = pair_pool_decode(
            kv,
            score,
            pos,
            core.raw_out_loc,
            core.c2_out_loc,
            req,
            state.kv_score_buffer.kv,
            state.kv_score_buffer.score,
            state.kv_score_buffer.shape[0] - 1,
            ring_size=state.ring_size,
        )
        self._low_ratio_write_group(
            layer,
            pooled,
            slots,
            group_pos,
            fuse_index_store=_is_sm100_or_newer(),
        )

    def _low_ratio_compress_decode_fused(self, layer, x, req, pos) -> None:
        """A low-ratio layer's whole decode write in three kernels: the fused compress
        and main-KV store, the indexer's `wk` GEMM, and the fused index-K store.

        Every array is passed in the dtype the metadata holds and both write kernels
        suppress padded stores, so nothing else runs on the path.
        """
        from sglang.kernels.ops.attention.dsv4.c1 import c1_decode_norm_rope_store
        from sglang.kernels.ops.attention.dsv4.c2 import c2_decode_norm_rope_store
        from sglang.kernels.ops.attention.dsv4.fp4_rope import (
            index_k_norm_rope_pack_store,
        )
        from sglang.kernels.ops.attention.dsv4.fp4_rope_hip import (
            index_k_norm_rope_pack_store_split,
        )

        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        compressor = layer.compressor
        layer_id = layer.layer_id
        # `freqs_cis` is contiguous complex64, so this is a pure view onto the
        # same storage -- no copy, no launch. It is the layout every DSV4 rope
        # kernel already takes; see `elementwise.py`.
        freqs_cis = torch.view_as_real(layer.freqs_cis).flatten(-2)
        kv_cache = pool.get_extra_key_buffer(layer_id)
        page_size = pool.get_extra_key_page_size(layer_id)
        assert kv_cache is not None

        if layer.compress_ratio == 1:
            # `c1_out_loc` is `raw_out_loc` at ratio 1 -- the metadata divides by
            # the ratio -- so pass the original (`out_cache_loc`, int64) as-is;
            # the kernel indexes either width.
            latent = c1_decode_norm_rope_store(
                compressor.wkv(x),
                compressor.norm.weight.data,
                pos,
                core.raw_out_loc,
                compressor.norm.eps,
                freqs_cis,
                kv_cache,
                page_size=page_size,
            )
            out_loc = core.c1_out_loc
        else:
            # the pending pair is `| kv | score |` at `req * ring_size + pos % ring_size`, derived in-kernel
            state = pool.get_attention_compress_states(layer_id)
            latent = c2_decode_norm_rope_store(
                compressor.project_fused(x),
                state.kv_score_buffer.kv_score,
                compressor.norm.weight.data,
                pos,
                req,
                core.raw_out_loc,
                compressor.norm.eps,
                freqs_cis,
                kv_cache,
                page_size=page_size,
                ring_size=state.ring_size,
            )
            out_loc = core.c2_out_loc

        indexer = layer.indexer
        if indexer is not None and indexer.owns_k:
            # `out_loc` is -1 on a row completing no group and 0 on a padded one;
            # the kernel publishes nothing for either, so the `where` that would
            # fold both onto the dummy slot is not needed.
            assert out_loc is not None
            k = indexer.forward_wk(latent)
            if pool.low_ratio_index_k_is_split(layer_id):
                # ROCm keeps payload and scale apart (FlyDSL layout); same bytes as store_fp4_index_k_cache_split
                index_k_norm_rope_pack_store_split(
                    k,
                    indexer.k_norm.weight.data,
                    indexer.k_norm.eps,
                    freqs_cis,
                    pos,
                    out_loc,
                    pool.get_index_k_fp4_payload_buffer(layer_id),
                    pool.get_index_k_fp4_scale_buffer(layer_id),
                    ratio=layer.compress_ratio,
                )
            else:
                index_k_norm_rope_pack_store(
                    k,
                    indexer.k_norm.weight.data,
                    indexer.k_norm.eps,
                    freqs_cis,
                    pos,
                    out_loc,
                    pool.get_index_k_with_scale_buffer(layer_id),
                    ratio=layer.compress_ratio,
                )

    def _low_ratio_compress_torch(self, layer, x, req, pos, projected=None) -> None:
        core = self.forward_metadata.core_metadata
        num_tokens = pos.shape[0]
        kv, score = projected if projected is not None else layer.compressor.project(x)
        if not num_tokens:
            return
        if layer.compress_ratio == 1:
            self._low_ratio_write_group(layer, kv, core.c1_out_loc[:num_tokens], pos)
            return

        partner_kv, partner_score = self._low_ratio_pair_partners(
            layer_id=layer.layer_id,
            kv=kv,
            score=score,
            req=req,
            pos=pos,
            pad=core.raw_out_loc[:num_tokens] == 0,
        )
        pooled = layer.compressor.pool_pairs(
            torch.stack([partner_kv, kv], dim=1),
            torch.stack([partner_score, score], dim=1),
        )
        group_pos = torch.where(pos % 2 == 1, pos - 1, pos)
        out_loc = core.c2_out_loc[:num_tokens]
        slots = torch.where(out_loc >= 0, out_loc, torch.zeros_like(out_loc))
        self._low_ratio_write_group(layer, pooled, slots, group_pos)

    def _low_ratio_pair_partners(
        self, *, layer_id, kv, score, req, pos, pad
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read the preceding token, then publish this batch's trailing ring window.

        Each request occupies consecutive rows and positions. Keeping only its
        last ring_size rows gives every write a distinct live slot, including
        when a prefill chunk is longer than the ring. Reads precede all writes
        so wrapping cannot overwrite a cross-chunk partner.
        """
        state = self.token_to_kv_pool.get_attention_compress_states(layer_id)
        ring = state.ring_size
        num_tokens = pos.shape[0]
        if not num_tokens:
            return kv, score

        read_pos = (pos - 1).masked_fill(pad, -1)
        carried = state.get_state_by_state_loc(
            state.translate_from_req_position_to_state_loc(req, read_pos)
        )
        in_batch = torch.zeros_like(pad)
        in_batch[1:] = (
            (req[1:] == req[:-1]) & (pos[1:] == pos[:-1] + 1) & ~pad[1:] & ~pad[:-1]
        )
        partner_kv = torch.where(in_batch[:, None], torch.roll(kv, 1, 0), carried.kv)
        partner_score = torch.where(
            in_batch[:, None], torch.roll(score, 1, 0), carried.score
        )

        keep = ~pad
        if num_tokens > ring:
            keep[:-ring] &= (req[:-ring] != req[ring:]) | pad[ring:]
        write_pos = pos.masked_fill(~keep, -1)
        state.set_state_by_state_loc(
            state.translate_from_req_position_to_state_loc(req, write_pos),
            KVAndScore.from_kv_score(kv=kv, score=score),
        )
        return partner_kv, partner_score

    def _low_ratio_write_group(
        self, layer, pooled, slots, group_pos, *, fuse_index_store=False
    ) -> None:
        pool = self.token_to_kv_pool
        latent = layer.compressor.finish(pooled)
        freqs = layer.freqs_cis[group_pos]
        # Index keys come from the pre-RoPE latent, so publish them first. Stored
        # as fp4 (per-32 ue8m0, no hadamard), matching the reference indexer.
        if layer.indexer is not None and layer.indexer.owns_k:
            if (
                fuse_index_store
                and latent.is_cuda
                and torch.version.cuda is not None
                and latent.dtype == torch.bfloat16
                and layer.indexer.index_head_dim == 128
            ):
                from sglang.kernels.ops.attention.dsv4.rope_pack_indexer import (
                    rope_fake_quant_pack_indexer,
                )

                indexer = layer.indexer
                k = indexer.k_norm(indexer.forward_wk(latent))
                rope_fake_quant_pack_indexer(
                    k,
                    freqs,
                    indexer.rope_head_dim,
                    cache=pool.get_index_k_with_scale_buffer(layer.layer_id),
                    loc=slots,
                )
            else:
                pool.set_index_k_fp4(
                    layer_id=layer.layer_id,
                    loc=slots,
                    cache_k=layer.indexer.index_keys(latent, freqs),
                )
        # The FlashMLA cache requantizes the FP4/E4M3 latent into its FP8 layout.
        latent = _rope_fq4(latent, freqs, layer.rope_head_dim, compressed_kv=True)
        pool.set_extra_key_buffer_fused(
            layer_id=layer.layer_id, loc=slots, cache_k=latent
        )

    def _low_ratio_index_topk(self, layer, x, q_lora, req, pos, forward_batch) -> None:
        from sglang.srt.model_executor.runner_utils.capture_mode import (
            skip_low_ratio_indexer,
        )

        if skip_low_ratio_indexer(layer.compress_ratio):
            # The compressor still writes index K for later, longer contexts.
            return
        is_decode_or_verify = (
            forward_batch.forward_mode.is_decode()
            or forward_batch.forward_mode.is_target_verify()
        )
        if is_decode_or_verify and _is_sm100_or_newer():
            self._low_ratio_index_topk_decode(layer, x, q_lora, pos)
        elif (
            self._use_dense_fp4_prefill_indexer(forward_batch) and _is_sm100_or_newer()
        ):
            self._low_ratio_index_topk_extend(layer, x, q_lora, pos, forward_batch)
        elif is_decode_or_verify:
            self._low_ratio_index_topk_sm90_decode(layer, x, q_lora, req, pos)
        else:
            self._low_ratio_index_topk_torch(layer, x, q_lora, req, pos)

    @staticmethod
    def _use_dense_fp4_prefill_indexer(forward_batch) -> bool:
        return (
            not envs.SGLANG_DSV41_TORCH_PREFILL_INDEXER.get()
            and _has_dense_fp4_indexer()
            and forward_batch.forward_mode.is_extend()
            and forward_batch.seq_lens_cpu is not None
            and forward_batch.extend_seq_lens_cpu is not None
        )

    def _low_ratio_index_topk_extend(
        self, layer, x, q_lora, pos, forward_batch
    ) -> None:
        tail = self.forward_metadata.late_layer_tail
        if tail is not None:
            q_lens, q_lens_cpu = tail.extend_seq_lens, tail.extend_seq_lens_cpu
        else:
            q_lens = forward_batch.extend_seq_lens
            q_lens_cpu = _as_int_list(forward_batch.extend_seq_lens_cpu)
        assert q_lens_cpu is not None
        self._low_ratio_index_topk_dense(
            layer, x, q_lora, pos, forward_batch, q_lens, q_lens_cpu
        )

    def _low_ratio_index_topk_dense(
        self, layer, x, q_lora, pos, forward_batch, q_lens, q_lens_cpu
    ) -> None:
        """Dense fp4 indexer over rows laid out request after request, `q_lens[b]` each."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            quantize_fp4_indexer_tensor,
        )

        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        indexer = layer.indexer
        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        page_indices.fill_(-1)
        if raw_indices is not None:
            raw_indices.fill_(-1)

        seq_lens_cpu = _as_int_list(forward_batch.seq_lens_cpu)
        assert seq_lens_cpu is not None
        device = pos.device
        # Visible compressed positions per request at its newest token; the
        # per-token count (pos + 1) // ratio bounds each row below.
        lc_per_req = [s // ratio for s in seq_lens_cpu]
        req_pool_indices = forward_batch.req_pool_indices.to(torch.int64)
        slot_chunks, starts = self._low_ratio_extend_k_slots(
            ratio=ratio,
            lc_per_req=lc_per_req,
            req_pool_indices=req_pool_indices,
            device=device,
        )
        empty_mask = torch.zeros(0, 0, dtype=torch.bool, device=device)
        num_tokens = pos.shape[0]
        if not slot_chunks or num_tokens == 0:
            if indexer.is_candidate_source:
                self.candidate_masks = [empty_mask for _ in lc_per_req]
            return
        k_slots = torch.cat(slot_chunks)
        k_fp4, k_sf = pool.get_low_ratio_index_k_fp4(layer.layer_id, k_slots)

        q = indexer.queries(q_lora, layer.freqs_cis[pos])  # [T, H, 128] fp4 grid
        num_heads = q.shape[1]
        q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
        q_fp4 = q_fp4.view(num_tokens, num_heads, 64)
        q_sf = q_sf.view(num_tokens, num_heads)
        weights = indexer.head_weights(x).float()
        compress_lens = ((pos + 1) // ratio).to(torch.int32)
        ks = torch.repeat_interleave(
            torch.tensor(starts, dtype=torch.int32, device=device),
            q_lens.to(torch.int64),
            output_size=num_tokens,
        )
        logits = _dense_fp4_mqa_logits(
            (q_fp4, q_sf),
            (k_fp4, k_sf),
            weights,
            ks,
            ks + compress_lens,
            # the fused top-k reads score rows through 16-byte vectors
            ceil_align(max(lc_per_req), 4),
        )
        if indexer.is_candidate_source or indexer.uses_candidates:
            self._publish_or_consume_candidates(
                indexer, logits, compress_lens, lc_per_req, q_lens_cpu, empty_mask
            )
        topk = indexer.index_topk
        selected = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
        topk_transform_ragged_v2(
            logits, compress_lens, out_offsets=ks, out_indices=selected
        )
        if indexer.uses_candidates and not indexer.is_candidate_source:
            selected = mask_topk_scores(logits, selected, ks)
        # ascending positions, padding last: the layout the consumers expect
        unselected = torch.iinfo(torch.int32).max
        selected = selected.masked_fill(selected < 0, unselected).sort(dim=-1).values
        chosen = selected != unselected
        page_indices[:num_tokens, :topk] = torch.where(
            chosen, k_slots[selected.clamp_max(k_slots.shape[0] - 1)], -1
        ).to(torch.int32)
        if raw_indices is not None:
            raw_indices[:num_tokens, :topk] = torch.where(
                chosen, selected - ks[:, None], -1
            )

    def _publish_or_consume_candidates(
        self, indexer, logits, compress_lens, lc_per_req, q_lens_cpu, empty_mask
    ) -> None:
        """Level one of the two-level top-k: publish block masks, or sink non-candidates."""
        publish = [] if indexer.is_candidate_source else None
        j = torch.arange(logits.shape[1], device=logits.device)
        tok_start = 0
        for b, (lc, t_len) in enumerate(zip(lc_per_req, q_lens_cpu)):
            rows = slice(tok_start, tok_start + t_len)
            tok_start += t_len
            if lc == 0 or t_len == 0:
                if publish is not None:
                    publish.append(empty_mask)
                continue
            scores = logits[rows, :lc]
            if publish is None:
                scores.masked_fill_(~self.candidate_masks[b], -torch.inf)
                continue
            lens = compress_lens[rows, None]
            # the block selection tells unreachable positions apart by -inf
            scores.masked_fill_(j[None, :lc] >= lens, -torch.inf)
            # the block selection pads and pools a copy of its rows; bound that copy
            step = max(1, _TORCH_INDEXER_SCORE_BUDGET_BYTES // (lc * 4))
            masks = [
                select_candidate_blocks(
                    scores[start : start + step],
                    lens[start : start + step],
                    topk_blocks=indexer.candidate_topk_blocks,
                    block_size=indexer.candidate_block_size,
                )
                for start in range(0, t_len, step)
            ]
            publish.append(masks[0] if len(masks) == 1 else torch.cat(masks))
        if publish is not None:
            self.candidate_masks = publish

    def _low_ratio_extend_k_slots(self, *, ratio, lc_per_req, req_pool_indices, device):
        """Per request, the c1/c2 pool slots of its visible compressed positions, and
        each request's start offset in their concatenation."""
        slot_chunks, starts, start = [], [], 0
        for r, lc in enumerate(lc_per_req):
            starts.append(start)
            if lc == 0:
                continue
            j = torch.arange(lc, device=device)
            slot_chunks.append(
                self.req_to_token[req_pool_indices[r], j * ratio].to(torch.int64)
                // ratio
            )
            start += lc
        return slot_chunks, starts

    def _low_ratio_index_topk_prefill_graph(self, layer, pos, q, w) -> None:
        """Extend indexer inside the graph: paged fp4 logits on per-token metadata
        of a static width, then the eager path's selection (torch top-k over the
        reachable columns, ascending) so the two agree bitwise. The paged top-k
        kernel breaks ties by launch order and is not replay-stable. `q` and `w`
        come from the source-projection break, computed on the live rows."""
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            quantize_fp4_indexer_tensor,
        )

        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        indexer = layer.indexer
        metadata = (
            self.forward_metadata.c1_indexer_metadata
            if ratio == 1
            else self.forward_metadata.c2_indexer_metadata
        )
        assert metadata is not None, f"no prefill graph indexer metadata for {ratio = }"
        assert indexer.n_local_heads == indexer.n_heads
        width = metadata.max_c4_seq_len
        if indexer.uses_candidates or indexer.is_candidate_source:
            # Every reachable block is a candidate inside the window, so the
            # two-level selection collapses to the plain top-k below.
            assert (
                width <= indexer.candidate_topk_blocks * indexer.candidate_block_size
            ), f"prefill graph indexer width {width} exceeds the candidate window"

        num_tokens, num_heads = q.shape[0], q.shape[1]
        q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
        q_fp4 = q_fp4.view(num_tokens, 1, num_heads, 64)
        q_sf = q_sf.view(num_tokens, 1, num_heads)
        weights = w.float()

        k_cache = pool.get_index_k_with_scale_buffer(layer.layer_id)
        assert k_cache.dim() == 2
        page_size = metadata.c4_page_size
        k_cache = k_cache.view(k_cache.shape[0], page_size, 1, 68)

        lens = metadata.c4_seq_lens
        page_table = metadata.page_table
        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        topk = min(indexer.index_topk, width)
        columns = torch.arange(width, device=lens.device)
        for rows, plan in metadata.row_chunks():
            logits = _fp4_paged_mqa_logits(
                (q_fp4[rows], q_sf[rows]),
                k_cache,
                weights[rows],
                lens[rows],
                page_table[rows],
                plan,
                width,
            )
            lens_c = lens[rows].unsqueeze(-1)
            # Columns past a row's length hold garbage.
            s = logits.masked_fill(columns[None, :] >= lens_c, -torch.inf)
            idx = s.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
            reach = idx < lens_c
            slots = page_table[rows].gather(-1, idx // page_size) * page_size + (
                idx % page_size
            )
            page_indices[rows, :topk] = torch.where(reach, slots, -1).to(torch.int32)
            if raw_indices is not None:
                raw_indices[rows, :topk] = torch.where(reach, idx, -1).to(torch.int32)

    def _low_ratio_index_topk_decode(self, layer, x, q_lora, pos) -> None:
        from sglang.kernels.ops.attention.dsv4.fp4_indexer import (
            quantize_fp4_indexer_tensor,
        )

        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        indexer = layer.indexer
        metadata = (
            self.forward_metadata.c1_indexer_metadata
            if ratio == 1
            else self.forward_metadata.c2_indexer_metadata
        )
        assert metadata is not None, f"no decode indexer metadata for {ratio = }"

        # fp4 query as (payload, scale), kernel layout [bs, 1, n_heads, dim]. The
        # kernel sums head scores locally, so the indexer heads must be replicated.
        assert indexer.n_local_heads == indexer.n_heads
        if (
            x.is_cuda
            and torch.version.cuda is not None
            and x.dtype == torch.bfloat16
            and indexer.index_head_dim == 128
        ):
            from sglang.kernels.ops.attention.dsv4.fp4_rope import (
                index_q_rope_pack_weights,
            )

            q, _ = indexer.wq_b(q_lora)
            q = q.view(q.shape[0], indexer.n_local_heads, indexer.index_head_dim)
            # One launch for the query's RoPE tail, both fp4 stages and the pack,
            # plus the head weights `head_weights(x).float()` as its epilogue
            # (bitwise: same fp32 multiply, same bf16 rounding). The kernel
            # indexes `freqs_cis` by position itself; the real/imag view of the
            # complex table is a free view, as at the c1/c2 write.
            q_fp4, q_sf, weights = index_q_rope_pack_weights(
                q,
                torch.view_as_real(layer.freqs_cis).flatten(-2),
                pos,
                indexer.head_weights_raw(x),  # [bs, n_local] bf16, n32k5120
                indexer.head_weight_scale,
            )
        else:
            q = indexer.queries(q_lora, layer.freqs_cis[pos])
            q_fp4, q_sf = quantize_fp4_indexer_tensor(q.flatten(0, 1), rne=True)
            weights = indexer.head_weights(x).float()  # [bs, n_local]
        bs = q.shape[0]
        q_fp4 = q_fp4.view(bs, 1, indexer.n_local_heads, 64)
        q_sf = q_sf.view(bs, 1, indexer.n_local_heads)

        k_cache = pool.get_index_k_with_scale_buffer(layer.layer_id)
        assert k_cache.dim() == 2
        # Index pool page (64 slots); metadata.page_table is expanded to match.
        page_size = metadata.c4_page_size
        k_cache = k_cache.view(
            k_cache.shape[0], page_size, 1, 68
        )  # fp4: 64 payload + 4 scale

        logits = _fp4_paged_mqa_logits(
            (q_fp4, q_sf),
            k_cache,
            weights,
            metadata.c4_seq_lens,
            metadata.page_table,
            metadata.deep_gemm_metadata,
            metadata.max_c4_seq_len,
        )

        logits, published = two_level_decode_logits(
            logits,
            metadata.c4_seq_lens,
            is_candidate_source=indexer.is_candidate_source,
            uses_candidates=indexer.uses_candidates,
            topk_blocks=indexer.candidate_topk_blocks,
            block_size=indexer.candidate_block_size,
            published=self.candidate_masks,
        )
        if published is not None:
            self.candidate_masks = published

        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        filter_candidates = indexer.uses_candidates and not indexer.is_candidate_source
        selected = torch.empty_like(page_indices) if filter_candidates else raw_indices
        if metadata.use_topk_v2 and raw_indices is None:
            topk_transform_paged_v2(
                logits,
                metadata.c4_seq_lens,
                None if filter_candidates else metadata.page_table,
                selected if filter_candidates else page_indices,
                page_size,
                metadata.topk_metadata,
            )
        else:
            topk_transform_paged(
                logits,
                metadata.c4_seq_lens,
                metadata.page_table,
                page_indices,
                page_size,
                selected,
            )
        if filter_candidates:
            selected = mask_topk_scores(logits, selected)
            columns = selected.clamp_min(0).to(torch.int64)
            slots = metadata.page_table.gather(1, columns // page_size) * page_size
            slots = slots + columns % page_size
            page_indices.copy_(torch.where(selected >= 0, slots, -1))
            if raw_indices is not None:
                raw_indices.copy_(selected)

    def _low_ratio_index_topk_sm90_decode(self, layer, x, q_lora, req, pos) -> None:
        """Hopper decode indexer: one token per request, every request scored at
        once against its visible compressed positions straight off the fp4 page
        table (Triton), then the same candidate / top-k contract as the DeepGEMM
        path: level-one candidate blocks where the layer publishes or consumes them,
        and -1 padded slots with the valid prefix first."""
        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        indexer = layer.indexer
        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        page_indices.fill_(-1)
        if raw_indices is not None:
            raw_indices.fill_(-1)
        bs = req.shape[0]
        assert pos.shape[0] == bs, (
            f"decode expects one token per request, {pos.shape=} {bs=}"
        )
        if bs == 0:
            return
        lens = (pos + 1) // ratio
        metadata = (
            self.forward_metadata.c1_indexer_metadata
            if ratio == 1
            else self.forward_metadata.c2_indexer_metadata
        )
        assert metadata is not None
        # V4 reserves the replay bound in metadata; visibility stays on device.
        # A capture-time length read would both synchronize and truncate replay.
        lmax = min(metadata.max_c4_seq_len, self.req_to_token.shape[1] // ratio)
        if lmax == 0:
            return
        q = indexer.queries(q_lora, layer.freqs_cis[pos])
        weights = indexer.head_weights(x)
        j = torch.arange(lmax, device=pos.device)
        valid = j[None, :] < lens[:, None]
        slots = (
            self.req_to_token[req[:, None], (j * ratio)[None, :]].to(torch.int64)
            // ratio
        )
        slots = slots.masked_fill(~valid, 0)
        table = pool.get_index_k_with_scale_buffer(layer.layer_id)
        s = fp4_index_logits_decode(
            q, weights, slots, lens, table, table.shape[1] // 68
        )
        if indexer.is_candidate_source:
            self.candidate_masks = select_candidate_blocks(
                s,
                lens[:, None],
                topk_blocks=indexer.candidate_topk_blocks,
                block_size=indexer.candidate_block_size,
            )
        elif indexer.uses_candidates:
            # Published this step by the candidate-source layer's decode pass above.
            consume = self.candidate_masks
            assert torch.is_tensor(consume) and consume.shape[0] == bs, (
                "candidate mask missing for decode"
            )
            s = s.masked_fill(~consume[:, :lmax], -torch.inf)
        k = min(indexer.index_topk, lmax)
        idx = s.topk(k, dim=-1, sorted=False).indices
        if indexer.uses_candidates and not indexer.is_candidate_source:
            idx = mask_topk_scores(s, idx)
            idx = idx.masked_fill(idx < 0, lmax)
        idx = idx.sort(dim=-1).values
        reach = idx < lens[:, None]
        page_indices[:bs, :k] = torch.where(
            reach, slots.gather(1, idx.clamp_max(lmax - 1)), -1
        ).to(torch.int32)
        if raw_indices is not None:
            raw_indices[:bs, :k] = torch.where(reach, idx, -1).to(torch.int32)

    def _low_ratio_index_topk_torch(self, layer, x, q_lora, req, pos) -> None:
        pool = self.token_to_kv_pool
        core = self.forward_metadata.core_metadata
        ratio = layer.compress_ratio
        indexer = layer.indexer
        # Attention scans sparse_topk_lengths slots and skips -1 entries.
        page_indices = core.sparse_page_indices(ratio)
        raw_indices = core.sparse_raw_indices(ratio)
        page_indices.fill_(-1)
        if raw_indices is not None:
            raw_indices.fill_(-1)
        q = indexer.queries(q_lora, layer.freqs_cis[pos])
        weights = indexer.head_weights(x)
        # A compressed position is visible once the query has passed its last token.
        compress_lens = (pos + 1) // ratio
        topk = indexer.index_topk
        publish = [] if indexer.is_candidate_source else None
        consume = self.candidate_masks if indexer.uses_candidates else None
        for b, r in enumerate(torch.unique_consecutive(req).tolist()):
            tok = (req == r).nonzero().squeeze(1)
            lens = compress_lens[tok]
            lc = int(lens.max().item())
            if lc == 0:
                # Consumers address masks by request position, including empty requests.
                if publish is not None:
                    publish.append(
                        torch.zeros(0, 0, dtype=torch.bool, device=pos.device)
                    )
                continue
            j = torch.arange(lc, device=pos.device)
            slots_j = self.req_to_token[r, j * ratio].to(torch.int64) // ratio
            # Dequantize only this request's visible K rows; the full table is
            # pool-sized.
            index_k = pool.get_low_ratio_index_k_dequant(layer.layer_id, slots_j)
            k = min(topk, lc)
            # Every step below is per query row; chunk rows so the [rows, heads, lc]
            # bf16 scores stay under the budget (16 GiB at once for a 16k-token prompt).
            rows_per_chunk = max(
                1,
                _TORCH_INDEXER_SCORE_BUDGET_BYTES // (q.shape[1] * lc * 2),
            )
            masks = [] if publish is not None else None
            for start in range(0, tok.numel(), rows_per_chunk):
                rows = slice(start, start + rows_per_chunk)
                tok_c, lens_c = tok[rows], lens[rows]
                s = indexer.scores(q[tok_c], index_k, weights[tok_c])
                s = s.masked_fill(j[None, :] >= lens_c[:, None], -torch.inf)
                if masks is not None:
                    masks.append(
                        select_candidate_blocks(
                            s,
                            lens_c[:, None],
                            topk_blocks=indexer.candidate_topk_blocks,
                            block_size=indexer.candidate_block_size,
                        )
                    )
                elif consume is not None:
                    s = s.masked_fill(~consume[b][rows], -torch.inf)
                idx = s.topk(k, dim=-1, sorted=False).indices
                if consume is not None and masks is None:
                    idx = mask_topk_scores(s, idx)
                    idx = idx.masked_fill(idx < 0, lc)
                idx = idx.sort(dim=-1).values
                reach = idx < lens_c[:, None]
                page_indices[tok_c, :k] = torch.where(
                    reach, slots_j[idx.clamp_max(lc - 1)], -1
                ).to(torch.int32)
                if raw_indices is not None:
                    raw_indices[tok_c, :k] = torch.where(reach, idx, -1).to(torch.int32)
            if masks is not None:
                publish.append(torch.cat(masks) if len(masks) > 1 else masks[0])
        if publish is not None:
            self.candidate_masks = publish
