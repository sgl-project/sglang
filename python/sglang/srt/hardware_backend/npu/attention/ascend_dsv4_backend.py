from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.layers.attention.dsv4.compressor import CompressorBackendMixin
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc, ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner



def _walsh_hadamard_matrix(n: int, dtype: torch.dtype, device) -> torch.Tensor:
    # n**-0.5 norm is baked in via the sqrt(2) division per doubling; _apply_hadamard is a plain matmul
    cache = _walsh_hadamard_matrix._cache
    key = (n, str(device))
    cached = cache.get(key)
    if cached is not None:
        return cached
    if not ((n & (n - 1) == 0) and (n > 0)):
        raise ValueError(f"n must be a positive power of 2, got {n}")
    had = torch.ones(1, 1, dtype=torch.bfloat16, device=device)
    while had.shape[0] != n:
        had = torch.cat((torch.cat([had, had], 1), torch.cat([had, -had], 1)), 0)
        had /= math.sqrt(2)
    had = had.contiguous()
    cache[key] = had
    return had


_walsh_hadamard_matrix._cache = {}


def _apply_hadamard(inp: torch.Tensor, hadamard_matrix: torch.Tensor) -> torch.Tensor:
    init_shape = inp.shape
    flat = inp.view(-1, hadamard_matrix.shape[0])
    return flat.matmul(hadamard_matrix).view(init_shape).to(torch.bfloat16)


def _overlap_transform(
    tensor: torch.Tensor, value: float, head_dim: int
) -> torch.Tensor:
    # Build (n_chunks, 2*ratio, d) from (n_chunks, ratio, coff*d): first ratio rows
    # = current chunk left half (:d), last ratio rows = previous chunk right half (d:);
    # first chunk's right half filled with `value`.
    n_chunks, r, _ = tensor.shape
    d = head_dim
    out = tensor.new_full((n_chunks, 2 * r, d), value)
    out[:, r:] = tensor[..., d:]
    out[1:, :r] = tensor[:-1, :, :d]
    return out


class CompressorAscendBackendMixin(CompressorBackendMixin):

    def _build_npu_compress_metadata(self, forward_batch: ForwardBatch) -> None:
        fm = self.forward_metadata
        is_decode = forward_batch.forward_mode.is_decode()
        is_verify = forward_batch.forward_mode.is_target_verify()
        _verify_compress = is_verify and bool(self._dsv4_compress_ratios)
        _seq_lens = forward_batch.seq_lens.to(torch.int32)
        if _verify_compress:
            _seq_lens = _seq_lens + self.speculative_num_draft_tokens
        result = self._compute_compress_locs(
            pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=_seq_lens,
            out_cache_loc=forward_batch.out_cache_loc,
            is_decode=is_decode,
            bs=forward_batch.batch_size,
            device=forward_batch.seq_lens.device,
            req_to_token_pool=self.req_to_token_pool,
            out_cache_loc_dsv4=forward_batch.out_cache_loc_dsv4,
        )
        for k, v in result.items():
            setattr(fm, k, v)
        if not is_decode:
            for ratio in self._dsv4_compress_ratios:
                if ratio in (4, 128):
                    if f"c{ratio}_state_loc" not in result:
                        setattr(fm, f"c{ratio}_state_loc", None)
                    if f"c{ratio}_loc" not in result:
                        setattr(fm, f"c{ratio}_loc", None)

        if _verify_compress:
            self._build_npu_compress_metadata_verify(forward_batch)

    def _build_npu_compress_metadata_prefill(self, forward_batch: ForwardBatch) -> None:
        # eager-only: prefill is never graph-captured, host reads (cu_cpu) are safe here
        fm = self.forward_metadata
        device = forward_batch.seq_lens.device
        positions = forward_batch.positions
        t = positions.shape[0]
        bs = forward_batch.batch_size
        cu = fm.actual_seq_lengths_q_pa

        cu_cpu = cu.cpu().tolist()
        ratio_lists: dict = {
            r: [] for r in self._dsv4_unique_compress_ratios if r in (4, 128)
        }
        for idx in range(bs):
            start = int(cu_cpu[idx])
            end = int(cu_cpu[idx + 1])
            if end == start:
                continue
            seq = end - start
            req_positions = positions[start:end]
            for ratio in ratio_lists:
                cutoff = seq - (seq % ratio)
                if cutoff > 0:
                    ratio_lists[ratio].append(req_positions[:cutoff:ratio])

        for ratio in (4, 128):
            if ratio not in ratio_lists:
                continue
            padding_size = min(t, t // ratio + bs)
            padding = torch.zeros(padding_size, dtype=torch.int64, device=device)
            if ratio_lists[ratio]:
                cat = torch.cat(ratio_lists[ratio], dim=0).to(torch.int64)
                assert cat.numel() <= padding.numel(), (
                    f"positions_cmp_padding_c{ratio} overflow: "
                    f"{cat.numel()} > {padding.numel()}"
                )
                padding[: cat.shape[0]].copy_(cat)
            setattr(fm, f"positions_cmp_padding_c{ratio}", padding)

        # start_pos=0: chunked prefill unsupported; seqused=None -> op derives lens from cu_seqlens
        fm.start_pos = torch.zeros(bs, dtype=torch.int32, device=device)
        fm.seqused = None

        # bundle out_c*_loc is densely packed in batch order (matches cmp_kv); invalid under chunked prefill
        bundle = forward_batch.out_cache_loc_dsv4
        for ratio in (4, 128):
            if ratio not in ratio_lists:
                continue
            bundle_loc = None
            if bundle is not None:
                bundle_loc = bundle.out_c4_loc if ratio == 4 else bundle.out_c128_loc
            setattr(
                fm,
                f"c{ratio}_loc",
                bundle_loc.to(torch.int32) if bundle_loc is not None else None,
            )

        # req_to_token_c*_state is not re-zeroed on slot reuse; zero pre-tail page cols so the kernel block-0 skip masks stale entries
        page_size = self.page_size
        for ratio in (4, 128):
            spt = getattr(fm, f"c{ratio}_state_page_table", None)
            if spt is None:
                continue
            for idx in range(bs):
                seqlen = int(cu_cpu[idx + 1] - cu_cpu[idx])
                if seqlen == 0:
                    continue
                tail = seqlen % 128
                if ratio == 4:
                    c_alloc_len = tail + 128 if (tail <= 3 and seqlen >= 128) else tail
                else:
                    c_alloc_len = tail
                c_alloc_offset = seqlen - c_alloc_len
                first_tail_page = c_alloc_offset // page_size
                if first_tail_page > 0:
                    spt[idx, :first_tail_page] = 0

    def _compute_compress_locs(
        self,
        *,
        pool,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        out_cache_loc: torch.Tensor,
        is_decode: bool,
        bs: int,
        device: torch.device,
        req_to_token_pool,
        out_cache_loc_dsv4,
        is_graph: bool = False,
        seq_lens_max_override: Optional[int] = None,
    ) -> dict:
        result: dict = {}
        req_pool = req_pool_indices

        if seq_lens_max_override is not None:
            seq_lens_max = int(seq_lens_max_override)
        else:
            seq_lens_max = int(seq_lens.max().item()) if bs > 0 else 0
        n_pages = max(1, (seq_lens_max + self.page_size - 1) // self.page_size)

        for ratio in self._dsv4_unique_compress_ratios:
            if ratio not in (4, 128):
                continue
            state_table = (
                req_to_token_pool.req_to_token_c4_state
                if ratio == 4
                else req_to_token_pool.req_to_token_c128_state
            )
            state_slots_2d = state_table[
                req_pool.to(torch.int64), : n_pages * self.page_size
            ]
            state_page_2d = (state_slots_2d[:, :: self.page_size] // self.page_size).to(
                torch.int32
            )

            if is_decode:
                state_loc_decode = None
                if out_cache_loc_dsv4 is not None:
                    state_loc_decode = (
                        out_cache_loc_dsv4.out_c4_state_loc
                        if ratio == 4
                        else out_cache_loc_dsv4.out_c128_state_loc
                    )
                if state_loc_decode is None:
                    state_loc_decode = torch.zeros(
                        bs,
                        dtype=torch.int32,
                        device=device,
                    )
                else:
                    state_loc_decode = state_loc_decode.to(torch.int32)
                compress_out_loc = torch.zeros(
                    bs,
                    dtype=torch.int32,
                    device=device,
                )
                # bundle_loc and cmp_kv are both densely packed in batch order, so
                # write them densely; indexing by batch slot would misalign them.
                if out_cache_loc_dsv4 is not None:
                    bundle_loc = (
                        out_cache_loc_dsv4.out_c4_loc
                        if ratio == 4
                        else out_cache_loc_dsv4.out_c128_loc
                    )
                    n_compress = bundle_loc.numel()
                    if n_compress > 0:
                        compress_out_loc[:n_compress] = bundle_loc.to(torch.int32)

            result[f"c{ratio}_state_page_table"] = state_page_2d
            if is_decode:
                result[f"c{ratio}_state_loc"] = state_loc_decode
                result[f"c{ratio}_loc"] = compress_out_loc

            c_table = (
                req_to_token_pool.req_to_token_c4
                if ratio == 4
                else req_to_token_pool.req_to_token_c128
            )
            if is_graph:
                n_c_tokens = seq_lens_max // ratio
            else:
                n_c_tokens = max(1, seq_lens_max // ratio)
            slots = c_table[req_pool.to(torch.int64), :n_c_tokens]
            c_page_table = (slots[:, :: self.page_size] // self.page_size).to(
                torch.int32
            )
            result[f"c{ratio}_page_table"] = c_page_table

        if is_decode:
            valid = seq_lens > 0
            positions_last = torch.clamp(seq_lens - 1, min=0)
            for ratio in self._dsv4_unique_compress_ratios:
                if ratio not in (4, 128):
                    continue
                padding_size = min(bs, bs // ratio + bs)
                padding = torch.zeros(padding_size, dtype=torch.int64, device=device)
                should_compress = ((seq_lens % ratio) == 0) & valid
                pos_cmp = positions_last[should_compress].to(torch.int64) + (1 - ratio)
                if pos_cmp.numel() > 0:
                    padding[: pos_cmp.shape[0]].copy_(pos_cmp)
                result[f"positions_cmp_padding_c{ratio}"] = padding

            result["start_pos"] = positions_last.to(torch.int32)
            result["seqused"] = valid.to(torch.int32)

        return result

    def forward_core_compressor(
        self,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
        layer_id: int,
        compressor,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        compressor(x, forward_batch)

    def forward_compress(
        self,
        compressor,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        if (
            forward_batch.forward_mode.is_prefill()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            return self._forward_compress_native(compressor, x, forward_batch)

        from sglang.srt.layers.deepseek_v4_rope import (
            get_fused_compressor_rope_cos_sin,
        )

        ratio = compressor.ratio
        coff = 1 + int(compressor.overlap)
        device = x.device
        self._ensure_compressor_hadamard(compressor, device)
        self._ensure_fused_caches(compressor)

        fm = self.forward_metadata
        positions_cmp = getattr(fm, f"positions_cmp_padding_c{ratio}", None)
        page_table = getattr(fm, f"c{ratio}_state_page_table", None)
        start_pos = getattr(fm, "start_pos", None)
        seqused = getattr(fm, "seqused", None)
        cu_seqlens = getattr(fm, "actual_seq_lengths_q_cmp", None)
        if cu_seqlens is None:
            cu_seqlens = getattr(fm, "actual_seq_lengths_q_pa", None)
        assert positions_cmp is not None and page_table is not None, (
            "fused compressor needs backend metadata "
            "(positions_cmp_padding / c*_state_page_table) — make sure "
            "_build_npu_compress_metadata ran before this forward."
        )
        assert start_pos is not None, "fused compressor needs start_pos"
        assert cu_seqlens is not None, "fused compressor needs cu_seqlens"

        pool = self.token_to_kv_pool
        state_cache = pool.get_state_cache(
            compressor.layer_id, compressor.is_in_indexer
        )

        cos, sin = get_fused_compressor_rope_cos_sin(
            compressor.freqs_cis, positions_cmp, dtype=torch.float32
        )

        cmp_kv = torch.ops.custom.compressor(
            x,
            compressor._fused_wkv_w,
            compressor._fused_wgate_w,
            state_cache,
            compressor.ape,
            compressor._fused_norm_weight_fp32,
            rope_sin=sin,
            rope_cos=cos,
            rope_head_dim=compressor.rope_head_dim,
            cmp_ratio=ratio,
            state_block_table=page_table,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            start_pos=start_pos,
            coff=coff,
            norm_eps=compressor.norm.variance_epsilon,
            rotary_mode=2,
            cache_mode=1,
        )

        loc = getattr(fm, f"c{ratio}_loc", None)
        is_prefill = (
            forward_batch.forward_mode.is_prefill()
            and not forward_batch.forward_mode.is_target_verify()
        )
        if loc is not None:
            if is_prefill and loc.numel() < cmp_kv.shape[0]:
                cmp_kv = cmp_kv[: loc.numel()]
            elif loc.numel() != cmp_kv.shape[0]:
                raise RuntimeError(
                    "DSV4 NPU fused compressor loc/kv length mismatch before "
                    f"epilog: mode={forward_batch.forward_mode}, ratio={ratio}, "
                    f"loc={loc.numel()}, kv={cmp_kv.shape[0]}"
                )

        if self.graph_mode or cmp_kv.shape[0] > 0:
            if compressor.rotate:
                cmp_kv = _apply_hadamard(cmp_kv, compressor.hadamard_matrix)
            self._compressor_epilog_npu(compressor, cmp_kv, forward_batch)

    def _forward_compress_native(
        self,
        compressor,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        """Per-request unfused compress path.

        * Prefill: split seq into ``cutoff = seqlen - seqlen % ratio`` to compress
          + ``remainder`` stashed as state (overlap/ratio=4 also stashes the last
          ``ratio`` of the cutoff). State writes via ``set_state_buffer``; cutoff gets
          ape-weighted softmax over ratio, sum, norm+rope+(opt) hadamard, then write.
        * Non-prefill (one token/req): append (kv, score) to the state ring; if it
          completes a ratio-aligned chunk, gather the chunk (overlap: 2*ratio, else
          ratio), ape-weighted softmax + sum, and write via ``set_compress_buffer``.
        """
        import torch_npu  # local: NPU-only, used for npu_rotary_mul below

        positions = forward_batch.positions
        ratio, overlap, d = compressor.ratio, compressor.overlap, compressor.head_dim
        device = x.device
        self._ensure_compressor_hadamard(compressor, device)
        dtype = x.dtype
        x_f32 = x.float()
        # wkv + wgate are fused into one wkv_gate.weight [2*coff*head_dim, hidden_size]
        # (kv concatenated before wgate); split along the output dim to recover each.
        coff = 1 + int(overlap)
        W = compressor.wkv_gate.weight.float()
        kv_full = F.linear(x_f32, W[: coff * d])  # [T, coff*d]
        score_full = F.linear(x_f32, W[coff * d :])  # [T, coff*d]

        seq_lens_cpu = forward_batch.seq_lens_cpu
        is_prefill = forward_batch.forward_mode.is_prefill()
        token_to_kv_pool = self.token_to_kv_pool
        backend_fm = self.forward_metadata
        if ratio == 4:
            page_table = backend_fm.c4_state_page_table
        else:
            page_table = backend_fm.c128_state_page_table

        kv_out_list: list[torch.Tensor] = []
        kv_state_to_be_cached: list[torch.Tensor] = []
        score_state_to_be_cached: list[torch.Tensor] = []
        state_loc_list: list[torch.Tensor] = []
        kv_out_positions: list[torch.Tensor] = []
        # Per-token write loc: record (req_idx_in_batch, compressed_seq_pos_in_req)
        # to derive the c{N}_kv_pool slot from the slab allocator, not out_cache_loc
        # // ratio (correct only when raw kv allocation aligns to ratio).
        write_req_indices: list[torch.Tensor] = []
        write_pos_in_req: list[torch.Tensor] = []
        seqlen_offset = 0
        # Running offset into the tail-only state bundle, flat layout
        # ``[req0_alloc_len_slots, ...]`` where ``alloc_len_i = seqlen_i -
        # c{ratio}_state_alloc_offset_i`` (NOT raw seqlen; see
        # ScheduleBatch._compute_dsv4_state_lens_extend).
        state_bundle_offset = 0

        for idx, seqlen in enumerate(seq_lens_cpu):
            seqlen = int(seqlen)
            if seqlen == 0:
                continue
            if is_prefill:
                pos_req = positions[seqlen_offset : seqlen_offset + seqlen]

                # Per-req tail-only state alloc range; same formula as
                # ScheduleBatch._compute_dsv4_state_lens_extend (recomputed to
                # avoid threading another tensor through forward_batch).
                tail_128 = seqlen % 128
                if ratio == 4:
                    c_alloc_len = (
                        tail_128 + 128
                        if (tail_128 <= 3 and seqlen >= 128)
                        else tail_128
                    )
                else:  # ratio == 128
                    c_alloc_len = tail_128
                c_alloc_offset = seqlen - c_alloc_len

                # Bundle slice for this req. The NPU paged state pool emits real
                # slot ids (no ring-hash); slice by ``state_bundle_offset`` (cumulative
                # alloc_len), NOT ``seqlen_offset`` (cumulative raw seqlen).
                bundle = forward_batch.out_cache_loc_dsv4
                assert bundle is not None, (
                    "unfused compress prefill on NPU needs the DSV4 "
                    "alloc bundle; expected maybe_write_dsv4_extend to have "
                    "populated batch.out_cache_loc_dsv4 before forward."
                )
                bundle_state_loc = (
                    bundle.out_c4_state_loc if ratio == 4 else bundle.out_c128_state_loc
                )
                if c_alloc_len > 0:
                    # Require a populated bundle only when this req allocates
                    # slots. A 128-aligned ratio==128 prefill has c_alloc_len==0
                    # (no partial tail), so an all-128-aligned batch legitimately
                    # yields an empty bundle. Empty while c_alloc_len > 0 means
                    # c{ratio}_state_attn_allocator was never initialized.
                    assert (
                        bundle_state_loc is not None and bundle_state_loc.numel() > 0
                    ), (
                        f"unfused compress prefill: bundle.out_c{ratio}_state_loc "
                        f"is empty/None — DSV4NPUTokenToKVPoolAllocator's "
                        f"c{ratio}_state_attn_allocator was not initialized (check "
                        f"pool_configurator's NPU branch + npu_state_pool_size)."
                    )
                    out_cache_loc = bundle_state_loc[
                        state_bundle_offset : state_bundle_offset + c_alloc_len
                    ]
                    state_bundle_offset += c_alloc_len
                else:
                    # No tail to cache: empty slot view, never indexed below.
                    # Only reached for c128 (c4's c_alloc_len is always > 0).
                    out_cache_loc = torch.empty((0,), dtype=torch.int64, device=device)
                remainder = seqlen % ratio
                cutoff = seqlen - remainder
                # ``cutoff`` is raw coords; subtract ``c_alloc_offset`` for
                # slice-relative indexing into the per-req bundle slice.
                cutoff_in_slice = cutoff - c_alloc_offset
                should_compress = cutoff >= ratio
                # ratio-strided positions for the cutoff chunks (one rope pos per token).
                pos_compressed = pos_req[:cutoff:ratio]
                kv = kv_full[seqlen_offset : seqlen_offset + seqlen]
                score = score_full[seqlen_offset : seqlen_offset + seqlen]

                if overlap and cutoff >= ratio:
                    # Stash the trailing ratio tokens of the cutoff so the next
                    # decode step can do overlap compression across the boundary
                    # (for ratio=4 this window is inside the state alloc range).
                    kv_state_to_be_cached.append(kv[cutoff - ratio : cutoff])
                    score_state_to_be_cached.append(
                        score[cutoff - ratio : cutoff] + compressor.ape
                    )
                    state_loc_list.append(
                        out_cache_loc[cutoff_in_slice - ratio : cutoff_in_slice]
                    )
                if remainder > 0:
                    kv_cut, kv_rem = kv.split([cutoff, remainder], dim=0)
                    score_cut, score_rem = score.split([cutoff, remainder], dim=0)
                    kv_state_to_be_cached.append(kv_rem)
                    score_state_to_be_cached.append(
                        score_rem + compressor.ape[:remainder]
                    )
                    state_loc_list.append(out_cache_loc[-remainder:])
                    kv = kv_cut
                    score = score_cut

                if should_compress:
                    kv = kv.unflatten(0, (-1, ratio))  # [n_chunks, ratio, coff*d]
                    score = score.unflatten(0, (-1, ratio)) + compressor.ape
                    if overlap:
                        kv = _overlap_transform(kv, value=0.0, head_dim=d)
                        score = _overlap_transform(
                            score, value=float("-inf"), head_dim=d
                        )
                    kv_compressed = (kv * score.softmax(dim=1)).sum(
                        dim=1
                    )  # [n_chunks, d]
                    n_compressed_this_req = kv_compressed.shape[0]
                    kv_out_list.append(kv_compressed)
                    kv_out_positions.append(pos_compressed)
                    write_req_indices.append(
                        torch.full(
                            (n_compressed_this_req,),
                            idx,
                            dtype=torch.int64,
                            device=device,
                        )
                    )
                    write_pos_in_req.append(
                        torch.arange(
                            n_compressed_this_req,
                            dtype=torch.int64,
                            device=device,
                        )
                    )
                seqlen_offset += seqlen
            else:
                # Decode: append (kv, score+ape[pos%r]) to the state ring at
                # c{4,128}_state_loc[idx]; if this completes a ratio-aligned
                # chunk, gather it and produce one compressed kv via ape-softmax-sum.
                start_pos = seqlen - 1
                should_compress = (start_pos + 1) % ratio == 0
                pos_req = positions[idx : idx + 1] + (1 - ratio)
                kv = kv_full[idx : idx + 1]
                score = score_full[idx : idx + 1] + compressor.ape[start_pos % ratio]
                if ratio == 4:
                    state_loc_decode = backend_fm.c4_state_loc
                else:
                    state_loc_decode = backend_fm.c128_state_loc
                token_to_kv_pool.set_state_buffer(
                    compressor.layer_id,
                    state_loc_decode[idx : idx + 1],
                    kv.view(1, 1, -1),
                    score.view(1, 1, -1),
                    compressor.is_in_indexer,
                )
                if should_compress:
                    if overlap:
                        kv_indices = _get_kv_indices(
                            forward_batch, 2 * ratio, page_table, idx, seqlen
                        )
                        kv_state, score_state = token_to_kv_pool.get_state_buffer(
                            compressor.layer_id, compressor.is_in_indexer, kv_indices
                        )
                        # kv_state / score_state: [2*r, 1, coff*d] → [2*r, d]
                        kv_state = kv_state.squeeze(1)
                        score_state = score_state.squeeze(1)
                        kv_state = torch.cat(
                            [kv_state[:ratio, :d], kv_state[ratio:, d:]], dim=0
                        )
                        score_state = torch.cat(
                            [score_state[:ratio, :d], score_state[ratio:, d:]],
                            dim=0,
                        )
                        kv_compressed = (kv_state * score_state.softmax(dim=0)).sum(
                            dim=0, keepdim=True
                        )
                    else:
                        kv_indices = _get_kv_indices(
                            forward_batch, ratio, page_table, idx, seqlen
                        )
                        kv_state, score_state = token_to_kv_pool.get_state_buffer(
                            compressor.layer_id, compressor.is_in_indexer, kv_indices
                        )
                        kv_compressed = (
                            kv_state[:, 0] * score_state[:, 0].softmax(dim=0)
                        ).sum(dim=0, keepdim=True)
                    kv_out_list.append(kv_compressed)
                    kv_out_positions.append(pos_req)
                    # Decode: 1 compressed token at compressed_seq_pos = seqlen//ratio - 1
                    decode_pos = seqlen // ratio - 1
                    write_req_indices.append(
                        torch.tensor([idx], dtype=torch.int64, device=device)
                    )
                    write_pos_in_req.append(
                        torch.tensor([decode_pos], dtype=torch.int64, device=device)
                    )

        # Flush the prefill state stash to the pool in one shot.
        if kv_state_to_be_cached:
            kv_state_cat = torch.cat(kv_state_to_be_cached, dim=0).unsqueeze(1)
            score_state_cat = torch.cat(score_state_to_be_cached, dim=0).unsqueeze(1)
            state_loc_cat = torch.cat(state_loc_list, dim=0)
            token_to_kv_pool.set_state_buffer(
                compressor.layer_id,
                state_loc_cat,
                kv_state_cat,
                score_state_cat,
                compressor.is_in_indexer,
            )

        # Norm + rope + optional hadamard on the freshly compressed tokens,
        # then write via _compressor_epilog_npu with explicit slab-derived locs.
        if kv_out_list:
            kv_out = torch.cat(kv_out_list, dim=0).to(dtype)
            pos_out = torch.cat(kv_out_positions, dim=0)
            kv_out = compressor.norm(kv_out)
            # npu_rotary_mul wants cos/sin in repeat_interleave(2) layout, reshaped
            # to (T, 1, 1, rope_dim); cos=real, sin=imag of the complex freqs_cis.
            rope_dim = compressor.rope_head_dim
            # Use the same contig cache as the outer rope path; .real/.imag on a
            # complex tensor are strided views and aclnnIndex over them triggers
            # StridedSlice (see _get_contig_freqs_real_imag in deepseek_v4_rope.py).
            from sglang.srt.layers.deepseek_v4_rope import (
                _get_contig_freqs_real_imag,
            )

            freqs_real, freqs_imag = _get_contig_freqs_real_imag(compressor.freqs_cis)
            cos_half = freqs_real[pos_out].to(kv_out.dtype)
            sin_half = freqs_imag[pos_out].to(kv_out.dtype)
            cos = (
                cos_half.repeat_interleave(2, dim=-1)
                .view(-1, 1, 1, rope_dim)
                .contiguous()
            )
            sin = (
                sin_half.repeat_interleave(2, dim=-1)
                .view(-1, 1, 1, rope_dim)
                .contiguous()
            )
            rope_slice = kv_out[..., -rope_dim:]
            rope_view = rope_slice.unsqueeze(-2).unsqueeze(1)  # (T, 1, 1, rope_dim)
            rope_rot = torch_npu.npu_rotary_mul(
                rope_view, cos, sin, rotary_mode="interleave"
            )
            rope_slice.copy_(rope_rot.view_as(rope_slice))
            if compressor.rotate:
                kv_out = _apply_hadamard(kv_out, compressor.hadamard_matrix)
            # c{N}_kv_pool slot per compressed token. DSV4NPUReqToTokenPool's
            # token-level slot id table is indexed directly by compressed-seq
            # position (elements already are c-pool slot ids; no page indirection).
            req_indices_flat = torch.cat(write_req_indices, dim=0)
            pos_in_req_flat = torch.cat(write_pos_in_req, dim=0)
            req_pool_flat = forward_batch.req_pool_indices[req_indices_flat]
            c_table = (
                self.req_to_token_pool.req_to_token_c4
                if ratio == 4
                else self.req_to_token_pool.req_to_token_c128
            )
            write_locs = c_table[
                req_pool_flat.to(torch.int64), pos_in_req_flat.to(torch.int64)
            ].to(torch.int32)
            self._compressor_epilog_npu(
                compressor, kv_out, forward_batch, override_loc=write_locs
            )
        return None

    def _ensure_compressor_hadamard(self, compressor, device: torch.device) -> None:
        if getattr(compressor, "hadamard_matrix", None) is None:
            H = _walsh_hadamard_matrix(compressor.head_dim, torch.float32, device)
            compressor.register_buffer("hadamard_matrix", H, persistent=False)

    def _ensure_fused_caches(self, compressor) -> None:
        if getattr(compressor, "_fused_wkv_w", None) is not None:
            return
        coff = 1 + int(compressor.overlap)
        split = coff * compressor.head_dim
        w = compressor.wkv_gate.weight
        assert (
            w.shape[0] == 2 * split
        ), f"wkv_gate.weight rows={w.shape[0]} != 2*coff*head_dim={2*split}"
        compressor._fused_wkv_w = w[:split]
        compressor._fused_wgate_w = w[split:]
        compressor._fused_norm_weight_fp32 = compressor.norm.weight.to(torch.float32)

    def _compressor_epilog_npu(
        self,
        compressor,
        kv: torch.Tensor,
        forward_batch: ForwardBatch,
        override_loc: Optional[torch.Tensor] = None,
    ) -> None:
        kv_scale: Optional[torch.Tensor] = None
        li_kv_dtype = getattr(compressor, "li_kv_dtype", "bf16")
        if li_kv_dtype == "int8" and compressor.is_in_indexer:
            import torch_npu

            kv, kv_scale = torch_npu.npu_dynamic_quant(kv)
            kv_scale = kv_scale.to(torch.float16)

        if override_loc is not None:
            loc = override_loc
        else:
            backend_fm = self.forward_metadata
            loc = backend_fm.c4_loc if compressor.ratio == 4 else backend_fm.c128_loc
        if loc is not None:
            if loc.numel() != kv.shape[0]:
                raise RuntimeError(
                    "DSV4 NPU fused compressor epilog loc/kv length mismatch: "
                    f"mode={forward_batch.forward_mode}, "
                    f"ratio={compressor.ratio}, loc={loc.numel()}, kv={kv.shape[0]}"
                )
            if forward_batch.forward_mode.is_target_verify():
                valid = loc != 0
                if self.graph_mode:
                    kv_mask = valid.to(kv.dtype).view(
                        valid.shape[0], *([1] * (kv.dim() - 1))
                    )
                    kv = kv * kv_mask
                    if kv_scale is not None:
                        scale_mask = valid.to(kv_scale.dtype).view(
                            valid.shape[0], *([1] * (kv_scale.dim() - 1))
                        )
                        kv_scale = kv_scale * scale_mask
                else:
                    loc = loc[valid]
                    kv = kv[valid]
                    if kv_scale is not None:
                        kv_scale = kv_scale[valid]
        self.token_to_kv_pool.set_compress_buffer(
            compressor.layer_id,
            loc,
            kv,
            kv_scale,
            compressor.is_in_indexer,
        )


class C4IndexerAscendBackendMixin(C4IndexerBackendMixin):

    def init_forward_metadata_indexer(self, core_attn_metadata):
        # li_quant_metadata is built in _compute_kernel_metadata; None satisfies the mixin contract
        return None

    def forward_c4_indexer_npu(
        self,
        c4_indexer,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: ForwardBatch,
        skip_compressor: bool = False,
    ) -> torch.Tensor:
        assert (
            not skip_compressor
        ), "skip_compressor=True is not supported by forward_c4_indexer_npu"
        from sglang.srt.layers.dp_attention import get_attention_tp_group

        ratio = c4_indexer.compressor.ratio
        device = x.device
        self._ensure_npu_c4_indexer(c4_indexer, device)
        bs = x.shape[0]
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
        )

        q = self._compute_q_npu(c4_indexer, q_lora, forward_batch.positions)

        weights, _ = c4_indexer.weights_proj(x)
        weights = weights * (c4_indexer.softmax_scale * c4_indexer.n_heads**-0.5)

        if not skip_compressor:
            c4_indexer.compressor(x, forward_batch)

        li_kv_dtype = getattr(c4_indexer.compressor, "li_kv_dtype", "bf16")
        if li_kv_dtype == "int8":
            # Empty/idle rank (T=0) must skip the indexer kernel; test is_idle
            # rather than .item() since a host sync is illegal during capture.
            if bs == 0 or forward_batch.forward_mode.is_idle():
                return torch.full(
                    (bs, self._dsv4_index_topk),
                    -1,
                    dtype=torch.int32,
                    device=device,
                )
            li_cmp_kv = self.token_to_kv_pool.get_compress_buffer(
                c4_indexer.layer_id, True
            )
            li_kv_scale = self.token_to_kv_pool.get_compress_dequant_scale_buffer(
                c4_indexer.layer_id, True
            )
            return self._forward_npu_fused(
                c4_indexer, q, li_cmp_kv, li_kv_scale, weights, forward_batch
            )

        seqlens_cpu = forward_batch.seq_lens_cpu
        end_pos = forward_batch.seq_lens.cumsum(dim=0)
        page_table = self.forward_metadata.c4_page_table
        attn_tp_size = get_attention_tp_size()
        topk_idxs: list[torch.Tensor] = []
        for i, _end_token in enumerate(end_pos):
            seq_i = int(seqlens_cpu[i])
            kv_indices = _get_kv_indices(
                forward_batch, seq_i // ratio, page_table, i, seq_i // ratio
            )
            kv_cache_value = self.token_to_kv_pool.get_compress_buffer(
                c4_indexer.layer_id, True, kv_indices
            )
            if is_prefill:
                start = 0 if i == 0 else int(end_pos[i - 1])
                end = int(end_pos[i])
                index_score = torch.einsum(
                    "shd,td->sht",
                    q[start:end, ...],
                    kv_cache_value.squeeze(1),
                )
                index_score = (
                    index_score.relu_() * weights.unsqueeze(-1)[start:end, ...]
                ).sum(dim=1)
                if attn_tp_size > 1 and getattr(c4_indexer, "enable_indexer_tp", False):
                    get_attention_tp_group().all_reduce(index_score)
                arange_kv = torch.arange(seq_i // ratio, device=device)
                arange_q = torch.arange(1, seq_i + 1, device=device).unsqueeze(1)
                causal = arange_kv.repeat(seq_i, 1) >= (arange_q // ratio)
                index_score += torch.where(
                    causal, float("-inf"), torch.zeros((), device=device)
                )
                topk_idx = index_score.topk(
                    min(self._dsv4_index_topk, seq_i // ratio), dim=-1
                )[1]
                drop = topk_idx >= (
                    torch.arange(1, seq_i + 1, device=device).unsqueeze(1) // ratio
                )
                topk_idx = torch.where(drop, -1, topk_idx)
            else:
                index_score = torch.einsum(
                    "shd,td->sht",
                    q[i : i + 1, ...],
                    kv_cache_value.squeeze(1),
                )
                index_score = (index_score.relu_() * weights.unsqueeze(-1)[i]).sum(
                    dim=1
                )
                topk_idx = index_score.topk(
                    min(self._dsv4_index_topk, seq_i // ratio), dim=-1
                )[1]
            topk_idx = F.pad(
                topk_idx,
                (0, self._dsv4_index_topk - topk_idx.shape[-1]),
                mode="constant",
                value=-1,
            )
            topk_idxs.append(topk_idx)
        return torch.cat(topk_idxs, dim=0).to(dtype=torch.int32)

    def _ensure_npu_c4_indexer(self, c4_indexer, device: torch.device) -> None:
        c4_indexer.compressor.li_kv_dtype = "int8"
        if getattr(c4_indexer, "hadamard_matrix", None) is None:
            H = _walsh_hadamard_matrix(c4_indexer.head_dim, torch.float32, device)
            c4_indexer.register_buffer("hadamard_matrix", H, persistent=False)

    def _compute_q_npu(
        self, c4_indexer, q_lora: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        from sglang.srt.layers.deepseek_v4_rope import v4_rope_inplace_npu

        bs = q_lora.shape[0]
        q, _ = c4_indexer.wq_b(q_lora)
        q = q.view(bs, c4_indexer.n_local_heads, c4_indexer.head_dim)
        v4_rope_inplace_npu(
            q[..., -c4_indexer.rope_head_dim :],
            None,
            c4_indexer.freqs_cis,
            positions,
        )
        return _apply_hadamard(q, c4_indexer.hadamard_matrix)

    def _forward_npu_fused(
        self,
        c4_indexer,
        q: torch.Tensor,
        k: torch.Tensor,
        k_scale: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        import torch_npu

        q_int8, q_scale = torch_npu.npu_dynamic_quant(q)
        fm = self.forward_metadata
        li_quant_metadata = fm.kernel_metadata["li_quant_metadata"]
        kwargs = dict(
            query=q_int8,
            key=k,
            key_dequant_scale=k_scale.squeeze(-2),
            actual_seq_lengths_query=fm.actual_seq_lengths_q,
            actual_seq_lengths_key=fm.actual_seq_lengths_kv,
            block_table=fm.c4_page_table,
            layout_query="TND",
            layout_key="PA_BSND",
            weights=weights.to(torch.float16),
            query_dequant_scale=q_scale.to(torch.float16),
            cmp_ratio=4,
            query_quant_mode=0,
            key_quant_mode=0,
            sparse_mode=3,
            sparse_count=self._dsv4_index_topk,
            metadata=li_quant_metadata,
        )
        topk_idxs, _ = torch.ops.custom.npu_quant_lightning_indexer(**kwargs)
        return topk_idxs.view(-1, self._dsv4_index_topk)

    def forward_c4_indexer(
        self,
        *,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: ForwardBatch,
        c4_indexer=None,
        alt_streams=None,
        enable_multi_stream: bool = False,
        q_lora_ready=None,
        skip_compressor: bool = False,
    ) -> None:
        if forward_batch.forward_mode.is_idle():
            return
        topk_idxs = self.forward_c4_indexer_npu(
            c4_indexer, x, q_lora, forward_batch, skip_compressor=skip_compressor
        )
        self.forward_metadata.c4_topk_indices = topk_idxs


class DeepseekV4AscendAttnBackend(
    AscendAttnBackend, C4IndexerAscendBackendMixin, CompressorAscendBackendMixin
):

    def __init__(
        self,
        model_runner: ModelRunner,
        speculative_step_id: int = 0,
    ):
        super().__init__(model_runner, speculative_step_id=speculative_step_id)
        # DSV4 custom sparse attention uses page tables plus ori_mask_mode and
        # never consumes ForwardMetadata.swa_mask.
        self.use_graph_swa_mask = False
        cfg = model_runner.model_config
        self._dsv4_config = cfg
        tp_size = get_attention_tp_size()
        self._dsv4_q_head_num = cfg.num_attention_heads // tp_size
        self._dsv4_kv_head_num = 1
        self._dsv4_head_dim = cfg.head_dim
        hf = getattr(cfg, "hf_config", cfg)
        self._dsv4_index_topk = hf.index_topk
        self._dsv4_index_n_heads = hf.index_n_heads
        self._dsv4_index_head_dim = hf.index_head_dim
        self._dsv4_compress_ratios = hf.compress_ratios
        if getattr(model_runner, "is_draft_worker", False):
            self._dsv4_compress_ratios = type(hf.compress_ratios)()
        self._dsv4_has_c4 = 4 in self._dsv4_compress_ratios
        self._dsv4_has_c128 = 128 in self._dsv4_compress_ratios
        self._dsv4_sliding_window_size = (
            cfg.sliding_window_size if cfg.sliding_window_size is not None else 128
        )
        self._dsv4_unique_compress_ratios = list(
            dict.fromkeys(self._dsv4_compress_ratios)
        )

    def _init_dsv4_graph_buffers(self, *, max_bs: int, max_num_tokens: int) -> None:
        device = self.device
        block_tables_shape = self.graph_metadata["block_tables"].shape
        max_pages = block_tables_shape[1]

        # -1 = invalid-page sentinel; full max_pages width keeps the replay
        # in-place copy shape-aligned across seq lengths.
        self.graph_metadata["swa_page_table"] = torch.full(
            (max_bs, max_pages), -1, dtype=torch.int32, device=device
        )

        self.graph_metadata["c4_page_table"] = torch.full(
            (max_bs, max_pages), -1, dtype=torch.int32, device=device
        )
        self.graph_metadata["c128_page_table"] = torch.full(
            (max_bs, max_pages), -1, dtype=torch.int32, device=device
        )
        # state_block_table uses 0 as the skip sentinel; real pages start at 1.
        self.graph_metadata["c4_state_page_table"] = torch.zeros(
            (max_bs, max_pages), dtype=torch.int32, device=device
        )
        self.graph_metadata["c128_state_page_table"] = torch.zeros(
            (max_bs, max_pages), dtype=torch.int32, device=device
        )

        for key in (
            "kernel_metadata_c1a",
            "kernel_metadata_c4a",
            "kernel_metadata_c128a",
            "kernel_metadata_li_quant",
        ):
            self.graph_metadata[key] = torch.zeros(
                1024, dtype=torch.int32, device=device
            )

        self.graph_metadata["c4_topk_indices"] = torch.full(
            (max_num_tokens, self._dsv4_index_topk),
            -1,
            dtype=torch.int32,
            device=device,
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        # Parent refreshes shared (block_tables / seq_lens) metadata; we layer DSV4
        # fields on top: capture allocates+zeros, replay refreshes them in place.
        super().init_forward_metadata_out_graph(forward_batch, in_capture=in_capture)
        bs = forward_batch.batch_size
        if in_capture:
            self._init_dsv4_graph_metadata(bs, forward_batch.forward_mode)
        else:
            self._apply_dsv4_graph_metadata(forward_batch)

    def _init_dsv4_graph_metadata(self, bs: int, forward_mode: ForwardMode) -> None:
        metadata = self.graph_metadata[bs]
        device = self.device

        if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
            tokens_per_bs = self.speculative_num_draft_tokens
        else:
            tokens_per_bs = 1

        metadata.actual_seq_lengths_q_pa = torch.arange(
            0,
            bs * tokens_per_bs + tokens_per_bs,
            tokens_per_bs,
            dtype=torch.int32,
            device=device,
        )
        metadata.actual_seq_lengths_q_cmp = metadata.actual_seq_lengths_q_pa.clone()

        metadata.actual_seq_lengths_kv = torch.ones(
            bs,
            dtype=torch.int32,
            device=device,
        )

        metadata.swa_page_table = self.graph_metadata["swa_page_table"][:bs, :]
        metadata.c4_page_table = self.graph_metadata["c4_page_table"][:bs, :]
        metadata.c128_page_table = self.graph_metadata["c128_page_table"][:bs, :]
        metadata.c4_state_page_table = self.graph_metadata["c4_state_page_table"][
            :bs, :
        ]
        metadata.c128_state_page_table = self.graph_metadata["c128_state_page_table"][
            :bs, :
        ]

        n_tok = bs * tokens_per_bs
        c4_pad = min(n_tok, n_tok // 4 + bs)
        c128_pad = min(n_tok, n_tok // 128 + bs)
        metadata.swa_loc = torch.zeros(n_tok, dtype=torch.int64, device=device)
        metadata.c4_loc = torch.zeros(c4_pad, dtype=torch.int64, device=device)
        metadata.c128_loc = torch.zeros(c128_pad, dtype=torch.int64, device=device)
        metadata.c4_state_loc = torch.zeros(n_tok, dtype=torch.int64, device=device)
        metadata.c128_state_loc = torch.zeros(n_tok, dtype=torch.int64, device=device)

        metadata.positions_cmp_padding_c4 = torch.zeros(
            c4_pad, dtype=torch.int64, device=device
        )
        metadata.positions_cmp_padding_c128 = torch.zeros(
            c128_pad, dtype=torch.int64, device=device
        )
        metadata.start_pos = torch.zeros(bs, dtype=torch.int32, device=device)
        metadata.seqused = torch.zeros(bs, dtype=torch.int32, device=device)

        metadata.kernel_metadata = {
            "c1a_metadata": self.graph_metadata["kernel_metadata_c1a"],
            "c4a_metadata": self.graph_metadata["kernel_metadata_c4a"],
            "c128a_metadata": self.graph_metadata["kernel_metadata_c128a"],
            "li_quant_metadata": self.graph_metadata["kernel_metadata_li_quant"],
        }

        T = bs * tokens_per_bs
        metadata.c4_topk_indices = self.graph_metadata["c4_topk_indices"][:T, :]

        self.forward_metadata = metadata

    def _apply_dsv4_graph_metadata(self, forward_batch: ForwardBatch) -> None:
        fm = self.forward_metadata
        forward_mode = (
            getattr(forward_batch, "global_forward_mode", None)
            or forward_batch.forward_mode
        )
        actual_forward_mode = getattr(forward_batch, "actual_forward_mode", None)
        if actual_forward_mode is None:
            actual_forward_mode = forward_batch.forward_mode
        bs = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        req_pool_indices = forward_batch.req_pool_indices
        device = seq_lens.device

        if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
            tokens_per_bs = self.speculative_num_draft_tokens
        else:
            tokens_per_bs = 1

        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert seq_lens_cpu is not None, "V4 graph replay requires seq_lens_cpu."
        if forward_mode.is_target_verify():
            # In graph replay, buffers.seq_lens already contains the attention KV
            # length (live length + draft tokens). Padded rows therefore show up as
            # tokens_per_bs instead of 0. Use the CPU live lengths as the source of
            # truth so padded rows stay masked out.
            live_seq_lens = seq_lens_cpu[:bs].to(device=device, dtype=torch.int32)
        elif seq_lens is not None and seq_lens.device.type != "cpu":
            live_seq_lens = seq_lens[:bs].to(dtype=torch.int32)
        else:
            live_seq_lens = seq_lens_cpu[:bs].to(device=device, dtype=torch.int32)
        attn_seq_lens = live_seq_lens
        if forward_mode.is_target_verify():
            valid_verify_rows = live_seq_lens > 0
            attn_seq_lens = live_seq_lens + int(tokens_per_bs)
            attn_seq_lens = torch.where(valid_verify_rows, attn_seq_lens, live_seq_lens)
            fm.seq_lens_cpu_int = (seq_lens_cpu[:bs] + int(tokens_per_bs)).int()
            fm.seq_lens_cpu_int = torch.where(
                seq_lens_cpu[:bs] > 0,
                fm.seq_lens_cpu_int,
                seq_lens_cpu[:bs].int(),
            )
        fm.actual_seq_lengths_kv.copy_(attn_seq_lens.clamp(min=1))

        pool = self.token_to_kv_pool
        out_cache_loc = forward_batch.out_cache_loc

        _verify_compress = (
            forward_mode.is_target_verify()
            and actual_forward_mode.is_target_verify()
            and bool(self._dsv4_compress_ratios)
        )
        _compress_seq_lens = live_seq_lens
        _compress_seq_lens_max = int(seq_lens_cpu[:bs].max()) if bs > 0 else 0
        if _verify_compress:
            _compress_seq_lens = live_seq_lens + int(tokens_per_bs)
            _compress_seq_lens_max += int(tokens_per_bs)

        result = self._compute_compress_locs(
            pool=pool,
            req_to_token=self.req_to_token,
            req_pool_indices=req_pool_indices[:bs],
            seq_lens=_compress_seq_lens,
            out_cache_loc=out_cache_loc,
            is_decode=forward_mode.is_decode(),
            bs=bs,
            device=device,
            req_to_token_pool=self.req_to_token_pool,
            out_cache_loc_dsv4=getattr(forward_batch, "out_cache_loc_dsv4", None),
            is_graph=True,
            seq_lens_max_override=_compress_seq_lens_max,
        )

        def _copy_2d(dst: torch.Tensor, src: torch.Tensor, val: int) -> None:
            dst.fill_(val)
            dst[: src.shape[0], : src.shape[1]].copy_(src)

        def _copy_1d(dst: torch.Tensor, src: torch.Tensor) -> None:
            dst.fill_(0)
            assert src.shape[0] <= dst.shape[0], (
                f"graph replay 1D metadata overflow: src={src.shape[0]} > "
                f"dst={dst.shape[0]}"
            )
            dst[: src.shape[0]].copy_(src)

        for key in (
            "c4_page_table",
            "c128_page_table",
            "c4_state_page_table",
            "c128_state_page_table",
        ):
            if key in result:
                _copy_2d(getattr(fm, key), result[key], 0 if "state" in key else -1)
        for key in ("c4_loc", "c128_loc", "c4_state_loc", "c128_state_loc"):
            if key in result:
                _copy_1d(getattr(fm, key), result[key])

        for key in (
            "positions_cmp_padding_c4",
            "positions_cmp_padding_c128",
            "start_pos",
            "seqused",
        ):
            if key in result and hasattr(fm, key) and getattr(fm, key) is not None:
                _copy_1d(getattr(fm, key), result[key])

        if _verify_compress:
            verify_seq_lens_cpu = seq_lens_cpu[:bs] + int(tokens_per_bs)
            verify_seq_lens_cpu = torch.where(
                seq_lens_cpu[:bs] > 0,
                verify_seq_lens_cpu,
                seq_lens_cpu[:bs],
            )
            self._fill_verify_positions_cmp_padding_one(
                forward_batch.positions,
                fm.positions_cmp_padding_c4,
                4,
                verify_seq_lens_cpu,
                n_draft=tokens_per_bs,
            )
            self._fill_verify_positions_cmp_padding_one(
                forward_batch.positions,
                fm.positions_cmp_padding_c128,
                128,
                verify_seq_lens_cpu,
                n_draft=tokens_per_bs,
            )
            fm.start_pos.copy_(live_seq_lens.to(torch.int32))
            valid = live_seq_lens[:bs] > 0
            fm.seqused.copy_(
                (valid.to(torch.int32) * int(tokens_per_bs)).to(device=device)
            )
            _bundle = getattr(forward_batch, "out_cache_loc_dsv4", None)
            if _bundle is not None:
                for ratio in self._dsv4_unique_compress_ratios:
                    if ratio not in (4, 128):
                        continue
                    bl = _bundle.out_c4_loc if ratio == 4 else _bundle.out_c128_loc
                    if bl is not None:
                        dst_loc = getattr(fm, f"c{ratio}_loc", None)
                        if dst_loc is not None:
                            dst_loc.zero_()
                            bl32 = bl.to(torch.int32)
                            assert bl32.numel() <= dst_loc.numel(), (
                                f"replay verify c{ratio}_loc overflow: "
                                f"{bl32.numel()} > {dst_loc.numel()}"
                            )
                            dst_loc[: bl32.numel()].copy_(bl32)

        elif (
            forward_mode.is_target_verify()
            # The graph may replay a target-verify capture for an idle/padded
            # DP rank. There is no real DSV4 allocation bundle in that case;
            # zero the compressor metadata so captured writes land in the
            # reserved dummy slot instead of reusing stale locs.
            and not actual_forward_mode.is_target_verify()
            and bool(self._dsv4_compress_ratios)
        ):
            for tensor in (
                fm.positions_cmp_padding_c4,
                fm.positions_cmp_padding_c128,
                fm.c4_loc,
                fm.c128_loc,
                fm.c4_state_loc,
                fm.c128_state_loc,
            ):
                if tensor is not None:
                    tensor.zero_()
            fm.start_pos.zero_()
            fm.seqused.zero_()

        swa_loc = pool.translate_loc_from_full_to_swa(out_cache_loc).to(torch.int64)
        _copy_1d(fm.swa_loc, swa_loc)

        swa_src = (
            fm.block_tables_swa if fm.block_tables_swa is not None else fm.block_tables
        )
        _copy_2d(fm.swa_page_table, swa_src, -1)
        if bs > 0:
            _spec = int(getattr(self, "speculative_num_draft_tokens", 0) or 0)
            max_len = int(seq_lens_cpu[:bs].max()) + _spec
            max_seq_pages = (max_len + self.page_size - 1) // self.page_size
            if 0 < max_seq_pages < fm.swa_page_table.shape[1]:
                fm.swa_page_table[:, max_seq_pages:].fill_(-1)

        kernel_metadata_new = self._kernel_metadata_from_parts(
            bs=bs,
            actual_seq_lengths_q_pa=fm.actual_seq_lengths_q_pa,
            actual_seq_lengths_kv=fm.actual_seq_lengths_kv,
            block_tables=fm.block_tables,
            max_seqlen_q=tokens_per_bs,
            is_nextn=False,
        )
        for key in (
            "c1a_metadata",
            "c4a_metadata",
            "c128a_metadata",
            "li_quant_metadata",
        ):
            if key in kernel_metadata_new:
                fm.kernel_metadata[key].copy_(kernel_metadata_new[key])

        fm.c4_topk_indices.fill_(-1)

        self.forward_metadata = fm

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        super().init_forward_metadata(forward_batch)
        fm = self.forward_metadata

        # Idle DP-attention ranks have zero seq_lens, which the metadata kernel
        # cannot handle; skip it and leave the fields cleared but well-typed.
        if forward_batch.forward_mode.is_idle():
            fm.actual_seq_lengths_q = None
            fm.actual_seq_lengths_q_pa = None
            fm.actual_seq_lengths_q_cmp = None
            fm.kernel_metadata = {}
            return

        device = forward_batch.seq_lens.device
        # cu_seqlens_q must hold per-request QUERY token counts, not KV lengths.
        if (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_draft_extend_v2()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            seq_lens_cpu = forward_batch.extend_seq_lens_cpu
            if isinstance(seq_lens_cpu, list):
                seq_lens_cpu = torch.tensor(seq_lens_cpu, dtype=torch.int32)
            else:
                seq_lens_cpu = seq_lens_cpu.int()
            actual_q = torch.cumsum(seq_lens_cpu, dim=0).int().to(device)
            fm.actual_seq_lengths_q = actual_q
            fm.actual_seq_lengths_q_pa = torch.cat(
                [torch.zeros(1, dtype=torch.int32, device=device), actual_q],
                dim=0,
            )
        elif forward_batch.forward_mode.is_decode():
            B = forward_batch.batch_size
            fm.actual_seq_lengths_q = torch.arange(
                1, B + 1, dtype=torch.int32, device=device
            )
            fm.actual_seq_lengths_q_pa = torch.arange(
                0, B + 1, dtype=torch.int32, device=device
            )
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            B = forward_batch.batch_size
            from sglang.srt.server_args import get_global_server_args

            n_draft = get_global_server_args().speculative_num_draft_tokens or 1
            actual_q = torch.arange(
                n_draft, B * n_draft + 1, n_draft, dtype=torch.int32, device=device
            )
            fm.actual_seq_lengths_q = actual_q
            fm.actual_seq_lengths_q_pa = torch.cat(
                [torch.zeros(1, dtype=torch.int32, device=device), actual_q],
                dim=0,
            )
        elif forward_batch.forward_mode.is_idle():
            B = forward_batch.batch_size
            fm.actual_seq_lengths_q = torch.arange(
                1, B + 1, dtype=torch.int32, device=device
            )
            fm.actual_seq_lengths_q_pa = torch.arange(
                0, B + 1, dtype=torch.int32, device=device
            )
            fm.actual_seq_lengths_kv = torch.ones(B, dtype=torch.int32, device=device)
        else:
            fm.actual_seq_lengths_q = None
            fm.actual_seq_lengths_q_pa = None

        fm.actual_seq_lengths_q_cmp = (
            fm.actual_seq_lengths_q_pa.clone()
            if fm.actual_seq_lengths_q_pa is not None
            else None
        )

        fm.swa_page_table = (
            fm.block_tables_swa if fm.block_tables_swa is not None else fm.block_tables
        )

        if fm.actual_seq_lengths_kv is None:
            if fm.seq_lens_cpu_int is not None:
                fm.actual_seq_lengths_kv = fm.seq_lens_cpu_int.to(
                    device=forward_batch.seq_lens.device, dtype=torch.int32
                )
            else:
                fm.actual_seq_lengths_kv = forward_batch.seq_lens.to(torch.int32)

        fm.kernel_metadata = self._compute_kernel_metadata(forward_batch)

        if self._dsv4_compress_ratios:
            self._build_npu_compress_metadata(forward_batch)

    def _compute_kernel_metadata(self, forward_batch: ForwardBatch) -> dict:
        fm = self.forward_metadata
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            from sglang.srt.server_args import get_global_server_args

            max_seqlen_q = get_global_server_args().speculative_num_draft_tokens or 1
        else:
            max_seqlen_q = 1
        return self._kernel_metadata_from_parts(
            bs=forward_batch.batch_size,
            actual_seq_lengths_q_pa=fm.actual_seq_lengths_q_pa,
            actual_seq_lengths_kv=fm.actual_seq_lengths_kv,
            block_tables=fm.block_tables,
            max_seqlen_q=max_seqlen_q,
            is_nextn=False,
        )

    def _kernel_metadata_from_parts(
        self,
        *,
        bs: int,
        actual_seq_lengths_q_pa: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
        block_tables: torch.Tensor,
        max_seqlen_q: int,
        is_nextn: bool,
    ) -> dict:
        common = {
            "cu_seqlens_q": actual_seq_lengths_q_pa,
            "seqused_kv": actual_seq_lengths_kv,
            "cmp_ratio": 1,
            "ori_mask_mode": 4,
            "cmp_mask_mode": 3,
            "ori_win_left": self._dsv4_sliding_window_size - 1,
            "ori_win_right": 0,
            "layout_q": "TND",
            "layout_kv": "PA_ND",
        }
        base_kwargs = {
            "batch_size": bs,
            "num_heads_q": self._dsv4_q_head_num,
            "num_heads_kv": self._dsv4_kv_head_num,
            "head_dim": self._dsv4_head_dim,
            "has_ori_kv": True,
            "has_cmp_kv": False,
        }
        c1a_kwargs = base_kwargs | common
        kernel_metadata = {
            "c1a_metadata": torch.ops.custom.npu_sparse_attn_sharedkv_metadata(
                **c1a_kwargs
            )
        }

        if self._dsv4_has_c4:
            c4a_overrides = {
                "cmp_ratio": 4,
                "has_cmp_kv": True,
                "cmp_topk": self._dsv4_index_topk,
            }
            c4a_kwargs = c1a_kwargs | c4a_overrides
            kernel_metadata["c4a_metadata"] = (
                torch.ops.custom.npu_sparse_attn_sharedkv_metadata(**c4a_kwargs)
            )

            if actual_seq_lengths_q_pa is not None:
                actual_q = actual_seq_lengths_q_pa[1:].clone()
            else:
                actual_q = actual_seq_lengths_kv
            kernel_metadata["li_quant_metadata"] = (
                torch.ops.custom.npu_quant_lightning_indexer_metadata(
                    device=str(actual_q.device),
                    actual_seq_lengths_query=actual_q,
                    actual_seq_lengths_key=actual_seq_lengths_kv,
                    layout_key="PA_BSND",
                    sparse_count=self._dsv4_index_topk,
                    sparse_mode=3,
                    layout_query="TND",
                    cmp_ratio=4,
                    key_quant_mode=0,
                    query_quant_mode=0,
                    num_heads_q=self._dsv4_index_n_heads,
                    num_heads_k=1,
                    head_dim=self._dsv4_index_head_dim,
                )
            )

        if self._dsv4_has_c128:
            c128a_overrides = {"cmp_ratio": 128, "has_cmp_kv": True}
            c128a_kwargs = c1a_kwargs | c128a_overrides
            kernel_metadata["c128a_metadata"] = (
                torch.ops.custom.npu_sparse_attn_sharedkv_metadata(**c128a_kwargs)
            )

        return kernel_metadata

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        *,
        compress_ratio: int = 0,
        attn_sink: Optional[torch.Tensor] = None,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if compress_ratio not in (0, 4, 128):
            raise ValueError(
                f"V4 attention expects compress_ratio in (0, 4, 128); got {compress_ratio}"
            )
        if forward_batch.forward_mode.is_idle():
            return torch.zeros_like(q)
        if save_kv_cache:
            self.store_cache(
                layer_id=layer.layer_id, swa_k=k, forward_batch=forward_batch
            )
        if compress_ratio == 0:
            return self._forward_dense(q, layer, forward_batch, attn_sink)
        return self._forward_compressed(
            q, layer, forward_batch, attn_sink, compress_ratio
        )

    def _forward_dense(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        attn_sink: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """ratio=0 dense layers — sliding-window attention via
        npu_sparse_attn_sharedkv with has_cmp_kv=False."""
        fm = self.forward_metadata
        pool = self.token_to_kv_pool
        ori_kv = pool.get_swa_buffer(layer.layer_id)

        attn_kwargs = dict(
            cu_seqlens_q=fm.actual_seq_lengths_q_pa,
            seqused_kv=fm.actual_seq_lengths_kv,
            ori_mask_mode=4,
            ori_win_left=self._dsv4_sliding_window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            q=q,
            ori_kv=ori_kv,
            ori_block_table=fm.swa_page_table,
            sinks=attn_sink,
            metadata=fm.kernel_metadata["c1a_metadata"],
            softmax_scale=layer.scaling,
        )
        out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(**attn_kwargs)
        return out

    def _forward_compressed(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        attn_sink: Optional[torch.Tensor],
        compress_ratio: int,
    ) -> torch.Tensor:
        fm = self.forward_metadata
        pool = self.token_to_kv_pool
        metadata = fm.kernel_metadata.get(f"c{compress_ratio}a_metadata")
        cmp_kv = pool.get_compress_buffer(layer.layer_id, False)

        if metadata is None or cmp_kv is None:
            raise RuntimeError(
                "DeepseekV4AscendAttnBackend._forward_compressed: missing "
                f"required state for layer_id={layer.layer_id} "
                f"compress_ratio={compress_ratio}. "
                f"metadata({'present' if metadata is not None else 'MISSING'}), "
                f"cmp_kv({'present' if cmp_kv is not None else 'MISSING'}). "
                f"Available kernel_metadata keys: {list(fm.kernel_metadata.keys())}. "
                "This indicates a configuration / pool-init bug — silently "
                "returning zeros would corrupt model output."
            )

        ori_kv = pool.get_swa_buffer(layer.layer_id)

        ori_page_size = ori_kv.shape[1]
        cmp_native_page_size = cmp_kv.shape[1]
        cmp_block_table = getattr(fm, f"c{compress_ratio}_page_table")
        assert cmp_native_page_size == ori_page_size, (
            f"cmp page_size={cmp_native_page_size} != ori page_size={ori_page_size}; "
            "c{N}_kv_pool must be allocated with the global page_size on NPU "
            "(see NPUDeepSeekV4SingleKVPool.kernel_page_size)"
        )

        attn_kwargs = dict(
            cu_seqlens_q=fm.actual_seq_lengths_q_pa,
            seqused_kv=fm.actual_seq_lengths_kv,
            ori_mask_mode=4,
            ori_win_left=self._dsv4_sliding_window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            q=q,
            ori_kv=ori_kv,
            ori_block_table=fm.swa_page_table,
            sinks=attn_sink,
            metadata=metadata,
            softmax_scale=layer.scaling,
            cmp_ratio=compress_ratio,
            cmp_mask_mode=3,
            cmp_kv=cmp_kv,
            cmp_block_table=cmp_block_table,
        )
        if compress_ratio == 4:
            topk = fm.c4_topk_indices
            attn_kwargs["cmp_sparse_indices"] = topk.view(-1, 1, topk.shape[-1])
        else:
            attn_kwargs["cmp_sparse_indices"] = None
        out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(**attn_kwargs)
        return out

    def store_cache(self, *, layer_id: int, swa_k: torch.Tensor, forward_batch):
        pool = self.token_to_kv_pool
        swa_loc = pool.translate_loc_from_full_to_swa(forward_batch.out_cache_loc)
        pool.set_swa_buffer(
            layer_id=layer_id,
            loc=swa_loc,
            cache=swa_k,
        )

    def _build_npu_compress_metadata_verify(self, forward_batch: ForwardBatch) -> None:
        fm = self.forward_metadata
        device = forward_batch.seq_lens.device
        positions = forward_batch.positions
        t = positions.shape[0]
        bs = forward_batch.batch_size
        n_draft = int(
            getattr(
                getattr(forward_batch, "spec_info", None),
                "draft_token_num",
                self.speculative_num_draft_tokens,
            )
        )
        verify_seq_lens_cpu = forward_batch.seq_lens_cpu[:bs] + int(n_draft)
        padding_sizes = {}
        for ratio in (4, 128):
            if ratio not in self._dsv4_compress_ratios:
                continue
            padding_size = max(1, min(t, t // ratio + bs))
            padding_sizes[ratio] = padding_size
            padding = torch.zeros(padding_size, dtype=torch.int64, device=device)
            self._fill_verify_positions_cmp_padding_one(
                positions, padding, ratio, verify_seq_lens_cpu, n_draft=n_draft
            )
            setattr(fm, f"positions_cmp_padding_c{ratio}", padding)
        fm.start_pos = forward_batch.seq_lens.to(torch.int32)
        valid = forward_batch.seq_lens[:bs] > 0
        fm.seqused = valid.to(torch.int32) * int(n_draft)
        _bundle = getattr(forward_batch, "out_cache_loc_dsv4", None)
        if _bundle is not None:
            for ratio in self._dsv4_unique_compress_ratios:
                if ratio not in (4, 128):
                    continue
                bl = _bundle.out_c4_loc if ratio == 4 else _bundle.out_c128_loc
                if bl is None:
                    loc = None
                else:
                    padding_size = padding_sizes[ratio]
                    loc = torch.zeros(padding_size, dtype=torch.int32, device=device)
                    if bl.numel() > 0:
                        assert bl.numel() <= padding_size, (
                            f"verify c{ratio}_loc overflow: "
                            f"{bl.numel()} > {padding_size}"
                        )
                        loc[: bl.numel()].copy_(bl.to(torch.int32))
                setattr(fm, f"c{ratio}_loc", loc)

    def _fill_verify_positions_cmp_padding(
        self,
        positions: torch.Tensor,
        c4_positions: torch.Tensor,
        c128_positions: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor] = None,
    ) -> None:
        c4_positions.fill_(0)
        c128_positions.fill_(0)
        if positions.numel() == 0:
            return

        n_draft = self.speculative_num_draft_tokens
        request_num = positions.shape[0] // n_draft
        if request_num == 0:
            return

        fm = self.forward_metadata
        if seq_lens_cpu is None:
            seq_lens_cpu = getattr(fm, "seq_lens_cpu", None)
        if seq_lens_cpu is None:
            seq_lens_cpu = getattr(fm, "seq_lens_cpu_int", None)
        if seq_lens_cpu is None:
            raise RuntimeError(
                "DSV4 verify buffer refresh requires seq_lens_cpu or "
                "seq_lens_cpu_int on forward metadata."
            )
        seq_lens_cpu = seq_lens_cpu[:request_num]
        if seq_lens_cpu.device.type != "cpu":
            seq_lens_cpu = seq_lens_cpu.cpu()

        start_positions = seq_lens_cpu - n_draft + 1
        abs_positions = start_positions.view(-1, 1) + torch.arange(
            n_draft, dtype=start_positions.dtype
        ).view(1, -1)
        mask_c4 = (abs_positions % 4) != 0
        mask_c128 = (abs_positions % 128) != 0

        gather_shape_c4 = min(
            positions.shape[0], mask_c4.numel(), c4_positions.shape[0]
        )
        gather_shape_c128 = min(
            positions.shape[0], mask_c128.numel(), c128_positions.shape[0]
        )
        sorted_indices_c4 = (
            torch.argsort(mask_c4.flatten(), dim=0, stable=True)[:gather_shape_c4]
            .pin_memory()
            .to(device=positions.device, non_blocking=True)
        )
        sorted_indices_c128 = (
            torch.argsort(mask_c128.flatten(), dim=0, stable=True)[:gather_shape_c128]
            .pin_memory()
            .to(device=positions.device, non_blocking=True)
        )

        c4_positions[:gather_shape_c4].copy_(
            torch.gather(positions, 0, sorted_indices_c4)
        )
        c128_positions[:gather_shape_c128].copy_(
            torch.gather(positions, 0, sorted_indices_c128)
        )

    def _fill_verify_positions_cmp_padding_one(
        self,
        positions: torch.Tensor,
        dst: torch.Tensor,
        ratio: int,
        seq_lens_cpu: torch.Tensor,
        n_draft: Optional[int] = None,
    ) -> None:
        dst.zero_()
        if ratio not in self._dsv4_compress_ratios or positions.numel() == 0:
            return

        if n_draft is None:
            n_draft = self.speculative_num_draft_tokens
        n_draft = int(n_draft)
        request_num = positions.shape[0] // n_draft
        if request_num == 0:
            return
        seq_lens_cpu = seq_lens_cpu[:request_num]
        if seq_lens_cpu.device.type != "cpu":
            seq_lens_cpu = seq_lens_cpu.cpu()

        start_positions = seq_lens_cpu - n_draft + 1
        abs_positions = start_positions.view(-1, 1) + torch.arange(
            n_draft, dtype=start_positions.dtype
        ).view(1, -1)
        boundary_mask = abs_positions % ratio == 0
        indices = torch.nonzero(boundary_mask.flatten(), as_tuple=False).flatten()

        if indices.numel() == 0:
            return
        # This tiny H2D copy runs on the verify metadata path. Keep it blocking:
        # on NPU, a non-blocking copy from a short-lived pinned CPU tensor can
        # surface later as an unrelated CopyKernel stream failure.
        indices = indices[: dst.numel()].to(device=positions.device)
        dst[: indices.numel()].copy_(torch.gather(positions, 0, indices))

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info, cuda_graph_bs: Optional[int]
    ):
        fm = self.forward_metadata
        positions = spec_info.positions
        c4_positions = getattr(fm, "positions_cmp_padding_c4", None)
        c128_positions = getattr(fm, "positions_cmp_padding_c128", None)
        if c4_positions is None or c128_positions is None:
            return

        n_draft = int(
            getattr(spec_info, "draft_token_num", self.speculative_num_draft_tokens)
        )
        seq_lens_cpu = getattr(fm, "seq_lens_cpu_int", None)
        if seq_lens_cpu is None:
            seq_lens_cpu = getattr(spec_info, "seq_lens_cpu", None)
            if seq_lens_cpu is None:
                raise RuntimeError(
                    "DSV4 verify buffer refresh requires seq_lens_cpu_int on "
                    "forward metadata or seq_lens_cpu on spec_info."
                )
            seq_lens_cpu = seq_lens_cpu + n_draft

        self._fill_verify_positions_cmp_padding_one(
            positions, c4_positions, 4, seq_lens_cpu, n_draft=n_draft
        )
        self._fill_verify_positions_cmp_padding_one(
            positions, c128_positions, 128, seq_lens_cpu, n_draft=n_draft
        )


def _get_kv_indices(
    forward_batch: ForwardBatch,
    kv_len: int,
    page_table: torch.Tensor,
    req_idx: int,
    seqlen: int,
) -> torch.Tensor:
    logic_start = max(0, seqlen - kv_len)
    logic_end = seqlen
    page_size = get_attn_backend().page_size
    if page_size == 1:
        return page_table[req_idx, logic_start:logic_end]
    logic_pos = torch.arange(logic_start, logic_end, device=page_table.device)
    block_id = logic_pos // page_size
    offset_in_block = logic_pos % page_size
    return page_table[req_idx, block_id] * page_size + offset_in_block


class DeepseekV4AscendMultiStepDraftBackend:

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.attn_backends = [
            DeepseekV4AscendAttnBackend(model_runner, speculative_step_id=step_id)
            for step_id in range(speculative_num_steps)
        ]

    def common_template(self, forward_batch: ForwardBatch, call_fn):
        assert forward_batch.spec_info is not None

        for i in range(self.speculative_num_steps - 1):
            call_fn(i, forward_batch)

    def _step_out_cache_loc(self, forward_batch: ForwardBatch, step_id: int):
        out_cache_loc = forward_batch.out_cache_loc
        if out_cache_loc is None:
            return None

        single_step_width = forward_batch.batch_size * self.topk
        if out_cache_loc.numel() <= single_step_width:
            return out_cache_loc

        step_layout_width = self.topk * self.speculative_num_steps
        if step_layout_width == 0 or out_cache_loc.numel() % step_layout_width != 0:
            return out_cache_loc

        from sglang.srt.speculative.eagle_utils import per_step_draft_out_cache_loc

        batch_size = out_cache_loc.numel() // step_layout_width
        return per_step_draft_out_cache_loc(
            out_cache_loc,
            batch_size,
            self.topk,
            self.speculative_num_steps,
        )[step_id]

    def _step_out_cache_loc_dsv4(self, forward_batch: ForwardBatch, step_id: int):
        bundle = forward_batch.out_cache_loc_dsv4
        if bundle is None or forward_batch.out_cache_loc is None:
            return None

        step_width = forward_batch.batch_size * self.topk
        total_width = step_width * self.speculative_num_steps
        raw_total_width = bundle.out_full_loc.numel()
        if (
            raw_total_width < total_width
            and raw_total_width % self.speculative_num_steps == 0
            and (raw_total_width // self.speculative_num_steps) % self.topk == 0
        ):
            step_width = raw_total_width // self.speculative_num_steps
            total_width = raw_total_width
        if step_width == 0 or bundle.out_full_loc.numel() < total_width:
            return bundle

        full_steps = bundle.out_full_loc[:total_width].reshape(
            step_width // self.topk, self.topk, self.speculative_num_steps
        )
        full_steps = full_steps.permute((2, 0, 1)).reshape(
            self.speculative_num_steps, -1
        )
        swa_steps = bundle.out_swa_loc[:total_width].reshape(
            step_width // self.topk, self.topk, self.speculative_num_steps
        )
        swa_steps = swa_steps.permute((2, 0, 1)).reshape(self.speculative_num_steps, -1)

        def step_state(loc):
            if loc is None or loc.numel() < total_width:
                return loc
            steps = loc[:total_width].reshape(
                step_width // self.topk, self.topk, self.speculative_num_steps
            )
            return steps.permute((2, 0, 1)).reshape(self.speculative_num_steps, -1)[
                step_id
            ]

        def step_compress(loc, ratio: int):
            if loc is None or loc.numel() == 0:
                return loc
            raw_bs = step_width // self.topk
            seq_lens = forward_batch.seq_lens[:raw_bs].to(torch.int64)
            positions = seq_lens[:, None, None] + torch.arange(
                self.speculative_num_steps,
                device=seq_lens.device,
                dtype=seq_lens.dtype,
            )
            positions = positions.expand(-1, self.topk, -1)
            should_compress = ((positions + 1) % ratio) == 0
            counts = should_compress.reshape(-1).to(torch.int64)
            offsets = torch.cumsum(counts, dim=0) - counts
            step_mask = should_compress[:, :, step_id].reshape(-1)
            step_offsets = offsets.reshape(
                raw_bs, self.topk, self.speculative_num_steps
            )[:, :, step_id].reshape(-1)
            return loc[step_offsets[step_mask].to(torch.int64)]

        return DSV4OutCacheLoc(
            out_full_loc=full_steps[step_id],
            out_swa_loc=swa_steps[step_id],
            out_c4_loc=step_compress(bundle.out_c4_loc, 4),
            out_c128_loc=step_compress(bundle.out_c128_loc, 128),
            out_c4_state_loc=step_state(bundle.out_c4_state_loc),
            out_c128_state_loc=step_state(bundle.out_c128_state_loc),
        )

    def _with_step_cache_locs(self, forward_batch: ForwardBatch, step_id: int, call_fn):
        old_out_cache_loc = forward_batch.out_cache_loc
        old_out_cache_loc_dsv4 = forward_batch.out_cache_loc_dsv4
        step_out_cache_loc = self._step_out_cache_loc(forward_batch, step_id)
        if step_out_cache_loc is not None:
            forward_batch.out_cache_loc = step_out_cache_loc
        forward_batch.out_cache_loc_dsv4 = self._step_out_cache_loc_dsv4(
            forward_batch, step_id
        )
        try:
            return call_fn()
        finally:
            forward_batch.out_cache_loc = old_out_cache_loc
            forward_batch.out_cache_loc_dsv4 = old_out_cache_loc_dsv4

    def _build_step_forward_batch(
        self, forward_batch: ForwardBatch, step_id: int
    ) -> ForwardBatch:
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        step_fb = build_inner_fb_view(
            forward_batch,
            bs=forward_batch.batch_size,
            forward_mode=ForwardMode.DECODE,
        )
        old_bundle = forward_batch.out_cache_loc_dsv4
        step_out_cache_loc = self._step_out_cache_loc(forward_batch, step_id)
        step_bundle = self._step_out_cache_loc_dsv4(forward_batch, step_id)
        step_fb.out_cache_loc_dsv4 = step_bundle
        step_fb.global_forward_mode = getattr(
            forward_batch, "global_forward_mode", None
        )
        if (
            step_bundle is not None
            and step_bundle is not old_bundle
            and step_bundle.out_full_loc is not None
        ):
            step_fb.out_cache_loc = step_bundle.out_full_loc
        elif step_out_cache_loc is not None:
            step_fb.out_cache_loc = step_out_cache_loc
        return step_fb

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self._with_step_cache_locs(
                forward_batch,
                i,
                lambda: self.attn_backends[i].init_forward_metadata(forward_batch),
            )

        self.common_template(forward_batch, call_fn)

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        for i in range(self.speculative_num_steps):
            self.attn_backends[i].init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_out_graph(
                self._build_step_forward_batch(forward_batch, i),
                in_capture=in_capture,
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_in_graph(forward_batch)

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_capture_cuda_graph(self, forward_batch: ForwardBatch):
        def call_fn(i, forward_batch):
            self.attn_backends[i].init_forward_metadata_capture_cuda_graph(
                forward_batch.batch_size,
                forward_batch.batch_size * self.topk,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                encoder_lens=None,
                forward_mode=ForwardMode.DECODE,
                spec_info=forward_batch.spec_info,
            )

        self.common_template(forward_batch, call_fn)

    def init_forward_metadata_replay_cuda_graph(
        self, forward_batch: ForwardBatch, bs: int
    ):
        def call_fn(i, forward_batch):
            old_oc = forward_batch.out_cache_loc
            old_bundle = forward_batch.out_cache_loc_dsv4
            step_bundle = self._step_out_cache_loc_dsv4(forward_batch, i)
            forward_batch.out_cache_loc_dsv4 = step_bundle
            if (
                step_bundle is not None
                and step_bundle is not old_bundle
                and step_bundle.out_full_loc is not None
            ):
                forward_batch.out_cache_loc = step_bundle.out_full_loc
            self.attn_backends[i]._replay_forward_batch = forward_batch
            try:
                self.attn_backends[i].init_forward_metadata_replay_cuda_graph(
                    bs,
                    forward_batch.req_pool_indices,
                    forward_batch.seq_lens,
                    seq_lens_sum=-1,
                    encoder_lens=None,
                    forward_mode=ForwardMode.DECODE,
                    spec_info=forward_batch.spec_info,
                    seq_lens_cpu=forward_batch.seq_lens_cpu,
                )
            finally:
                self.attn_backends[i]._replay_forward_batch = None
                forward_batch.out_cache_loc = old_oc
                forward_batch.out_cache_loc_dsv4 = old_bundle

        self.common_template(forward_batch, call_fn)
