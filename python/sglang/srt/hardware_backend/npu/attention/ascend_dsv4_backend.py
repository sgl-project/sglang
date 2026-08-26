from __future__ import annotations

import logging
import math
from types import SimpleNamespace
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F
import torch_npu

from sglang.kernels.ops.speculative.dspark.dspark_attn_metadata import (
    BuildBlockSeqLensCausal,
    BuildDsparkSwaPageIndices,
    ComputeDsparkWindowGather,
)
from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc, ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


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


def _build_explicit_state_block_table(
    *,
    compress_ratio: int,
    coff: int,
    state_pool,
    token_to_kv_pool,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    start_pos: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seqused: torch.Tensor,
    max_input_capacity: int,
) -> torch.Tensor:
    """Adapt GPU-style state locations to the A3 cache_mode=2 table ABI."""
    req_pool_indices = req_pool_indices.to(torch.int64)
    capacities = cu_seqlens[1:] - cu_seqlens[:-1]
    history_size = coff * compress_ratio
    width = history_size + max_input_capacity
    columns = torch.arange(width, dtype=torch.int64, device=req_to_token.device)
    positions = start_pos[:, None] - history_size + columns
    within_capacity = columns[None, :] < history_size + capacities[:, None]
    valid = (seqused[:, None] > 0) & within_capacity & (positions >= 0)

    if compress_ratio == 4:
        # Masked history/ragged columns are still indexed before torch.where.
        safe_positions = positions.clamp(0, req_to_token.shape[1] - 1)
        full_locs = req_to_token[req_pool_indices[:, None], safe_positions]
        swa_locs = token_to_kv_pool.translate_loc_from_full_to_swa(full_locs)
        state_locs = state_pool.translate_from_swa_loc_to_state_loc(swa_locs)
    else:
        state_locs = state_pool.translate_from_req_position_to_state_loc(
            req_pool_indices[:, None], positions
        )

    return torch.where(
        valid,
        state_locs.to(torch.int32),
        state_pool.dummy_state_loc,
    ).contiguous()


class CompressorAscendBackendMixin:

    @staticmethod
    def _to_cpu_int_list(values) -> Optional[list[int]]:
        if values is None:
            return None
        if isinstance(values, torch.Tensor):
            values = values.cpu().tolist()
        return [int(v) for v in values]

    def _extend_prefix_lens_cpu(
        self, forward_batch: ForwardBatch
    ) -> Optional[list[int]]:
        prefix_lens = self._to_cpu_int_list(
            getattr(forward_batch, "extend_prefix_lens_cpu", None)
        )
        if prefix_lens is not None:
            return prefix_lens

        seq_lens = self._to_cpu_int_list(getattr(forward_batch, "seq_lens_cpu", None))
        extend_lens = self._to_cpu_int_list(
            getattr(forward_batch, "extend_seq_lens_cpu", None)
        )
        if seq_lens is None or extend_lens is None or len(seq_lens) != len(extend_lens):
            return None
        return [
            max(0, seq_len - extend_len)
            for seq_len, extend_len in zip(seq_lens, extend_lens)
        ]

    def _build_npu_compress_metadata(self, forward_batch: ForwardBatch) -> None:
        fm = self.forward_metadata
        is_decode = forward_batch.forward_mode.is_decode()
        is_verify = forward_batch.forward_mode.is_target_verify()
        fm.dsv4_explicit_state_block_tables = {}
        fm.dsv4_max_input_capacity = 1 if is_decode else None
        _verify_compress = is_verify and bool(self._dsv4_compress_ratios)
        _seq_lens = forward_batch.seq_lens.to(torch.int32)
        if _verify_compress:
            n_draft = int(forward_batch.spec_info.draft_token_num)
            _seq_lens = _seq_lens + n_draft
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
                    if f"c{ratio}_loc" not in result:
                        setattr(fm, f"c{ratio}_loc", None)
            # _compute_compress_locs builds positions_cmp_padding / start_pos /
            # seqused only for decode. Every eager prefill uses the fused compressor,
            # so build its global block positions, state metadata, and output locs
            # here. Exclude speculative modes so the cu.cpu() host read never
            # runs for target_verify / draft_extend (potentially graph-captured).
            if forward_batch.forward_mode.is_extend_without_speculative():
                self._build_npu_compress_metadata_prefill(forward_batch)

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
        fm.dsv4_max_input_capacity = max(
            1,
            max(
                (int(cu_cpu[idx + 1]) - int(cu_cpu[idx]) for idx in range(bs)),
                default=0,
            ),
        )
        prefix_cpu = self._extend_prefix_lens_cpu(forward_batch)
        ratio_lists: dict = {
            r: [] for r in self._dsv4_unique_compress_ratios if r in (4, 128)
        }
        for idx in range(bs):
            start = int(cu_cpu[idx])
            end = int(cu_cpu[idx + 1])
            if end == start:
                continue
            prefix = (
                int(prefix_cpu[idx])
                if prefix_cpu is not None and idx < len(prefix_cpu)
                else 0
            )
            total = prefix + (end - start)
            for ratio in ratio_lists:
                first_k = prefix // ratio
                last_k = total // ratio
                if last_k > first_k:
                    ratio_lists[ratio].append(
                        torch.arange(first_k, last_k, device=device, dtype=torch.int64)
                        * ratio
                    )

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

        # start_pos is each request's global chunk start. cache_mode=2 uses it
        # together with the explicit table to align history/current columns.
        if forward_batch.extend_prefix_lens is not None:
            fm.start_pos = forward_batch.extend_prefix_lens.to(
                device=device, dtype=torch.int32
            )
        elif prefix_cpu is not None:
            fm.start_pos = torch.tensor(
                prefix_cpu[:bs] + [0] * max(0, bs - len(prefix_cpu)),
                dtype=torch.int32,
                device=device,
            )
        else:
            fm.start_pos = torch.zeros(bs, dtype=torch.int32, device=device)
        fm.seqused = (cu[1:] - cu[:-1]).to(torch.int32)

        # bundle out_c*_loc = the NEW c-pool slots allocated this extend (incremental),
        # densely packed in batch order to match cmp_kv. Valid under chunked prefill:
        # each chunk writes only the ratio-blocks it newly completed.
        bundle = forward_batch.out_cache_loc_dsv4
        for ratio in (4, 128):
            if ratio not in ratio_lists:
                continue
            bundle_loc = (
                (bundle.out_c4_loc if ratio == 4 else bundle.out_c128_loc)
                if bundle is not None
                else None
            )
            setattr(
                fm,
                f"c{ratio}_loc",
                bundle_loc.to(torch.int32) if bundle_loc is not None else None,
            )

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
        req_pool_64 = req_pool.to(torch.int64)

        if seq_lens_max_override is not None:
            seq_lens_max = int(seq_lens_max_override)
        else:
            seq_lens_max = int(seq_lens.max().item()) if bs > 0 else 0
        for ratio in self._dsv4_unique_compress_ratios:
            if ratio not in (4, 128):
                continue
            bundle_loc = None

            if is_decode:
                # bundle_loc and cmp_kv are both densely packed in batch order, so
                # write them densely; indexing by batch slot would misalign them.
                if out_cache_loc_dsv4 is not None:
                    bundle_loc = (
                        out_cache_loc_dsv4.out_c4_loc
                        if ratio == 4
                        else out_cache_loc_dsv4.out_c128_loc
                    )

            if is_decode:
                compress_out_loc = torch.zeros(
                    bs,
                    dtype=torch.int32,
                    device=device,
                )
                if bundle_loc is not None:
                    n_compress = bundle_loc.numel()
                    if n_compress > 0:
                        compress_out_loc[:n_compress] = bundle_loc.to(torch.int32)
                result[f"c{ratio}_loc"] = compress_out_loc

            # graph: keep shape aligned with the preallocated buffer; eager: clamp >=1 so kernels see a column
            if is_graph:
                n_c_tokens = seq_lens_max // ratio
            else:
                n_c_tokens = max(1, seq_lens_max // ratio)
            if ratio == 4:
                slots = req_to_token[req_pool_64, : n_c_tokens * ratio]
                c_page_table = (slots[:, :: self.page_size] // self.page_size).to(
                    torch.int32
                )
            else:
                c128_page_size = req_to_token_pool.c128_page_size
                n_groups = (n_c_tokens + c128_page_size - 1) // c128_page_size
                c_page_table = req_to_token_pool.req_to_c128_sidecar[
                    req_pool_64, :n_groups
                ].to(torch.int32)
            result[f"c{ratio}_page_table"] = c_page_table

        if is_decode:
            valid = seq_lens > 0
            positions_last = torch.clamp(seq_lens - 1, min=0)
            for ratio in self._dsv4_unique_compress_ratios:
                if ratio not in (4, 128):
                    continue
                padding_size = min(bs, bs // ratio + bs)
                should_compress = ((seq_lens % ratio) == 0) & valid
                pos_cmp = positions_last[should_compress].to(torch.int64) + (1 - ratio)
                padding = torch.zeros(padding_size, dtype=torch.int64, device=device)
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

        ratio = compressor.ratio
        coff = 1 + int(compressor.overlap)
        device = x.device
        self._ensure_compressor_hadamard(compressor, device)
        self._ensure_fused_caches(compressor)

        fm = self.forward_metadata
        pool = self.token_to_kv_pool
        state_pool = pool._get_state_pool(compressor.layer_id, compressor.is_in_indexer)
        state_cache = state_pool.state_cache_3d
        table_cache = fm.dsv4_explicit_state_block_tables
        if ratio not in table_cache:
            table_cache[ratio] = _build_explicit_state_block_table(
                compress_ratio=ratio,
                coff=coff,
                state_pool=state_pool,
                token_to_kv_pool=pool,
                req_to_token=self.req_to_token,
                req_pool_indices=forward_batch.req_pool_indices,
                start_pos=fm.start_pos,
                cu_seqlens=fm.actual_seq_lengths_q_pa,
                seqused=fm.seqused,
                max_input_capacity=fm.dsv4_max_input_capacity,
            )
        state_block_table = table_cache[ratio]

        cos, sin = Dsv4NpuRoPE.for_freqs(
            compressor.freqs_cis, getattr(compressor, "rotary_emb", None)
        ).get_cos_sin(
            getattr(fm, f"positions_cmp_padding_c{ratio}"),
            torch.float32,
            view_4d=False,
            allow_build=False,
        )

        cmp_kv = torch.ops.npu.compressor(
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
            state_block_table=state_block_table,
            cu_seqlens=fm.actual_seq_lengths_q_pa,
            seqused=fm.seqused,
            start_pos=fm.start_pos,
            coff=coff,
            norm_eps=compressor.norm.variance_epsilon,
            rotary_mode=2,
            cache_mode=2,
        )

        # prefill output may be padded; trim to loc length
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


class C4IndexerAscendBackendMixin:

    def init_forward_metadata_indexer(self, core_attn_metadata):
        # li_quant_metadata is built in _compute_kernel_metadata; None satisfies the mixin contract
        return None

    def _forward_prepare(
        self,
        c4_indexer,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._compute_q_npu(c4_indexer, q_lora, forward_batch.positions)
        weights, _ = c4_indexer.weights_proj(x)
        weights = weights * (c4_indexer.softmax_scale * c4_indexer.n_heads**-0.5)
        c4_indexer.compressor(x, forward_batch)
        return q, weights

    def _can_use_indexer_multi_stream(self) -> bool:
        return envs.SGLANG_NPU_USE_MULTI_STREAM.get()

    def _get_npu_indexer_q_stream(self):
        s = getattr(self, "_npu_indexer_q_stream_obj", None)
        if s is None:
            s = torch.npu.Stream()
            self._npu_indexer_q_stream_obj = s
        return s

    def _forward_prepare_multi_stream(
        self,
        c4_indexer,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: ForwardBatch,
        q_lora_ready,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from sglang.srt.hardware_backend.npu.utils import (
            get_indexer_weight_stream,
        )

        cur = torch.npu.current_stream()
        stream_q = self._get_npu_indexer_q_stream()
        stream_w = get_indexer_weight_stream()

        # q_lora/x are produced on cur; workers wait for them.
        stream_q.wait_stream(cur)
        stream_w.wait_stream(cur)

        # route-KV write on cur; ordered before the topk read by cur's program order.
        c4_indexer.compressor(x, forward_batch)

        # weights_proj + scale on stream_w.
        with torch.npu.stream(stream_w):
            weights = c4_indexer.weights_proj(x)[0]
            weights = weights * (c4_indexer.softmax_scale * c4_indexer.n_heads**-0.5)
            weights.record_stream(stream_w)

        # q (wq_b + rope + hadamard) on stream_q.
        with torch.npu.stream(stream_q):
            if q_lora_ready is not None:
                stream_q.wait_event(q_lora_ready)
            q = self._compute_q_npu(c4_indexer, q_lora, forward_batch.positions)
            q.record_stream(stream_q)

        cur.wait_stream(stream_w)
        cur.wait_stream(stream_q)
        return q, weights

    def _forward_indexer(
        self,
        c4_indexer,
        x: torch.Tensor,
        q: torch.Tensor,
        weights: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        ratio = c4_indexer.compressor.ratio
        device = x.device
        bs = x.shape[0]
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
        )

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

        # bf16 fallback: per-request einsum + topk, slow but architecture-faithful
        seqlens_cpu = forward_batch.seq_lens_cpu
        end_pos = forward_batch.seq_lens.cumsum(dim=0)
        page_table = self.forward_metadata.c4_page_table
        attn_tp_size = get_parallel().attn_tp_size
        topk_idxs: list[torch.Tensor] = []
        for i, _end_token in enumerate(end_pos):
            seq_i = int(seqlens_cpu[i])
            kv_indices = _get_kv_indices(
                forward_batch,
                seq_i // ratio,
                page_table,
                i,
                seq_i // ratio,
                page_size=self.page_size // ratio,
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
                    get_parallel().attn_tp_group.all_reduce(index_score)
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

        bs = q_lora.shape[0]
        q, _ = c4_indexer.wq_b(q_lora)
        q = q.view(bs, c4_indexer.n_local_heads, c4_indexer.head_dim)
        qk_nope = c4_indexer.head_dim - c4_indexer.rope_head_dim
        # Position-gathered RoPE values are forward-local.  The rotary embedding
        # object is shared, so retaining them there can leak target positions into
        # NextN (or a previous graph replay) when the next batch has the same shape.
        cos4, sin4 = Dsv4NpuRoPE.for_freqs(
            c4_indexer.freqs_cis, getattr(c4_indexer, "rotary_emb", None)
        ).get_cos_sin(
            positions,
            q.dtype,
            view_4d=True,
            allow_build=False,
            cache_dtype=torch.float32,
        )
        Dsv4NpuRoPE.apply_rotary_mul_inplace(
            q,
            None,
            cos4,
            sin4,
            qk_nope_dim=qk_nope,
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
        assert (
            not skip_compressor
        ), "skip_compressor=True is not supported on the NPU indexer path"
        self._ensure_npu_c4_indexer(c4_indexer, x.device)
        if self._can_use_indexer_multi_stream():
            q, weights = self._forward_prepare_multi_stream(
                c4_indexer, x, q_lora, forward_batch, q_lora_ready
            )
        else:
            q, weights = self._forward_prepare(c4_indexer, x, q_lora, forward_batch)
        topk_idxs = self._forward_indexer(c4_indexer, x, q, weights, forward_batch)
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
        self.use_graph_swa_mask = False
        cfg = model_runner.model_config
        self._dsv4_config = cfg
        tp_size = get_parallel().attn_tp_size
        self._dsv4_q_head_num = cfg.num_attention_heads // tp_size
        self._dsv4_kv_head_num = 1  # V4 MQA / latent
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
        self._is_dspark_algorithm = bool(
            model_runner.spec_algorithm is not None
            and model_runner.spec_algorithm.is_dspark()
        )
        self._is_dspark_draft_worker = bool(
            getattr(model_runner, "is_draft_worker", False)
            and self._is_dspark_algorithm
        )
        self._dsv4_graph_tokens_per_req = int(model_runner.decode_num_tokens_per_req())
        self._dsv4_state_pools_by_ratio = {
            pool.ratio: pool
            for pool in self.token_to_kv_pool.compress_state_pools
            if pool is not None
        }

    def _is_dspark_draft_block(self, forward_batch: ForwardBatch) -> bool:
        spec_algorithm = forward_batch.spec_algorithm
        return (
            self._is_dspark_draft_worker
            and forward_batch.forward_mode.is_target_verify()
            and spec_algorithm is not None
            and spec_algorithm.is_dspark()
        )

    def _init_dspark_sparse_metadata(self, forward_batch: ForwardBatch) -> None:
        """Build block-noncausal SWA slot ids for a DSpark draft forward.

        Every token in a DSpark draft block attends to the trailing SWA
        context and to the whole current draft block.  The Ascend
        sparse-attention operator consumes physical SWA slot ids with shape
        [T, N_kv, K], where K must be 128-aligned.
        """
        fm = self.forward_metadata
        fm.ori_sparse_indices = None
        fm.ori_win_left = self._dsv4_sliding_window_size - 1
        fm.ori_win_right = 0

        if not self._is_dspark_draft_block(forward_batch):
            return

        block_size = int(forward_batch.spec_info.draft_token_num)
        out_cache_loc = forward_batch.out_cache_loc

        ori_sparse_indices = self._build_dspark_sparse_indices(
            seq_lens=forward_batch.seq_lens,
            req_pool_indices=forward_batch.req_pool_indices,
            out_cache_loc=out_cache_loc,
            block_size=block_size,
        )
        ori_sparse_indices = ori_sparse_indices.unsqueeze(1).contiguous()

        fm.ori_sparse_indices = ori_sparse_indices
        fm.ori_win_left = self._dsv4_sliding_window_size + block_size - 1
        fm.ori_win_right = 0

    def _build_dspark_sparse_indices(
        self,
        *,
        seq_lens: torch.Tensor,
        req_pool_indices: torch.Tensor,
        out_cache_loc: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Return [bs * block_size, K] physical SWA slots for one draft block.

        This helper is deliberately allocation-producing.  Eager forwards use
        the returned tensor directly; graph replay copies it into the stable
        graph-owned ``ori_sparse_indices`` storage.
        """
        bs = int(seq_lens.shape[0])
        expected_tokens = bs * block_size

        seq_lens_causal = BuildBlockSeqLensCausal.execute(
            seq_lens=seq_lens,
            block_size=block_size,
            device=seq_lens.device,
        )
        req_pool_indices_repeated = req_pool_indices.repeat_interleave(block_size)
        gather = ComputeDsparkWindowGather.execute(
            seq_lens_casual=seq_lens_causal,
            req_pool_indices_repeated=req_pool_indices_repeated,
            block_size=block_size,
            swa_window=self._dsv4_sliding_window_size,
        )
        ori_sparse_indices, _ = BuildDsparkSwaPageIndices.execute(
            req_to_token=self.req_to_token,
            full_to_swa_mapping=self.token_to_kv_pool.full_to_swa_index_mapping,
            req_pool_indices_per_request=gather.req_pool_indices_per_request,
            offsets=gather.offsets,
            invalid=gather.invalid,
            out_loc=out_cache_loc[:expected_tokens],
            context_lens=gather.context_lens,
            block_size=block_size,
            swa_window=self._dsv4_sliding_window_size,
            page_index_aligned_size=128,
        )
        return ori_sparse_indices

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

        # 1024 int32 per kernel-metadata buffer (fixed op metadata size)
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

        if self._is_dspark_draft_worker:
            block_size = self._dsv4_graph_tokens_per_req
            sparse_width = (
                (self._dsv4_sliding_window_size + block_size + 127) // 128 * 128
            )
            self.graph_metadata["ori_sparse_indices"] = torch.full(
                (
                    max_bs * block_size,
                    self._dsv4_kv_head_num,
                    sparse_width,
                ),
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
            tokens_per_req = self._dsv4_graph_tokens_per_req
        else:
            tokens_per_req = 1

        metadata.actual_seq_lengths_q_pa = torch.arange(
            0,
            bs * tokens_per_req + tokens_per_req,
            tokens_per_req,
            dtype=torch.int32,
            device=device,
        )
        # q_pa is constant per graph shape (never rewritten at replay), so the
        # CPU mirror for the host metadata op is built once here.
        metadata.actual_seq_lengths_q_pa_cpu = torch.arange(
            0,
            bs * tokens_per_req + tokens_per_req,
            tokens_per_req,
            dtype=torch.int32,
        )

        # init >=1 so the captured kernel records valid attention work; replay overwrites in-place
        metadata.actual_seq_lengths_kv = torch.ones(
            bs,
            dtype=torch.int32,
            device=device,
        )

        metadata.swa_page_table = self.graph_metadata["swa_page_table"][:bs, :]
        metadata.c4_page_table = self.graph_metadata["c4_page_table"][:bs, :]
        metadata.c128_page_table = self.graph_metadata["c128_page_table"][:bs, :]

        n_tok = bs * tokens_per_req
        c4_pad = min(n_tok, n_tok // 4 + bs)
        c128_pad = min(n_tok, n_tok // 128 + bs)
        metadata.swa_loc = torch.zeros(n_tok, dtype=torch.int64, device=device)
        metadata.c4_loc = torch.zeros(c4_pad, dtype=torch.int64, device=device)
        metadata.c128_loc = torch.zeros(c128_pad, dtype=torch.int64, device=device)
        metadata.dsv4_max_input_capacity = tokens_per_req
        metadata.dsv4_explicit_state_block_tables = {
            ratio: torch.full(
                (
                    bs,
                    (2 if ratio == 4 else 1) * ratio + tokens_per_req,
                ),
                state_pool.dummy_state_loc,
                dtype=torch.int32,
                device=device,
            )
            for ratio, state_pool in self._dsv4_state_pools_by_ratio.items()
        }

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

        T = bs * tokens_per_req
        metadata.c4_topk_indices = self.graph_metadata["c4_topk_indices"][:T, :]

        metadata.ori_sparse_indices = None
        metadata.ori_win_left = self._dsv4_sliding_window_size - 1
        metadata.ori_win_right = 0
        if self._is_dspark_draft_worker and forward_mode.is_target_verify():
            metadata.ori_sparse_indices = self.graph_metadata["ori_sparse_indices"][:T]
            metadata.ori_sparse_indices.fill_(-1)
            metadata.ori_sparse_indices[:, :, 0] = 0
            metadata.ori_win_left = self._dsv4_sliding_window_size + tokens_per_req - 1

        self.forward_metadata = metadata

    @staticmethod
    def _copy_2d_with_tail(dst: torch.Tensor, src: torch.Tensor, val: int) -> None:
        # Graph replay metadata buffers are sliced to the active bs; only the
        # page-column tail needs the sentinel refresh.
        r, c = src.shape
        dst[:r, :c].copy_(src)
        if c < dst.shape[1]:
            dst[:, c:].fill_(val)

    @staticmethod
    def _copy_1d_with_zero_tail(dst: torch.Tensor, src: Optional[torch.Tensor]) -> None:
        if src is None:
            dst.zero_()
            return
        n = src.numel()
        assert (
            n <= dst.shape[0]
        ), f"graph replay 1D metadata overflow: src={n} > dst={dst.shape[0]}"
        if n > 0:
            if src.dtype != dst.dtype:
                src = src.to(dst.dtype)
            dst[:n].copy_(src)
        if n < dst.shape[0]:
            dst[n:].fill_(0)

    def _build_dsv4_graph_replay_ctx(self, forward_batch: ForwardBatch):
        graph_mode = forward_batch.forward_mode
        runtime_mode = getattr(forward_batch, "actual_forward_mode", None) or graph_mode
        bs = forward_batch.batch_size
        num_padding = int(getattr(forward_batch, "num_padding", 0) or 0)

        raw_bs = bs - num_padding
        seq_lens_cpu = forward_batch.seq_lens_cpu
        assert seq_lens_cpu is not None, "V4 graph replay requires seq_lens_cpu."

        device = forward_batch.seq_lens.device
        tokens_per_bs = (
            self._dsv4_graph_tokens_per_req
            if graph_mode.is_target_verify() or graph_mode.is_draft_extend_v2()
            else 1
        )

        raw_seq_lens_cpu = seq_lens_cpu[:bs]
        is_idle_replay = runtime_mode.is_idle()
        if graph_mode.is_target_verify():
            explicit_live_cpu = getattr(
                getattr(forward_batch, "spec_info", None),
                "live_seq_lens_cpu",
                None,
            )
            if is_idle_replay:
                live_seq_lens_cpu = torch.zeros_like(raw_seq_lens_cpu)
                final_seq_lens_cpu = live_seq_lens_cpu
            elif self._is_dspark_algorithm or explicit_live_cpu is not None:
                # DSpark/DFLASH temporarily expand seq_lens_cpu to the final
                # target-verify length and carry the committed/live prefix
                # separately. Graph replay batches are lightweight namespace
                # views and do not necessarily retain spec_algorithm.
                final_seq_lens_cpu = raw_seq_lens_cpu
                if explicit_live_cpu is None:
                    live_seq_lens_cpu = torch.clamp(
                        final_seq_lens_cpu - int(tokens_per_bs), min=0
                    )
                else:
                    explicit_live_cpu = torch.as_tensor(
                        explicit_live_cpu,
                        dtype=final_seq_lens_cpu.dtype,
                        device=final_seq_lens_cpu.device,
                    ).flatten()
                    live_seq_lens_cpu = torch.zeros_like(final_seq_lens_cpu)
                    num_live_rows = min(bs, explicit_live_cpu.numel())
                    if num_live_rows > 0:
                        live_seq_lens_cpu[:num_live_rows].copy_(
                            explicit_live_cpu[:num_live_rows]
                        )
            else:
                # EAGLE and the other uniform verify callers keep
                # seq_lens_cpu at the committed/live prefix length.
                live_seq_lens_cpu = raw_seq_lens_cpu
                final_seq_lens_cpu = live_seq_lens_cpu + int(tokens_per_bs)
            live_seq_lens = live_seq_lens_cpu.to(device=device, dtype=torch.int32)
        elif (
            forward_batch.seq_lens is not None
            and forward_batch.seq_lens.device.type != "cpu"
        ):
            live_seq_lens_cpu = seq_lens_cpu[:bs]
            final_seq_lens_cpu = live_seq_lens_cpu
            live_seq_lens = forward_batch.seq_lens[:bs].to(dtype=torch.int32)
        else:
            live_seq_lens_cpu = seq_lens_cpu[:bs]
            final_seq_lens_cpu = live_seq_lens_cpu
            live_seq_lens = live_seq_lens_cpu.to(device=device, dtype=torch.int32)
        has_compress = self._dsv4_has_c4 or self._dsv4_has_c128
        active_target_verify = (
            graph_mode.is_target_verify() and not is_idle_replay and has_compress
        )
        compress_seq_lens = live_seq_lens
        compress_seq_lens_max = int(final_seq_lens_cpu.max()) if bs > 0 else 0
        if active_target_verify:
            compress_seq_lens = final_seq_lens_cpu.to(device=device, dtype=torch.int32)

        return SimpleNamespace(
            forward_batch=forward_batch,
            fm=self.forward_metadata,
            graph_mode=graph_mode,
            runtime_mode=runtime_mode,
            is_idle_replay=is_idle_replay,
            has_compress=has_compress,
            active_target_verify=active_target_verify,
            bs=bs,
            raw_bs=raw_bs,
            tokens_per_bs=tokens_per_bs,
            device=device,
            seq_lens_cpu=seq_lens_cpu,
            final_seq_lens_cpu=final_seq_lens_cpu,
            live_seq_lens_cpu=live_seq_lens_cpu,
            live_seq_lens=live_seq_lens,
            compress_seq_lens=compress_seq_lens,
            compress_seq_lens_max=compress_seq_lens_max,
        )

    def _refresh_graph_seq_metadata(self, ctx) -> None:
        fm = ctx.fm
        attn_seq_lens = ctx.live_seq_lens
        if ctx.graph_mode.is_target_verify():
            valid_verify_rows = ctx.live_seq_lens > 0
            final_seq_lens = ctx.final_seq_lens_cpu.to(
                device=ctx.device, dtype=torch.int32
            )
            attn_seq_lens = torch.where(
                valid_verify_rows, final_seq_lens, ctx.live_seq_lens
            )
            fm.seq_lens_cpu_int = ctx.final_seq_lens_cpu.int()
            fm.seq_lens_cpu_int = torch.where(
                ctx.live_seq_lens_cpu > 0,
                fm.seq_lens_cpu_int,
                ctx.live_seq_lens_cpu.int(),
            )
        else:
            # CPU mirror of the kv buffer written below, from its CPU source — the
            # host metadata op reads this instead of a D2H sync.
            fm.seq_lens_cpu_int = ctx.final_seq_lens_cpu[: ctx.bs].int().clamp(min=1)
        fm.actual_seq_lengths_kv.copy_(attn_seq_lens.clamp(min=1))

    def _refresh_graph_compress_page_tables_direct(self, ctx) -> None:
        result = self._compute_compress_locs(
            pool=self.token_to_kv_pool,
            req_to_token=self.req_to_token,
            req_pool_indices=ctx.forward_batch.req_pool_indices[: ctx.bs],
            seq_lens=ctx.compress_seq_lens,
            out_cache_loc=ctx.forward_batch.out_cache_loc,
            is_decode=False,
            bs=ctx.bs,
            device=ctx.device,
            req_to_token_pool=self.req_to_token_pool,
            out_cache_loc_dsv4=getattr(ctx.forward_batch, "out_cache_loc_dsv4", None),
            is_graph=True,
            seq_lens_max_override=ctx.compress_seq_lens_max,
        )
        for key in ("c4_page_table", "c128_page_table"):
            if key in result:
                self._copy_2d_with_tail(getattr(ctx.fm, key), result[key], -1)

    def _refresh_graph_decode_compress_1d_direct(self, ctx) -> None:
        fm = ctx.fm
        bundle = getattr(ctx.forward_batch, "out_cache_loc_dsv4", None)
        for ratio in self._dsv4_unique_compress_ratios:
            if ratio not in (4, 128):
                continue
            loc = None
            if bundle is not None:
                loc = bundle.out_c4_loc if ratio == 4 else bundle.out_c128_loc
            self._copy_1d_with_zero_tail(getattr(fm, f"c{ratio}_loc"), loc)

        valid = ctx.live_seq_lens > 0
        positions_last = torch.clamp(ctx.live_seq_lens - 1, min=0)
        for ratio in self._dsv4_unique_compress_ratios:
            if ratio not in (4, 128):
                continue
            should_compress = ((ctx.live_seq_lens % ratio) == 0) & valid
            pos_cmp = positions_last[should_compress].to(torch.int64) + (1 - ratio)
            self._copy_1d_with_zero_tail(
                getattr(fm, f"positions_cmp_padding_c{ratio}"), pos_cmp
            )
        fm.start_pos.copy_(positions_last.to(torch.int32))
        fm.seqused.copy_(valid.to(torch.int32))

    def _refresh_graph_target_verify_compress_1d_direct(self, ctx) -> None:
        fm = ctx.fm
        verify_seq_lens_cpu = ctx.final_seq_lens_cpu
        verify_seq_lens_cpu = torch.where(
            ctx.live_seq_lens_cpu > 0,
            verify_seq_lens_cpu,
            ctx.live_seq_lens_cpu,
        )
        self._fill_verify_positions_cmp_padding_one(
            ctx.forward_batch.positions,
            fm.positions_cmp_padding_c4,
            4,
            verify_seq_lens_cpu,
            n_draft=ctx.tokens_per_bs,
        )
        self._fill_verify_positions_cmp_padding_one(
            ctx.forward_batch.positions,
            fm.positions_cmp_padding_c128,
            128,
            verify_seq_lens_cpu,
            n_draft=ctx.tokens_per_bs,
        )
        fm.start_pos.copy_(ctx.live_seq_lens.to(torch.int32))
        valid = ctx.live_seq_lens[: ctx.bs] > 0
        fm.seqused.copy_(
            (valid.to(torch.int32) * int(ctx.tokens_per_bs)).to(device=ctx.device)
        )
        bundle = getattr(ctx.forward_batch, "out_cache_loc_dsv4", None)
        if bundle is None:
            return
        for ratio in self._dsv4_unique_compress_ratios:
            if ratio not in (4, 128):
                continue
            loc = bundle.out_c4_loc if ratio == 4 else bundle.out_c128_loc
            self._copy_1d_with_zero_tail(getattr(fm, f"c{ratio}_loc"), loc)

    @staticmethod
    def _clear_graph_target_verify_metadata(ctx) -> None:
        fm = ctx.fm
        for tensor in (
            fm.positions_cmp_padding_c4,
            fm.positions_cmp_padding_c128,
            fm.c4_loc,
            fm.c128_loc,
        ):
            if tensor is not None:
                tensor.zero_()
        fm.start_pos.zero_()
        fm.seqused.zero_()

    def _refresh_graph_explicit_state_block_tables(self, ctx) -> None:
        fm = ctx.fm
        for ratio, fixed_table in fm.dsv4_explicit_state_block_tables.items():
            fixed_table.copy_(
                _build_explicit_state_block_table(
                    compress_ratio=ratio,
                    coff=2 if ratio == 4 else 1,
                    state_pool=self._dsv4_state_pools_by_ratio[ratio],
                    token_to_kv_pool=self.token_to_kv_pool,
                    req_to_token=self.req_to_token,
                    req_pool_indices=ctx.forward_batch.req_pool_indices[: ctx.bs],
                    start_pos=fm.start_pos,
                    cu_seqlens=fm.actual_seq_lengths_q_pa,
                    seqused=fm.seqused,
                    max_input_capacity=fm.dsv4_max_input_capacity,
                )
            )

    def _refresh_graph_swa_metadata_direct(self, ctx) -> None:
        fm = ctx.fm
        swa_loc = self.token_to_kv_pool.translate_loc_from_full_to_swa(
            ctx.forward_batch.out_cache_loc
        ).to(torch.int64)
        self._copy_1d_with_zero_tail(fm.swa_loc, swa_loc)

        swa_src = (
            fm.block_tables_swa if fm.block_tables_swa is not None else fm.block_tables
        )
        if ctx.bs > 0:
            max_len = int(ctx.final_seq_lens_cpu.max())
            max_seq_pages = (max_len + self.page_size - 1) // self.page_size
            if 0 < max_seq_pages < swa_src.shape[1]:
                swa_src = swa_src[:, :max_seq_pages]
        self._copy_2d_with_tail(fm.swa_page_table, swa_src, -1)

    def _refresh_graph_dspark_sparse_metadata(self, ctx) -> None:
        if not (self._is_dspark_draft_worker and ctx.graph_mode.is_target_verify()):
            return

        dst = getattr(ctx.fm, "ori_sparse_indices", None)
        if dst is None:
            raise RuntimeError(
                "DSpark NPU graph replay is missing its captured "
                "ori_sparse_indices buffer."
            )

        if ctx.is_idle_replay:
            dst.fill_(-1)
            dst[:, :, 0] = 0
            return

        src = self._build_dspark_sparse_indices(
            # ``out_cache_loc`` is deliberately left unpadded by the graph
            # replay view.  Build sparse indices for real requests only, then
            # populate the remaining rows of the captured buffer below.
            seq_lens=ctx.live_seq_lens[: ctx.raw_bs],
            req_pool_indices=ctx.forward_batch.req_pool_indices[: ctx.raw_bs],
            out_cache_loc=ctx.forward_batch.out_cache_loc,
            block_size=ctx.tokens_per_bs,
        ).unsqueeze(1)
        if (
            src.ndim != dst.ndim
            or src.shape[0] > dst.shape[0]
            or src.shape[1:] != dst.shape[1:]
        ):
            raise RuntimeError(
                "DSpark NPU graph sparse-index shape mismatch: "
                f"runtime={tuple(src.shape)}, captured={tuple(dst.shape)}."
            )
        # A graph bucket can contain padded request rows.  Keep those rows
        # valid for capture/replay while replacing only the live prefix.
        dst.fill_(-1)
        dst[: src.shape[0]].copy_(src)
        if src.shape[0] < dst.shape[0]:
            dst[src.shape[0] :, :, 0] = 0

    def _refresh_graph_kernel_metadata(self, ctx) -> None:
        fm = ctx.fm
        kernel_metadata_new = self._kernel_metadata_from_parts(
            bs=ctx.bs,
            actual_seq_lengths_q_pa=fm.actual_seq_lengths_q_pa,
            actual_seq_lengths_kv=fm.actual_seq_lengths_kv,
            block_tables=fm.block_tables,
            max_seqlen_q=ctx.tokens_per_bs,
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

    def _apply_dsv4_graph_metadata(self, forward_batch: ForwardBatch) -> None:
        ctx = self._build_dsv4_graph_replay_ctx(forward_batch)

        self._refresh_graph_seq_metadata(ctx)
        self._refresh_graph_compress_page_tables_direct(ctx)

        if ctx.graph_mode.is_decode():
            self._refresh_graph_decode_compress_1d_direct(ctx)
        elif ctx.graph_mode.is_target_verify():
            if ctx.is_idle_replay and ctx.has_compress:
                self._clear_graph_target_verify_metadata(ctx)
            elif ctx.active_target_verify:
                self._refresh_graph_target_verify_compress_1d_direct(ctx)

        self._refresh_graph_explicit_state_block_tables(ctx)

        self._refresh_graph_swa_metadata_direct(ctx)
        self._refresh_graph_dspark_sparse_metadata(ctx)
        self._refresh_graph_kernel_metadata(ctx)

        self.forward_metadata = ctx.fm

    def init_forward_metadata(self, forward_batch: ForwardBatch) -> None:
        super().init_forward_metadata(forward_batch)
        fm = self.forward_metadata

        # Idle DP-attention ranks have zero seq_lens, which the metadata kernel
        # cannot handle; skip it and leave the fields cleared but well-typed.
        if forward_batch.forward_mode.is_idle():
            fm.actual_seq_lengths_q = None
            fm.actual_seq_lengths_q_pa = None
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
            fm.actual_seq_lengths_q_pa_cpu = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32),
                    torch.cumsum(seq_lens_cpu, dim=0).int(),
                ],
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
            fm.actual_seq_lengths_q_pa_cpu = torch.arange(0, B + 1, dtype=torch.int32)
        elif (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            B = forward_batch.batch_size
            n_draft = (
                int(forward_batch.spec_info.draft_token_num)
                if forward_batch.forward_mode.is_target_verify()
                else self.speculative_num_draft_tokens
            )
            actual_q = torch.arange(
                n_draft, B * n_draft + 1, n_draft, dtype=torch.int32, device=device
            )
            fm.actual_seq_lengths_q = actual_q
            fm.actual_seq_lengths_q_pa = torch.cat(
                [torch.zeros(1, dtype=torch.int32, device=device), actual_q],
                dim=0,
            )
            fm.actual_seq_lengths_q_pa_cpu = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32),
                    torch.arange(n_draft, B * n_draft + 1, n_draft, dtype=torch.int32),
                ],
                dim=0,
            )
        else:
            fm.actual_seq_lengths_q = None
            fm.actual_seq_lengths_q_pa = None
            fm.actual_seq_lengths_q_pa_cpu = None

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

        self._init_dspark_sparse_metadata(forward_batch)
        fm.kernel_metadata = self._compute_kernel_metadata(forward_batch)

        if self._dsv4_compress_ratios:
            self._build_npu_compress_metadata(forward_batch)

    def _compute_kernel_metadata(self, forward_batch: ForwardBatch) -> dict:
        fm = self.forward_metadata
        if (
            forward_batch.forward_mode.is_target_verify()
            or forward_batch.forward_mode.is_draft_extend_v2()
        ):
            max_seqlen_q = (
                int(forward_batch.spec_info.draft_token_num)
                if forward_batch.forward_mode.is_target_verify()
                else self.speculative_num_draft_tokens
            )
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
        fm = self.forward_metadata
        common = {
            "cmp_ratio": 1,
            "ori_mask_mode": 4,
            "cmp_mask_mode": 3,
            "ori_win_left": getattr(
                fm, "ori_win_left", self._dsv4_sliding_window_size - 1
            ),
            "ori_win_right": getattr(fm, "ori_win_right", 0),
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
        # The host metadata op reads CPU int32 mirrors — never a D2H sync of the
        # device tensors (that would drain the stream and stall overlapped prep).
        c1a_kwargs = base_kwargs | common
        if self._is_dspark_draft_worker:
            cu_q_cpu = fm.actual_seq_lengths_q_pa_cpu
            if cu_q_cpu is not None and cu_q_cpu.numel() > bs + 1:
                cu_q_cpu = cu_q_cpu[: bs + 1]
            host_inputs = {"seqused_kv": fm.seq_lens_cpu_int[:bs].int()}
            if cu_q_cpu is not None:
                host_inputs["cu_seqlens_q"] = cu_q_cpu
            c1a_kwargs = c1a_kwargs | host_inputs
            metadata_op = torch.ops.npu.sparse_attn_sharedkv_metadata_host
        else:
            # The device-side op requires tensor args for backend dispatch; pass
            # the device mirrors just like the pre-refactor call did.
            c1a_kwargs = c1a_kwargs | {
                "cu_seqlens_q": actual_seq_lengths_q_pa,
                "seqused_kv": actual_seq_lengths_kv,
            }
            metadata_op = torch.ops.custom.npu_sparse_attn_sharedkv_metadata
        kernel_metadata = {"c1a_metadata": metadata_op(**c1a_kwargs)}

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
                # the indexer metadata op wants a fresh contiguous tensor without the leading 0
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
        # idle ranks only feed the MoE collectives; skip attn + store_cache and return zeros
        if forward_batch.forward_mode.is_idle():
            return torch.zeros_like(q)
        # MQALayer prepass already stores K and passes save_kv_cache=False; True callers still get the write
        if save_kv_cache:
            self.store_cache(
                layer_id=layer.layer_id, swa_k=k, forward_batch=forward_batch
            )
        if compress_ratio == 0:
            return self._forward_swa(q, layer, forward_batch, attn_sink)
        return self._forward_compressed(
            q, layer, forward_batch, attn_sink, compress_ratio
        )

    def _forward_swa(
        self,
        q: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        attn_sink: Optional[torch.Tensor],
    ) -> torch.Tensor:
        fm = self.forward_metadata
        pool = self.token_to_kv_pool
        ori_kv = pool.get_swa_buffer(layer.layer_id)

        attn_kwargs = dict(
            cu_seqlens_q=fm.actual_seq_lengths_q_pa,
            seqused_kv=fm.actual_seq_lengths_kv,
            ori_mask_mode=4,
            ori_win_left=getattr(
                fm, "ori_win_left", self._dsv4_sliding_window_size - 1
            ),
            ori_win_right=getattr(fm, "ori_win_right", 0),
            layout_q="TND",
            layout_kv="PA_ND",
            q=q,
            ori_kv=ori_kv,
            ori_block_table=fm.swa_page_table,
            sinks=attn_sink,
            metadata=fm.kernel_metadata["c1a_metadata"],
            softmax_scale=layer.scaling,
            cmp_ratio=1,
        )
        if self._is_dspark_draft_worker:
            attn_kwargs["cu_seqlens_ori_kv"] = fm.actual_seq_lengths_q_pa
        ori_sparse_indices = getattr(fm, "ori_sparse_indices", None)
        if ori_sparse_indices is not None:
            attn_kwargs["ori_sparse_indices"] = ori_sparse_indices
        q_arg = attn_kwargs.pop("q")
        out, _ = torch.ops.npu.sparse_attn_sharedkv(q_arg, **attn_kwargs)
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
        expected_cmp_page_size = (
            ori_page_size // 4
            if compress_ratio == 4
            else pool.c128_kv_pool.kernel_page_size
        )
        assert cmp_native_page_size == expected_cmp_page_size, (
            f"c{compress_ratio} page_size={cmp_native_page_size} != "
            f"expected={expected_cmp_page_size} for ori page_size={ori_page_size}; "
            "C4 and C128 must use their configured native page layouts "
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
        # c4 attends via indexer topk; c128 reads the full compressed history
        if compress_ratio == 4:
            topk = fm.c4_topk_indices
            attn_kwargs["cmp_sparse_indices"] = topk.view(-1, 1, topk.shape[-1])
        else:
            attn_kwargs["cmp_sparse_indices"] = None
        q_arg = attn_kwargs.pop("q")
        out, _ = torch.ops.npu.sparse_attn_sharedkv(q_arg, **attn_kwargs)
        return out

    def get_swa_out_cache_loc(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Return the SWA KV write locations used by DeepSeek-V4 draft layers.

        During NPU graph capture/replay, ``swa_loc`` is stable graph storage
        whose contents are refreshed by ``_apply_dsv4_graph_metadata``. Eager
        forwards do not necessarily build that metadata, so translate the
        current full-pool locations on demand as a fallback.
        """
        out_cache_loc = forward_batch.out_cache_loc
        metadata = self.forward_metadata
        cached = getattr(metadata, "swa_loc", None)
        if (
            cached is not None
            and not forward_batch.forward_mode.is_idle()
            and cached.shape[0] == out_cache_loc.shape[0]
        ):
            return cached
        return self.token_to_kv_pool.translate_loc_from_full_to_swa(out_cache_loc).to(
            torch.int64
        )

    def store_cache(self, *, layer_id: int, swa_k: torch.Tensor, forward_batch):
        pool = self.token_to_kv_pool
        swa_loc = self.get_swa_out_cache_loc(forward_batch)
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
        n_draft = int(forward_batch.spec_info.draft_token_num)
        # The parent backend normalizes this to final KV lengths for every
        # algorithm: it adds n_draft for EAGLE/NGRAM, while DSpark/DFLASH
        # already pass expanded lengths and are not incremented again.
        verify_seq_lens_cpu = fm.seq_lens_cpu_int[:bs]
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
        fm.dsv4_max_input_capacity = max(1, n_draft)
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

    def _fill_verify_positions_cmp_padding_one(
        self,
        positions: torch.Tensor,
        dst: torch.Tensor,
        ratio: int,
        seq_lens_cpu: torch.Tensor,
        n_draft: int,
    ) -> None:
        dst.zero_()
        if ratio not in self._dsv4_compress_ratios or positions.numel() == 0:
            return

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

        n_draft = int(spec_info.draft_token_num)
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
    page_size: Optional[int] = None,
) -> torch.Tensor:
    logic_start = max(0, seqlen - kv_len)
    logic_end = seqlen
    if page_size is None:
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
