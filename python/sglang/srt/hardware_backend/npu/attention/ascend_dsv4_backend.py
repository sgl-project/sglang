from __future__ import annotations

import functools
import logging
import math
import os
from contextlib import contextmanager
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
from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE, rope_cos_sin
from sglang.srt.hardware_backend.npu.utils import is_npu_arch35
from sglang.srt.layers.cp.base import get_cp_strategy
from sglang.srt.model_executor.forward_batch_info import DSV4OutCacheLoc, ForwardMode
from sglang.srt.model_executor.forward_context import get_attn_backend
from sglang.srt.runtime_context import get_parallel
from sglang.srt.state_capturer.indexer_topk import maybe_capture_indexer_topk

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


# A5 kv-quant KV layout: nope is quantized in groups of 64 and the RoPE half is
# stored unquantized, so the kernels need both dimensions spelled out.
_NPU_ARCH35_KV_TILE_SIZE = 64
_NPU_ARCH35_KV_ROPE_HEAD_DIM = 64


def _sparse_attn_ops(is_dspark=False):
    """(metadata op, attention op) for the DSV4 shared-KV sparse attention.

    ``is_dspark`` selects the DSpark draft-worker pair: it reads its sequence
    lengths from CPU mirrors (host metadata) and, on A5, uses a dedicated
    quantized sparse-attention kernel rather than a flag on the pre-A5 one.
    """
    if is_npu_arch35():
        if is_dspark:
            from sgl_kernel_npu.operators.attention import (
                kv_quant_sparse_attn_sharedkv,
                kv_quant_sparse_attn_sharedkv_metadata_host,
            )

            return (
                kv_quant_sparse_attn_sharedkv_metadata_host,
                kv_quant_sparse_attn_sharedkv,
            )
        return (
            torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv_metadata,
            torch.ops.custom.npu_kv_quant_sparse_attn_sharedkv,
        )
    if is_dspark:
        return (
            torch.ops.npu.sparse_attn_sharedkv_metadata_host,
            torch.ops.npu.sparse_attn_sharedkv,
        )
    return (
        torch.ops.custom.npu_sparse_attn_sharedkv_metadata,
        torch.ops.custom.npu_sparse_attn_sharedkv,
    )


def _sparse_attn_kv_quant_kwargs() -> dict:
    """Extra kwargs the A5 kv-quant kernels need to interpret the KV layout."""
    if not is_npu_arch35():
        return {}
    return {
        "kv_quant_mode": 1,
        "tile_size": _NPU_ARCH35_KV_TILE_SIZE,
        "rope_head_dim": _NPU_ARCH35_KV_ROPE_HEAD_DIM,
    }


@functools.cache
def _ds41_ops():
    """The `cann_ops_transformer.ops.ds41` module with the V4.1 Python operators.

    Only `compressor`, `indexer_prologue_k`, `quant_lightning_indexer` and
    `quant_sparse_lightning_indexer` are exposed as Python ops (they need
    Python-side argument marshalling); the remaining V4.1 operators go through
    `torch.ops.*` and are called inline.

    `ds41` is NOT in `cann_ops_transformer.ops`'s static registration table, so
    importing this module is also what registers the
    `torch.ops.cann_ops_transformer.ds41.*` namespace (SMLA, mhc_post, the
    `_metadata` variants). Call this once during backend init, before any
    `torch.ops.cann_ops_transformer.ds41.*` use.
    """
    from cann_ops_transformer.ops import ds41

    return ds41


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


def _build_cycle_state_block_table(req_pool_indices: torch.Tensor) -> torch.Tensor:
    """Build the Atlas A5 cache_mode=2 request-bank table.

    A5 interprets this input as one bank id per request and computes the
    in-bank ring offset itself.  It must never receive the A3 explicit
    per-token location table.
    """
    if req_pool_indices.ndim != 1:
        raise ValueError(
            "Atlas A5 compressor requires a 1-D request-bank table, got "
            f"shape={tuple(req_pool_indices.shape)}"
        )
    return req_pool_indices.to(dtype=torch.int32).contiguous()


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
        fm.dsv4_cycle_state_block_table = (
            _build_cycle_state_block_table(forward_batch.req_pool_indices)
            if is_npu_arch35()
            else None
        )
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

        # DeepSeek-V4.1 ratio 1/2 source layers reuse the same three-segment
        # contract, but there is no allocator bundle to read slots from (see
        # LowRatioCompressAscendBackendMixin._build_low_ratio_compress_metadata).
        # V4-only deployments skip this entirely.
        if getattr(self, "_dsv4_has_low_ratio", False):
            self._build_low_ratio_compress_metadata(forward_batch)

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
                col_idx = torch.arange(
                    0,
                    n_c_tokens * ratio,
                    self.page_size,
                    device=req_to_token.device,
                )
                slots = torch.index_select(
                    torch.index_select(req_to_token, 1, col_idx), 0, req_pool_64
                )
                c_page_table = (slots // self.page_size).to(torch.int32)
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

    # The NPU fused compressor writes c4/indexer payloads identically.
    forward_indexer_compressor = forward_core_compressor

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
        if is_npu_arch35():
            # A5 cache_mode=2 is CYCLE: one request bank per row.  The
            # compressor derives the in-bank offset from start_pos; passing
            # the A3 explicit [B, width] table here would be an ABI violation.
            state_block_table = fm.dsv4_cycle_state_block_table
        else:
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

        # TODO: torch.ops.npu.compressor does not support Atlas A5 yet.
        compressor_op = (
            torch.ops.custom.compressor if is_npu_arch35() else torch.ops.npu.compressor
        )
        cmp_kv = compressor_op(
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
        assert w.shape[0] == 2 * split, (
            f"wkv_gate.weight rows={w.shape[0]} != 2*coff*head_dim={2 * split}"
        )
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
        # A5 quantizes and scatters in one fused kernel, so the dequant scale is
        # produced inside indexer_compress_epilog rather than here.
        fused_fp8_indexer_write = li_kv_dtype == "float8" and compressor.is_in_indexer
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

        # Eager verify keeps no row when no request completed a compression block
        # this step (loc is then all-zero, the skip sentinel), and prefill can hand
        # us an empty chunk. Nothing to write: the pre-A5 scatter treated that as a
        # no-op, while both A5 fused epilog kernels reject a zero-row input. Unlike
        # the `loc is None` check below (missing metadata = a bug), an empty write is
        # a legitimate step outcome. Static shape read, so graph capture is unaffected.
        if kv.shape[0] == 0:
            return

        if fused_fp8_indexer_write:
            if loc is None:
                raise RuntimeError(
                    "DSV4 A5 fused indexer epilog needs a slot mapping, but "
                    f"loc is None (mode={forward_batch.forward_mode}, "
                    f"ratio={compressor.ratio}). Writing nothing here would "
                    "leave the indexer KV cache stale."
                )
            torch.ops.custom.indexer_compress_epilog(
                indexer_compress_cache=self.token_to_kv_pool.get_compress_buffer(
                    compressor.layer_id, True
                ),
                indexer_compress_scale=self.token_to_kv_pool.get_compress_dequant_scale_buffer(
                    compressor.layer_id, True
                ),
                x=kv,
                slot_mapping=loc.to(torch.int32),
            )
            return

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
        skip_compressor: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self._compute_q_npu(c4_indexer, q_lora, forward_batch)
        weights, _ = c4_indexer.weights_proj(x)
        weights = weights * (c4_indexer.softmax_scale * c4_indexer.n_heads**-0.5)
        if not skip_compressor:
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
        skip_compressor: bool = False,
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
        if not skip_compressor:
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
            q = self._compute_q_npu(c4_indexer, q_lora, forward_batch)
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
        if li_kv_dtype in ("int8", "float8"):
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
        # A5's lightning indexer consumes FP8 K + fp32 scales; pre-A5 stays int8.
        c4_indexer.compressor.li_kv_dtype = "float8" if is_npu_arch35() else "int8"
        if getattr(c4_indexer, "hadamard_matrix", None) is None:
            H = _walsh_hadamard_matrix(c4_indexer.head_dim, torch.float32, device)
            c4_indexer.register_buffer("hadamard_matrix", H, persistent=False)

    def _compute_q_npu(
        self, c4_indexer, q_lora: torch.Tensor, forward_batch: ForwardBatch
    ) -> torch.Tensor:

        positions = self._cp_local_positions(forward_batch)
        bs = q_lora.shape[0]
        q, _ = c4_indexer.wq_b(q_lora)
        q = q.view(bs, c4_indexer.n_local_heads, c4_indexer.head_dim)
        qk_nope = c4_indexer.head_dim - c4_indexer.rope_head_dim
        # Per-forward memo keyed on the c4 layers' freqs_cis (the indexer
        # shares it), so every c4 layer reads one gather instead of its own.
        cos4, sin4 = rope_cos_sin(
            c4_indexer.freqs_cis,
            getattr(c4_indexer, "rotary_emb", None),
            forward_batch,
            positions,
            q.dtype,
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
        import torch_npu

        if k.dtype == torch.float8_e4m3fn:
            # A5: block-quantize Q to FP8 so it matches the FP8 K buffer; scales
            # stay fp32 and the kernel wants one scale per (token, head).
            q_quant, q_scale = torch_npu.npu_dynamic_block_quant(
                q.view(-1, q.shape[-1]), dst_type=k.dtype
            )
            q_quant = q_quant.view(-1, c4_indexer.n_heads, c4_indexer.head_dim)
            q_scale = q_scale.view(-1, c4_indexer.n_heads)
        else:
            q_quant, q_scale = torch_npu.npu_dynamic_quant(q)
            q_scale = q_scale.to(torch.float16)

        fm = self.forward_metadata
        li_quant_metadata = fm.kernel_metadata["li_quant_metadata"]
        kwargs = dict(
            query=q_quant,
            key=k,
            key_dequant_scale=k_scale.squeeze(-2).to(q_scale.dtype),
            actual_seq_lengths_query=fm.actual_seq_lengths_q,
            actual_seq_lengths_key=fm.actual_seq_lengths_kv,
            block_table=fm.c4_page_table,
            layout_query="TND",
            layout_key="PA_BSND",
            weights=weights.to(q_scale.dtype),
            query_dequant_scale=q_scale,
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
        # skip_compressor=True is the CP full-metadata protocol: the compressor
        # already ran via forward_indexer_compressor.
        self._ensure_npu_c4_indexer(c4_indexer, x.device)
        if self._can_use_indexer_multi_stream():
            q, weights = self._forward_prepare_multi_stream(
                c4_indexer, x, q_lora, forward_batch, q_lora_ready, skip_compressor
            )
        else:
            q, weights = self._forward_prepare(
                c4_indexer, x, q_lora, forward_batch, skip_compressor
            )
        topk_idxs = self._forward_indexer(c4_indexer, x, q, weights, forward_batch)
        self.forward_metadata.c4_topk_indices = topk_idxs
        compress_layer_id = self.token_to_kv_pool.layer_mapping[
            c4_indexer.layer_id
        ].compress_layer_id
        maybe_capture_indexer_topk(compress_layer_id, topk_idxs)

    def _cp_local_positions(self, forward_batch: ForwardBatch) -> torch.Tensor:
        """Per-rank positions under CP-v2 (the batch keeps full-length ones)."""
        local = getattr(forward_batch, "dsv4_cp_local_positions", None)
        return local if local is not None else forward_batch.positions


class LowRatioCompressAscendBackendMixin:
    """NPU implementation of the DeepSeek-V4.1 low-ratio (1 / 2) source path.

    Mirrors ``DeepseekV4AttnBackend.forward_low_ratio_sources``
    (layers/attention/deepseek_v4_backend.py L3203-L3281) but keeps its own
    orchestration rather than sharing the CUDA backend's, for the same reason
    :class:`C4IndexerAscendBackendMixin` and
    :class:`CompressorAscendBackendMixin` do: ``deepseek_v4_backend`` imports
    Triton kernels at module scope, and the two generations differ in the
    kernels and the cache row formats, not in the skeleton.

    Deliberately dropped relative to the CUDA source (unreachable on NPU):
      * the context-parallel branch (``_forward_low_ratio_sources_cp``);
      * the prefill-graph branch — :attr:`low_ratio_prefill_graph` stays False.

    Intentional deviation from the CUDA source: seam A hands its pre-RoPE
    ``latent`` to seam B, because ``indexer_prologue_k`` projects the index K
    from that latent (``wk: [index_head_dim, head_dim]``).  CUDA needs no such
    handoff: it writes the index K *inside* its compress step, where the same
    latent already exists (``_low_ratio_write_group`` ->
    ``indexer.forward_wk(latent)``, deepseek_v4_backend.py L3613-L3643),
    whereas on NPU the fused ``indexer_prologue_k`` is a separate op call.

    Each method notes its CUDA anchor; keep them in sync when upstream
    semantics change (see the plan's drift checklist).
    """

    # CUDA-only prefill graph (BCG). The model reads this via
    # getattr(attn_backend, "low_ratio_prefill_graph", False)
    # (deepseek_v4.py L2391) to choose the two-call side-stream contract over
    # the single call, so declaring False is what selects L2397-2403 on NPU.
    low_ratio_prefill_graph = False
    # Written by the model / replay-tail path exactly as on CUDA
    # (deepseek_v4_backend.py L2012 / L2067, deepseek_v4_replay_tail.py L637).
    tail_source_kv_layer_id: Optional[int] = None

    def forward_low_ratio_sources(
        self,
        *,
        layer,
        x,
        q_lora,
        positions,
        forward_batch,
        run_compressor: bool = True,
        run_indexer: bool = True,
    ) -> None:
        """Runs on every ratio 1/2 layer before its attention.

        Anchor: DeepseekV4AttnBackend.forward_low_ratio_sources
        (deepseek_v4_backend.py L3203-L3281).
        """
        if forward_batch.forward_mode.is_idle():
            return
        if forward_batch.encoder_swa_replay:
            run_compressor = False
        if layer.layer_id == self.tail_source_kv_layer_id:
            run_compressor = False

        pos = positions
        latent = None
        if run_compressor and layer.compressor is not None:
            latent = self._low_ratio_compress(layer, x, pos, forward_batch)
        if run_indexer and layer.indexer is not None:
            # Seam B needs the latent only to produce and store the indexer K
            # (a layer that does not own K, e.g. a tail layer whose compressor
            # was skipped, only scores against the stored one), so the check
            # lives there rather than here.
            self._low_ratio_index_topk(layer, latent, x, q_lora, pos, forward_batch)

    def _low_ratio_compress(self, layer, x, pos, forward_batch):
        """Seam A: project, normalize and RoPE one ratio-1/2 compressed latent,
        then write it into the ratio's compressed pool.

        Anchor: DeepseekV4AttnBackend._low_ratio_compress_torch
        (deepseek_v4_backend.py L3531). Returns the pre-RoPE latent for seam B.
        """
        ratio = layer.compress_ratio
        compressor = layer.compressor
        fm = self.forward_metadata
        if ratio == 1:
            # No gate and no pairing for ratio 1; the recipes branch is a plain
            # linear as well (compressor.py L127).
            kv, _ = compressor.project(x)
        else:
            assert is_npu_arch35(), (
                "the fused DeepSeek-V4.1 compressor needs Atlas A5 "
                "(kv_compress_epilog_v2 packing); A3 is not supported."
            )
            # `DeepseekV41Compressor` only builds the fused `wkv_gate` where the
            # CUDA-only fused-compress path is available; on NPU it must expose
            # the two separate projections this call reads. Fail here rather than
            # with an AttributeError on `compressor.wkv` deep inside the forward.
            assert not compressor.use_fused_gate, (
                "DeepSeek-V4.1 ratio-2 compression on NPU needs the unfused "
                "compressor (separate wkv/wgate), but the module has wkv_gate."
            )
            ds41 = _ds41_ops()
            pool = self.token_to_kv_pool
            # Mirrors forward_compress (L453-L517) for the metadata side: A5
            # cache_mode=2 is CYCLE, so the compressor gets one request bank per
            # row and derives the in-bank offset from start_pos.
            # This is the layer's attention compressor, never the nested c4-indexer
            # one, so it always reads the attention state pool. (DeepseekV41Compressor
            # has no `is_in_indexer`; only V4's Compressor does.)
            state_pool = pool._get_state_pool(layer.layer_id, False)
            kv = ds41.compressor(
                x=x.view(-1, x.shape[-1]),
                wkv=compressor.wkv.weight,
                wgate=compressor.wgate.weight,
                seqused=fm.seqused,
                start_pos=fm.start_pos,
                cmp_ratio=ratio,
                cu_seqlens=fm.actual_seq_lengths_q_pa,
                state_cache=state_pool.state_cache_3d,
                state_block_table=fm.dsv4_cycle_state_block_table,
            )
        out_dim = kv.shape[-1]
        kv = torch_npu.npu_rms_norm(kv, compressor.norm.weight, compressor.norm.eps)[0]
        latent = kv.clone()

        from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE

        # DeepseekV41Compressor carries no freqs_cis of its own — unlike the V4
        # Compressor, which is handed the layer's — so the layer's table is the
        # rope for its compressed latents (and for the indexer K/Q below). A
        # compress_ratio != 0 layer already builds that table with
        # compress_rope_theta + YaRN (deepseek_v4.py L1053-L1074), which is the
        # reference model's compress_rotary_emb.
        freqs_cis = layer.freqs_cis
        table = Dsv4NpuRoPE.for_freqs(freqs_cis, getattr(layer, "rotary_emb", None))
        rope_dim = freqs_cis.shape[-1] * 2
        cos, sin = table.get_cos_sin(
            getattr(fm, f"positions_cmp_padding_c{ratio}"),
            kv.dtype,
            view_4d=True,
            # The fp32 table is built at model init (deepseek_v4.py L1287), so a
            # captured decode never runs repeat_interleave inside the graph.
            allow_build=False,
            cache_dtype=torch.float32,
        )
        Dsv4NpuRoPE.apply_rotary_mul_inplace(
            kv.view(-1, 1, out_dim),
            None,
            cos,
            sin,
            qk_nope_dim=out_dim - rope_dim,
        )

        slot = getattr(fm, f"c{ratio}_loc", None)
        if slot is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 ratio {ratio} has no c-pool slot metadata: "
                "the compress-loc builder has not been run for this mode yet."
            )
        self.token_to_kv_pool.set_compress_buffer(layer.layer_id, slot, kv, None, False)
        return latent

    def _low_ratio_index_topk(
        self,
        layer,
        latent,
        x,
        q_lora,
        pos,
        forward_batch,
        *,
        rows_per_request=None,  # NOTE: only used under CP (unsupported here)
    ) -> None:
        """Seam B: store the indexer K from ``latent`` and score it with QLI.

        Anchor: DeepseekV4AttnBackend._low_ratio_index_topk
        (deepseek_v4_backend.py L3657-L3705) for the role dispatch, and
        cann-recipes-infer models/deepseek_v4_1/models/modules/indexer.py
        L270-L400 for the NPU operator ABI.
        """
        indexer = layer.indexer
        ratio = layer.compress_ratio
        fm = self.forward_metadata
        if indexer.owns_k:
            self._dsv41_index_k_store(layer, latent)
        if indexer.is_candidate_source and not os.environ.get(
            "SGLANG_DSV41_DISABLE_QSLI"
        ):
            # Level one: the plain top-k this layer attends with, plus the block
            # candidates every later indexer layer rescoring inside.
            topk, blocks = self._dsv41_index_full_topk(
                layer, x, q_lora, pos, forward_batch, candidate=True
            )
            fm.dsv41_candidate_blocks = blocks
        elif indexer.uses_candidates and not os.environ.get(
            "SGLANG_DSV41_DISABLE_QSLI"
        ):
            topk = self._dsv41_index_qsli_topk(layer, x, q_lora, pos, forward_batch)
        else:
            topk, _ = self._dsv41_index_full_topk(layer, x, q_lora, pos, forward_batch)
        setattr(fm, f"c{ratio}_topk_indices", topk)
        # No indexer-topk capture here: get_num_indexer_layers() sizes that
        # capturer from the ratio-4 layers (model_config.py L482-L484), which
        # DeepSeek-V4.1 has none of, so the capturer is disabled for it -- and
        # compress_layer_id is a per-ratio counter that collides across ratios
        # anyway (both V4.1 sources are index 0).

    def _dsv41_index_k_store(self, layer, latent) -> None:
        """Project, norm, RoPE and quantize the index K straight into the split
        index pool (the cache QLI reads).

        Recipes: indexer.py L270-L279 (``write_prologue_k`` with storage_mode=0).
        """
        if latent is None:
            raise RuntimeError(
                "DeepSeek-V4.1 index K needs the compressor latent, but the "
                f"compressor did not run for layer {layer.layer_id}."
            )
        import torch_npu

        indexer = layer.indexer
        ratio = layer.compress_ratio
        fm = self.forward_metadata
        pool = self.token_to_kv_pool
        # -1 rows are the compressor's padding rows; the operator skips them, so
        # the padded (garbage) latent rows are dropped for free.
        cache_index = getattr(fm, f"c{ratio}_loc", None)
        if cache_index is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 index K for layer {layer.layer_id} has no "
                f"c{ratio}_loc slot metadata to write into."
            )
        positions = getattr(fm, f"positions_cmp_padding_c{ratio}", None)
        if positions is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 index K for layer {layer.layer_id} has no "
                f"positions_cmp_padding_c{ratio} metadata."
            )
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE

        # indexer_prologue_k wants flat FP32 [T, Dr] RoPE tensors, unlike
        # inplace_partial_rotary_mul's [T, 1, 1, Dr] bf16.
        cos, sin = Dsv4NpuRoPE.for_freqs(
            layer.freqs_cis, getattr(layer, "rotary_emb", None)
        ).get_cos_sin(
            positions,
            torch.float32,
            view_4d=False,
            allow_build=False,
            cache_dtype=torch.float32,
        )
        ds41 = _ds41_ops()
        # The operator reads wk in FRACTAL_NZ, and the cast is deliberately
        # *inside* the traced region: the NZ tensor is then either an
        # intermediate node consumed straight by the op, or a constant cast
        # at trace time -- in neither case a *graph input*, so aclgraph's
        # input clone cannot degrade it back to ND.  The README's "ACLGraph
        # must disable input clone" note is about a pre-cast NZ tensor that
        # arrives *as an input*: that is the recipes layout, where
        # modeling_deepseek.py L2854-2863 casts attn.indexer.wk in place to
        # FRACTAL_NZ at load time and passes the parameter straight through.
        # Casting here yields the same format without depending on that
        # framework flag, which sglang never sets.  Do NOT hoist this cast
        # into a cached attribute: a long-lived NZ tensor can be lifted as a
        # graph input, which is exactly the case the README warns about.
        # Both storage modes share every argument but the destination.
        common = dict(
            latent=latent.reshape(-1, latent.shape[-1]),
            wk=torch_npu.npu_format_cast(
                indexer.wk.weight, torch_npu.Format.FRACTAL_NZ
            ),
            norm_weight=indexer.k_norm.weight.to(torch.float32),
            rope_sin=sin,
            rope_cos=cos,
            cache_index=cache_index.to(torch.int64),
            norm_eps=indexer.k_norm.eps,
        )
        ds41.indexer_prologue_k(
            **common,
            k_cache=pool.get_index_k(layer.layer_id),
            k_scale_cache=pool.get_index_scale(layer.layer_id),
            storage_mode=0,
        )
        if indexer.is_candidate_source and not os.environ.get(
            "SGLANG_DSV41_DISABLE_QSLI_CACHE"
        ):
            # Level-two K, read by QSLI: the same rows again, eight to a record,
            # fused with their scales (indexer_prologue_k_dsl README L49-L52).
            ds41.indexer_prologue_k(
                **common,
                k_cache=self._dsv41_qsli_cache(layer),
                k_scale_cache=None,
                storage_mode=1,
                combined_block_size=indexer.candidate_block_size,
            )

    def _dsv41_index_q(self, layer, x, q_lora, pos):
        """Packed MXFP4 Q, its E8M0 scales and the head weights, for both indexers.

        Recipes: indexer.py L341-L357 and L410-L415 — QLI and QSLI take the same
        Q side, so it is built once here for both.
        """
        import torch_npu

        indexer = layer.indexer
        t = int(q_lora.shape[0])
        packed_dim = indexer.index_head_dim // 2

        q, _ = indexer.wq_b(q_lora)
        q = q.view(t, indexer.n_heads, indexer.index_head_dim)
        from sglang.srt.hardware_backend.npu.dsv4.dsv4_rope import Dsv4NpuRoPE

        # Like the compressor, DeepseekV41Indexer is built without its own
        # freqs_cis/rotary_emb (deepseek_v4.py L1366-1372), so the layer's own
        # table is the rope for its Q (and for the index K in
        # _dsv41_index_k_store). A compress_ratio != 0 layer builds that table
        # with compress_rope_theta + YaRN (L1053-L1074), which is exactly the
        # reference model's compress_rotary_emb (recipes indexer.py L234 uses
        # cos_sin["comp"] here).
        table = Dsv4NpuRoPE.for_freqs(
            layer.freqs_cis, getattr(layer, "rotary_emb", None)
        )
        cos, sin = table.get_cos_sin(
            pos,
            torch.bfloat16,
            view_4d=True,
            allow_build=False,
            cache_dtype=torch.bfloat16,
        )
        Dsv4NpuRoPE.apply_rotary_mul_inplace(
            q,
            None,
            cos,
            sin,
            qk_nope_dim=indexer.index_head_dim - indexer.rope_head_dim,
        )
        q_packed, q_scale = torch_npu.npu_dynamic_mx_quant(
            q, dst_type=torch_npu.float4_e2m1fn_x2
        )
        q_packed = q_packed.view(torch.uint8).view(t, indexer.n_heads, packed_dim)
        # QLI wants the E8M0 scales observed as [..., 2, 2]; the pool stores the
        # same 4 bytes per row, so both are pure reshapes.
        descale_q = q_scale.view(torch.uint8).view(t, indexer.n_heads, 2, 2)
        w = indexer.head_weights(x).view(-1, indexer.n_heads).float()
        return q_packed, descale_q, w

    def _dsv41_index_full_topk(
        self, layer, x, q_lora, pos, forward_batch, *, candidate: bool = False
    ):
        """Score the compressed history with ``quant_lightning_indexer``.

        With ``candidate`` the same operator call also emits the level-one block
        candidates, and its third and fourth returns stop being empty tensors.
        Recipes: indexer.py L338-L400.
        """
        indexer = layer.indexer
        ratio = layer.compress_ratio
        fm = self.forward_metadata
        pool = self.token_to_kv_pool
        t = int(q_lora.shape[0])
        packed_dim = indexer.index_head_dim // 2
        q_packed, descale_q, w = self._dsv41_index_q(layer, x, q_lora, pos)

        k = pool.get_index_k(layer.layer_id)
        k_scale = pool.get_index_scale(layer.layer_id)
        # PA_ND rows are [pages, page_rows, 1, dim]; QLI reads the packed fp4
        # payload and its E8M0 scales as [..., 2, 2], both pure reshapes of the
        # buffers the prologue wrote.
        pages, page_rows = k.shape[0], k.shape[1]
        k = k.view(pages, page_rows, 1, packed_dim)
        descale_k = k_scale.view(torch.uint8).view(pages, page_rows, 1, 2, 2)

        page_table = getattr(fm, f"c{ratio}_index_page_table", None)
        if page_table is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 QLI for layer {layer.layer_id} is missing the "
                "compressed index page table."
            )
        # QLI's seqused_k is the compressed history length and cmp_residual_k is
        # its unfinished-group remainder; recipes computes them as
        # ``actual_seq_k // ratio`` and ``actual_seq_k % ratio`` (indexer.py
        # L359-L366 feeding attention_data.py L907-L918). Derive them from this
        # batch's seq_lens rather than from a metadata tensor: seq_lens is a
        # graph input (refreshed every replay), while a captured metadata buffer
        # would keep the capture-time lengths for every later step.
        seq_lens = forward_batch.seq_lens
        # Same generation-agnostic `index_topk` the QLI metadata op and the graph
        # buffer are sized with (L1336 / L1761); the op both selects and reshapes
        # with it.
        topk = self._dsv4_index_topk
        key = (
            f"c{ratio}a_qli_candidate_metadata"
            if candidate
            else f"c{ratio}a_qli_metadata"
        )
        metadata = fm.kernel_metadata.get(key)
        if metadata is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 QLI for layer {layer.layer_id} has no {key}; the "
                "kernel-metadata builder must run before the source layers."
            )
        result = _ds41_ops().quant_lightning_indexer(
            q=q_packed,
            k=k,
            w=w,
            descale_q=descale_q,
            descale_k=descale_k,
            topk=topk,
            quant_mode=1,
            cu_seqlens_q=fm.actual_seq_lengths_q_pa,
            seqused_q=None,
            seqused_k=(seq_lens // ratio).to(torch.int32),
            cmp_residual_k=(None if ratio == 1 else (seq_lens % ratio).to(torch.int32)),
            block_table=page_table.to(torch.int32),
            max_seqlen_q=-1,
            mask_mode=3,
            cmp_ratio=ratio,
            layout_q="TND",
            layout_k="PA_BBND",
            return_value=False,
            metadata=metadata,
            # -1 leaves the operator's third and fourth returns empty.
            candidate_topk_blocks=indexer.candidate_topk_blocks if candidate else -1,
            candidate_block_size=indexer.candidate_block_size if candidate else -1,
        )
        indices, _, candidate_indices, candidate_lengths = result
        scored = indices.view(t, topk).to(torch.int32)
        if not candidate:
            return scored, None
        return scored, (candidate_indices, candidate_lengths)

    def _dsv41_qsli_cache(self, layer) -> torch.Tensor:
        """The combined index-K record QSLI reads: ``[pages, page_size/8, 1, 544]``.

        ``indexer_prologue_k`` in ``storage_mode=1`` packs the same K rows eight
        to a record -- K[8, 64] as 512B, then their e8m0 scales as 32B -- which
        is exactly the PA_BBND K layout QSLI takes
        (quant_sparse_lightning_indexer_dsl/README.md L12-L29). Only the
        candidate source layer writes it, so it is allocated per (ratio, source
        slot) on first use instead of for every layer of the ratio; the consumer
        layers reach the source's buffer through the same ``layer_mapping`` slot.
        """
        pool = self.token_to_kv_pool
        item = pool.layer_mapping[layer.layer_id]
        key = (item.compress_ratio, item.compress_layer_id)
        cache = self._dsv41_qsli_caches.get(key)
        if cache is None:
            index_k = pool.get_index_k(layer.layer_id)
            block = self._dsv41_candidate_block_size
            pages, page_rows = int(index_k.shape[0]), int(index_k.shape[1])
            if block <= 0 or page_rows % block:
                raise RuntimeError(
                    "DeepSeek-V4.1 QSLI needs candidate_block_size to divide the "
                    f"compressed page rows, got block={block} rows={page_rows}."
                )
            head_dim = int(index_k.shape[-1]) * 2
            # 32 = the prologue's e4m3 scale group (indexer_prologue_k_dsl
            # README L40-L52: 64 payload bytes + 4 scale bytes per 128-dim row).
            record = block * (head_dim // 2 + head_dim // 32)
            cache = torch.zeros(
                pages,
                page_rows // block,
                1,
                record,
                dtype=torch.uint8,
                device=index_k.device,
            )
            self._dsv41_qsli_caches[key] = cache
        return cache

    def _dsv41_index_qsli_topk(self, layer, x, q_lora, pos, forward_batch):
        """Level two: rescore the published candidate blocks with this layer's Q.

        The candidates are logical ``candidate_block_size`` blocks of compressed
        tokens; the operator expands them itself, so the block ids pass through
        unchanged. Recipes: indexer.py L402-L453.
        """
        indexer = layer.indexer
        ratio = layer.compress_ratio
        fm = self.forward_metadata
        t = int(q_lora.shape[0])
        blocks = getattr(fm, "dsv41_candidate_blocks", None)
        if blocks is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 QSLI for layer {layer.layer_id} has no published "
                "candidate blocks; the candidate source layer must run before "
                "its consumers."
            )
        candidate_indices, candidate_lengths = blocks
        if candidate_indices.shape[0] != t:
            raise RuntimeError(
                "DeepSeek-V4.1 QSLI candidate blocks are stale: "
                f"{candidate_indices.shape[0]} rows published, {t} needed."
            )
        q_packed, descale_q, w = self._dsv41_index_q(layer, x, q_lora, pos)
        page_table = getattr(fm, f"c{ratio}_index_page_table", None)
        if page_table is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 QSLI for layer {layer.layer_id} is missing the "
                "compressed index page table."
            )
        metadata = fm.kernel_metadata.get(f"c{ratio}a_qsli_metadata")
        if metadata is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 QSLI for layer {layer.layer_id} has no "
                f"c{ratio}a_qsli_metadata; the kernel-metadata builder must run "
                "before the index layers."
            )
        topk = self._dsv4_index_topk
        qsli_k = self._dsv41_qsli_cache(layer)
        indices, _ = _ds41_ops().quant_sparse_lightning_indexer(
            q_packed,
            # QSLI takes the record rows without the singleton head axis.
            qsli_k.squeeze(-2),
            w,
            descale_q,
            candidate_indices,
            candidate_lengths,
            topk,
            1,
            indexer.candidate_block_size,
            descale_k=None,
            cu_seqlens_q=fm.actual_seq_lengths_q_pa,
            seqused_q=None,
            seqused_k=(forward_batch.seq_lens // ratio).to(torch.int32),
            # Only mask_mode=3 with cmp_ratio != 1 needs it, and the consumers
            # are ratio-1 layers (README "公共可选参数").
            cmp_residual_k=None,
            block_table=page_table.to(torch.int32),
            output_idx_offset=None,
            metadata=metadata,
            max_seqlen_q=-1,
            mask_mode=3,
            cmp_ratio=ratio,
            layout_q="TND",
            layout_k="PA_BBND",
            return_value=False,
        )
        scored = indices.view(t, topk).to(torch.int32)
        if os.environ.get("SGLANG_DSV41_DEBUG_QSLI"):
            # What MQSMLA will be told: exactly the helper's buffer (position
            # derived, dead rows zeroed).  Any -1 of `scored` inside that prefix
            # is the wild-address violation -- the CPU golden raises on it
            # (mqsmla_golden.py L215) and the kernel does not skip it -- so print
            # every occurrence, not just the first call per layer.
            lens = (
                self._dsv41_cmp_topk_length(forward_batch, ratio)
                .reshape(-1)
                .to(torch.long)
            )
            in_prefix = torch.arange(topk, device=scored.device) < lens.unsqueeze(1)
            neg_in = int(((scored < 0) & in_prefix).sum())
            cov = (
                candidate_lengths.reshape(-1).to(torch.long)[:t]
                * int(indexer.candidate_block_size)
            ).clamp_max(int(topk))
            short = bool((cov < lens).any())
            key = (layer.layer_id, "qsli")
            if neg_in or short or key not in self._dsv41_debugged:
                self._dsv41_debugged.add(key)
                print(
                    f"[dsv41][{self._dsv41_rank_tag()}] qsli layer={layer.layer_id} "
                    f"ratio={ratio} rows={t} "
                    f"neg_in_prefix={neg_in} len_max={int(lens.max())} "
                    f"vis_max={int((forward_batch.seq_lens // ratio).max())} "
                    f"cov_max={int(cov.max())} "
                    f"cand={tuple(candidate_indices.shape)} "
                    f"block_len={int(candidate_lengths.min())}/"
                    f"{int(candidate_lengths.max())} "
                    f"scored_min={int(scored.min())} "
                    f"scored_max={int(scored.max())} "
                    f"k={tuple(qsli_k.shape)} "
                    f"pt={tuple(page_table.shape)}/{int(page_table.max())}",
                    flush=True,
                )
        return scored

    def _dsv41_index_kernel_metadata(self, forward_batch: ForwardBatch) -> dict:
        """QLI dispatch tables, one per low ratio (the ``[1024]`` int32 tensor the
        compute entry point requires; QLI README "Metadata 参数与返回值").

        Recipes: indexer.py L368-L394 reads
        ``attn_metadata["kernel_metadata"][f"c{ratio}a_qli_metadata"]``. Built from
        ``forward_batch.seq_lens`` directly so it does not depend on the
        compress-metadata pass running first.
        """
        fm = self.forward_metadata
        seq_lens = forward_batch.seq_lens.to(torch.int64)
        cu_seqlens_q = fm.actual_seq_lengths_q_pa
        if cu_seqlens_q is None:
            return {}
        ds41 = _ds41_ops()
        out: dict = {}

        def keep(key: str, metadata: torch.Tensor) -> None:
            # aclgraph binds the address at capture: refresh in place.
            buf = fm.kernel_metadata.get(key)
            if buf is not None and buf.shape == metadata.shape:
                buf.copy_(metadata)
                metadata = buf
            out[key] = metadata

        batch_size = int(forward_batch.batch_size)
        for ratio in self._dsv41_low_ratios():
            cmp_lens = (seq_lens // ratio).to(torch.int32)
            keep(
                f"c{ratio}a_qli_metadata",
                ds41.quant_lightning_indexer_metadata(
                    cu_seqlens_q=cu_seqlens_q,
                    seqused_q=None,
                    seqused_k=cmp_lens,
                    # mask_mode=3 + cmp_ratio != 1 requires the residual (README).
                    cmp_residual_k=(
                        None if ratio == 1 else (seq_lens % ratio).to(torch.int32)
                    ),
                    batch_size=batch_size,
                    num_heads_q=self._dsv4_index_n_heads,
                    num_heads_k=1,
                    head_dim=self._dsv4_index_head_dim,
                    topk=self._dsv4_index_topk,
                    mask_mode=3,
                    cmp_ratio=ratio,
                    layout_q="TND",
                    layout_k="PA_BBND",
                ),
            )
        candidate_ratio = self._dsv41_candidate_ratio()
        if candidate_ratio is not None:
            block = self._dsv41_candidate_block_size
            if block <= 0 or self._dsv41_candidate_topk_blocks <= 0:
                raise RuntimeError(
                    "DeepSeek-V4.1 has a candidate source layer but no "
                    f"candidate_topk_blocks/candidate_block_size "
                    f"({self._dsv41_candidate_topk_blocks}/{block})."
                )
            cmp_lens = (seq_lens // candidate_ratio).to(torch.int32)
            # Level one publishes blocks, so its dispatch table is a separate
            # instance from the plain per-ratio one (recipes indexer.py L388-L393).
            keep(
                f"c{candidate_ratio}a_qli_candidate_metadata",
                ds41.quant_lightning_indexer_metadata(
                    cu_seqlens_q=cu_seqlens_q,
                    seqused_q=None,
                    seqused_k=cmp_lens,
                    cmp_residual_k=(
                        None
                        if candidate_ratio == 1
                        else (seq_lens % candidate_ratio).to(torch.int32)
                    ),
                    batch_size=batch_size,
                    num_heads_q=self._dsv4_index_n_heads,
                    num_heads_k=1,
                    head_dim=self._dsv4_index_head_dim,
                    topk=self._dsv4_index_topk,
                    mask_mode=3,
                    cmp_ratio=candidate_ratio,
                    layout_q="TND",
                    layout_k="PA_BBND",
                    candidate_topk_blocks=self._dsv41_candidate_topk_blocks,
                    candidate_block_size=block,
                ),
            )
        if candidate_ratio is not None and self._dsv41_has_candidate_consumers():
            # QSLI only needs a scheduling bound for the candidate prefix, so the
            # length is the visible block count per row, computed the way the
            # reference does it (modeling_deepseek.py L3241-L3245).
            visible_blocks = (
                torch.div(
                    forward_batch.positions.reshape(-1) + candidate_ratio * block,
                    candidate_ratio * block,
                    rounding_mode="floor",
                )
                .clamp_min(1)
                .clamp_max(self._dsv41_candidate_topk_blocks)
            )
            keep(
                f"c{candidate_ratio}a_qsli_metadata",
                ds41.quant_sparse_lightning_indexer_metadata(
                    visible_blocks.to(torch.int32).view(-1, 1),
                    cu_seqlens_q=cu_seqlens_q,
                    seqused_q=None,
                    seqused_k=(seq_lens // candidate_ratio).to(torch.int32),
                    cmp_residual_k=None,
                    batch_size=batch_size,
                    num_heads_q=self._dsv4_index_n_heads,
                    num_heads_k=1,
                    head_dim=self._dsv4_index_head_dim,
                    topk=self._dsv4_index_topk,
                    quant_mode=1,
                    candidate_block_size=block,
                    mask_mode=3,
                    cmp_ratio=candidate_ratio,
                    layout_q="TND",
                    layout_k="PA_BBND",
                ),
            )
        return out

    # ---- Seam D: mixed-quant sparse MLA ---------------------------------------
    #
    # DeepSeek-V4.1 runs a *different* attention operator from V4's
    # npu_kv_quant_sparse_attn_sharedkv: ``ds41.mixed_quant_sparse_flash_mla``,
    # which takes the window/compressed sides purely as explicit sparse indices +
    # topk lengths -- it has no mask, window or compress-ratio parameters at all
    # (mixed_quant_sparse_flash_mla_dsl/README.md L208).  Pool row widths and
    # group sizes line up with the pools built in dsv4_memory_pool.py:
    # ori 544B = FP8E4M3 group32 + BF16 scales, cmp 320B = FP4E2M1 group16 +
    # BF16 scales (README L185-L192).

    def _dsv41_window_indices(self, forward_batch: ForwardBatch):
        """Explicit causal sliding-window indices + lengths for the window side.

        Recipes: attention_data.py L333-L352 (``generate_win_topk_ids``) and
        L614-L618.  Both are pure device arithmetic, so this is aclgraph-safe;
        the results are written into stable buffers because the captured graph
        holds their addresses.
        """
        fm = self.forward_metadata
        positions = forward_batch.positions.reshape(-1)
        t = int(positions.numel())
        topk = int(self._dsv4_sliding_window_size)
        indices = getattr(fm, "dsv41_win_sparse_indices", None)
        lengths = getattr(fm, "dsv41_win_topk_length", None)
        if indices is None or indices.shape[0] != t:
            indices = torch.empty(
                (t, 1, topk), dtype=torch.int32, device=positions.device
            )
            lengths = torch.empty((t, 1), dtype=torch.int32, device=positions.device)
            setattr(fm, "dsv41_win_sparse_indices", indices)
            setattr(fm, "dsv41_win_topk_length", lengths)
        offsets = torch.arange(topk, dtype=positions.dtype, device=positions.device)
        valid_len = torch.clamp(positions + 1, max=topk)
        # Dead rows (batch padding / idle slots: seq_lens == 0, positions == 0)
        # have no KV at all, but (positions + 1) claims one window token.  The
        # indexer scores nothing for them and fills the row with -1, so the
        # claimed prefix would feed MQSMLA an in-prefix -1 (contract violation,
        # mqsmla_golden.py L215).  Zero the length instead: README L211 makes a
        # zero length skip the side and emit 0.
        seq_lens = forward_batch.seq_lens
        if seq_lens is not None:
            seq_lens = seq_lens.reshape(-1)
            if seq_lens.numel() == t:
                valid_len = torch.where(
                    seq_lens > 0, valid_len, torch.zeros_like(valid_len)
                )
        window_start = positions - valid_len + 1
        ids = window_start.unsqueeze(1) + offsets.unsqueeze(0)
        valid = offsets.unsqueeze(0) < valid_len.unsqueeze(1)
        ids = torch.where(valid, ids, ids.new_full((), -1))
        indices.copy_(ids.unsqueeze(1).to(torch.int32))
        lengths.copy_(valid_len.to(torch.int32).unsqueeze(1))
        return indices, lengths

    def _dsv41_cmp_topk_length(self, forward_batch: ForwardBatch, ratio: int):
        """Per-query valid compressed top-k length, ``min((p+1)//ratio, index_topk)``.

        Recipes: attention_data.py L891-L904.
        """
        fm = self.forward_metadata
        positions = forward_batch.positions.reshape(-1)
        t = int(positions.numel())
        buf = getattr(fm, f"dsv41_cmp_topk_length_c{ratio}", None)
        if buf is None or buf.shape[0] != t:
            buf = torch.empty((t, 1), dtype=torch.int32, device=positions.device)
            setattr(fm, f"dsv41_cmp_topk_length_c{ratio}", buf)
        lens = ((positions + 1) // ratio).clamp_max(int(self._dsv4_index_topk))
        # Same dead-row rule as _dsv41_window_indices: a seq_lens == 0 row has
        # no compressed tokens, so its length must be 0, not (0 + 1) // ratio.
        seq_lens = forward_batch.seq_lens
        if seq_lens is not None:
            seq_lens = seq_lens.reshape(-1)
            if seq_lens.numel() == t:
                lens = torch.where(seq_lens > 0, lens, torch.zeros_like(lens))
        buf.copy_(lens.to(torch.int32).view(-1, 1))
        return buf

    def _dsv41_attention_kernel_metadata(self, forward_batch: ForwardBatch) -> dict:
        """MQSMLA dispatch tables: ``win`` (ratio 0) plus ``c{r}a`` (ratio 1/2).

        Recipes: modeling_deepseek.py L3154-L3176.
        """
        fm = self.forward_metadata
        _, win_lengths = self._dsv41_window_indices(forward_batch)
        ds41 = _ds41_ops()
        cu_seqlens_q = fm.actual_seq_lengths_q_pa
        head_num = int(self._dsv4_q_head_num)
        # Take the width from the backend's own config, like the graph path
        # (``_init_dsv41_graph_metadata``) and the V4 eager path
        # (``_compute_kernel_metadata``).  The DSV4 pool exposes
        # ``qk_nope_head_dim``/``qk_rope_head_dim`` but has no ``head_dim``
        # attribute, so reading it off the pool is an AttributeError.
        head_dim = int(self._dsv4_head_dim)
        out = {}
        for ratio in (0, *self._dsv41_low_ratios()):
            attention_type = "win" if ratio == 0 else f"c{ratio}a"
            if ratio == 0:
                cmp_lengths = torch.zeros_like(win_lengths)
            else:
                cmp_lengths = self._dsv41_cmp_topk_length(forward_batch, ratio)
            key = f"{attention_type}_metadata"
            metadata = ds41.mixed_quant_sparse_flash_mla_metadata(
                win_lengths,
                cmp_lengths,
                cu_seqlens_q=cu_seqlens_q,
                num_heads_q=head_num,
                num_heads_kv=1,
                head_dim=head_dim,
                quant_mode=1,
                layout_q="TND",
                layout_kv="PA_BBND",
                has_ori_kv=True,
                has_cmp_kv=ratio > 0,
            )
            # aclgraph binds the tensor address at capture, so refresh the
            # graph-allocated buffer in place instead of rebinding.
            buf = fm.kernel_metadata.get(key)
            if buf is not None and buf.shape == metadata.shape:
                buf.copy_(metadata)
                metadata = buf
            out[key] = metadata
        return out

    def _dsv41_attention_sinks(self, attn_sink, device, head_num: int):
        """MQSMLA always needs a float32 ``(N1,)`` sink (README L110/L219)."""
        if attn_sink is None:
            cached = getattr(self, "_dsv41_zero_sinks", None)
            if cached is None or cached.numel() != head_num:
                cached = torch.zeros(head_num, dtype=torch.float32, device=device)
                setattr(self, "_dsv41_zero_sinks", cached)
            return cached
        return attn_sink.reshape(-1).to(torch.float32)

    def _dsv41_rank_tag(self) -> str:
        """Per-rank tag for every [dsv41] debug line.

        Multi-rank logs interleave, so each line carries its origin; without it
        the same step number from two DP ranks looks like time travel.
        """
        cached = getattr(self, "_dsv41_rank_tag_cache", None)
        if cached is None:
            try:
                p = get_parallel()
                cached = (
                    f"dp{getattr(p, 'attn_dp_rank', '?')}"
                    f".tp{getattr(p, 'tp_rank', '?')}"
                )
            except Exception:
                cached = "dp?.tp?"
            self._dsv41_rank_tag_cache = cached
        return cached

    def _dsv41_regenerate_smla_metadata(self, kw: dict):
        """Re-run the metadata op on the exact tensors this SMLA call receives.

        The two-stage contract says the dispatch table must be derived from this
        step's lengths; a table that differs from a fresh regeneration is a
        stale buffer on the sglang side, while an equal-but-broken table points
        at the operator.  Debug tooling only: never mask the underlying fault.
        """
        try:
            ori_len = kw["ori_topk_length"]
            has_cmp = "cmp_topk_length" in kw
            cmp_len = kw["cmp_topk_length"] if has_cmp else torch.zeros_like(ori_len)
            return _ds41_ops().mixed_quant_sparse_flash_mla_metadata(
                ori_len,
                cmp_len,
                cu_seqlens_q=kw.get("cu_seqlens_q"),
                num_heads_q=int(self._dsv4_q_head_num),
                num_heads_kv=1,
                head_dim=int(self._dsv4_head_dim),
                quant_mode=1,
                layout_q="TND",
                layout_kv="PA_BBND",
                has_ori_kv=True,
                has_cmp_kv=has_cmp,
            )
        except Exception as err:
            print(
                f"[dsv41][{self._dsv41_rank_tag()}] metadata regeneration "
                f"failed: {err}",
                flush=True,
            )
            return None

    def _dsv41_audit_smla(self, layer, q, forward_batch, kw, ratio: int) -> None:
        """Row-accurate MQSMLA contract audit.

        The device contract (mixed_quant_sparse_flash_mla_dsl/README.md L209-L214,
        enforced by the CPU golden mqsmla_golden.py L211-L216) is: for every query
        row, every sparse index inside the ``topk_length`` prefix must be a valid
        logical token index of that row's batch, and must map through
        ``block_table[b, idx // block_size]`` to a column inside the table and a
        page inside the pool.  The kernel does no bounds checking, so any
        violation is exactly the wild GM address of the aicore fault.  Every
        check is per-row and names the offending rows, so the last VIOLATION
        line before the async fault pinpoints the layer / side / row that fed
        the bad address.  A run that crashes with no VIOLATION line (but with
        the cfg/snapshot lines present) moves the suspicion off the values and
        onto the operator/dispatch internals.
        """
        phase = "decode" if forward_batch.forward_mode.is_decode() else "extend"
        step = getattr(self, "_dsv41_audit_step", 0) + 1
        self._dsv41_audit_step = step
        if ("cfg",) not in self._dsv41_debugged:
            self._dsv41_debugged.add(("cfg",))
            print(
                f"[dsv41][{self._dsv41_rank_tag()}] cfg "
                f"index_topk={self._dsv4_index_topk} "
                f"sliding_window={self._dsv4_sliding_window_size} "
                f"candidate_topk_blocks={self._dsv41_candidate_topk_blocks} "
                f"candidate_block_size={self._dsv41_candidate_block_size} "
                f"low_ratios={self._dsv41_low_ratios()}",
                flush=True,
            )
        rows = int(q.shape[0])
        positions = forward_batch.positions.reshape(-1).to(torch.long)
        seq_lens = forward_batch.seq_lens.reshape(-1).to(torch.long)
        device = positions.device
        cu = kw.get("cu_seqlens_q")
        problems = []
        if rows != int(positions.numel()):
            problems.append(f"q rows {rows} != positions {int(positions.numel())}")
        if cu is not None and int(cu[-1]) != rows:
            problems.append(f"cu span {int(cu[-1])} != q rows {rows}")
        # row -> batch map (decode: one token per request; extend: cu bucketing)
        if cu is not None and rows == int(cu[-1]) and rows != cu.numel() - 1:
            row_b = torch.repeat_interleave(
                torch.arange(cu.numel() - 1, device=device),
                (cu[1:] - cu[:-1]).to(torch.long),
            )
        else:
            row_b = torch.arange(rows, device=device)
        # The dispatch table's padded tail is NOT a valid check target: healthy
        # single-curl runs show the same garbage there, so the wheel pads the
        # (2*blocks,) table to a fixed 1024 and leaves the padding
        # uninitialized while the kernel only consumes the head region
        # (blockDim=32 in the fault dumps).  Compare only the head against a
        # fresh regeneration, and dump the layout once so the real format is
        # on record.
        md = kw.get("metadata")
        if md is not None and md.numel() % 2 == 0:
            half = md.shape[0] // 2
            fresh = self._dsv41_regenerate_smla_metadata(kw)
            if fresh is not None and fresh.shape == md.shape:
                if not bool(torch.equal(md[:64], fresh[:64])):
                    problems.append(
                        "metadata head differs from a fresh regeneration: "
                        f"buf={md[:8].tolist()} fresh={fresh[:8].tolist()}"
                    )
                if ("md_layout",) not in self._dsv41_debugged:
                    self._dsv41_debugged.add(("md_layout",))
                    diff = (md != fresh).nonzero().flatten()
                    rt = self._dsv41_rank_tag()
                    print(
                        f"[dsv41][{rt}] md_layout rows={rows} blocks={half} "
                        f"buf[:8]={md[:8].tolist()} "
                        f"fresh[:8]={fresh[:8].tolist()}\n"
                        f"[dsv41][{rt}]   buf_half_head={md[half : half + 8].tolist()} "
                        f"fresh_half_head={fresh[half : half + 8].tolist()} "
                        f"buf_tail_sum={int(md[half:].sum())} "
                        f"fresh_tail_sum={int(fresh[half:].sum())} "
                        f"first_diff={int(diff[0]) if diff.numel() else -1} "
                        f"n_diff={int(diff.numel())}",
                        flush=True,
                    )
        for prefix in ("ori", "cmp"):
            kv = kw.get(f"{prefix}_kv")
            bt = kw.get(f"{prefix}_block_table")
            idx_t = kw.get(f"{prefix}_sparse_indices")
            lens_t = kw.get(f"{prefix}_topk_length")
            if kv is None or bt is None or idx_t is None or lens_t is None:
                continue
            # Per-row visible length: extend rows outnumber batches, so the
            # batch-level lengths must be gathered through the row->batch map.
            vis_batch = seq_lens if prefix == "ori" else seq_lens // max(1, ratio)
            vis = vis_batch[row_b]
            idx = idx_t.reshape(rows, -1).to(torch.long)
            lens = lens_t.reshape(-1).to(torch.long)
            k = idx.shape[-1]
            block = max(1, int(kv.shape[1]))
            width = int(bt.shape[1])
            pages = int(kv.shape[0])
            in_prefix = torch.arange(k, device=device) < lens.unsqueeze(1)
            col = torch.div(idx, block, rounding_mode="floor")
            page = bt.to(torch.long)[row_b.unsqueeze(1), col.clamp(0, width - 1)]
            checks = (
                # golden rule: the prefix must be exact -- a -1 inside it is
                # consumed as a logical token index and becomes a wild address
                ("neg_in_prefix", (idx < 0) & in_prefix),
                # in-prefix index at/after the row's visible logical length
                ("idx_ge_visible", (idx >= vis.clamp_min(1).unsqueeze(1)) & in_prefix),
                # in-prefix index maps to a column outside the block table
                ("bt_col_oob", (col >= width) & in_prefix),
                # the page at that column is outside the pool (or negative)
                ("page_oob", ((page < 0) | (page >= pages)) & in_prefix),
                # outside the prefix only the -1 filler is allowed
                ("garbage_tail", idx < -1),
            )
            for name, mask in checks:
                n = int(mask.sum())
                if not n:
                    continue
                r0 = int(mask.any(dim=1).nonzero()[0, 0])
                j = mask[r0].nonzero()[:4, 0].tolist()
                problems.append(
                    f"{prefix}.{name} n={n} first_row={r0} "
                    f"(b={int(row_b[r0])} pos={int(positions[r0])} "
                    f"seq={int(seq_lens[row_b[r0]])} len={int(lens[r0])} "
                    f"bt={width}x pages={pages} block={block}) "
                    + " ".join(
                        f"[j={a} idx={int(idx[r0, a])} col={int(idx[r0, a]) // block} "
                        f"page={int(page[r0, a])}]"
                        for a in j
                    )
                )
            # window-side staleness: the prefix must end at the row position.
            # Dead rows (length 0) are skipped; their gathered tail is filler.
            if prefix == "ori" and rows == int(positions.numel()):
                last = idx.gather(1, (lens - 1).clamp_min(0).unsqueeze(1)).squeeze(1)
                stale = (lens > 0) & (last != positions)
                if bool(stale.any()):
                    bad = stale.nonzero()[:8, 0].tolist()
                    problems.append(
                        f"ori.window_stale rows={bad} last={last[bad].tolist()} "
                        f"pos={positions[bad].tolist()}"
                    )
            # direct test of the QSLI prefix hypothesis: the length passed to
            # MQSMLA must not exceed the candidate tokens QSLI could fill
            if prefix == "cmp":
                cand = getattr(self.forward_metadata, "dsv41_candidate_blocks", None)
                if cand is not None and self._dsv41_candidate_block_size > 0:
                    cov = (
                        cand[1].reshape(-1).to(torch.long)[:rows]
                        * self._dsv41_candidate_block_size
                    ).clamp_max(self._dsv4_index_topk)
                    over = lens > cov
                    if bool(over.any()):
                        r = over.nonzero()[:8, 0].tolist()
                        problems.append(
                            f"cmp.len_gt_candidate rows={r} "
                            f"len={lens[r].tolist()} cov={cov[r].tolist()} "
                            f"(QSLI pads -1 inside this prefix)"
                        )
        if problems:
            print(
                f"[dsv41][{self._dsv41_rank_tag()}] VIOLATION step={step} "
                f"layer={layer.layer_id} ratio={ratio} "
                f"phase={phase} bs={forward_batch.batch_size} rows={rows}: "
                + " | ".join(problems),
                flush=True,
            )
        key = (layer.layer_id, ratio, phase)
        if key not in self._dsv41_debugged:
            self._dsv41_debugged.add(key)
            shapes = {n: tuple(v.shape) for n, v in kw.items() if torch.is_tensor(v)}
            rt = self._dsv41_rank_tag()
            print(
                f"[dsv41][{rt}] AUDIT snapshot step={step} layer={layer.layer_id} "
                f"ratio={ratio} phase={phase} bs={forward_batch.batch_size} "
                f"q={tuple(q.shape)}\n"
                f"[dsv41][{rt}]   {shapes}\n"
                f"[dsv41][{rt}]   pos=[{int(positions.min())},{int(positions.max())}] "
                f"seq=[{int(seq_lens.min())},{int(seq_lens.max())}] "
                f"dead={int((seq_lens <= 0).sum())}\n"
                f"[dsv41][{rt}]   ori_len_max={int(kw['ori_topk_length'].max())} "
                f"ori_idx_max={int(kw['ori_sparse_indices'].max())} "
                f"ori_bt={tuple(kw['ori_block_table'].shape)} "
                f"ori_bt_max={int(kw['ori_block_table'].max())} "
                f"ori_pages={kw['ori_kv'].shape[0]}"
                + (
                    f"\n[dsv41][{rt}]   cmp_len_max={int(kw['cmp_topk_length'].max())} "
                    f"cmp_idx_max={int(kw['cmp_sparse_indices'].max())} "
                    f"cmp_bt={tuple(kw['cmp_block_table'].shape)} "
                    f"cmp_bt_max={int(kw['cmp_block_table'].max())} "
                    f"cmp_pages={kw['cmp_kv'].shape[0]}"
                    if "cmp_kv" in kw
                    else ""
                ),
                flush=True,
            )

    def _forward_low_ratio_attention(
        self,
        q: torch.Tensor,
        layer,
        forward_batch: ForwardBatch,
        attn_sink: Optional[torch.Tensor],
        compress_ratio: int,
    ) -> torch.Tensor:
        """The V4.1 attention body: window side always, compressed side when
        ``compress_ratio > 0``.

        Recipes: modeling_deepseek.py L1189-L1238 (``prepare_fa_kwargs``).
        """
        fm = self.forward_metadata
        pool = self.token_to_kv_pool
        win_indices, win_lengths = self._dsv41_window_indices(forward_batch)
        metadata = fm.kernel_metadata.get(
            "win_metadata" if compress_ratio == 0 else f"c{compress_ratio}a_metadata"
        )
        if metadata is None:
            raise RuntimeError(
                "DeepSeek-V4.1 attention metadata missing for "
                f"layer_id={layer.layer_id} compress_ratio={compress_ratio}; "
                f"available: {list(fm.kernel_metadata.keys())}."
            )
        attn_kwargs = dict(
            ori_kv=pool.get_swa_buffer(layer.layer_id).view(torch.uint8),
            ori_sparse_indices=win_indices,
            ori_topk_length=win_lengths,
            ori_block_table=fm.swa_page_table.to(torch.int32),
            cu_seqlens_q=fm.actual_seq_lengths_q_pa,
            seqused_q=None,
            # Reserved by the operator (README L106): accepted, not consumed.
            seqused_ori_kv=fm.actual_seq_lengths_kv,
            sinks=self._dsv41_attention_sinks(
                attn_sink, q.device, int(self._dsv4_q_head_num)
            ),
            metadata=metadata,
            quant_mode=1,
            # ``layer`` here is the ``RadixAttention`` the model passes to
            # ``forward`` (``self.attn_mqa``), *not* the ``MQALayer`` that
            # ``forward_low_ratio_sources`` receives.  ``scaling`` is the
            # RadixAttention attribute holding ``MQALayer.softmax_scale``
            # (= head_dim**-0.5, deepseek_v4.py L1374-1377); the MQALayer spells
            # the same number ``softmax_scale``.  Do not "fix" this to
            # ``layer.softmax_scale``: that would be the AttributeError.
            softmax_scale=layer.scaling,
            layout_q="TND",
            layout_kv="PA_BBND",
            return_softmax_lse=False,
        )
        if compress_ratio > 0:
            topk = getattr(fm, f"c{compress_ratio}_topk_indices", None)
            if topk is None:
                raise RuntimeError(
                    f"DeepSeek-V4.1 ratio-{compress_ratio} attention has no "
                    f"c{compress_ratio}_topk_indices; the indexer source layer "
                    "must run before this layer."
                )
            # Reserved by the operator (README L107); derived like the rest so
            # the value is live rather than capture-time.
            compressed_lens = (forward_batch.seq_lens // compress_ratio).to(torch.int32)
            cmp_kv = pool.get_compress_buffer(layer.layer_id, False).view(torch.uint8)
            cmp_bt = getattr(fm, f"c{compress_ratio}_page_table").to(torch.int32)
            if (
                os.environ.get("SGLANG_DSV41_DEBUG_QSLI")
                and ("smla", layer.layer_id) not in self._dsv41_debugged
            ):
                self._dsv41_debugged.add(("smla", layer.layer_id))
                rows_per_page = int(cmp_kv.shape[1]) if cmp_kv.dim() > 1 else 1
                need_cols = int((compressed_lens.max() - 1) // rows_per_page) + 1
                # Per-row page maxima: a row above the pool size is the row the
                # kernel turns into a wild address, and a -1 is the sentinel
                # leaking out of the write path.
                row_max = cmp_bt.max(dim=1).values[:16].tolist()
                rt = self._dsv41_rank_tag()
                print(
                    f"[dsv41][{rt}] smla layer={layer.layer_id} ratio={compress_ratio}\n"
                    f"[dsv41][{rt}]   cmp_kv={tuple(cmp_kv.shape)} rows_per_page={rows_per_page}\n"
                    f"[dsv41][{rt}]   bt={tuple(cmp_bt.shape)} bt_max={int(cmp_bt.max())} "
                    f"need_cols={need_cols} bt_cols_ok="
                    f"{cmp_bt.shape[1] >= need_cols}\n"
                    f"[dsv41][{rt}]   page_max_ok={int(cmp_bt.max()) < cmp_kv.shape[0]} "
                    f"negs={int((cmp_bt < 0).sum())}\n"
                    f"[dsv41][{rt}]   row_max={row_max}\n"
                    f"[dsv41][{rt}]   visible={compressed_lens[:16].tolist()}\n"
                    f"[dsv41][{rt}]   cmp_topk={tuple(topk.shape)} "
                    f"topk_max={int(topk.max())} topk_neg={int((topk < 0).sum())}",
                    flush=True,
                )
            attn_kwargs.update(
                cmp_kv=cmp_kv,
                cmp_sparse_indices=topk.view(-1, 1, topk.shape[-1]).to(torch.int32),
                cmp_block_table=cmp_bt,
                # seqused_cmp_kv is reserved by the operator (README L107).
                seqused_cmp_kv=compressed_lens,
                # One length per query row, from the position, capped by
                # index_topk (recipes attention_data.py L892-L905). The reference
                # uses this same upper bound for the QSLI layers too, so the -1
                # suffix inside it is expected to be skipped.
                cmp_topk_length=self._dsv41_cmp_topk_length(
                    forward_batch, compress_ratio
                ),
            )
        # Audit every ratio (0 included): a win-only layer can fault through the
        # ori side too, and the audit must name the last call before the async
        # aicore fault.
        if os.environ.get("SGLANG_DSV41_DEBUG_SMLA"):
            self._dsv41_audit_smla(
                layer, q, forward_batch, attn_kwargs, compress_ratio
            )
        attn_out, _lse = _ds41_ops().mixed_quant_sparse_flash_mla(q, **attn_kwargs)
        return attn_out

    # ---- V4.1 metadata assembly ------------------------------------------------
    #
    # ``ds41.compressor`` emits ``min(T, T // ratio + B)`` rows laid out as
    # <batch0>compressed + <batch1>compressed + ... + pad
    # (ops-transformer experimental/attention/compressor_v2/docs/
    # torchapi_compressor.md L96).  Both the RoPE positions and the slot list
    # handed to seam A therefore carry exactly that many rows: live rows first,
    # filler after.  Slots use -1 as filler because that is
    # ``kv_compress_epilog_v2``'s skip sentinel (cann-recipes-infer ops/ascendc/
    # docs/custom-npu_kv_compress_epilog_v2.md L89); positions use 0, which is
    # never stored because its row is skipped.
    #
    # V4 reads its slots from the allocator bundle.  V4.1 has no bundle
    # (``out_cache_loc_dsv4`` stays None), so a slot is derived as
    # ``full_loc // ratio``: that is exactly how the compressed pools are sized
    # and addressed (size = full_size // ratio, pages line up with the full
    # pool), so nothing beyond ``out_cache_loc`` is needed.

    _DSV41_SLOT_SKIP = -1

    def _dsv41_low_ratios(self) -> list:
        return [r for r in self._dsv4_unique_compress_ratios if r in (1, 2)]

    def _dsv41_candidate_ratio(self) -> Optional[int]:
        """The compression ratio of the layer that publishes candidate blocks."""
        layer_id = self._dsv41_candidate_source_layer
        if layer_id < 0 or layer_id >= len(self._dsv4_compress_ratios):
            return None
        return self._dsv4_compress_ratios[layer_id]

    def _dsv41_has_candidate_consumers(self) -> bool:
        """Whether any later index source layer rescoring inside the candidates."""
        source = self._dsv41_candidate_source_layer
        return self._dsv41_candidate_ratio() is not None and any(
            layer > source for layer in self._dsv41_index_source_layers
        )

    @staticmethod
    def _dsv41_compact_1d(
        dst: torch.Tensor,
        values: torch.Tensor,
        keep: torch.Tensor,
        fill: int,
    ) -> None:
        """``_stable_compact_1d`` with a caller-chosen tail filler.

        Same device-only ordinal scatter (no NonZero, no device-to-host size
        read), so it stays aclgraph-safe; only the untouched tail differs.

        The scatter *adds* into ``dst``, so the destination must start at zero:
        filling it with ``fill`` first would make every written slot
        ``value + fill`` (with the ``-1`` skip sentinel that is an off-by-one
        row). The tail is masked in afterwards instead, which keeps both the
        assignment semantics for written slots and a device-only implementation.
        """
        values = values.reshape(-1)
        keep = keep.reshape(-1)
        if values.numel() != keep.numel():
            raise ValueError(
                "dsv41 stable compact requires value/mask size equality, got "
                f"{values.numel()} and {keep.numel()}"
            )
        dst.zero_()
        if dst.numel() == 0 or values.numel() == 0:
            dst.fill_(fill)
            return
        ranks = torch.cumsum(keep.to(torch.int64), dim=0) - 1
        in_bounds = keep & (ranks < dst.numel())
        safe_ranks = ranks.clamp(min=0, max=dst.numel() - 1)
        compact = torch.where(
            in_bounds,
            values.to(dst.dtype),
            torch.zeros_like(values, dtype=dst.dtype),
        )
        dst.scatter_add_(0, safe_ranks, compact)
        if fill != 0:
            # Written slots are exactly ``[0, kept)``: ranks are dense over the
            # kept elements. Summing the mask keeps the count on device.
            kept = in_bounds.to(torch.int64).sum()
            tail = torch.arange(dst.numel(), device=dst.device) >= kept
            dst.masked_fill_(tail, fill)

    def _dsv41_debug_dump_cmp_rows(
        self, prefix: str, loc: torch.Tensor, ratio: int
    ) -> None:
        """Read back the compressed rows this step claims to have written."""
        pool = self.token_to_kv_pool
        layer_id = None
        for lid, r in enumerate(self._dsv4_compress_ratios):
            if r == ratio:
                layer_id = lid
                break
        if layer_id is None:
            return
        try:
            buf = pool.get_compress_buffer(layer_id)
        except Exception as exc:  # debug path: never break the run
            print(f"{prefix} cmp_buf_unavailable={exc!r}", flush=True)
            return
        slots = loc.reshape(-1).to(torch.int64)
        total = int(buf.shape[0]) * int(buf.shape[1])
        rows = slots[(slots >= 0) & (slots < total)]
        if rows.numel() == 0:
            print(f"{prefix} cmp_rows=0 (nothing written)", flush=True)
            return
        flat = buf.reshape(-1, buf.shape[-1])
        sample = flat[rows[:4]].to(torch.float32)
        print(
            f"{prefix} cmp_rows={int(rows.numel())} slots={rows[:4].tolist()} "
            f"payload_abs_sum={float(sample.abs().sum()):.1f}",
            flush=True,
        )

    def _dsv41_debug_dump_1d(
        self,
        tag: str,
        positions: torch.Tensor,
        out_loc: torch.Tensor,
        forward_batch: ForwardBatch,
        ratio: int,
        *,
        short_out_loc: bool = False,
    ) -> None:
        """Env-gated per-step dump of the low-ratio 1-D metadata.

        ``SGLANG_DSV41_DEBUG_REPLAY=1`` prints the first five steps in each mode
        so the eager and graph-replay values can be diffed field by field.  It
        also counts rows where the eager ``positions`` disagree with the replay's
        ``seq_len - 1`` derivation, and reads the compressed rows back, so one
        run separates "wrong metadata" from "wrong write".
        """
        if not os.environ.get("SGLANG_DSV41_DEBUG_REPLAY"):
            return
        # Per-(mode, ratio) counter: with two ratios each "step" fires twice, and
        # eager/replay run in separate processes anyway.
        key = (tag, ratio)
        counters = getattr(self, "_dsv41_dump_steps", None)
        if counters is None:
            counters = {}
            self._dsv41_dump_steps = counters
        counter = counters.get(key, 0)
        if counter >= 5:
            return
        counters[key] = counter + 1
        fm = self.forward_metadata
        prefix = f"[dsv41-rp][{self._dsv41_rank_tag()}][{tag}] r{ratio}"
        pos = positions.reshape(-1).to(torch.int64)
        oloc = out_loc.reshape(-1).to(torch.int64)
        live = forward_batch.seq_lens.reshape(-1).to(torch.int64)
        if pos.numel() == live.numel():
            derived = torch.clamp(live - 1, min=0)
            mismatch = int((pos != derived).sum())
        else:
            mismatch = -1
        n = int(pos.numel())
        print(
            f"{prefix} step={counter} n={n} short_out_loc={int(short_out_loc)} "
            f"pos_ne_live-1={mismatch} live={live[:4].tolist()} "
            f"out_loc={oloc[:4].tolist()} pos={pos[:4].tolist()}",
            flush=True,
        )
        loc = getattr(fm, f"c{ratio}_loc", None)
        pad = getattr(fm, f"positions_cmp_padding_c{ratio}", None)
        topk = getattr(fm, f"dsv41_cmp_topk_length_c{ratio}", None)
        print(
            f"{prefix} start_pos={fm.start_pos[:4].tolist()} "
            f"seqused={fm.seqused[:4].tolist()} "
            f"loc={None if loc is None else loc[:8].tolist()} "
            f"pad={None if pad is None else pad[:8].tolist()} "
            f"cmp_topk={None if topk is None else topk[:4].tolist()}",
            flush=True,
        )
        if loc is not None:
            self._dsv41_debug_dump_cmp_rows(prefix, loc, ratio)

    def _build_low_ratio_compress_metadata(self, forward_batch: ForwardBatch) -> None:
        """Eager half of the V4.1 metadata contract (V4 ``_compute_compress_locs``
        counterpart).  aclgraph allocates these buffers at capture
        (:meth:`_init_dsv41_graph_metadata`) and refreshes them in place on
        replay (:meth:`_refresh_graph_dsv41_1d`), so this runs eager-only.
        """
        ratios = self._dsv41_low_ratios()
        if not ratios:
            return
        forward_mode = forward_batch.forward_mode
        if forward_mode.is_decode() or forward_mode.is_extend_without_speculative():
            for ratio in ratios:
                self._build_dsv41_ratio_metadata(forward_batch, ratio)

    def _build_dsv41_ratio_metadata(
        self, forward_batch: ForwardBatch, ratio: int
    ) -> None:
        fm = self.forward_metadata
        device = forward_batch.positions.device
        positions = forward_batch.positions.reshape(-1).to(torch.int64)
        out_loc = forward_batch.out_cache_loc.reshape(-1).to(torch.int64)
        n = min(int(positions.numel()), int(out_loc.numel()))
        positions, out_loc = positions[:n], out_loc[:n]
        bs = int(forward_batch.batch_size)
        # One compressed row per token that closes a `ratio`-token group, i.e.
        # the token whose global position is `ratio - 1` (mod ratio).
        # Slots below one page are the reserved dummy page, so filtering on
        # `out_loc >= page_size` drops padding rows -- the same criterion recipes
        # uses (`slots >= block_size`, attention_data.py L878-L879).
        completes = ((positions % ratio) == (ratio - 1)) & (
            out_loc >= int(self.page_size)
        )
        if ratio == 1:
            # Ratio 1 has no fused compressor: the latent is the plain `wkv`
            # projection of every token row, so its row i is token i and the
            # slot row must keep that order -- mask in place (excluded rows get
            # -1 and the operator skips them) instead of compacting, exactly like
            # the graph refresh.  `pad == n` here (n // 1 + bs >= n).
            pos_row = positions.clone()
            pos_row.masked_fill_(~completes, 0)
            slot_row = out_loc.to(torch.int32)
            slot_row.masked_fill_(~completes, self._DSV41_SLOT_SKIP)
        else:
            # The fused compressor emits exactly ``min(n, n // ratio + bs)``
            # rows, dense in (request, position) order with the pad rows last,
            # so the slots are the completed groups compacted into that order.
            pad = min(n, n // ratio + bs)
            pos_row = torch.empty(pad, dtype=torch.int64, device=device)
            slot_row = torch.empty(pad, dtype=torch.int32, device=device)
            self._dsv41_compact_1d(
                pos_row, (positions // ratio) * ratio, completes, fill=0
            )
            self._dsv41_compact_1d(
                slot_row, out_loc // ratio, completes, fill=self._DSV41_SLOT_SKIP
            )
        setattr(fm, f"positions_cmp_padding_c{ratio}", pos_row)
        setattr(fm, f"c{ratio}_loc", slot_row)
        self._dsv41_debug_dump_1d("eager", positions, out_loc, forward_batch, ratio)
        # Compressed pages carry the full pool's page ids (compressed page j is
        # full page j), so the full page table doubles as the compressed page
        # table read with page_size // ratio.
        block_tables = getattr(fm, "block_tables", None)
        setattr(fm, f"c{ratio}_page_table", block_tables)
        # The indexer-K pool pages at DSV41_INDEX_PAGE_SIZE slots, so its block
        # table is the FULL page table expanded to that granularity: full page p
        # owns the consecutive index pages [k*p, k*p + k), where
        # k = (page_size // ratio) // DSV41_INDEX_PAGE_SIZE (1 for ratio 2, 2 for
        # ratio 1). This is the mapping the CUDA backends build with the shared
        # ``expand_index_page_table`` helper, and the index-K write path
        # (``cache_index = full_loc // ratio``) depends on it.
        index_page_table = None
        if block_tables is not None:
            from sglang.srt.layers.attention.dsv4.metadata import (
                expand_index_page_table,
            )
            from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
                DSV41_INDEX_PAGE_SIZE,
            )

            index_page_table = expand_index_page_table(
                block_tables,
                full_page_size=int(self.page_size),
                compress_ratio=ratio,
                index_page_size=DSV41_INDEX_PAGE_SIZE,
            )
        setattr(fm, f"c{ratio}_index_page_table", index_page_table)

    def _init_dsv41_graph_metadata(
        self, metadata, bs: int, tokens_per_req: int, device
    ) -> None:
        """Allocate the aclgraph-persistent low-ratio buffers for one bs bucket."""
        t = bs * tokens_per_req
        for ratio in self._dsv41_low_ratios():
            pad = min(t, t // ratio + bs)
            setattr(
                metadata,
                f"positions_cmp_padding_c{ratio}",
                torch.zeros(pad, dtype=torch.int64, device=device),
            )
            setattr(
                metadata,
                f"c{ratio}_loc",
                torch.full(
                    (pad,),
                    self._DSV41_SLOT_SKIP,
                    dtype=torch.int32,
                    device=device,
                ),
            )
            setattr(
                metadata,
                f"c{ratio}_page_table",
                getattr(metadata, "block_tables", None),
            )
            setattr(
                metadata,
                f"c{ratio}_index_page_table",
                self._dsv41_graph_index_page_table(metadata, ratio),
            )
            setattr(
                metadata,
                f"dsv41_cmp_topk_length_c{ratio}",
                torch.zeros((t, 1), dtype=torch.int32, device=device),
            )
        # The graph binds these tensor addresses at capture, so they must live in
        # fixed buffers that the replay refresh writes into.
        setattr(
            metadata,
            "dsv41_win_sparse_indices",
            torch.empty(
                (t, 1, int(self._dsv4_sliding_window_size)),
                dtype=torch.int32,
                device=device,
            ),
        )
        setattr(
            metadata,
            "dsv41_win_topk_length",
            torch.zeros((t, 1), dtype=torch.int32, device=device),
        )
        if not hasattr(metadata, "kernel_metadata") or metadata.kernel_metadata is None:
            metadata.kernel_metadata = {}
        # The dispatch tables are variable-size, so ask the metadata ops for one
        # probe tensor per attention type and keep a same-shaped buffer.
        ds41 = _ds41_ops()
        cu_seqlens_q = (
            torch.arange(bs + 1, dtype=torch.int32, device=device) * tokens_per_req
        )
        probe_cmp_lengths = torch.zeros((t, 1), dtype=torch.int32, device=device)
        for ratio in (0, *self._dsv41_low_ratios()):
            attention_type = "win" if ratio == 0 else f"c{ratio}a"
            probe = ds41.mixed_quant_sparse_flash_mla_metadata(
                metadata.dsv41_win_topk_length,
                probe_cmp_lengths,
                cu_seqlens_q=cu_seqlens_q,
                num_heads_q=int(self._dsv4_q_head_num),
                num_heads_kv=1,
                head_dim=int(self._dsv4_head_dim),
                quant_mode=1,
                layout_q="TND",
                layout_kv="PA_BBND",
                has_ori_kv=True,
                has_cmp_kv=ratio > 0,
            )
            key = f"kernel_metadata_{attention_type}"
            buf = self.graph_metadata.get(key)
            if buf is None or buf.shape != probe.shape:
                buf = torch.zeros_like(probe)
                self.graph_metadata[key] = buf
            metadata.kernel_metadata[f"{attention_type}_metadata"] = buf
        for ratio in self._dsv41_low_ratios():
            probe = ds41.quant_lightning_indexer_metadata(
                cu_seqlens_q=cu_seqlens_q,
                seqused_q=None,
                seqused_k=torch.zeros(bs, dtype=torch.int32, device=device),
                cmp_residual_k=(
                    None
                    if ratio == 1
                    else torch.zeros(bs, dtype=torch.int32, device=device)
                ),
                batch_size=bs,
                num_heads_q=int(self._dsv4_index_n_heads),
                num_heads_k=1,
                head_dim=int(self._dsv4_index_head_dim),
                topk=int(self._dsv4_index_topk),
                mask_mode=3,
                cmp_ratio=ratio,
                layout_q="TND",
                layout_k="PA_BBND",
            )
            key = f"kernel_metadata_c{ratio}a_qli"
            buf = self.graph_metadata.get(key)
            if buf is None or buf.shape != probe.shape:
                buf = torch.zeros_like(probe)
                self.graph_metadata[key] = buf
            metadata.kernel_metadata[f"c{ratio}a_qli_metadata"] = buf

        # The candidate table and the QSLI table are separate metadata instances
        # (see _dsv41_index_kernel_metadata): a capture that only seeds the plain
        # per-ratio tables makes the candidate source layer fail on its first
        # forward, because the in-graph builder never runs.  Probe and cache both
        # here as well; replay refreshes them in place by the same key match.
        candidate_ratio = self._dsv41_candidate_ratio()
        if candidate_ratio is None:
            return
        block = int(self._dsv41_candidate_block_size)
        if block <= 0 or self._dsv41_candidate_topk_blocks <= 0:
            raise RuntimeError(
                "DeepSeek-V4.1 has a candidate source layer but no "
                f"candidate_topk_blocks/candidate_block_size "
                f"({self._dsv41_candidate_topk_blocks}/{block})."
            )
        local_cmp_lens = torch.zeros(bs, dtype=torch.int32, device=device)
        probe = ds41.quant_lightning_indexer_metadata(
            cu_seqlens_q=cu_seqlens_q,
            seqused_q=None,
            seqused_k=local_cmp_lens,
            cmp_residual_k=(
                None
                if candidate_ratio == 1
                else torch.zeros(bs, dtype=torch.int32, device=device)
            ),
            batch_size=bs,
            num_heads_q=int(self._dsv4_index_n_heads),
            num_heads_k=1,
            head_dim=int(self._dsv4_index_head_dim),
            topk=int(self._dsv4_index_topk),
            mask_mode=3,
            cmp_ratio=candidate_ratio,
            layout_q="TND",
            layout_k="PA_BBND",
            candidate_topk_blocks=int(self._dsv41_candidate_topk_blocks),
            candidate_block_size=block,
        )
        key = f"kernel_metadata_c{candidate_ratio}a_qli_candidate"
        buf = self.graph_metadata.get(key)
        if buf is None or buf.shape != probe.shape:
            buf = torch.zeros_like(probe)
            self.graph_metadata[key] = buf
        metadata.kernel_metadata[f"c{candidate_ratio}a_qli_candidate_metadata"] = buf

        if self._dsv41_has_candidate_consumers():
            # Only the record count matters for the capture probe; replay rewrites
            # it from positions.  1 block per row is inside the documented range.
            visible_blocks = torch.ones((bs, 1), dtype=torch.int32, device=device)
            probe = ds41.quant_sparse_lightning_indexer_metadata(
                visible_blocks,
                cu_seqlens_q=cu_seqlens_q,
                seqused_q=None,
                seqused_k=local_cmp_lens,
                cmp_residual_k=None,
                batch_size=bs,
                num_heads_q=int(self._dsv4_index_n_heads),
                num_heads_k=1,
                head_dim=int(self._dsv4_index_head_dim),
                topk=int(self._dsv4_index_topk),
                quant_mode=1,
                candidate_block_size=block,
                mask_mode=3,
                cmp_ratio=candidate_ratio,
                layout_q="TND",
                layout_k="PA_BBND",
            )
            key = f"kernel_metadata_c{candidate_ratio}a_qsli"
            buf = self.graph_metadata.get(key)
            if buf is None or buf.shape != probe.shape:
                buf = torch.zeros_like(probe)
                self.graph_metadata[key] = buf
            metadata.kernel_metadata[f"c{candidate_ratio}a_qsli_metadata"] = buf

    def _dsv41_graph_index_page_table(self, metadata, ratio: int):
        """Captured buffer for one ratio's indexer-K block table.

        The indexer-K pool pages at ``DSV41_INDEX_PAGE_SIZE`` slots, so its block
        table is the FULL page table expanded to that granularity: full page p
        owns the consecutive index pages ``[k*p, k*p + k)`` with
        ``k = (page_size // ratio) // DSV41_INDEX_PAGE_SIZE`` (1 for ratio 2,
        2 for ratio 1).  That widens the row when ``k > 1``, so it always gets a
        dedicated buffer rather than aliasing ``block_tables``; aclgraph binds
        the address at capture and :meth:`_refresh_graph_dsv41_1d` rewrites it in
        place on every replay.
        """
        block_tables = getattr(metadata, "block_tables", None)
        if block_tables is None:
            return None
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DSV41_INDEX_PAGE_SIZE,
        )

        slots_per_page = int(self.page_size) // ratio
        assert slots_per_page % DSV41_INDEX_PAGE_SIZE == 0, (
            f"ratio-{ratio} index pages of {DSV41_INDEX_PAGE_SIZE} slots do "
            f"not tile a FULL page of {slots_per_page} slots"
        )
        blocks_per_page = slots_per_page // DSV41_INDEX_PAGE_SIZE
        shape = (block_tables.shape[0], block_tables.shape[1] * blocks_per_page)
        key = f"dsv41_index_page_table_c{ratio}"
        buf = self.graph_metadata.get(key)
        if buf is None or tuple(buf.shape) != shape:
            buf = torch.zeros(shape, dtype=torch.int32, device=block_tables.device)
            self.graph_metadata[key] = buf
        return buf

    def _refresh_graph_dsv41_1d(self, ctx) -> None:
        """Decode replay refresh: derive the rows from ``out_cache_loc``."""
        fm = ctx.fm
        n = int(ctx.bs)
        live = ctx.live_seq_lens[:n].to(torch.int64)
        raw_out_loc = ctx.forward_batch.out_cache_loc.reshape(-1).to(torch.int64)
        # The replay view's out_cache_loc is deliberately UNPADDED (raw_bs
        # entries; ``build_replay_fb_view`` passes the raw batch's), while
        # live/n span the padded bucket.  Only a truly empty vector is the
        # DP-idle case: a short-but-nonempty one is a padded live replay whose
        # first raw_bs entries are real slots, so scatter them into a padded
        # vector and let the mask below skip only the tail rows (out_loc 0 <
        # page_size).  Broadcasting the raw vector against `live` (the pre-fix
        # behavior) treats every padded replay as idle and silently drops the
        # whole step's compression / index-K writes.
        short_out_loc = raw_out_loc.numel() < n
        out_loc = torch.zeros_like(live)
        if raw_out_loc.numel() > 0:
            m = min(int(raw_out_loc.numel()), n)
            out_loc[:m] = raw_out_loc[:m]
        completes = (live > 0) & (out_loc >= int(self.page_size))
        positions = torch.clamp(live - 1, min=0)
        for ratio in self._dsv41_low_ratios():
            keep = completes & ((positions % ratio) == (ratio - 1))
            pos_dst = getattr(fm, f"positions_cmp_padding_c{ratio}")
            slot_dst = getattr(fm, f"c{ratio}_loc")
            if ratio == 1:
                # Per-token rows (native projection): identity, not compacted.
                pos_dst.copy_(torch.where(keep, positions, torch.zeros_like(positions)))
                slot_dst.copy_(
                    torch.where(
                        keep,
                        out_loc,
                        torch.full_like(out_loc, self._DSV41_SLOT_SKIP),
                    ).to(slot_dst.dtype)
                )
            else:
                self._dsv41_compact_1d(
                    pos_dst,
                    (positions // ratio) * ratio,
                    keep,
                    fill=0,
                )
                self._dsv41_compact_1d(
                    slot_dst,
                    out_loc // ratio,
                    keep,
                    fill=self._DSV41_SLOT_SKIP,
                )
            self._dsv41_debug_dump_1d(
                "replay",
                positions,
                out_loc,
                ctx.forward_batch,
                ratio,
                short_out_loc=short_out_loc,
            )

        # The indexer-K block table is the FULL page table expanded to the index
        # pool's page granularity (full page p -> index pages [k*p, k*p + k)).
        # The shared ``block_tables`` view has already been refreshed by the
        # parent, so re-derive the expansion into the captured buffers here.
        block_tables = getattr(fm, "block_tables", None)
        if block_tables is None:
            return
        from sglang.srt.layers.attention.dsv4.metadata import (
            expand_index_page_table,
        )
        from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
            DSV41_INDEX_PAGE_SIZE,
        )

        for ratio in self._dsv41_low_ratios():
            dst = getattr(fm, f"c{ratio}_index_page_table", None)
            if dst is None:
                continue
            dst.copy_(
                expand_index_page_table(
                    block_tables,
                    full_page_size=int(self.page_size),
                    compress_ratio=ratio,
                    index_page_size=DSV41_INDEX_PAGE_SIZE,
                )
            )


class DeepseekV4AscendAttnBackend(
    AscendAttnBackend,
    C4IndexerAscendBackendMixin,
    CompressorAscendBackendMixin,
    LowRatioCompressAscendBackendMixin,
):
    _DSV4_CP_LOCAL_FIELDS = (
        "actual_seq_lengths_q",
        "actual_seq_lengths_q_pa",
        "actual_seq_lengths_kv",
        "block_tables",
        "swa_page_table",
        "c4_page_table",
        "c128_page_table",
        "kernel_metadata",
        "c4_topk_indices",
        "positions_cmp_padding_c4",
        "positions_cmp_padding_c128",
        "c4_state_page_table",
        "c128_state_page_table",
        "c4_loc",
        "c128_loc",
        "c4_state_loc",
        "c128_state_loc",
        "start_pos",
        "seqused",
    )

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
        # Two-level indexer: the layer that publishes candidate blocks and the
        # consumer layers that rescore inside them. -1 / 0 on a plain V4 model.
        self._dsv41_candidate_source_layer = int(
            getattr(hf, "candidate_source_layer_id", -1)
        )
        self._dsv41_candidate_topk_blocks = int(getattr(hf, "candidate_topk_blocks", 0))
        self._dsv41_candidate_block_size = int(getattr(hf, "candidate_block_size", 0))
        # A layer that publishes candidates is only useful when a later index
        # source layer consumes them; read through _dsv41_has_candidate_consumers
        # because a draft worker empties the ratio list below.
        self._dsv41_index_source_layers = tuple(
            getattr(hf, "index_source_layer_ids", ())
        )
        # QSLI's 544B/8-token K records, allocated on first write; see
        # _dsv41_qsli_cache.
        self._dsv41_qsli_caches: dict = {}
        # Layers whose QSLI geometry has already been dumped (SGLANG_DSV41_DEBUG_QSLI).
        self._dsv41_debugged: set = set()
        if getattr(model_runner, "is_draft_worker", False):
            self._dsv4_compress_ratios = type(hf.compress_ratios)()
        self._dsv4_has_c4 = 4 in self._dsv4_compress_ratios
        self._dsv4_has_c128 = 128 in self._dsv4_compress_ratios
        # DeepSeek-V4.1 generation signal. Both generations report KVLayout.V4
        # on NPU, so the ratios themselves are the only discriminator.
        self._dsv4_has_low_ratio = any(
            ratio in (1, 2) for ratio in self._dsv4_compress_ratios
        )
        if self._dsv4_has_low_ratio:
            # MQSMLA fixes N1 = 64 and does not shard the query heads over TP
            # (mixed_quant_sparse_flash_mla_dsl/README.md L399 "N1=64 ... 固定").
            # A narrower per-rank width would feed the op the wrong head count,
            # so reject it loudly here instead of at the first attention call.
            assert self._dsv4_q_head_num == 64, (
                "DeepSeek-V4.1 on NPU needs all 64 query heads on every rank "
                f"(got {self._dsv4_q_head_num}): check num_attention_heads and "
                "the attention TP size."
            )
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
        self._is_eagle_algorithm = bool(
            model_runner.spec_algorithm is not None
            and model_runner.spec_algorithm.is_eagle()
            and not model_runner.spec_algorithm.is_frozen_kv_mtp()
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
        # High-water mark of written page-table columns per shared graph
        # buffer; see _copy_page_table_into_graph.
        self._graph_table_high_water: dict[str, int] = {}

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
        # The query is padded to an attn_tp multiple (prepare_mlp_sync_batch),
        # but seq_lens is not, so pad the tail with -1 ("skip") to match.
        num_padded_tokens = out_cache_loc.shape[0]
        if ori_sparse_indices.shape[0] < num_padded_tokens:
            pad_rows = num_padded_tokens - ori_sparse_indices.shape[0]
            pad = torch.full(
                (pad_rows, ori_sparse_indices.shape[1]),
                -1,
                dtype=ori_sparse_indices.dtype,
                device=ori_sparse_indices.device,
            )
            ori_sparse_indices = torch.cat([ori_sparse_indices, pad], dim=0)
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

    def prepare_dsv4_cp_metadata(self, forward_batch: ForwardBatch) -> None:
        if getattr(forward_batch, "dsv4_cp_metadata_prepared", False):
            return
        if getattr(forward_batch, "attn_cp_metadata", None) is None:
            return
        if not forward_batch.forward_mode.is_context_parallel_extend():
            return
        if forward_batch.forward_mode.is_target_verify():
            return

        strategy = get_cp_strategy()
        if strategy is None or strategy.cp_size <= 1:
            return

        fm = self.forward_metadata
        global_positions = forward_batch.positions
        if global_positions is None:
            return

        device = global_positions.device
        num_tokens = int(global_positions.shape[0])
        local_idx = strategy.local_q_indices(num_tokens, forward_batch).to(
            device=device, dtype=torch.long
        )
        if local_idx.numel() > 0:
            # Same bound the runner uses to shard model inputs (x[:total_seq_lens]).
            shard_bound = int(
                getattr(forward_batch.attn_cp_metadata, "total_seq_lens", num_tokens)
            )
            local_idx = local_idx[local_idx < shard_bound]
        local_positions = global_positions.index_select(0, local_idx)
        # Sharded model inputs are padded to per_rank_actual_token; pad rows get
        # position 0 so per-token metadata matches the sharded q length.
        from sglang.srt.layers.cp.padding import pad_local_rows

        local_positions = pad_local_rows(
            local_positions, forward_batch.attn_cp_metadata, dim=0
        )

        extend_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if extend_lens is None:
            seq_lens_cpu = getattr(forward_batch, "seq_lens_cpu", None)
            if seq_lens_cpu is not None:
                extend_lens = seq_lens_cpu.tolist()
            else:
                extend_lens = [num_tokens]
        extend_lens = [int(x) for x in extend_lens]
        real_num_tokens = min(sum(extend_lens), num_tokens)

        batch_ids_parts = []
        for batch_id, length in enumerate(extend_lens):
            if length <= 0:
                continue
            batch_ids_parts.append(
                torch.full((length,), batch_id, dtype=torch.long, device=device)
            )
        if batch_ids_parts:
            batch_ids = torch.cat(batch_ids_parts, dim=0)
        else:
            batch_ids = torch.empty(0, dtype=torch.long, device=device)
        if batch_ids.shape[0] < num_tokens:
            pad_len = num_tokens - batch_ids.shape[0]
            batch_ids = torch.cat(
                [batch_ids, torch.zeros(pad_len, dtype=torch.long, device=device)],
                dim=0,
            )
        elif batch_ids.shape[0] > num_tokens:
            batch_ids = batch_ids[:num_tokens]

        local_batch_ids = batch_ids.index_select(0, local_idx)
        valid_rows = local_idx < real_num_tokens
        # Pad rows reuse request 0's page table (in-bounds) and stay invalid.
        pad_rows = local_positions.shape[0] - local_batch_ids.shape[0]
        if pad_rows > 0:
            local_batch_ids = torch.cat(
                [local_batch_ids, local_batch_ids.new_zeros(pad_rows)]
            )
            valid_rows = torch.cat([valid_rows, valid_rows.new_zeros(pad_rows)])
        seqused_kv = torch.where(
            valid_rows,
            local_positions.to(torch.int32) + 1,
            torch.ones_like(local_positions, dtype=torch.int32),
        ).clamp(min=1)

        full_fields = {
            field: getattr(fm, field, None) for field in self._DSV4_CP_LOCAL_FIELDS
        }
        setattr(fm, "dsv4_cp_full_fields", full_fields)

        def _select_rows(table: Optional[torch.Tensor]):
            if table is None:
                return None
            if local_batch_ids.numel() == 0:
                return table.new_empty((0, *table.shape[1:]))
            return table.index_select(0, local_batch_ids)

        fm.block_tables = _select_rows(full_fields["block_tables"])
        fm.swa_page_table = _select_rows(full_fields["swa_page_table"])
        if self._dsv4_has_c4:
            fm.c4_page_table = _select_rows(full_fields["c4_page_table"])
        if self._dsv4_has_c128:
            fm.c128_page_table = _select_rows(full_fields["c128_page_table"])

        local_t = int(local_positions.shape[0])
        fm.actual_seq_lengths_q = torch.arange(
            1, local_t + 1, dtype=torch.int32, device=device
        )
        fm.actual_seq_lengths_q_pa = torch.arange(
            0, local_t + 1, dtype=torch.int32, device=device
        )
        fm.actual_seq_lengths_kv = seqused_kv
        fm.kernel_metadata = self._kernel_metadata_from_parts(
            bs=local_t,
            actual_seq_lengths_q_pa=fm.actual_seq_lengths_q_pa,
            actual_seq_lengths_kv=fm.actual_seq_lengths_kv,
        )
        if self._dsv4_has_c4:
            fm.c4_topk_indices = torch.full(
                (local_t, self._dsv4_index_topk),
                -1,
                dtype=torch.int32,
                device=device,
            )

        forward_batch.dsv4_cp_metadata_prepared = True
        forward_batch.dsv4_cp_global_positions = global_positions
        forward_batch.dsv4_cp_local_positions = local_positions

    @contextmanager
    def use_dsv4_cp_full_metadata(self, forward_batch: ForwardBatch):
        fm = self.forward_metadata
        full_fields = getattr(fm, "dsv4_cp_full_fields", None)
        if not full_fields:
            yield
            return

        local_fields = {
            field: getattr(fm, field, None) for field in self._DSV4_CP_LOCAL_FIELDS
        }
        previous_positions = getattr(forward_batch, "positions", None)
        try:
            for field, value in full_fields.items():
                setattr(fm, field, value)
            global_positions = getattr(forward_batch, "dsv4_cp_global_positions", None)
            if global_positions is not None:
                forward_batch.positions = global_positions
            yield
        finally:
            for field, value in local_fields.items():
                setattr(fm, field, value)
            forward_batch.positions = previous_positions

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
        if is_npu_arch35():
            metadata.dsv4_cycle_state_block_table = torch.zeros(
                bs, dtype=torch.int32, device=device
            )
            # A5 addresses state by request bank (CYCLE), so the explicit
            # per-ratio location tables are never read. Leave them empty: the
            # DeepSeek-V4.1 ratio-2 pair pool is a plain CompressStatePool and
            # has no ``dummy_state_loc`` sentinel to seed them with.
            metadata.dsv4_explicit_state_block_tables = {}
        else:
            metadata.dsv4_cycle_state_block_table = None
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

        if self._dsv4_has_low_ratio:
            # DeepSeek-V4.1: MQSMLA / QLI dispatch tables plus the window and
            # topk-length tensors the attention and indexer ops consume.
            self._init_dsv41_graph_metadata(metadata, bs, tokens_per_req, device)

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

    def _copy_page_table_into_graph(self, key: str, src: torch.Tensor) -> None:
        # Graph page tables live in buffers shared across bs buckets (each
        # bucket's metadata holds a row slice), refreshed in place on replay.
        full = self.graph_metadata[key]
        r, c = src.shape
        full[:r, :c].copy_(src)
        high = self._graph_table_high_water.get(key, 0)
        if c < high:
            # Full height, not just this replay's slice: other buckets'
            # replays may have written rows beyond this bucket's row count.
            full[:, c:high].fill_(-1)
        elif c > high:
            self._graph_table_high_water[key] = c

    @staticmethod
    def _copy_1d_with_zero_tail(dst: torch.Tensor, src: Optional[torch.Tensor]) -> None:
        if src is None:
            dst.zero_()
            return
        n = src.numel()
        assert n <= dst.shape[0], (
            f"graph replay 1D metadata overflow: src={n} > dst={dst.shape[0]}"
        )
        if n > 0:
            if src.dtype != dst.dtype:
                src = src.to(dst.dtype)
            dst[:n].copy_(src)
        if n < dst.shape[0]:
            dst[n:].fill_(0)

    @staticmethod
    def _stable_compact_1d(
        dst: torch.Tensor, values: torch.Tensor, keep: torch.Tensor
    ) -> None:
        """Compact selected values into a fixed graph buffer without NonZero.

        For every selected element, ``cumsum(keep) - 1`` is exactly its ordinal
        in ``nonzero(keep)``.  Selected elements therefore have unique scatter
        destinations, while rejected elements contribute integer zero only.
        This preserves the stable boolean-index order without a dynamic output
        shape or a device-to-host size read.
        """
        dst.zero_()
        if dst.numel() == 0 or values.numel() == 0:
            return

        values = values.reshape(-1)
        keep = keep.reshape(-1)
        if values.numel() != keep.numel():
            raise ValueError(
                "stable compact requires value/mask size equality, got "
                f"{values.numel()} and {keep.numel()}"
            )

        ranks = torch.cumsum(keep.to(torch.int64), dim=0) - 1
        in_bounds = keep & (ranks < dst.numel())
        safe_ranks = ranks.clamp(min=0, max=dst.numel() - 1)
        compact_values = torch.where(
            in_bounds,
            values.to(dst.dtype),
            torch.zeros_like(values, dtype=dst.dtype),
        )
        dst.scatter_add_(0, safe_ranks, compact_values)

    def _fill_verify_positions_cmp_padding_one_device(
        self,
        positions: torch.Tensor,
        dst: torch.Tensor,
        ratio: int,
        live_seq_lens: torch.Tensor,
        n_draft: int,
    ) -> None:
        """Device-only fixed-shape equivalent of the eager CPU reference path."""
        if ratio not in self._dsv4_compress_ratios or positions.numel() == 0:
            dst.zero_()
            return

        n_draft = int(n_draft)
        request_num = positions.shape[0] // n_draft
        if request_num == 0:
            dst.zero_()
            return
        if live_seq_lens.device != positions.device:
            raise ValueError(
                "device verify compression metadata requires live_seq_lens and "
                "positions on the same device"
            )

        live_seq_lens = live_seq_lens[:request_num]
        token_offsets = torch.arange(
            1,
            n_draft + 1,
            dtype=live_seq_lens.dtype,
            device=live_seq_lens.device,
        )
        absolute_lengths = live_seq_lens.view(-1, 1) + token_offsets.view(1, -1)
        boundary_mask = ((absolute_lengths % ratio) == 0) & (
            live_seq_lens.view(-1, 1) > 0
        )
        # Match the CPU reference exactly: select the boundary token from the
        # request-major positions array, then move RoPE to the group's first token.
        values = positions[: request_num * n_draft].reshape(-1) + (1 - ratio)
        self._stable_compact_1d(dst, values, boundary_mask.reshape(-1))

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
                self._copy_page_table_into_graph(key, result[key])

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
            dst = getattr(fm, f"positions_cmp_padding_c{ratio}")
            self._stable_compact_1d(
                dst,
                positions_last.to(torch.int64) + (1 - ratio),
                should_compress,
            )
        fm.start_pos.copy_(positions_last.to(torch.int32))
        fm.seqused.copy_(valid.to(torch.int32))

        if self._dsv4_has_low_ratio:
            self._refresh_graph_dsv41_1d(ctx)

    def _refresh_graph_target_verify_compress_1d_direct(self, ctx) -> None:
        fm = ctx.fm
        fm.start_pos.copy_(ctx.live_seq_lens.to(torch.int32))
        if self._is_eagle_algorithm:
            self._fill_verify_positions_cmp_padding_one_device(
                ctx.forward_batch.positions,
                fm.positions_cmp_padding_c4,
                4,
                ctx.live_seq_lens,
                n_draft=ctx.tokens_per_bs,
            )
            self._fill_verify_positions_cmp_padding_one_device(
                ctx.forward_batch.positions,
                fm.positions_cmp_padding_c128,
                128,
                ctx.live_seq_lens,
                n_draft=ctx.tokens_per_bs,
            )
        else:
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
        self._copy_page_table_into_graph("swa_page_table", swa_src)

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
        if self._dsv4_has_low_ratio:
            # V4.1: the ds41 metadata ops are recomputed and copied into the
            # graph buffers by the mixin helpers themselves (see their
            # ``kernel_metadata.get(key)`` refresh).
            previous = self.forward_metadata
            self.forward_metadata = fm
            try:
                self._dsv41_attention_kernel_metadata(ctx.forward_batch)
                self._dsv41_index_kernel_metadata(ctx.forward_batch)
            finally:
                self.forward_metadata = previous
            return
        kernel_metadata_new = self._kernel_metadata_from_parts(
            bs=ctx.bs,
            actual_seq_lengths_q_pa=fm.actual_seq_lengths_q_pa,
            actual_seq_lengths_kv=fm.actual_seq_lengths_kv,
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

        if is_npu_arch35():
            ctx.fm.dsv4_cycle_state_block_table.copy_(
                ctx.forward_batch.req_pool_indices[: ctx.bs]
            )

        self._refresh_graph_seq_metadata(ctx)
        self._refresh_graph_compress_page_tables_direct(ctx)

        if ctx.graph_mode.is_decode():
            self._refresh_graph_decode_compress_1d_direct(ctx)
        elif ctx.graph_mode.is_target_verify():
            if ctx.is_idle_replay and ctx.has_compress:
                self._clear_graph_target_verify_metadata(ctx)
            elif ctx.active_target_verify:
                self._refresh_graph_target_verify_compress_1d_direct(ctx)
            elif self._dsv4_has_low_ratio and not ctx.is_idle_replay:
                # DeepSeek-V4.1 has no low-ratio verify refresh: the branch above
                # is gated on c4/c128 (`has_compress`), so `_refresh_graph_dsv41_1d`
                # only ever runs from the decode arm.  A verify replay would then
                # scatter the compressed rows of these tokens to the *previous*
                # step's slots and silently corrupt the compressed / index pools
                # (the eager path already fails loudly with a missing c{r}_loc).
                # A5 DSV4.1 has no speculative decoding today; fail loudly rather
                # than corrupt if one is ever enabled.
                raise RuntimeError(
                    "DeepSeek-V4.1 low-ratio graphs have no target-verify "
                    "metadata refresh (that arm is gated on c4/c128); "
                    "speculative decoding is not supported for DSV4.1 on NPU."
                )

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
        # The V4.1 builders below reuse the previous step's graph-allocated
        # buffers, matched by key and shape, instead of rebinding them; the
        # attribute therefore has to exist before they read it, while
        # ``init_forward_metadata`` assigns the result only afterwards.
        if getattr(fm, "kernel_metadata", None) is None:
            fm.kernel_metadata = {}
        if self._dsv4_has_low_ratio:
            # V4.1 metadata comes from the ds41 operators only: MQSMLA drives the
            # attention and QLI drives the indexer. The V4 entries (built from
            # npu_kv_quant_sparse_attn_sharedkv metadata) do not apply.
            return self._dsv41_attention_kernel_metadata(
                forward_batch
            ) | self._dsv41_index_kernel_metadata(forward_batch)
        return self._kernel_metadata_from_parts(
            bs=forward_batch.batch_size,
            actual_seq_lengths_q_pa=fm.actual_seq_lengths_q_pa,
            actual_seq_lengths_kv=fm.actual_seq_lengths_kv,
        )

    def _kernel_metadata_from_parts(
        self,
        *,
        bs: int,
        actual_seq_lengths_q_pa: torch.Tensor,
        actual_seq_lengths_kv: torch.Tensor,
    ) -> dict:
        fm = self.forward_metadata
        common = {
            **_sparse_attn_kv_quant_kwargs(),
            "cu_seqlens_q": actual_seq_lengths_q_pa,
            "seqused_kv": actual_seq_lengths_kv,
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
        c1a_kwargs = base_kwargs | common
        if self._is_dspark_draft_worker:
            cu_q_cpu = fm.actual_seq_lengths_q_pa_cpu
            seq_kv_cpu = fm.seq_lens_cpu_int
            if (
                cu_q_cpu is None
                or seq_kv_cpu is None
                or cu_q_cpu.device.type != "cpu"
                or seq_kv_cpu.device.type != "cpu"
            ):
                raise RuntimeError(
                    "DSpark host metadata requires CPU query and KV length mirrors"
                )
            if cu_q_cpu.numel() > bs + 1:
                cu_q_cpu = cu_q_cpu[: bs + 1]
            host_inputs = {
                "seqused_kv": seq_kv_cpu[:bs].to(dtype=torch.int32).contiguous()
            }
            host_inputs["cu_seqlens_q"] = cu_q_cpu.to(dtype=torch.int32).contiguous()
            metadata_op, _ = _sparse_attn_ops(is_dspark=True)
            c1a_metadata = metadata_op(**(c1a_kwargs | host_inputs))
        else:
            metadata_op, _ = _sparse_attn_ops()
            c1a_metadata = metadata_op(**c1a_kwargs)
        kernel_metadata = {"c1a_metadata": c1a_metadata}

        if self._dsv4_has_c4:
            c4a_overrides = {
                "cmp_ratio": 4,
                "has_cmp_kv": True,
                "cmp_topk": self._dsv4_index_topk,
            }
            c4a_kwargs = c1a_kwargs | c4a_overrides
            metadata_op, _ = _sparse_attn_ops()
            kernel_metadata["c4a_metadata"] = metadata_op(**c4a_kwargs)

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
            metadata_op, _ = _sparse_attn_ops()
            kernel_metadata["c128a_metadata"] = metadata_op(**c128a_kwargs)

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
        # Generation-exclusive: V4.1 only ever declares compress_ratios in
        # (0, 1, 2) (recipes configuration_deepseek.py L218), V4 only in
        # (0, 4, 128). Mixing the two would silently route a ratio to an
        # operator that has no metadata for it.
        if compress_ratio not in (
            (0, 1, 2) if self._dsv4_has_low_ratio else (0, 4, 128)
        ):
            raise ValueError(
                "V4 attention expects compress_ratio in (0, 4, 128); "
                f"V4.1 in (0, 1, 2); got {compress_ratio}"
            )
        # idle ranks only feed the MoE collectives; skip attn + store_cache and return zeros
        if forward_batch.forward_mode.is_idle():
            return torch.zeros_like(q)
        # MQALayer prepass already stores K and passes save_kv_cache=False; True callers still get the write
        if save_kv_cache:
            self.store_cache(
                layer_id=layer.layer_id, swa_k=k, forward_batch=forward_batch
            )
        if self._dsv4_has_low_ratio:
            # V4.1 uses ds41.mixed_quant_sparse_flash_mla for every ratio: it
            # expresses both the window and the compressed side as explicit
            # sparse indices, so ratio 0 is not the old "swa" op either.
            return self._forward_low_ratio_attention(
                q, layer, forward_batch, attn_sink, compress_ratio
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
            **_sparse_attn_kv_quant_kwargs(),
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
        ori_sparse_indices = getattr(fm, "ori_sparse_indices", None)
        if ori_sparse_indices is not None:
            attn_kwargs["ori_sparse_indices"] = ori_sparse_indices
        q_arg = attn_kwargs.pop("q")
        _, attn_op = _sparse_attn_ops(self._is_dspark_draft_worker)
        out, _ = attn_op(q_arg, **attn_kwargs)
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
            **_sparse_attn_kv_quant_kwargs(),
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
        _, attn_op = _sparse_attn_ops()
        out, _ = attn_op(q_arg, **attn_kwargs)
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
        boundary_mask = (abs_positions % ratio == 0) & (seq_lens_cpu.view(-1, 1) > 0)
        indices = torch.nonzero(boundary_mask.flatten(), as_tuple=False).flatten()

        if indices.numel() == 0:
            return
        # This tiny H2D copy runs on the verify metadata path. Keep it blocking:
        # on NPU, a non-blocking copy from a short-lived pinned CPU tensor can
        # surface later as an unrelated CopyKernel stream failure.
        indices = indices[: dst.numel()].to(device=positions.device)
        # ``indices`` selects the final token of each newly completed group.
        # The compressed KV applies RoPE at the group's first token, matching
        # the decode path and ``comp_pos = (position // ratio) * ratio``.
        compressed_positions = torch.gather(positions, 0, indices) + (1 - ratio)
        dst[: indices.numel()].copy_(compressed_positions)

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

        return DSV4OutCacheLoc(
            out_full_loc=full_steps[step_id],
            out_swa_loc=swa_steps[step_id],
            out_c4_loc=(
                None if bundle.out_c4_loc is None else bundle.out_c4_loc.new_empty((0,))
            ),
            out_c128_loc=(
                None
                if bundle.out_c128_loc is None
                else bundle.out_c128_loc.new_empty((0,))
            ),
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
