"""DeepSeek V4 attention backend on Ascend NPU.

NPU counterpart of the CUDA ``DeepseekV4AttnBackend``. Bridges V4 model code
(which expects ``CompressorBackendMixin`` + ``C4IndexerBackendMixin`` on top
of ``AttentionBackend``) with ``AscendAttnBackend``. MRO: ``AscendAttnBackend``
supplies the NPU forward / metadata surface; the V4 mixins add the c4 / c128
compress + indexer helpers. ``forward()`` routes by ``compress_ratio``:
0 / 1 → dense SWA (``_forward_dense``), 4 / 128 → sparse compressed
(``_forward_compressed`` via ``npu_sparse_attn_sharedkv``).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Optional

import torch

# custom_ops registers torch.ops.custom.npu_sparse_attn_sharedkv_metadata,
# npu_sparse_attn_sharedkv, npu_quant_lightning_indexer and friends. The
# V4 ascend backend has no pure-torch fallback for those ops, so if the
# import fails we must fail fast with a clear message rather than crash
# later with an opaque AttributeError on torch.ops.custom.<name>.
try:
    import custom_ops  # noqa: F401
except ImportError as e:
    raise ImportError(
        "DeepSeek-V4 ascend attention backend requires the `custom_ops` "
        "wheel that ships with the Ascend cann-8.5.0-a3 image (registers "
        "torch.ops.custom.npu_sparse_attn_sharedkv_*, "
        "npu_quant_lightning_indexer, npu_hc_pre/post, etc.). The package "
        "is normally at /usr/local/python*/site-packages/custom_ops. "
        f"Original ImportError: {e}"
    ) from e

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.attention.ascend_backend import AscendAttnBackend
from sglang.srt.layers.attention.dsv4.compressor import CompressorBackendMixin
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.layers.attention.dsv4.indexer import C4IndexerBackendMixin

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _walsh_hadamard_matrix(n: int, dtype: torch.dtype, device) -> torch.Tensor:
    # bf16 Sylvester matrix with the n**-0.5 norm factor baked in by dividing
    # by sqrt(2) at each doubling (log2(n) steps total). `dtype` is accepted
    # for backward-compat with callers; ascend builds in bf16 only.
    cache = _walsh_hadamard_matrix._cache  # type: ignore[attr-defined]
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


_walsh_hadamard_matrix._cache = {}  # type: ignore[attr-defined]


def _apply_hadamard(inp: torch.Tensor, hadamard_matrix: torch.Tensor) -> torch.Tensor:
    # The n**-0.5 scale is already baked into `hadamard_matrix` (see
    # _walsh_hadamard_matrix above), so this is just `inp @ H` then bf16 cast.
    init_shape = inp.shape
    flat = inp.view(-1, hadamard_matrix.shape[0])
    return flat.matmul(hadamard_matrix).view(init_shape).to(torch.bfloat16)


class DeepseekV4AscendAttnBackend(
    AscendAttnBackend, C4IndexerBackendMixin, CompressorBackendMixin
):
    """V4 attention dispatcher for Ascend NPU.

    Method resolution order is intentional: AscendAttnBackend ships the
    NPU-side ``init_forward_metadata`` / ``forward_extend`` / ``forward_decode``
    surface; the V4 mixins only add the c4/c128 compress + c4 indexer
    helpers. When both define a method (e.g. ``forward``), MRO picks
    Ascend's, which is what we want for the regular MQA path.
    """

    def __init__(
        self,
        model_runner: "ModelRunner",
        speculative_step_id: int = 0,
    ):
        super().__init__(model_runner, speculative_step_id=speculative_step_id)
        cfg = model_runner.model_config
        self._dsv4_config = cfg
        tp_size = get_attention_tp_size()
        self._dsv4_q_head_num = cfg.num_attention_heads // tp_size
        self._dsv4_kv_head_num = 1  # V4 MQA / latent
        # V4-Flash config.json sets head_dim=512 directly (qk_nope_head_dim is
        # null in the HF config); pass head_dim verbatim to the metadata kernel.
        self._dsv4_head_dim = cfg.head_dim
        hf = getattr(cfg, "hf_config", cfg)
        self._dsv4_index_topk = hf.index_topk
        self._dsv4_index_n_heads = hf.index_n_heads
        self._dsv4_index_head_dim = hf.index_head_dim
        self._dsv4_compress_ratios = hf.compress_ratios
        self._dsv4_has_c4 = 4 in self._dsv4_compress_ratios
        self._dsv4_has_c128 = 128 in self._dsv4_compress_ratios
        self._dsv4_sliding_window_size = (
            cfg.sliding_window_size if cfg.sliding_window_size is not None else 128
        )

    # ------------------------------------------------------------------
    # V4-specific metadata + dispatch.
    # ------------------------------------------------------------------

    def _init_dsv4_graph_buffers(self, *, max_bs: int, max_num_tokens: int) -> None:
        """Preallocate V4-Flash graph buffers reused across cuda graph capture.

        Called once from the base class init_cuda_graph_state. All buffers are
        zero-initialized on self.device; capture binds them to forward_metadata,
        replay copies into them in-place.
        """
        device = self.device
        # Maximum number of pages a single request can occupy in graph mode.
        # Pulled from the existing graph_metadata['block_tables'] shape, which the
        # base class already allocated.
        block_tables_shape = self.graph_metadata["block_tables"].shape  # [max_bs, max_pages]
        max_pages = block_tables_shape[1]

        # swa page table — same shape as block_tables (full kv pages).
        # Sentinel = -1 ("no valid page"); 0 would collide with legitimate
        # page id 0 and let the kernel read wrong tokens past the tail.
        self.graph_metadata["swa_page_table"] = torch.full(
            (max_bs, max_pages), -1, dtype=torch.int32, device=device
        )

        # c4 / c128 page tables — allocate the full max_pages width.
        # Using max_pages // R + 1 (an apparent "1/R" optimization) is unsafe:
        # _compute_compress_locs can emit more cols than pages/R at certain
        # seq-len alignments / SWA edges, causing aclnnInplaceCopy 161002
        # ("[2, 33] vs [2, 34] cannot broadcast") during replay.
        self.graph_metadata["c4_page_table"] = torch.full(
            (max_bs, max_pages), -1, dtype=torch.int32, device=device
        )
        self.graph_metadata["c128_page_table"] = torch.full(
            (max_bs, max_pages), -1, dtype=torch.int32, device=device
        )
        self.graph_metadata["c4_state_page_table"] = torch.zeros(
            (max_bs, max_pages), dtype=torch.int32, device=device
        )
        self.graph_metadata["c128_state_page_table"] = torch.zeros(
            (max_bs, max_pages), dtype=torch.int32, device=device
        )

        # 1 kernel_metadata slot per ratio, 1024 int32 entries per source ref.
        for key in (
            "kernel_metadata_c1a",
            "kernel_metadata_c4a",
            "kernel_metadata_c128a",
            "kernel_metadata_li_quant",
        ):
            self.graph_metadata[key] = torch.zeros(
                1024, dtype=torch.int32, device=device
            )

        # c4_topk_indices: per-token sparse-index buffer for the c4 indexer.
        # Shape is [total_tokens, index_topk] int32; preallocate at max_num_tokens
        # so any captured bs fits, and slice to [:T, :] in capture. -1 is the
        # "no valid index" sentinel that npu_sparse_attn_sharedkv expects.
        self.graph_metadata["c4_topk_indices"] = torch.full(
            (max_num_tokens, self._dsv4_index_topk),
            -1,
            dtype=torch.int32,
            device=device,
        )

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional["SpecInput"],
    ):
        """Capture-time metadata setup for V4-Flash on NPU.

        Calls the base class to populate generic fields (block_tables, seq_lens,
        actual_seq_lengths_q). Then attaches V4-specific graph buffers
        (preallocated in _init_dsv4_graph_buffers) to the per-bs ForwardMetadata
        and fills fixed-shape per-request tensors (actual_seq_lengths_q_pa /
        _kv / _q_cmp). The dynamic content of the loc/page_table buffers is
        written during replay (where the live forward_batch is available).
        """
        super().init_forward_metadata_capture_cuda_graph(
            bs=bs,
            num_tokens=num_tokens,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
        )
        metadata = self.graph_metadata[bs]
        device = self.device

        # tokens_per_bs: 1 for decode/idle, speculative_num_draft_tokens for
        # target_verify / draft_extend / draft_extend_v2. Match the base class's
        # branch on forward_mode (lines 496-508 of ascend_backend.py).
        if (
            forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
            or forward_mode.is_draft_extend()
        ):
            tokens_per_bs = self.speculative_num_draft_tokens
        else:
            tokens_per_bs = 1

        # actual_seq_lengths_q_pa: cumulative q-lengths WITH leading 0, length bs+1.
        # Decode: [0, 1, 2, ..., bs]. Target_verify: [0, n_draft, 2n_draft, ..., bs*n_draft].
        metadata.actual_seq_lengths_q_pa = torch.arange(
            0,
            bs * tokens_per_bs + tokens_per_bs,
            tokens_per_bs,
            dtype=torch.int32,
            device=device,
        )

        # actual_seq_lengths_kv: KV lengths per request; shape (bs,) to match
        # eager init_forward_metadata (lines 529/551/555) and the kernel's
        # expected layout (kernel infers bs from len(cu_seqlens_q) - 1 and
        # reads seqused_kv with that exact length). Initialized non-zero so the
        # captured kernel records valid attention work; replay overwrites with
        # real seq_lens in-place.
        metadata.actual_seq_lengths_kv = torch.ones(
            bs, dtype=torch.int32, device=device,
        )

        # Bind preallocated V4 page tables to metadata, sliced to [:bs, :].
        metadata.swa_page_table = self.graph_metadata["swa_page_table"][:bs, :]
        metadata.c4_page_table = self.graph_metadata["c4_page_table"][:bs, :]
        metadata.c128_page_table = self.graph_metadata["c128_page_table"][:bs, :]
        metadata.c4_state_page_table = self.graph_metadata["c4_state_page_table"][:bs, :]
        metadata.c128_state_page_table = self.graph_metadata["c128_state_page_table"][:bs, :]

        # Per-capture-bs loc buffers (size scales with tokens_per_bs and bs).
        # int64 matches what eager _compute_compress_locs produces for the loc
        # tensors via .to(torch.int64) in the same backend file.
        n_tok = bs * tokens_per_bs
        metadata.swa_loc = torch.zeros(n_tok, dtype=torch.int64, device=device)
        metadata.c4_loc = torch.zeros(n_tok, dtype=torch.int64, device=device)
        metadata.c128_loc = torch.zeros(n_tok, dtype=torch.int64, device=device)
        metadata.c4_state_loc = torch.zeros(n_tok, dtype=torch.int64, device=device)
        metadata.c128_state_loc = torch.zeros(n_tok, dtype=torch.int64, device=device)

        # Fused-compressor metadata buffers. Decode-mode size for
        # positions_cmp_padding is min(n_tok, n_tok//ratio + bs) which
        # equals bs for decode (tokens_per_bs=1). For target_verify /
        # draft modes, n_tok = bs * n_draft and the upper bound is
        # bs * (n_draft // ratio + 1) ≤ n_tok; size n_tok covers both
        # cases. int64 dtype matches what _compute_compress_locs emits.
        c4_pad   = min(n_tok, n_tok // 4   + bs)
        c128_pad = min(n_tok, n_tok // 128 + bs)
        metadata.positions_cmp_padding_c4 = torch.zeros(
            c4_pad, dtype=torch.int64, device=device
        )
        metadata.positions_cmp_padding_c128 = torch.zeros(
            c128_pad, dtype=torch.int64, device=device
        )
        metadata.start_pos = torch.zeros(bs, dtype=torch.int32, device=device)
        metadata.seqused = torch.zeros(bs, dtype=torch.int32, device=device)

        # kernel_metadata dict points at preallocated 1024-int32 buffers.
        metadata.kernel_metadata = {
            "c1a_metadata": self.graph_metadata["kernel_metadata_c1a"],
            "c4a_metadata": self.graph_metadata["kernel_metadata_c4a"],
            "c128a_metadata": self.graph_metadata["kernel_metadata_c128a"],
            "li_quant_metadata": self.graph_metadata["kernel_metadata_li_quant"],
        }

        # c4_topk_indices is preallocated in _init_dsv4_graph_buffers; bind a
        # [:T, :] slice so the lazy seed in _forward_compressed sees a non-None
        # tensor and skips its allocator. Real indexer.forward overwrites the
        # contents via npu_quant_lightning_indexer at runtime.
        T = bs * tokens_per_bs
        metadata.c4_topk_indices = self.graph_metadata["c4_topk_indices"][:T, :]

        self.forward_metadata = metadata

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional["SpecInput"],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """In-place metadata refresh for V4-Flash graph replay.

        Three phases:
          1. Base class fills block_tables / seq_lens.
          2. V4 kv lengths and q_cmp updated in place.
          3. _compute_compress_locs / _kernel_metadata_from_parts results copied
             into preallocated graph buffers.

        All copies are .copy_(src) into existing tensors; no fresh allocations
        on the graph stream. The live forward_batch is obtained from
        self._replay_forward_batch, set by cuda_graph_runner.replay_prepare
        before calling init_forward_metadata_replay_cuda_graph.
        """
        super().init_forward_metadata_replay_cuda_graph(
            bs=bs,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_sum=seq_lens_sum,
            encoder_lens=encoder_lens,
            forward_mode=forward_mode,
            spec_info=spec_info,
            seq_lens_cpu=seq_lens_cpu,
        )
        fm = self.forward_metadata

        forward_batch = getattr(self, "_replay_forward_batch", None)
        if forward_batch is None:
            raise RuntimeError(
                "V4 graph replay called without a forward_batch — "
                "cuda_graph_runner.replay_prepare must set "
                "attn_backend._replay_forward_batch before calling replay_cuda_graph."
            )

        # tokens_per_bs: matches base + capture branches on forward_mode.
        if (
            forward_mode.is_target_verify()
            or forward_mode.is_draft_extend_v2()
            or forward_mode.is_draft_extend()
        ):
            tokens_per_bs = self.speculative_num_draft_tokens
        else:
            tokens_per_bs = 1

        # Phase 2: kv lengths (and any spec-decode adjustment matches base).
        # The base class above already adjusted seq_lens in-place for target_verify
        # / decode+spec_info; copy the final adjusted values into fm.actual_seq_lengths_kv.
        # fm.actual_seq_lengths_kv has shape (bs,) (bound in capture); clamp to 1 so
        # padded slots (real seq_lens=0 beyond raw_bs) and capture-time zero seq_lens
        # don't trip the kernel's seqused_kv >= 1 validation.
        #
        # Source: the device-side `seq_lens` arg is `buffers.seq_lens[:bs]` which is
        # NOT populated under the NPU graph runner — that runner refreshes seq_lens
        # for the captured graph via Graph.update(cpu_update_input={...}) targeting
        # `actual_seq_lengths_kv` directly, leaving `buffers.seq_lens` at the init
        # fill (0). The CPU param `seq_lens_cpu` IS populated (CPU-side .copy_ in
        # populate_from_forward_batch is synchronous), so use it as the live source.
        assert seq_lens_cpu is not None, (
            "V4 graph replay requires seq_lens_cpu — buffers.seq_lens is stale on "
            "NPU (Graph.update only refreshes fm.actual_seq_lengths_kv inside the "
            "captured graph, not the device-side buffers.seq_lens)."
        )
        live_seq_lens = seq_lens_cpu[:bs].to(
            device=seq_lens.device, dtype=torch.int32
        )
        fm.actual_seq_lengths_kv.copy_(live_seq_lens.clamp(min=1))

        # Phase 3: compress locs via shared helper (Task 2 _compute_compress_locs).
        pool = forward_batch.token_to_kv_pool
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        out_cache_loc = forward_batch.out_cache_loc
        device = seq_lens.device

        result = self._compute_compress_locs(
            pool=pool,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices[:bs],
            seq_lens=live_seq_lens,
            out_cache_loc=out_cache_loc,
            is_decode=forward_mode.is_decode(),
            bs=bs,
            device=device,
            req_to_token_pool=forward_batch.req_to_token_pool,
            out_cache_loc_dsv4=forward_batch.out_cache_loc_dsv4,
            is_graph=True,
        )

        # In-place copy result into preallocated fm buffers.
        # Page tables are 2-D (max_bs, max_pages); -1 the tail per row so
        # unused slots are an invalid-page sentinel (matches reference impl
        # and the initial fill in _init_dsv4_graph_buffers).
        def _copy_2d(dst: torch.Tensor, src: torch.Tensor, val: int) -> None:
            dst.fill_(val)
            dst[: src.shape[0], : src.shape[1]].copy_(src)

        # Loc tensors are 1-D flat; zero the tail (loc arrays are token-level
        # offsets, 0 is a benign default).
        def _copy_1d(dst: torch.Tensor, src: torch.Tensor) -> None:
            dst.fill_(0)
            dst[: src.shape[0]].copy_(src)

        for key in (
            "c4_page_table", "c128_page_table",
            "c4_state_page_table", "c128_state_page_table",
        ):
            if key in result:
                _copy_2d(getattr(fm, key), result[key], 0 if 'state' in key else -1)
        for key in ("c4_loc", "c128_loc", "c4_state_loc", "c128_state_loc"):
            if key in result:
                _copy_1d(getattr(fm, key), result[key])

        # Fused-compressor metadata (decode-path only; capture skips other modes).
        # positions_cmp_padding tail stays 0 — which is a benign position used as
        # padding and masked away by seqused at op time.
        for key in (
            "positions_cmp_padding_c4",
            "positions_cmp_padding_c128",
            "start_pos",
            "seqused",
        ):
            if key in result and hasattr(fm, key) and getattr(fm, key) is not None:
                _copy_1d(getattr(fm, key), result[key])

        # swa_loc — eager path uses pool.translate_loc_from_full_to_swa(out_cache_loc).
        # _compute_compress_locs does NOT produce this (per Task 2's design); compute
        # here and copy into the preallocated buffer.
        swa_loc = pool.translate_loc_from_full_to_swa(out_cache_loc).to(torch.int64)
        _copy_1d(fm.swa_loc, swa_loc)

        # swa_page_table — eager init_forward_metadata sets this directly from
        # block_tables_swa or block_tables. Replay does the same in-place into
        # the preallocated swa_page_table buffer. Use fm.block_tables_swa if the
        # backend is hybrid-SWA (set by the base class replay), else fm.block_tables.
        swa_src = (
            fm.block_tables_swa if fm.block_tables_swa is not None else fm.block_tables
        )
        _copy_2d(fm.swa_page_table, swa_src, -1)
        # The base graph replay 0-pads block_tables{,_swa} past the real pages
        # (AscendAttnBackend: block_tables_swa[:bs, max_seq_pages:].fill_(0)). The
        # full-width copy above thus overwrites the -1 sentinel with page id 0 in
        # the tail, so the ori/swa sparse-attn kernel (oriMaxBlockNumPerBatch =
        # block_table.shape[1], full width) could read page 0 past a request's
        # real pages. Restore the -1 sentinel beyond the valid pages so swa
        # matches the c4/c128 page tables (and the reference impl, which copies a
        # tight src into a -1-filled buffer). max_seq_pages mirrors the base.
        if bs > 0:
            # Over-estimate max_len by the spec draft tokens (a safety margin):
            # over-estimating only leaves a few extra 0s in the UNREAD tail,
            # while under-estimating would -1 over valid pages. Plain decode
            # adds 0.
            _spec = int(getattr(self, "speculative_num_draft_tokens", 0) or 0)
            max_len = int(seq_lens_cpu[:bs].max()) + _spec
            max_seq_pages = (max_len + self.page_size - 1) // self.page_size
            if 0 < max_seq_pages < fm.swa_page_table.shape[1]:
                fm.swa_page_table[:, max_seq_pages:].fill_(-1)

        # Kernel metadata refresh via shared helper from Task 1.
        kernel_metadata_new = self._kernel_metadata_from_parts(
            bs=bs,
            actual_seq_lengths_q_pa=fm.actual_seq_lengths_q_pa,
            actual_seq_lengths_kv=fm.actual_seq_lengths_kv,
            block_tables=fm.block_tables,
            max_seqlen_q=tokens_per_bs,
            is_nextn=False,
        )
        # In-place copy each entry into the preallocated kernel_metadata buffer.
        for key in ("c1a_metadata", "c4a_metadata", "c128a_metadata", "li_quant_metadata"):
            if key in kernel_metadata_new:
                fm.kernel_metadata[key].copy_(kernel_metadata_new[key])

        # Reset c4_topk_indices to the -1 sentinel before the captured forward
        # runs. The indexer will overwrite valid entries; any unset rows must
        # read -1 (= "no sparse index") to keep npu_sparse_attn_sharedkv stable.
        fm.c4_topk_indices.fill_(-1)

        self.forward_metadata = fm

    def init_forward_metadata(self, forward_batch: "ForwardBatch") -> None:
        super().init_forward_metadata(forward_batch)
        fm = self.forward_metadata

        # DP-attention IDLE ranks get a padded batch (bs>0) but seq_lens are
        # all zero. The sparse-attn metadata kernel
        # (npu_sparse_attn_sharedkv_metadata) doesn't accept this shape; even
        # after clamping seqused_kv it tries to read the request's page table
        # at positions that were never written, which surfaces as an AICPU
        # exception (errcode 0x2a / runtime 507018) on the next sync.
        # The rest of the V4 backend already treats IDLE as a no-op (see
        # forward_compress / forward_c4_indexer below), so we mirror that
        # contract here: stash empty-but-typed defaults on fm so any later
        # attribute access stays well-defined, then return without invoking
        # any sparse-attn metadata kernels.
        if forward_batch.forward_mode.is_idle():
            fm.actual_seq_lengths_q = None
            fm.actual_seq_lengths_q_pa = None
            fm.kernel_metadata = {}
            return

        # Build TND cu_seqlens_q (= cumulative QUERY seq lens, int32 device tensor).
        # The kernel uses cu_seqlens_q to slice the q tensor by request, so
        # the per-request length here must equal the per-request token count
        # in q — NOT the KV/context length.
        #
        #   extend / prefill: q has extend_seq_lens_cpu tokens per request →
        #                     cumsum(extend_seq_lens_cpu).
        #   decode:           q has exactly 1 new token per request → [1, 1, ..., 1].
        #   target_verify /
        #   draft_extend:     q has speculative_num_draft_tokens per request.
        #
        # Earlier this branch fell back to `forward_batch.seq_lens_cpu` (the
        # full KV length) on the non-extend path, which made the kernel slice
        # q at offset = full_seq_len while q.shape[0] = batch_size for decode.
        # That is the V4-NPU root cause of token-1+ divergence — kernel
        # metadata says q has e.g. 257 tokens but q tensor only has 1.
        device = forward_batch.seq_lens.device
        if forward_batch.forward_mode.is_extend():
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
            or forward_batch.forward_mode.is_draft_extend(include_v2=True)
        ):
            B = forward_batch.batch_size
            from sglang.srt.utils.common import get_global_server_args

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
            # DP-attention IDLE: forward_batch is padded with B placeholder rows
            # for the MLP/MoE-sync collectives; the attention output on this rank
            # is discarded. We still go through model.forward, so downstream
            # AscendC/AICPU ops (npu_sparse_attn_sharedkv_metadata @ L312/L325,
            # npu_quant_lightning_indexer_metadata @ L336, and the per-layer
            # attention calls at L547/L634) consume these three fields and
            # reject None / zero-length entries with `execute kernel param
            # invalid`. Pre-fill with decode-shaped dummy values (1 padded
            # q-token + 1 padded kv-token per padded req) so every downstream
            # consumer sees valid inputs in one place. This subsumes the
            # earlier ad-hoc `seqused_kv.clamp(min=1)` workaround in
            # _compute_kernel_metadata.
            B = forward_batch.batch_size
            fm.actual_seq_lengths_q = torch.arange(
                1, B + 1, dtype=torch.int32, device=device
            )
            fm.actual_seq_lengths_q_pa = torch.arange(
                0, B + 1, dtype=torch.int32, device=device
            )
            fm.actual_seq_lengths_kv = torch.ones(
                B, dtype=torch.int32, device=device
            )
        else:
            # Unknown / unsupported mode — leave None so downstream fails
            # loudly instead of silently producing garbage.
            fm.actual_seq_lengths_q = None
            fm.actual_seq_lengths_q_pa = None

        # SWA page table -- populated by AscendAttnBackend when the model is
        # hybrid-SWA, else None. Aliased under the name forward_sparse uses.
        # Use explicit `is not None` check (not `or`) because
        # `bool(multi-element tensor)` raises.
        fm.swa_page_table = (
            fm.block_tables_swa if fm.block_tables_swa is not None else fm.block_tables
        )

        # actual_seq_lengths_kv defaults to None on main; the V4 metadata
        # kernel needs an int32 device tensor of per-request KV lengths.
        if fm.actual_seq_lengths_kv is None:
            if fm.seq_lens_cpu_int is not None:
                fm.actual_seq_lengths_kv = fm.seq_lens_cpu_int.to(
                    device=forward_batch.seq_lens.device, dtype=torch.int32
                )
            else:
                fm.actual_seq_lengths_kv = forward_batch.seq_lens.to(torch.int32)

        # Build kernel_metadata dict. For V4-Flash we mainly need c1a (no
        # compress KV) right now; c4a/c128a follow when we add those paths.
        fm.kernel_metadata = self._compute_kernel_metadata(forward_batch)

        # NPU compress metadata: per-request tensors consumed by
        # dsv4/{compressor,indexer}.py forward_npu. There is no per-ratio
        # req_to_token mapping in the request pool, so we compute the
        # equivalent on the fly from req_to_token + the V4 KV pool's
        # swa translation.
        if self._dsv4_compress_ratios:
            self._build_npu_compress_metadata(forward_batch)

    def _compute_kernel_metadata(self, forward_batch: "ForwardBatch") -> dict:
        fm = self.forward_metadata
        # DP-attention idle-rank safety lives in the is_idle() branch of
        # init_forward_metadata, which already sets actual_seq_lengths_{q,
        # q_pa,kv} to valid dummy values before any consumer runs here.
        if forward_batch.forward_mode.is_target_verify() or forward_batch.forward_mode.is_draft_extend(include_v2=True):
            from sglang.srt.utils.common import get_global_server_args

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
        """Pure kernel-metadata computation shared by eager and graph-capture paths.

        No reference to forward_batch or self.forward_metadata. The caller is
        responsible for passing already-shaped tensors. Returns the
        {c1a,c4a,c128a,li_quant}_metadata dict.

        ``actual_seq_lengths_q_pa`` is the padded cumulative query-length tensor
        with a leading 0 (length ``bs+1``); the lightning-indexer query lengths
        are derived as ``actual_seq_lengths_q_pa[1:]`` to mirror the
        ``actual_seq_lengths_q`` produced by eager ``init_forward_metadata``.

        ``max_seqlen_q`` / ``block_tables`` / ``is_nextn`` are accepted now for
        signature stability with the upcoming graph-capture path; the eager body
        does not consume them yet — capture/replay (Tasks 4/5) will branch on
        ``is_nextn`` and short-circuit construction on small ``max_seqlen_q``.
        """
        common = {
            "cu_seqlens_q": actual_seq_lengths_q_pa,
            "seqused_kv": actual_seq_lengths_kv,
            "cmp_ratio": 1,
            "ori_mask_mode": 4,  # sliding window
            "cmp_mask_mode": 3,  # causal
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

            # The lightning indexer is only attached to c4 layers. Pass
            # actual_seq_lengths_q as a fresh contiguous int32 device tensor
            # (no leading 0, B-element cumsum) — not a slice.
            # Eager builds this alongside actual_seq_lengths_q_pa; here we
            # recover it by dropping the leading 0 of the _pa form so the
            # capture/replay path doesn't need to pass two parallel tensors.
            # ``.clone()`` (not just ``[1:]``) preserves the "fresh tensor,
            # not a slice/view" semantics the kernel relies on.
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

    def _build_npu_compress_metadata(self, forward_batch: "ForwardBatch") -> None:
        """Populate c{4,128}_{page_table,state_page_table,state_loc,loc} on
        forward_metadata for the NPU compressor / indexer forward_npu paths.

        Thin eager shim around :meth:`_compute_compress_locs`; the pure helper
        returns a dict that graph replay can copy into preallocated buffers.
        """
        fm = self.forward_metadata
        is_decode = forward_batch.forward_mode.is_decode()
        result = self._compute_compress_locs(
            pool=forward_batch.token_to_kv_pool,
            req_to_token=forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            out_cache_loc=forward_batch.out_cache_loc,
            is_decode=is_decode,
            bs=forward_batch.batch_size,
            device=forward_batch.seq_lens.device,
            req_to_token_pool=forward_batch.req_to_token_pool,
            out_cache_loc_dsv4=forward_batch.out_cache_loc_dsv4,
        )
        for k, v in result.items():
            setattr(fm, k, v)
        # Non-decode path leaves c{ratio}_state_loc / c{ratio}_loc absent from
        # result; the eager contract was that those fm fields are None in
        # non-decode mode. Replay (Task 5) checks key presence instead.
        if not is_decode:
            for ratio in self._dsv4_compress_ratios:
                if ratio in (4, 128):
                    if f"c{ratio}_state_loc" not in result:
                        setattr(fm, f"c{ratio}_state_loc", None)
                    if f"c{ratio}_loc" not in result:
                        setattr(fm, f"c{ratio}_loc", None)

        # Fused-compressor metadata. Decode case lives in
        # _compute_compress_locs (graph-replay-compatible); prefill needs
        # per-request slicing of forward_batch.positions via cu_seqlens.
        # Prefill is not graph-captured today so we keep it here in the
        # eager-only shim. See cheat sheet B.2.
        #
        # Build only for non-chunked prefill; the fused compressor reads
        # positions_cmp_padding / start_pos / seqused from fm. target_verify
        # is excluded — NPU compress is fused-only now.
        if (
            forward_batch.forward_mode.is_prefill()
            and not forward_batch.forward_mode.is_target_verify()
            and self._dsv4_compress_ratios
        ):
            self._build_npu_compress_metadata_prefill(forward_batch)

    def _build_npu_compress_metadata_prefill(
        self, forward_batch: "ForwardBatch"
    ) -> None:
        fm = self.forward_metadata
        device = forward_batch.seq_lens.device
        positions = forward_batch.positions
        t = positions.shape[0]
        bs = forward_batch.batch_size
        # cu_seqlens (prefix-sum query lengths with leading 0, shape [bs+1]
        # int32). Set by init_forward_metadata in every mode.
        cu = fm.actual_seq_lengths_q_pa

        # Per-request positions: build positions_cmp_padding for each ratio
        # by slicing `positions[start:end][:cutoff:ratio]` per request and
        # concatenating. cu_seqlens reads sync to host; this is fine in
        # eager prefill (graph capture skips this branch).
        cu_cpu = cu.cpu().tolist()
        ratio_lists: dict = {r: [] for r in self._dsv4_compress_ratios if r in (4, 128)}
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

        # Prefill: start_pos = 0 per req (no chunk prefill support yet),
        # seqused = None (op falls back to cu_seqlens diff).
        fm.start_pos = torch.zeros(bs, dtype=torch.int32, device=device)
        fm.seqused = None

        # c{ratio}_loc for the fused-op epilog write (non-chunked prefill).
        # out_c{4,128}_loc is densely packed in batch order (one slot per
        # compressed token), same order as positions_cmp_padding above, so the
        # op's cmp_kv[k] -> c{ratio}_loc[k]. Equals forward_npu write_locs when
        # prefix_lens == 0. NOT valid under chunked prefill.
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

        # ROOT-CAUSE FIX: stale non-tail state-page-table entries
        # (req_to_token_c{N}_state is NOT re-zeroed on req-slot reuse, see
        # DSV4NPUReqToTokenPool.clear) would make the fused op's
        # WriteToCacheState (block-0-skip) write into another request's state
        # block. Force the page columns BEFORE each request's tail page to 0 so
        # they hit the kernel's idInBlockTable==0 skip. Decode is unaffected
        # (single-position write); prefill writes [0, seqlen). The tail per req
        # starts at c_alloc_offset (same formula as forward_npu's tail-state
        # alloc); the boundary page is left intact (its non-tail tokens write to
        # the request's OWN tail block, harmless and never read back).
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
    ) -> dict:
        """Pure compress-loc computation shared by eager and graph-replay paths.

        Sources for the c{ratio}_page_table / c{ratio}_loc fields:

          * ``c{ratio}_page_table`` is sliced from
            ``req_to_token_pool.req_to_token_c{4,128}`` (token-level c-pool
            slot ids), then ``[:, :: page_size] // page_size`` converts to
            page ids. This is the iforgetmyname/sglang dsv4_release pattern.
          * ``c{ratio}_loc`` (decode only) comes straight from the
            :class:`DSV4OutCacheLoc` bundle that
            :class:`DSV4NPUTokenToKVPoolAllocator` produced during alloc.
          * ``c{ratio}_state_page_table`` is sliced from
            ``req_to_token_pool.req_to_token_c{4,128}_state`` (per-raw-token
            state-pool slot ids written by the dsv4_common_hooks after each
            allocator call), then ``[:, :: page_size] // page_size`` →
            kernel-view block ids. State pool is paged (cache_mode=1
            on Atlas A3); ring-hash translation is not supported.
          * ``c{ratio}_state_loc`` (decode only) comes from the
            :class:`DSV4OutCacheLoc` bundle's ``out_c{4,128}_state_loc``
            field. One slot per raw decode token (state is per-token,
            not per-ratio).

        Returns dict keys:
          - c{ratio}_state_page_table  (always present for ratio in {4,128})
          - c{ratio}_state_loc         (decode only)
          - c{ratio}_loc               (decode only)
          - c{ratio}_page_table        (always present for ratio in {4,128})

        Caller responsibility: this helper calls ``seq_lens.max().item()``
        which synchronizes to host. Graph-replay callers must short-circuit
        or pre-compute on the host before capture.
        """
        result: dict = {}
        req_pool = req_pool_indices

        seq_lens_max = int(seq_lens.max().item()) if bs > 0 else 0
        n_pages = max(1, (seq_lens_max + self.page_size - 1) // self.page_size)

        for ratio in self._dsv4_compress_ratios:
            if ratio not in (4, 128):
                continue
            # State page table from req_to_token_c{ratio}_state. The table
            # stores one state-pool slot id per RAW token (state is not
            # ratio-compressed); take one slot per page boundary and divide
            # by page_size to get the kernel's block id. Unallocated tail
            # entries default to 0 — the kernel treats block 0 as
            # skip-sentinel (NPUCompressStatePool reserves it).
            state_table = (
                req_to_token_pool.req_to_token_c4_state
                if ratio == 4
                else req_to_token_pool.req_to_token_c128_state
            )
            state_slots_2d = state_table[
                req_pool.to(torch.int64), :n_pages * self.page_size
            ]
            state_page_2d = (
                state_slots_2d[:, :: self.page_size] // self.page_size
            ).to(torch.int32)

            if is_decode:
                # Decode-time state_loc for the new raw token: one slot per
                # req from the allocator bundle. None on idle DP-attention
                # ranks (alloc_decode short-circuited).
                state_loc_decode = None
                if out_cache_loc_dsv4 is not None:
                    state_loc_decode = (
                        out_cache_loc_dsv4.out_c4_state_loc
                        if ratio == 4
                        else out_cache_loc_dsv4.out_c128_state_loc
                    )
                if state_loc_decode is None:
                    # Fallback shape-correct buffer of zeros (block 0 dummy)
                    # for idle ranks so downstream kernels see a valid tensor.
                    state_loc_decode = torch.zeros(
                        bs, dtype=torch.int32, device=device,
                    )
                else:
                    state_loc_decode = state_loc_decode.to(torch.int32)
                # c{ratio}_loc comes from the DSV4OutCacheLoc bundle the
                # allocator stashed in alloc_decode. Densely-packed in the
                # prefix [0, n_compress): bundle_loc[k] is the c-pool slot
                # for the k-th boundary-hitting req (in batch order). The
                # fused compressor op (torch.ops.custom.compressor) returns
                # cmp_kv aligned the same way -- cmp_kv[k] is the compressed
                # token for the k-th entry of positions_cmp_padding (also
                # densely packed; see below). The epilog writes
                # buf_flat[c{ratio}_loc[i]] = cmp_kv[i] for i in [0, bs);
                # tail entries [n_compress, bs) stay 0 and land in the
                # allocator-reserved skip slot 0 (NPUPagedTokenToKVPool
                # free_pages starts at 1, so slot 0 is never read).
                #
                # NOTE: an earlier version scattered bundle_loc into
                # ``compress_out_loc[idx]`` using ``idx = nonzero(should_compress)``
                # (sparsely-padded by batch index). That broke multi-req
                # batches whenever a boundary-hitting req was not at the
                # front of the batch: cmp_kv[k] (densely packed) and
                # c{ratio}_loc[idx[k]] (batch-indexed) misaligned, so
                # compressed tokens were lost to slot 0 and real slots got
                # padding junk -- the c4/c128 attention then read corrupt
                # KV and produced garbled tokens (TP/multi-concurrency
                # regression).
                #
                # out_cache_loc_dsv4 is None on IDLE DP-attention ranks
                # (alloc_decode short-circuited because there's no real batch
                # to allocate). For those ranks the kernel still needs a
                # shape-correct compress_out_loc buffer, so emit all zeros --
                # the captured graph will run with no actual compress work.
                compress_out_loc = torch.zeros(
                    bs, dtype=torch.int32, device=device,
                )
                if out_cache_loc_dsv4 is not None:
                    bundle_loc = (
                        out_cache_loc_dsv4.out_c4_loc
                        if ratio == 4
                        else out_cache_loc_dsv4.out_c128_loc
                    )
                    n_compress = bundle_loc.numel()
                    if n_compress > 0:
                        compress_out_loc[:n_compress] = bundle_loc.to(
                            torch.int32
                        )

            result[f"c{ratio}_state_page_table"] = state_page_2d
            if is_decode:
                result[f"c{ratio}_state_loc"] = state_loc_decode
                result[f"c{ratio}_loc"] = compress_out_loc

            # c{ratio}_page_table — kernel-view page table for c{N}_kv_pool.
            # Slice req_to_token_c{ratio} (token-level c-pool slot ids), take
            # one slot per page (`::page_size`), convert to page id via
            # `// page_size`. iforgetmyname/sglang dsv4_release pattern.
            c_table = (
                req_to_token_pool.req_to_token_c4
                if ratio == 4
                else req_to_token_pool.req_to_token_c128
            )
            # Graph mode: raw `seq_lens_max // ratio` keeps the slice shape
            # aligned with the preallocated (max_bs, max_pages) buffer copy.
            # Eager: clamp to >=1 so downstream kernels always see a column.
            if is_graph:
                n_c_tokens = seq_lens_max // ratio
            else:
                n_c_tokens = max(1, seq_lens_max // ratio)
            slots = c_table[req_pool.to(torch.int64), :n_c_tokens]
            c_page_table = (
                slots[:, :: self.page_size] // self.page_size
            ).to(torch.int32)
            result[f"c{ratio}_page_table"] = c_page_table

        # Fused-compressor metadata. The torch.ops.custom.compressor op
        # consumes positions_cmp_padding_c{4,128} (the absolute positions
        # of the tokens being compressed this step), start_pos (where in
        # the sequence each request is writing), and seqused (per-req
        # valid token count this step). Decode path only — prefill is
        # computed in _build_npu_compress_metadata's prefill branch since
        # the per-request slicing pattern needs cu_seqlens host reads.
        # See cheat sheet section B.1 (decode) for the reference impl.
        if is_decode:
            valid = seq_lens > 0
            positions_last = torch.clamp(seq_lens - 1, min=0)
            for ratio in self._dsv4_compress_ratios:
                if ratio not in (4, 128):
                    continue
                # padding size = min(bs, bs // ratio + bs) which is just bs
                # for any reasonable bs / ratio combo in decode mode (1
                # query token per req, so t=bs).
                padding_size = min(bs, bs // ratio + bs)
                padding = torch.zeros(
                    padding_size, dtype=torch.int64, device=device
                )
                should_compress = ((seq_lens % ratio) == 0) & valid
                pos_cmp = positions_last[should_compress].to(torch.int64) + (
                    1 - ratio
                )
                if pos_cmp.numel() > 0:
                    padding[: pos_cmp.shape[0]].copy_(pos_cmp)
                result[f"positions_cmp_padding_c{ratio}"] = padding

            result["start_pos"] = positions_last.to(torch.int32)
            result["seqused"] = valid.to(torch.int32)

        return result

    def init_forward_metadata_indexer(self, core_attn_metadata):
        # li_quant_metadata is computed inside _compute_kernel_metadata; nothing
        # extra to do here. Return None to satisfy the mixin contract.
        return None

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        *,
        compress_ratio: int = 0,
        attn_sink: Optional[torch.Tensor] = None,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        if compress_ratio not in (0, 1, 4, 128):
            raise ValueError(
                f"V4 attention expects compress_ratio in (0, 1, 4, 128); got {compress_ratio}"
            )
        # DP-attention IDLE short-circuit. Idle ranks run model.forward only to
        # participate in the downstream MoE collective (deepep dispatch/combine
        # or the TP all-reduce in the nodeepep path); their attention output is
        # discarded. npu_sparse_attn_sharedkv / sharedkv_metadata trip on the
        # placeholder KV pages even after init_forward_metadata's is_idle()
        # branch supplies valid dummy length tensors, so skip the whole attn
        # compute and the K store_cache (K is computed from a padded zero
        # hidden state and has no future reader) and return a zero tensor of
        # q's shape so hc_post sees the right output shape.
        if forward_batch.forward_mode.is_idle():
            return torch.zeros_like(q)
        # Honor the save_kv_cache flag. MQALayer._forward_prepare already
        # writes K via store_cache on the compressor stream and passes
        # save_kv_cache=False here, so the dup-write is skipped by default;
        # any caller still passing True (e.g. a non-overlap path) gets the
        # write so decode never reads an unwritten swa_kv_pool.
        if save_kv_cache:
            self.store_cache(
                layer_id=layer.layer_id, swa_k=k, forward_batch=forward_batch
            )
        if compress_ratio in (0, 1):
            return self._forward_dense(q, layer, forward_batch, attn_sink)
        # ratio 4 / 128: sparse compressed-KV path via npu_sparse_attn_sharedkv
        # with has_cmp_kv=True.
        return self._forward_compressed(
            q, layer, forward_batch, attn_sink, compress_ratio
        )

    def _forward_dense(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_sink: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """ratio=1 / ratio=0 dense layers — sliding-window attention via
        npu_sparse_attn_sharedkv with has_cmp_kv=False."""
        fm = self.forward_metadata
        pool = forward_batch.token_to_kv_pool
        ori_kv = pool.get_swa_buffer(layer.layer_id)  # (num_pages, page_size, 1, dim)

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
        # for _k, _v in attn_kwargs.items():
        #     if isinstance(_v, torch.Tensor):
        #         print(f"[dsv4_dense] {_k}: shape={tuple(_v.shape)} dtype={_v.dtype}")
        #     else:
        #         print(f"[dsv4_dense] {_k}: {_v!r}")
        out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(**attn_kwargs)
        return out

    def _forward_compressed(
        self,
        q: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        attn_sink: Optional[torch.Tensor],
        compress_ratio: int,
    ) -> torch.Tensor:
        """ratio=4 / ratio=128 layers — sliding-window + compressed-KV sparse
        attention via npu_sparse_attn_sharedkv with has_cmp_kv=True. cmp_kv
        comes from the c4 / c128 pool (populated by the compressor) and
        cmp_sparse_indices for c4 comes from forward_metadata.c4_topk_indices
        (populated by the lightning indexer).
        """
        fm = self.forward_metadata
        pool = forward_batch.token_to_kv_pool
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
        # The c{4,128} kv pools are allocated with the global page_size (via
        # NPUDeepSeekV4SingleKVPool.kernel_page_size), so cmp_kv already shares
        # ori_kv's page_size and the kernel consumes it directly — no reshape.
        # Assert the invariant rather than papering over a page-size mismatch.
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
        # c4 attention reads the top-k compressed indices produced by the
        # lightning indexer; c128 uses cmp_sparse_indices=None and lets the
        # kernel read the full populated c128 history.
        if compress_ratio == 4:
            # The c4 lightning indexer (Indexer.forward_npu) populates
            # c4_topk_indices on every non-idle c4 layer before this attention
            # call, and idle ranks short-circuit in forward() before reaching
            # here — so a None topk is a real bug to surface, not seed around.
            topk = fm.c4_topk_indices
            attn_kwargs["cmp_sparse_indices"] = topk.view(-1, 1, topk.shape[-1])
        else:
            attn_kwargs["cmp_sparse_indices"] = None
        out, _ = torch.ops.custom.npu_sparse_attn_sharedkv(**attn_kwargs)
        return out

    def store_cache(self, *, layer_id: int, swa_k: torch.Tensor, forward_batch):
        """Write the SWA layer's K cache into the bf16 PA_ND buffer.

        ``swa_k`` arrives shaped (T, num_kv_heads=1, dim) where dim packs
        K_nope + K_rope in bf16 (same layout as get_swa_buffer returns).
        ``forward_batch.out_cache_loc`` is in FULL-pool index space (size
        = sum of all KV pools); the swa_kv_pool buffer is its own smaller
        space. We must translate full→swa first — otherwise the
        index_put hits the wrong slot (or wraps OOB), and decode reads
        garbage K back. This mirrors what the CUDA radix path does at
        set_swa_key_buffer_radix.
        """
        pool = forward_batch.token_to_kv_pool
        swa_loc = pool.translate_loc_from_full_to_swa(forward_batch.out_cache_loc)
        pool.set_swa_buffer(
            layer_id=layer_id,
            loc=swa_loc,
            cache=swa_k,
        )

    # c4/c128 compressor + indexer entry points. On NPU ``forward_compress``
    # runs the fused compressor inline (the CUDA mixin's set_extra_key_buffer
    # chain has no NPU equivalent).

    def forward_core_compressor(  # type: ignore[override]
        self,
        x: torch.Tensor,
        forward_batch: "ForwardBatch",
        layer_id: int,
        compressor,
    ) -> None:
        """Trigger the OUTER attention compressor on NPU.

        CUDA's ``forward_core_compressor`` does (compressor → set_extra_key_*).
        On NPU, ``Compressor.forward`` delegates to ``forward_compress``
        which writes the KV pool inline, so we just call the compressor.
        """
        if forward_batch.forward_mode.is_idle():
            return
        compressor(x, forward_batch)

    def forward_compress(  # type: ignore[override]
        self,
        compressor,
        x: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> None:
        """Fused-op NPU compressor path, lowered from ``Compressor``.

        A single ``torch.ops.custom.compressor`` call; the compressor's
        weights / ape / fused caches are read through ``compressor`` (mirrors
        ``C4Indexer.forward`` -> ``forward_c4_indexer``). State_cache writes
        happen inside the op, so only the compressed-kv epilog
        (:meth:`_compressor_epilog_npu`) runs on the returned tensor.
        """
        from sglang.srt.models.deepseek_v4 import get_fused_compressor_rope_cos_sin

        ratio = compressor.ratio
        coff = 1 + int(compressor.overlap)
        device = x.device
        self._ensure_compressor_hadamard(compressor, device)
        self._ensure_fused_caches(compressor)

        fm = forward_batch.attn_backend.forward_metadata
        positions_cmp = getattr(fm, f"positions_cmp_padding_c{ratio}", None)
        page_table = getattr(fm, f"c{ratio}_state_page_table", None)
        start_pos = getattr(fm, "start_pos", None)
        seqused = getattr(fm, "seqused", None)
        # cu_seqlens: prefix-sum query lengths with leading 0, [bs+1] int32.
        cu_seqlens = getattr(fm, "actual_seq_lengths_q_pa", None)
        assert positions_cmp is not None and page_table is not None, (
            "fused compressor needs backend metadata "
            "(positions_cmp_padding / c*_state_page_table) — make sure "
            "_build_npu_compress_metadata ran before this forward."
        )
        assert start_pos is not None, "fused compressor needs start_pos"
        assert cu_seqlens is not None, "fused compressor needs cu_seqlens"

        # state_cache: fp32 [block_num, page_size, 2*coff*D].
        pool = forward_batch.token_to_kv_pool
        state_cache = pool.get_state_cache(compressor.layer_id, compressor.is_in_indexer)

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

        # cmp_kv shape: [min(T, T//ratio + B), head_dim]. For prefill the loc
        # tensor may be shorter than the padded output; trim to len(loc).
        loc = getattr(fm, f"c{ratio}_loc", None)
        if loc is not None and loc.numel() < cmp_kv.shape[0]:
            cmp_kv = cmp_kv[: loc.numel()]

        if forward_batch.attn_backend.graph_mode or cmp_kv.shape[0] > 0:
            if compressor.rotate:
                cmp_kv = _apply_hadamard(cmp_kv, compressor.hadamard_matrix)
            self._compressor_epilog_npu(compressor, cmp_kv, forward_batch)

    def _ensure_compressor_hadamard(self, compressor, device: torch.device) -> None:
        # Register the Walsh-Hadamard matrix on the compressor module (used to
        # rotate cmp_kv after rope). Built lazily on first NPU forward.
        if getattr(compressor, "hadamard_matrix", None) is None:
            H = _walsh_hadamard_matrix(compressor.head_dim, torch.float32, device)
            compressor.register_buffer("hadamard_matrix", H, persistent=False)

    def _ensure_fused_caches(self, compressor) -> None:
        """Build the per-instance views the fused compressor op needs.

        Lazy because at __init__ time wkv_gate.weight is uninitialized and
        norm.weight hasn't been loaded yet; first forward fires after weight
        loading + device move, when these slices are stable.
        """
        if getattr(compressor, "_fused_wkv_w", None) is not None:
            return
        coff = 1 + int(compressor.overlap)
        split = coff * compressor.head_dim
        # wkv_gate.weight shape: [2*coff*head_dim, hidden_size]. Split row-wise.
        w = compressor.wkv_gate.weight
        assert w.shape[0] == 2 * split, (
            f"wkv_gate.weight rows={w.shape[0]} != 2*coff*head_dim={2*split}"
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
        # Quant + write — quant only when this is an indexer compressor with
        # int8 li_kv. For the bf16 indexer / attention compressor branches,
        # kv_scale is None.
        kv_scale: Optional[torch.Tensor] = None
        li_kv_dtype = getattr(compressor, "li_kv_dtype", "bf16")
        if li_kv_dtype == "int8" and compressor.is_in_indexer:
            import torch_npu  # local import: only available on NPU

            kv, kv_scale = torch_npu.npu_dynamic_quant(kv)
            kv_scale = kv_scale.to(torch.float16)

        if override_loc is not None:
            loc = override_loc
        else:
            backend_fm = forward_batch.attn_backend.forward_metadata
            loc = backend_fm.c4_loc if compressor.ratio == 4 else backend_fm.c128_loc
        forward_batch.token_to_kv_pool.set_compress_buffer(
            compressor.layer_id,
            loc,
            kv,
            kv_scale,
            compressor.is_in_indexer,
        )

    def forward_c4_indexer_npu(
        self,
        c4_indexer,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Compute c4 top-k sparse indices on NPU; returns a [T, index_topk]
        int32 tensor.

        Steps:
          1. Materialize q via wq_b + rope + hadamard.
          2. Run the inner c4_indexer.compressor (which on NPU writes the
             indexer c4 compress kv + state to the pool).
          3. Project x through weights_proj and apply softmax/head scale.
          4. Either `npu_quant_lightning_indexer` (int8 li_kv) or per-request
             einsum + topk (bf16 li_kv) to produce the top-k indices.
        """
        from sglang.srt.layers.dp_attention import get_attention_tp_group

        ratio = c4_indexer.compressor.ratio  # = 4 for the c4 indexer
        device = x.device
        self._ensure_npu_c4_indexer(c4_indexer, device)
        bs = x.shape[0]
        is_prefill = (
            forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
        )

        # q path
        q = self._compute_q_npu(c4_indexer, q_lora, forward_batch.positions)

        # weights path — keep the bf16 → bf16 projection and apply the
        # combined softmax + n_heads scaling here so the int8 indexer kernel
        # receives `weights * scale`.
        weights, _ = c4_indexer.weights_proj(x)
        weights = weights * (c4_indexer.softmax_scale * c4_indexer.n_heads**-0.5)

        # compressor path — writes c4 indexer compress kv + state on NPU.
        c4_indexer.compressor(x, forward_batch)

        # Prefer the fused int8 lightning indexer when the indexer KV is
        # quantized.
        li_kv_dtype = getattr(c4_indexer.compressor, "li_kv_dtype", "bf16")
        if li_kv_dtype == "int8":
            # Skip the indexer kernel call when this rank's batch is empty
            # (no tokens to score). DP attention can leave some ranks with
            # an empty batch in flight; calling npu_quant_lightning_indexer
            # with T=0 / kv_len=0 deadlocks async on an internal collective.
            # Return the sentinel topk so downstream _forward_compressed
            # sees a well-shaped tensor without entering the indexer kernel.
            # Use forward_mode.is_idle() instead of kv_lens.sum().item() to
            # stay graph-capture-safe (.item() forces a host sync that ACL
            # graph capture rejects with error 107027).
            if bs == 0 or forward_batch.forward_mode.is_idle():
                return torch.full(
                    (bs, self._dsv4_index_topk),
                    -1,
                    dtype=torch.int32,
                    device=device,
                )
            li_cmp_kv = forward_batch.token_to_kv_pool.get_compress_buffer(
                c4_indexer.layer_id, True
            )
            li_kv_scale = (
                forward_batch.token_to_kv_pool.get_compress_dequant_scale_buffer(
                    c4_indexer.layer_id, True
                )
            )
            return self._forward_npu_fused(
                c4_indexer, q, li_cmp_kv, li_kv_scale, weights, forward_batch
            )

        # bf16 li_kv path: per-request einsum + topk against the indexer
        # compress buffer — slow but architecture-faithful fallback.
        seqlens_cpu = forward_batch.seq_lens_cpu
        end_pos = forward_batch.seq_lens.cumsum(dim=0)
        page_table = forward_batch.attn_backend.forward_metadata.c4_page_table
        attn_tp_size = get_attention_tp_size()
        topk_idxs: list[torch.Tensor] = []
        for i, _end_token in enumerate(end_pos):
            seq_i = int(seqlens_cpu[i])
            kv_indices = _get_kv_indices(
                forward_batch, seq_i // ratio, page_table, i, seq_i // ratio
            )
            kv_cache_value = (
                forward_batch.token_to_kv_pool.get_compress_buffer(
                    c4_indexer.layer_id, True, kv_indices
                )
            )
            if is_prefill:
                start = 0 if i == 0 else int(end_pos[i - 1])
                end = int(end_pos[i])
                index_score = torch.einsum(
                    "shd,td->sht",
                    q[start:end, ...],
                    kv_cache_value.squeeze(1),
                )  # [s, n_heads, seq_i//ratio]
                index_score = (
                    index_score.relu_() * weights.unsqueeze(-1)[start:end, ...]
                ).sum(dim=1)
                if attn_tp_size > 1 and getattr(c4_indexer, "enable_indexer_tp", False):
                    get_attention_tp_group().all_reduce(index_score)
                # Causal mask in compressed-token coordinates.
                arange_kv = torch.arange(seq_i // ratio, device=device)
                arange_q = torch.arange(1, seq_i + 1, device=device).unsqueeze(1)
                causal = arange_kv.repeat(seq_i, 1) >= (arange_q // ratio)
                index_score += torch.where(
                    causal, float("-inf"), torch.zeros((), device=device)
                )
                topk_idx = index_score.topk(
                    min(self._dsv4_index_topk, seq_i // ratio), dim=-1
                )[1]
                # Drop the diagonal token (position seq_i % ratio == 0
                # leaves a self-loop after the // ratio division).
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
                index_score = (
                    index_score.relu_() * weights.unsqueeze(-1)[i]
                ).sum(dim=1)
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
        # One-time NPU setup for the c4 indexer module (CUDA does the
        # equivalent inside C4Indexer via triton). Tag the inner compressor
        # as int8 li_kv so its epilog quantizes through npu_dynamic_quant,
        # and register the Walsh-Hadamard matrix used for the q rotation
        # (keyed off the buffer's existence so registration runs once).
        c4_indexer.compressor.li_kv_dtype = "int8"
        if getattr(c4_indexer, "hadamard_matrix", None) is None:
            H = _walsh_hadamard_matrix(c4_indexer.head_dim, torch.float32, device)
            c4_indexer.register_buffer("hadamard_matrix", H, persistent=False)

    def _compute_q_npu(
        self, c4_indexer, q_lora: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        # wq_b → reshape to (T, n_heads, head_dim) → in-place rope on the
        # rope-slice → hadamard rotation (Walsh-Hadamard matmul). The CUDA
        # `compute_q` uses tvm_ffi `fused_rope` + triton `rotate_activation`;
        # both are absent on NPU.
        from sglang.srt.models.deepseek_v4 import v4_rope_inplace_npu

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
        # Single fused `npu_quant_lightning_indexer` call. Reads c4_page_table
        # and li_quant_metadata from the backend's forward_metadata.
        import torch_npu  # local import: NPU only

        q_int8, q_scale = torch_npu.npu_dynamic_quant(q)
        fm = forward_batch.attn_backend.forward_metadata
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

    def forward_c4_indexer(  # type: ignore[override]
        self,
        *,
        x: torch.Tensor,
        q_lora: torch.Tensor,
        forward_batch: "ForwardBatch",
        c4_indexer=None,
        alt_streams=None,
        enable_multi_stream: bool = False,
        q_lora_ready=None,
    ) -> None:
        """Populate ``forward_metadata.c4_topk_indices`` for c4 sparse attention.

        Mirrors the CUDA ``C4IndexerBackendMixin.forward_c4_indexer``: the
        compute lives on the backend but reads the indexer's weights /
        compressor through ``c4_indexer``. ``forward_c4_indexer_npu`` runs the
        npu_quant_lightning_indexer (or the bf16 einsum fallback) and returns
        the real top-k indices; idle / empty ranks are short-circuited inside it.
        """
        if forward_batch.forward_mode.is_idle():
            return
        topk_idxs = self.forward_c4_indexer_npu(c4_indexer, x, q_lora, forward_batch)
        self.forward_metadata.c4_topk_indices = topk_idxs


def _get_kv_indices(
    forward_batch: ForwardBatch,
    kv_len: int,
    page_table: torch.Tensor,
    req_idx: int,
    seqlen: int,
) -> torch.Tensor:
    # Duplicated in compressor.py — kept private here so indexer doesn't
    # re-import a private symbol from compressor.
    logic_start = max(0, seqlen - kv_len)
    logic_end = seqlen
    page_size = forward_batch.attn_backend.page_size
    if page_size == 1:
        return page_table[req_idx, logic_start:logic_end]
    logic_pos = torch.arange(logic_start, logic_end, device=page_table.device)
    block_id = logic_pos // page_size
    offset_in_block = logic_pos % page_size
    return page_table[req_idx, block_id] * page_size + offset_in_block
