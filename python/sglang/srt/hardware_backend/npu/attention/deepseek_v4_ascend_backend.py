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


def _build_hadamard_matrix(n: int, dtype: torch.dtype, device) -> torch.Tensor:
    """Sylvester-construction Walsh-Hadamard matrix of size n × n.

    n must be a power of 2 (asserted by callers). Caches per (n, dtype, device)
    on the function so repeated calls within a forward batch don't rebuild.
    """
    cache = _build_hadamard_matrix._cache  # type: ignore[attr-defined]
    key = (n, dtype, str(device))
    if key in cache:
        return cache[key]
    H = torch.tensor([[1.0]], dtype=torch.float32)
    while H.size(0) < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
    H = H.to(dtype=dtype, device=device).contiguous()
    cache[key] = H
    return H


_build_hadamard_matrix._cache = {}  # type: ignore[attr-defined]


def _compute_c4_q_npu(
    c4_indexer,
    q_lora: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """NPU equivalent of ``C4Indexer.compute_q``.

    ``compute_q`` does:
        q, _ = wq_b(q_lora)
        q = q.view(-1, n_local_heads, head_dim)
        fused_rope(q[..., -rope_head_dim:], None, freqs_cis, positions=...)
        q = rotate_activation(q)            # triton hadamard_transform

    On NPU, ``fused_rope`` is a tvm_ffi CUDA kernel and ``rotate_activation``
    is a triton hadamard. Replace with ``_v4_rope_inplace_npu`` and a torch
    Walsh-Hadamard matmul. Note: Sylvester ordering may not match the triton
    kernel's ordering — final consumer (``npu_quant_lightning_indexer``) is
    insensitive to the basis since both q and k are rotated by the same H.
    """
    from sglang.srt.models.deepseek_v4 import _v4_rope_inplace_npu

    q, _ = c4_indexer.wq_b(q_lora)
    q = q.view(-1, c4_indexer.n_local_heads, c4_indexer.head_dim)
    _v4_rope_inplace_npu(
        q[..., -c4_indexer.rope_head_dim :],
        None,
        c4_indexer.freqs_cis,
        positions,
    )
    H = _build_hadamard_matrix(c4_indexer.head_dim, torch.float32, q.device)
    scale = c4_indexer.head_dim**-0.5
    q_f32 = q.to(torch.float32)
    q_rotated = torch.matmul(q_f32, H) * scale
    return q_rotated.to(torch.bfloat16)


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
        self._dsv4_index_topk = getattr(hf, "index_topk", 512)
        self._dsv4_index_n_heads = getattr(hf, "index_n_heads", 64)
        self._dsv4_index_head_dim = getattr(hf, "index_head_dim", 128)
        self._dsv4_compress_ratios = getattr(hf, "compress_ratios", None)
        self._dsv4_has_c4 = (
            self._dsv4_compress_ratios is not None and 4 in self._dsv4_compress_ratios
        )
        self._dsv4_has_c128 = (
            self._dsv4_compress_ratios is not None and 128 in self._dsv4_compress_ratios
        )
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
        self.graph_metadata["swa_page_table"] = torch.zeros(
            (max_bs, max_pages), dtype=torch.int32, device=device
        )

        # c4 / c128 page tables — at ratio R, pages are at most max_pages // R + 1.
        # The kernel reads up to ceil(seq_len / ratio / page_size) pages anyway.
        c4_max_pages = max_pages // 4 + 1
        c128_max_pages = max_pages // 128 + 1

        self.graph_metadata["c4_page_table"] = torch.zeros(
            (max_bs, c4_max_pages), dtype=torch.int32, device=device
        )
        self.graph_metadata["c128_page_table"] = torch.zeros(
            (max_bs, c128_max_pages), dtype=torch.int32, device=device
        )
        self.graph_metadata["c4_state_page_table"] = torch.zeros(
            (max_bs, c4_max_pages), dtype=torch.int32, device=device
        )
        self.graph_metadata["c128_state_page_table"] = torch.zeros(
            (max_bs, c128_max_pages), dtype=torch.int32, device=device
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

        # actual_seq_lengths_kv: KV lengths per request; replay rewrites this in-place.
        # Allocate max_bs-sized zero buffer at capture so replay can `.fill_(0)` +
        # `[:bs].copy_(seq_len_for_this_batch)` without reallocating.
        max_bs = self.graph_metadata["block_tables"].shape[0]
        metadata.actual_seq_lengths_kv = torch.zeros(
            max_bs, dtype=torch.int32, device=device,
        )

        # actual_seq_lengths_q_cmp: same shape as _pa (bs+1). Replay mirrors
        # _pa into this (some compress-attn metadata kernels read it separately).
        metadata.actual_seq_lengths_q_cmp = torch.zeros(
            bs + 1, dtype=torch.int32, device=device,
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
        fm.actual_seq_lengths_kv.fill_(0)
        fm.actual_seq_lengths_kv[:bs].copy_(seq_lens[:bs].to(torch.int32))

        # actual_seq_lengths_q_cmp mirrors actual_seq_lengths_q_pa (captured already).
        fm.actual_seq_lengths_q_cmp.copy_(fm.actual_seq_lengths_q_pa)

        # Phase 3: compress locs via shared helper (Task 2 _compute_compress_locs).
        pool = forward_batch.token_to_kv_pool
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        out_cache_loc = forward_batch.out_cache_loc
        device = seq_lens.device

        result = self._compute_compress_locs(
            pool=pool,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices[:bs],
            seq_lens=seq_lens[:bs].to(torch.int32),
            out_cache_loc=out_cache_loc,
            is_decode=forward_mode.is_decode(),
            bs=bs,
            device=device,
        )

        # In-place copy result into preallocated fm buffers.
        # Page tables are 2-D (max_bs, c{ratio}_max_pages); zero the tail per row.
        def _copy_2d(dst: torch.Tensor, src: torch.Tensor) -> None:
            dst.fill_(0)
            dst[: src.shape[0], : src.shape[1]].copy_(src)

        # Loc tensors are 1-D flat; zero the tail.
        def _copy_1d(dst: torch.Tensor, src: torch.Tensor) -> None:
            dst.fill_(0)
            dst[: src.shape[0]].copy_(src)

        for key in (
            "c4_page_table", "c128_page_table",
            "c4_state_page_table", "c128_state_page_table",
        ):
            if key in result:
                _copy_2d(getattr(fm, key), result[key])
        for key in ("c4_loc", "c128_loc", "c4_state_loc", "c128_state_loc"):
            if key in result:
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
        swa_src = getattr(fm, "block_tables_swa", None)
        if swa_src is None:
            swa_src = fm.block_tables
        _copy_2d(fm.swa_page_table, swa_src)

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
        block_tables_swa = getattr(fm, "block_tables_swa", None)
        fm.swa_page_table = (
            block_tables_swa if block_tables_swa is not None else fm.block_tables
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
        result = self._compute_compress_locs(
            pool=forward_batch.token_to_kv_pool,
            req_to_token=forward_batch.req_to_token_pool.req_to_token,
            req_pool_indices=forward_batch.req_pool_indices,
            seq_lens=forward_batch.seq_lens.to(torch.int32),
            out_cache_loc=forward_batch.out_cache_loc,
            is_decode=forward_batch.forward_mode.is_decode(),
            bs=forward_batch.batch_size,
            device=forward_batch.seq_lens.device,
        )
        for k, v in result.items():
            setattr(fm, k, v)
        # Non-decode path leaves c{ratio}_state_loc / c{ratio}_loc absent from
        # result; the eager contract was that those fm fields are None in
        # non-decode mode. Replay (Task 5) checks key presence instead.
        if not forward_batch.forward_mode.is_decode():
            for ratio in self._dsv4_compress_ratios:
                if ratio in (4, 128):
                    if f"c{ratio}_state_loc" not in result:
                        setattr(fm, f"c{ratio}_state_loc", None)
                    if f"c{ratio}_loc" not in result:
                        setattr(fm, f"c{ratio}_loc", None)

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
    ) -> dict:
        """Pure compress-loc computation shared by eager and graph-replay paths.

        Computes on the fly from req_to_token + the V4 KV pool's swa
        translation, since the request pool has no per-ratio mapping tables.
        Slower than a prealloc per-request mapping but avoids cross-cutting
        allocator surgery on the request pool.

        Returns a dict keyed by the output field names that the eager shim
        assigns onto the forward metadata struct; graph replay copies each
        entry into preallocated buffers. For each ratio in {4, 128} (subject
        to self._dsv4_compress_ratios), produces:
          - c{ratio}_state_page_table
          - c{ratio}_state_loc        (omitted from dict when not decode)
          - c{ratio}_loc              (omitted from dict when not decode)
          - c{ratio}_page_table

        Non-decode mode skips writing the two ``state_loc`` / ``loc`` keys
        rather than emitting ``None`` values, so graph-replay callers can
        distinguish "field does not apply" from "valid empty tensor" via
        ``key in result``. The eager shim restores the old "attribute is
        None" contract by None-filling absent keys onto the metadata struct.

        Caller responsibility: this helper calls ``seq_lens.max().item()``
        which synchronizes to host. Graph-replay callers (cuda/acl graph
        capture) must bound ``seq_lens_max`` differently — typically from
        the preallocated buffer shapes — and short-circuit before reaching
        this helper, or pre-compute the needed page counts on the host
        before the capture window.
        """
        result: dict = {}
        req_pool = req_pool_indices

        seq_lens_max = int(seq_lens.max().item()) if bs > 0 else 0
        n_pages = max(1, (seq_lens_max + self.page_size - 1) // self.page_size)

        # State page tables — for each request, for each page, the state-buffer
        # page index. Use the FIRST token of each page as the representative
        # (tokens within the same SWA page produce contiguous state-buffer slots).
        page_starts = torch.arange(
            0, n_pages * self.page_size, self.page_size, device=device
        )  # [n_pages]
        # [bs, n_pages] flattened token positions; positions past seq_len are
        # clamped to 0 (will be masked out by _get_kv_indices' kv_len).
        page_starts_2d = page_starts.unsqueeze(0).expand(bs, n_pages)
        # Index req_to_token: [bs, n_pages] of full-kv-pool slot ids.
        raw_loc = req_to_token[
            req_pool.unsqueeze(1).expand(-1, n_pages), page_starts_2d
        ]

        for ratio in self._dsv4_compress_ratios:
            if ratio not in (4, 128):
                continue
            # State page table — translate each (bs, n_pages) raw kv slot to a
            # state-buffer page id. translate_kv_loc_to_compress_state_loc gives
            # the flat state slot; divide by page_size for the page id.
            state_loc_2d = pool.translate_kv_loc_to_compress_state_loc(raw_loc, ratio)
            state_page_2d = (state_loc_2d // self.page_size).to(torch.int32)

            # State loc — single state-buffer slot for the new decode token.
            # In decode, out_cache_loc has shape [bs] (one new token per req).
            if is_decode:
                state_loc_decode = pool.translate_kv_loc_to_compress_state_loc(
                    out_cache_loc, ratio
                )
                # Compressor write loc — step 5c slab allocator. For each
                # request that just completed a ratio-aligned chunk, the new
                # compressed token writes to slot
                #   k_seq = seqlen_after // ratio - 1     (compressed seq pos)
                #   slot  = req_to_token_c{N}_pages[req_pool_idx, k_seq // page_size]
                #           * page_size + k_seq % page_size
                # Replaces the old `raw_out_loc // ratio` formula which only
                # worked when the request happened to land on a page-aligned
                # raw kv slot (= almost never).
                pages_table = pool.get_req_to_token_c_pages(ratio)
                should_compress = (seq_lens % ratio) == 0
                k_seq = (seq_lens.to(torch.int64) // ratio - 1).clamp(min=0)
                page_seq = (k_seq // self.page_size).to(torch.int64)
                offset = (k_seq % self.page_size).to(torch.int64)
                kernel_page = pages_table[req_pool.to(torch.int64), page_seq].to(
                    torch.int64
                )
                compress_out_loc = (kernel_page * self.page_size + offset).to(
                    torch.int32
                )
                compress_out_loc = torch.where(
                    should_compress,
                    compress_out_loc,
                    torch.zeros_like(compress_out_loc),
                )

            result[f"c{ratio}_state_page_table"] = state_page_2d
            if is_decode:
                result[f"c{ratio}_state_loc"] = state_loc_decode
                result[f"c{ratio}_loc"] = compress_out_loc

            # c{ratio}_page_table — kernel-view page table for c{N}_kv_pool.
            # Reading directly from the slab gives each request its own
            # dedicated kernel pages, so cmp_kv reads at compressed seq
            # pos 0..N-1 land in the right physical slots regardless of how
            # the raw_kv allocator scattered the request's full pages.
            pages_table = pool.get_req_to_token_c_pages(ratio)
            n_pages_c = (n_pages + ratio - 1) // ratio
            n_pages_c = max(1, min(n_pages_c, pages_table.shape[1]))
            c_page_table = pages_table[req_pool.to(torch.int64), :n_pages_c].to(
                torch.int32
            )
            result[f"c{ratio}_page_table"] = c_page_table

        return result

    def init_forward_metadata_indexer(self, core_attn_metadata):
        # li_quant_metadata is computed inside _compute_kernel_metadata; nothing
        # extra to do here. Return None to satisfy the mixin contract.
        return None

    def _seed_c4_topk_indices(self, forward_batch: "ForwardBatch") -> torch.Tensor:
        """Allocate a [T, index_topk] int32 tensor on the compute device,
        filled with -1 (= "no valid sparse index" sentinel that npu_sparse_
        attn_sharedkv accepts). Real ``forward_c4_indexer`` will overwrite the
        contents via ``npu_quant_lightning_indexer``; until then this lets the
        c4 path of ``_forward_compressed`` consume a well-shaped tensor."""
        if forward_batch.input_ids is not None:
            T = forward_batch.input_ids.shape[0]
        else:
            T = int(forward_batch.seq_lens.sum().item())
        return torch.full(
            (T, self._dsv4_index_topk),
            -1,
            dtype=torch.int32,
            device=forward_batch.seq_lens.device,
        )

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

        # Reshape cmp_kv to share page_size with ori_kv before the kernel call.
        # main's V4 pool layout: c{N}_kv_pool buffer is (num_pages, page_size//
        # ratio, 1, dim) so each native page holds page_size//ratio compressed
        # tokens. The aclnn kernel expects cmp_kv to share its page_size with
        # ori_kv (=global page_size). We slice the buffer to a ratio-aligned
        # native-page count and view it as (N_kernel, global_page_size, 1, dim).
        #
        # cmp_block_table values: step 5c slab (`req_to_token_c{N}_pages`)
        # already gives kernel-view page indices in [0, N_kernel), so no
        # further `// page_ratio` divide is needed — the divide was a leftover
        # from step 5b when block_table came from raw kv pool page indices.
        ori_page_size = ori_kv.shape[1]
        cmp_native_page_size = cmp_kv.shape[1]
        cmp_block_table = getattr(
            fm, f"c{compress_ratio}_page_table", fm.swa_page_table
        )
        if cmp_native_page_size != ori_page_size:
            page_ratio = ori_page_size // cmp_native_page_size
            assert page_ratio == compress_ratio, (
                f"page_ratio={page_ratio} != compress_ratio={compress_ratio}; "
                "main's V4 pool keeps c{N}_native_page_size = global_page_size//ratio"
            )
            n_native = cmp_kv.shape[0]
            n_kernel = n_native // page_ratio
            cmp_kv = cmp_kv[: n_kernel * page_ratio].reshape(
                n_kernel, ori_page_size, *cmp_kv.shape[2:]
            )
            # Slab already in kernel-view page space — no divide.
            cmp_block_table = cmp_block_table.to(torch.int32)

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
            topk = fm.c4_topk_indices
            if topk is None:
                topk = self._seed_c4_topk_indices(forward_batch)
                fm.c4_topk_indices = topk
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

    # c4/c128 compressor + indexer entry points. ``forward_compress`` is a
    # no-op on NPU (Compressor.forward_npu writes the pool inline rather than
    # via the CUDA mixin set_extra_key_buffer chain).

    def forward_compress(self, *args, **kwargs):  # type: ignore[override]
        return None

    def forward_core_compressor(  # type: ignore[override]
        self,
        x: torch.Tensor,
        forward_batch: "ForwardBatch",
        layer_id: int,
        compressor,
    ) -> None:
        """Trigger the OUTER attention compressor on NPU.

        CUDA's ``forward_core_compressor`` does (compressor → set_extra_key_*).
        On NPU, ``Compressor.forward_npu`` writes the KV pool inline, so we
        just call the compressor.
        """
        if forward_batch.forward_mode.is_idle():
            return
        compressor(x, forward_batch)

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

        ``C4Indexer.forward_npu`` runs the npu_quant_lightning_indexer and
        writes the real top-k indices itself; we seed the -1 sentinel here
        so ``_forward_compressed`` reads a well-shaped tensor on idle ranks
        (where forward_npu is short-circuited).
        """
        if forward_batch.forward_mode.is_idle():
            return
        self.forward_metadata.c4_topk_indices = self._seed_c4_topk_indices(
            forward_batch
        )
