# SPDX-License-Identifier: Apache-2.0
"""KVarN attention backend for SGLang.

Implements compressed KV-cache attention with a dual-pool architecture
matching the KVarN algorithm:

  1. **Hadamard rotation** of Q, K, V before attention and storage.
  2. **Tail pool** (fp16, small): stores ROTATED K/V for in-progress blocks
     and sink blocks. Each block occupies one tail-pool slot:
     ``[pool_slots, group, num_kv_heads, head_dim]``.
  3. **Compressed cache** (int4, uint8): stores flushed (fully-written) blocks
     as quantized tiles: ``[num_blocks, num_kv_heads, tile_bytes]`` per layer.
  4. **Block-to-slot mapping**: translates scheduler page block_ids to
     tail-pool slots. Flushed blocks have their slot freed and live in int4.
  5. **Decode/Extend**: gather K/V from both sources (int4 dequant + fp16 tail
     pool), un-rotate, then run SDPA per request. This is the "slow path"
     (materialize + SDPA) — the fused Triton kernel is future work.
  6. **Un-rotate** the output after attention.

The scheduler sees ``max_total_num_tokens = num_blocks * page_size`` — the
compressed cache capacity — while the actual GPU memory is dominated by the
small fp16 tail pool + the int4 compressed cache, yielding the 3.5×
compression ratio. The standard ``token_to_kv_pool`` is a NoOp pool (large
logical size, tiny physical allocation) used only for the scheduler's
capacity accounting; the real K/V storage lives in the KVarN backend's
tail pool and compressed cache.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.quantization.kvarn.config import KVarNConfig
from sglang.srt.layers.quantization.kvarn.flush_manager import KVarNFlushManager
from sglang.srt.layers.quantization.kvarn.hadamard import build_hadamard

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Number of sink blocks (first pages) kept in fp16 tail pool, never flushed.
# Default: 1 sink block = first page.
KVaRN_SINK_BLOCKS = 1


class KVarNAttnBackend(AttentionBackend):
    """KVarN attention backend with dual-pool (tail + int4) architecture."""

    needs_cpu_seq_lens: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        """Block sizes (page sizes) the KVarN kernels support."""
        return [128, 64]

    @staticmethod
    def get_preferred_block_size() -> int:
        """Preferred block size for KVarN — matches the tile group size."""
        return 128

    @staticmethod
    def supports_head_size(head_size: int) -> bool:
        """KVarN supports power-of-2 head dimensions (Hadamard requirement)."""
        return head_size > 0 and (head_size & (head_size - 1)) == 0

    @staticmethod
    def supports_kv_cache_dtype(kv_cache_dtype: str) -> bool:
        """Check if a kv_cache_dtype string is supported by KVarN."""
        return kv_cache_dtype.startswith("kvarn_")

    def __init__(
        self, model_runner: ModelRunner, kvarn_config: Optional[KVarNConfig] = None
    ):
        super().__init__()

        self.model_runner = model_runner
        self.server_args = model_runner.server_args
        self.device = model_runner.device

        # Expose pool references for the scheduler / hybrid backend.
        # The token_to_kv_pool is a NoOpMHATokenToKVPool (large logical size,
        # tiny physical allocation) — real K/V storage lives in the KVarN
        # backend's tail pool + int4 compressed cache.
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token_pool = model_runner.req_to_token_pool

        # Get KVarN config
        if kvarn_config is not None:
            kvarn_cfg = kvarn_config
        else:
            kvarn_cfg = getattr(model_runner, "kvarn_config", None)
            if kvarn_cfg is None:
                quant_config = getattr(model_runner, "quant_config", None)
                if quant_config is not None:
                    kvarn_cfg = getattr(quant_config, "kvarn_config", None)

        if kvarn_cfg is None:
            raise RuntimeError(
                "KVarNAttnBackend requires a KVarNConfig. "
                "Set --quantization kvarn-k4v4g128 or similar."
            )
        self.cfg: KVarNConfig = kvarn_cfg
        self.group = self.cfg.group

        # Validate head_dim — KVarN supports power-of-2 head dims
        model_config = model_runner.model_config
        if model_config.head_dim & (model_config.head_dim - 1) != 0:
            raise RuntimeError(
                f"KVarN requires power-of-2 head_dim, got {model_config.head_dim}"
            )

        # Model config
        model_config = model_runner.model_config
        self.num_heads = (
            model_config.get_total_num_attention_heads() // model_runner.ps.tp_size
        )
        self.num_kv_heads = model_config.get_num_kv_heads(model_runner.ps.tp_size)
        self.head_dim = model_config.head_dim
        self.v_head_dim = model_config.v_head_dim
        if self.v_head_dim is None:
            self.v_head_dim = self.head_dim
        self.scale = (
            float(model_config.scaling)
            if hasattr(model_config, "scaling")
            else 1.0 / (self.head_dim**0.5)
        )

        # For hybrid models (e.g. Qwen3.5 GDN), only the full-attention layers
        # use the KV cache.  Linear/SSM layers have their own state and must NOT
        # be allocated compressed-cache buffers — otherwise we waste 4x memory
        # on models where 48/64 layers are linear attention.
        # Check both model_config and the HF config (hybrid GDN models like
        # Qwen3.5 expose full_attention_layer_ids on the HF config, not on
        # model_config since they're not "hybrid SWA" models).
        full_attn_ids = getattr(model_config, "full_attention_layer_ids", None)
        if full_attn_ids is None:
            # Hybrid GDN models (Qwen3.5) expose full_attention_layer_ids on
            # the HF text config, not on model_config.
            hf_config = getattr(model_config, "hf_config", None)
            if hf_config is not None:
                text_config = getattr(hf_config, "text_config", hf_config)
                full_attn_ids = getattr(text_config, "full_attention_layer_ids", None)
        if full_attn_ids is not None and len(full_attn_ids) > 0:
            self.num_layers = len(full_attn_ids)
            self.full_attn_layer_ids = list(full_attn_ids)
            # Map absolute layer_id -> compressed-cache index (0..N-1)
            self._layer_id_to_idx = {lid: i for i, lid in enumerate(full_attn_ids)}
        else:
            self.num_layers = model_config.num_attention_layers
            self.full_attn_layer_ids = None
            self._layer_id_to_idx = None  # identity mapping

        # Hadamard rotation matrix (shared across all layers)
        self._H: Optional[torch.Tensor] = None  # [D, D] fp32, lazy

        # Tail pool: per-layer fp16 buffers
        # Allocated in _init_pools, called from init_forward_metadata
        self.tail_K: list[torch.Tensor] = (
            []
        )  # [num_layers][pool_slots, group, Hk, D] fp16
        self.tail_V: list[torch.Tensor] = []
        self.pool_slots = 0

        # Compressed cache: per-layer uint8 buffers
        self.kv_cache_int4: list[torch.Tensor] = (
            []
        )  # [num_layers][num_blocks, Hk, tile_bytes] uint8
        self.num_blocks = 0

        # Block-to-slot mapping
        self._block_to_slot: dict[int, int] = {}  # block_id -> tail pool slot
        self._slot_to_block: dict[int, int] = {}  # tail pool slot -> block_id
        self._free_slots: list[int] = []
        self._block_fill: dict[int, int] = {}  # block_id -> token count in tail pool

        # Sink blocks: block_ids that stay in fp16, never flushed
        self._sink_block_ids: set[int] = set()
        # Retired sinks: finished requests' sink blocks, kept fp16-resident
        # for future prefix-cache hits. Lazily evicted (flushed to int4) when
        # pool slots run dry, oldest first.
        self._retired_sinks: dict[int, None] = {}

        # Flush manager
        self.flush_manager = KVarNFlushManager(
            cfg=self.cfg,
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            v_head_dim=self.v_head_dim,
            sink_blocks=KVaRN_SINK_BLOCKS,
        )

        # Page size (= group)
        self.page_size = model_runner.page_size

        logger.info(
            f"KVarNAttnBackend: group={self.group}, num_kv_heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, num_layers={self.num_layers}, "
            f"full_attn_layer_ids={self.full_attn_layer_ids}, "
            f"pool_slots={self.pool_slots} (allocated later), "
            f"page_size={self.page_size}"
        )

    def _li(self, layer_id: int) -> int:
        """Map absolute layer_id to compressed-cache index."""
        if self._layer_id_to_idx is not None:
            return self._layer_id_to_idx[layer_id]
        return layer_id

    def _get_hadamard(self, device: torch.device) -> torch.Tensor:
        """Get or build the Hadamard rotation matrix."""
        if self._H is None or self._H.device != device:
            self._H = build_hadamard(self.head_dim, device)
        return self._H

    def _init_pools(self):
        """Allocate tail pool and compressed cache tensors."""
        if self.pool_slots > 0:
            return  # Already initialized

        # Cache SM count for split-K auto-enable heuristic in the Triton kernel.
        if not hasattr(self, "_sm_count") or self._sm_count == 0:
            self._sm_count = torch.cuda.get_device_properties(
                self.device
            ).multi_processor_count

        mr = self.model_runner
        max_running = mr.server_args.max_running_requests or 256
        max_prefill_tokens = mr.server_args.max_prefill_tokens or 16384

        # Tail pool size: 2 * max_running (sink + in-progress tail per request)
        # + prefill_blocks + headroom.
        prefill_blocks = (max_prefill_tokens + self.group - 1) // self.group
        self.pool_slots = max(2 * max_running + prefill_blocks + 8, 8)

        # Compressed cache: one slot per scheduler page.
        # The paged allocator uses 1-based page_ids (page 0 is reserved as a
        # dummy/padding slot for padded tokens), so block_ids range from 1 to
        # num_pages inclusive. Size the compressed cache and block-to-slot
        # lookup tensor with +1 entry (matching the standard MHATokenToKVPool
        # convention of size + page_size) so block_id = num_pages is in range.
        self.num_blocks = mr.max_total_num_tokens // self.page_size + 1

        device = self.device

        # Allocate per-layer tail pool buffers
        self.tail_K = []
        self.tail_V = []
        for _ in range(self.num_layers):
            self.tail_K.append(
                torch.zeros(
                    self.pool_slots,
                    self.group,
                    self.num_kv_heads,
                    self.head_dim,
                    dtype=torch.float16,
                    device=device,
                )
            )
            self.tail_V.append(
                torch.zeros(
                    self.pool_slots,
                    self.group,
                    self.num_kv_heads,
                    self.v_head_dim,
                    dtype=torch.float16,
                    device=device,
                )
            )

        # Allocate per-layer compressed cache buffers
        self.kv_cache_int4 = []
        tile_bytes = self.cfg.tile_bytes_aligned
        for _ in range(self.num_layers):
            self.kv_cache_int4.append(
                torch.zeros(
                    self.num_blocks,
                    self.num_kv_heads,
                    tile_bytes,
                    dtype=torch.uint8,
                    device=device,
                )
            )

        # Initialize free slots
        self._free_slots = list(range(self.pool_slots))

        # Block-to-slot GPU lookup tensor (used by Triton kernels)
        self._block_lookup_size = self.num_blocks
        self._block_to_slot_t = torch.full(
            (self.num_blocks,),
            -1,
            dtype=torch.int32,
            device=device,
        )

        # Store tail pool strides for Triton kernel launches
        self._tail_K_stride0 = self.tail_K[0].stride(0)
        self._tail_K_stride1 = self.tail_K[0].stride(1)
        self._tail_K_stride2 = self.tail_K[0].stride(2)

        # Pre-allocated rotation scratch (avoids allocation during CUDA graph capture)
        max_batched_tokens = mr.server_args.max_prefill_tokens or 16384
        self._k_rot_scratch = torch.empty(
            max_batched_tokens,
            self.num_kv_heads,
            self.head_dim,
            dtype=torch.float16,
            device=device,
        )
        self._v_rot_scratch = torch.empty_like(self._k_rot_scratch)

        # Cached fp16 Hadamard for fast matmul in store path
        self._H_fp16 = self._get_hadamard(device).to(torch.float16).contiguous()

        # Pre-allocated Q rotation buffer (CUDA graph capture-safe)
        # The decode driver reshapes Q from [B, Hq, D] to [B*Hq, D] and does
        # torch.mm(q_flat, H16) → [B*Hq, D]. So the buffer must be sized
        # [max_decode_rows, D] where max_decode_rows = max_running * Hq
        # (sized for max decode batch).
        max_decode_tokens = max(max_running * 1, 1)
        max_prefill = max_prefill_tokens or 16384
        # Decode: B*Hq rows of D; Prefill: max_prefill rows of D.
        # Take the max of both to cover any code path.
        max_decode_rows = max(max_decode_tokens * self.num_heads, max_prefill)
        self._q_rot_fp16_buf = torch.empty(
            max_decode_rows,
            self.head_dim,
            dtype=torch.float16,
            device=device,
        )

        # Pre-allocated fused-decode output + split-K mid buffers
        # (CUDA graph capture-safe; reused across layers + steps).
        # Sized for the decode regime: max_running * Hq rows (flat N = B*Hq).
        # Layout: mid_o is [N, max_splits, D], mid_lse is [N, max_splits].
        _max_n = max(max_running * self.num_heads, 1)
        from sglang.srt.layers.attention.kvarn_ops.triton_decode import (
            KVARN_MAX_KV_SPLITS,
        )

        self._fused_out_buf = torch.empty(
            _max_n,
            self.head_dim,
            dtype=torch.float16,
            device=device,
        )
        self._mid_o_buf = torch.empty(
            _max_n,
            KVARN_MAX_KV_SPLITS,
            self.head_dim,
            dtype=torch.float32,
            device=device,
        )
        self._mid_lse_buf = torch.empty(
            _max_n,
            KVARN_MAX_KV_SPLITS,
            dtype=torch.float32,
            device=device,
        )

        logger.info(
            f"KVarN pools allocated: tail_pool_slots={self.pool_slots}, "
            f"compressed_blocks={self.num_blocks}, "
            f"tail_pool_bytes={self.pool_slots * self.group * self.num_kv_heads * self.head_dim * 4 * self.num_layers / 1e9:.2f} GB, "
            f"compressed_cache_bytes={self.num_blocks * tile_bytes * self.num_kv_heads * self.num_layers / 1e9:.2f} GB"
        )

    def _alloc_slot(self, block_id: int) -> int:
        """Allocate a tail pool slot for a block.

        If no free slots are available, evicts the oldest retired sink
        (flushes it to int4 so future prefix-cache hits still find a valid
        tile) and reuses its slot.
        """
        if not self._free_slots:
            # Evict oldest retired sink
            if self._retired_sinks:
                old = next(iter(self._retired_sinks))
                self._retired_sinks.pop(old)
                self._flush_block(old)
                self._sink_block_ids.discard(old)
            if not self._free_slots:
                raise RuntimeError("KVarN tail pool exhausted — no free slots")
        slot = self._free_slots.pop()
        self._block_to_slot[block_id] = slot
        self._slot_to_block[slot] = block_id
        self._block_fill[block_id] = 0
        # Update GPU lookup tensor
        if self._block_to_slot_t is not None and block_id < self._block_lookup_size:
            self._block_to_slot_t[block_id] = slot
        return slot

    def _free_slot(self, block_id: int):
        """Free a tail pool slot (block has been flushed to int4)."""
        slot = self._block_to_slot.pop(block_id, None)
        if slot is not None:
            self._slot_to_block.pop(slot, None)
            self._block_fill.pop(block_id, None)
            self._free_slots.append(slot)
            # Update GPU lookup tensor: -1 means block is in int4 cache
            if self._block_to_slot_t is not None and block_id < self._block_lookup_size:
                self._block_to_slot_t[block_id] = -1

    def get_slot_for_block(self, block_id: int) -> Optional[int]:
        """Get the tail pool slot for a block, or None if flushed."""
        return self._block_to_slot.get(block_id)

    def get_kv_cache_int4(self, layer_id: int) -> torch.Tensor:
        """Get the compressed int4 cache tensor for a layer."""
        return self.kv_cache_int4[self._li(layer_id)]

    def get_tail_pool(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the tail pool K/V tensors for a layer."""
        li = self._li(layer_id)
        return self.tail_K[li], self.tail_V[li]

    # ── AttentionBackend interface ─────────────────────────────────────────

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Eager entry point. Initialize pools, flush blocks, allocate slots."""
        if self.pool_slots == 0:
            self._init_pools()

        # Flush any full blocks from the tail pool to int4 cache.
        self._maybe_flush_blocks(forward_batch)

        # Pre-allocate tail pool slots for new blocks
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is not None:
            self._ensure_slots_for_tokens(out_cache_loc)

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        """Per-iter metadata prep (outside graph). Allocates slots, flushes
        blocks, and fills static block_table/seq_lens/cu_seqlens buffers."""
        if self.pool_slots == 0:
            self._init_pools()

        # Flush full blocks from tail pool to int4 cache
        self._maybe_flush_blocks(forward_batch)

        # Pre-allocate tail pool slots for new blocks in this batch
        # (must happen before the captured forward_decode scatter store)
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is not None:
            self._ensure_slots_for_tokens(out_cache_loc)

        # Build block_table/seq_lens/cu_seqlens into static buffers
        if hasattr(self, "_cg_block_table"):
            self._fill_cg_buffers(forward_batch)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch):
        """Graph-recordable GPU ops. No-op — the fused decode kernel reads
        from static buffers filled in init_forward_metadata_out_graph."""
        pass

    def _fill_cg_buffers(self, forward_batch: ForwardBatch):
        """Fill pre-allocated CUDA graph buffers with current batch data."""
        B = forward_batch.batch_size
        if B > self._cg_max_batch_size:
            raise RuntimeError(
                f"Batch size {B} exceeds CUDA graph max {self._cg_max_batch_size}"
            )

        seq_lens = forward_batch.seq_lens
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = self.model_runner.req_to_token_pool.req_to_token

        # Zero-fill unused entries
        self._cg_block_table[:B].zero_()
        self._cg_seq_lens[:B].zero_()
        self._cg_cu_seqlens.zero_()

        if B > 0:
            # Fill seq_lens
            self._cg_seq_lens[:B] = seq_lens.to(torch.int32)[:B]

            # Vectorized block_table construction
            max_blocks = min(self._cg_max_blocks, self._cg_block_table.shape[1])
            col_offsets = torch.arange(max_blocks, device=self.device) * self.page_size
            req_indices = req_pool_indices.long()
            token_positions = col_offsets.unsqueeze(0).expand(B, max_blocks)
            max_token_pos = req_to_token.shape[1] - 1
            token_positions = token_positions.clamp(max=max_token_pos)
            slots = req_to_token[req_indices.unsqueeze(1), token_positions]
            self._cg_block_table[:B, :max_blocks] = (slots // self.page_size).to(
                torch.int32
            )

            # Fill cu_seqlens
            self._cg_cu_seqlens[1 : B + 1] = torch.cumsum(self._cg_seq_lens[:B], dim=0)

            # Fill verify-specific buffers when this is a TARGET_VERIFY batch
            # (extend_seq_lens is populated by Fix 4 / dflash_info.prepare_for_verify).
            # All computation is eager (outside the graph); the captured
            # _fused_verify_path only reads the filled static buffers.
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
            if extend_seq_lens is not None:
                self._fill_cg_verify_buffers(B, extend_seq_lens)

    def _fill_cg_verify_buffers(self, B: int, extend_seq_lens: torch.Tensor) -> None:
        """Fill _cg_vq_req / _cg_vq_seqlen / _cg_qsl / _cg_committed for the
        captured verify path. Runs eagerly in init_forward_metadata_out_graph.
        """
        device = self.device
        qlens = extend_seq_lens.to(torch.long)[:B]
        NQ = int(qlens.sum().item())
        if NQ > self._cg_max_num_tokens:
            raise RuntimeError(
                f"Verify token count {NQ} exceeds CUDA graph max "
                f"{self._cg_max_num_tokens}"
            )

        # qsl: cumulative sum of extend_seq_lens [B+1]
        self._cg_qsl.zero_()
        self._cg_qsl[1 : B + 1] = torch.cumsum(qlens, dim=0)

        # vq_req: which request each query token belongs to [NQ]
        # repeat_interleave(arange(B), qlens) — eager fill into static buffer
        self._cg_vq_req[:NQ].zero_()
        self._cg_vq_seqlen[:NQ].zero_()
        if NQ > 0:
            vq_req_long = torch.repeat_interleave(torch.arange(B, device=device), qlens)
            self._cg_vq_req[:NQ] = vq_req_long.to(torch.int32)

            # committed = seq_len - extend_len (cached prefix per request)
            committed = self._cg_seq_lens[:B].to(torch.long) - qlens
            self._cg_committed[:B] = committed.to(torch.int32)

            # pos_in_req: position within each request's query block [NQ]
            pos_in_req = torch.arange(NQ, device=device) - self._cg_qsl[:B][vq_req_long]

            # vq_seqlen: causal length = committed[vq_req] + pos + 1
            self._cg_vq_seqlen[:NQ] = (committed[vq_req_long] + pos_in_req + 1).to(
                torch.int32
            )

    def init_cuda_graph_state(self, max_batch_size: int, max_num_token: int):
        """Pre-allocate CUDA graph state (static buffers for block_table, etc.)."""
        if self.pool_slots == 0:
            self._init_pools()

        # Pre-allocate static buffers for CUDA graph
        max_blocks_per_req = max(
            self.model_runner.model_config.context_len // self.page_size, 1
        )
        self._cg_block_table = torch.zeros(
            max_batch_size,
            max_blocks_per_req,
            dtype=torch.int32,
            device=self.device,
        )
        self._cg_seq_lens = torch.zeros(
            max_batch_size, dtype=torch.int32, device=self.device
        )
        self._cg_cu_seqlens = torch.zeros(
            max_batch_size + 1, dtype=torch.int32, device=self.device
        )
        self._cg_max_batch_size = max_batch_size
        self._cg_max_blocks = max_blocks_per_req

        # Verify-specific static buffers (for graph-capturable _fused_verify_path)
        # _cg_vq_req[NQ]: which request each query token belongs to
        # _cg_vq_seqlen[NQ]: causal length per query token
        # _cg_qsl[B+1]: cumulative sum of extend_seq_lens (query start locs)
        # _cg_committed[B]: cached prefix length per request (seq_len - extend_len)
        self._cg_vq_req = torch.zeros(
            max_num_token, dtype=torch.int32, device=self.device
        )
        self._cg_vq_seqlen = torch.zeros(
            max_num_token, dtype=torch.int32, device=self.device
        )
        self._cg_qsl = torch.zeros(
            max_batch_size + 1, dtype=torch.int64, device=self.device
        )
        self._cg_committed = torch.zeros(
            max_batch_size, dtype=torch.int32, device=self.device
        )
        self._cg_max_num_tokens = max_num_token

        # Warm up scatter store kernel (fast). The fused decode kernel has
        # autotuning configs that can take a long time on first run — let
        # it autotune during normal serving (first request is slow, then fast).
        self._warmup_kernels(max_batch_size, max_blocks_per_req)

    def _warmup_kernels(self, max_batch_size: int, max_blocks_per_req: int):
        """Warm up all Triton kernels with dummy inputs to trigger JIT compilation."""
        from sglang.srt.layers.attention.kvarn_ops.triton_decode import (
            kvarn_decode_attention,
            kvarn_scatter_store,
        )

        device = self.device
        B = min(max_batch_size, 4)
        D = self.head_dim
        Hk = self.num_kv_heads
        Hq = self.num_heads
        G = self.group

        # Warm up scatter store
        dummy_k = torch.zeros(B, Hk, D, dtype=torch.float16, device=device)
        dummy_v = torch.zeros(B, Hk, D, dtype=torch.float16, device=device)
        dummy_loc = torch.zeros(B, dtype=torch.int32, device=device)
        kvarn_scatter_store(
            dummy_k,
            dummy_v,
            dummy_loc,
            self._block_to_slot_t,
            self.tail_K[0],
            self.tail_V[0],
            G,
            D,
            self._block_lookup_size,
        )

        # Warm up fused decode
        dummy_q = torch.zeros(B, Hq, D, dtype=torch.float16, device=device)
        dummy_bt = torch.zeros(B, max_blocks_per_req, dtype=torch.int32, device=device)
        dummy_sl = torch.ones(B, dtype=torch.int32, device=device)
        dummy_cs = torch.zeros(B + 1, dtype=torch.int32, device=device)
        dummy_cs[1:] = torch.cumsum(dummy_sl, dim=0)
        try:
            _ = kvarn_decode_attention(
                query=dummy_q,
                kv_cache=self.kv_cache_int4[0],
                tail_K=self.tail_K[0],
                tail_V=self.tail_V[0],
                hadamard=self._H_fp16.float(),
                scale=self.scale,
                cfg=self.cfg,
                impl=self,
                block_table=dummy_bt,
                seq_lens=dummy_sl,
                cu_seqlens=dummy_cs,
            )
        except Exception as e:
            logger.warning(f"KVarN kernel warmup failed (non-fatal): {e}")

        # Warm up fused verify kernel (used by spec decode TARGET_VERIFY)
        try:
            from sglang.srt.layers.attention.kvarn_ops.triton_decode import (
                kvarn_verify_attention,
            )

            NQ_warmup = B * 16  # draft_token_num per request, rough
            dummy_vq_req = torch.zeros(NQ_warmup, dtype=torch.int32, device=device)
            dummy_vq_seqlen = torch.ones(NQ_warmup, dtype=torch.int32, device=device)
            _ = kvarn_verify_attention(
                query=torch.zeros(NQ_warmup, Hq, D, dtype=torch.float16, device=device),
                kv_cache=self.kv_cache_int4[0],
                tail_K=self.tail_K[0],
                tail_V=self.tail_V[0],
                hadamard=self._H_fp16.float(),
                scale=self.scale,
                cfg=self.cfg,
                impl=self,
                block_table=dummy_bt,
                vq_req=dummy_vq_req,
                vq_seqlen=dummy_vq_seqlen,
                max_ctx_blocks=max_blocks_per_req,
                sliding_window=0,
            )
        except Exception as e:
            logger.warning(f"KVarN verify kernel warmup failed (non-fatal): {e}")

        logger.info("KVarN Triton kernels warmed up")

    def get_cuda_graph_seq_len_fill_value(self) -> int:
        return 1

    def clear(self):
        """Clear all mappings. Called when the scheduler resets."""
        self._block_to_slot.clear()
        self._slot_to_block.clear()
        self._block_fill.clear()
        self._free_slots = list(range(self.pool_slots))
        self._sink_block_ids.clear()
        self._retired_sinks.clear()

    # ── Forward methods ─────────────────────────────────────────────────────

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        sinks=None,
    ) -> torch.Tensor:
        """Decode forward: one token per request.

        Uses the fused Triton decode kernel when KVARN_FUSED_DECODE=1 (default):
        dequant int4 in registers + online-softmax flash-decode.
        Falls back to Python gather + SDPA when KVARN_FUSED_DECODE=0.
        """
        from sglang.srt.layers.attention.kvarn_ops.triton_decode import (
            kvarn_decode_attention,
            kvarn_scatter_store,
        )

        layer_id = layer.layer_id
        N = q.shape[0]
        H = self._get_hadamard(self.device)

        q_3d = q.view(N, self.num_heads, self.head_dim)
        k_3d = k.view(N, self.num_kv_heads, self.head_dim)
        v_3d = v.view(N, self.num_kv_heads, self.v_head_dim)

        # Store K/V to tail pool (slots already allocated in init_forward_metadata_out_graph)
        if save_kv_cache:
            use_triton_store = True
            if use_triton_store:
                # Use pre-allocated scratch + cached fp16 Hadamard (capture-safe)
                k_rot = self._k_rot_scratch[:N]
                v_rot = self._v_rot_scratch[:N]
                torch.matmul(k_3d.to(torch.float16), self._H_fp16, out=k_rot)
                torch.matmul(v_3d.to(torch.float16), self._H_fp16, out=v_rot)
                kvarn_scatter_store(
                    k_rot,
                    v_rot,
                    forward_batch.out_cache_loc.to(torch.int32),
                    self._block_to_slot_t,
                    self.tail_K[self._li(layer_id)],
                    self.tail_V[self._li(layer_id)],
                    self.group,
                    self.head_dim,
                    self._block_lookup_size,
                )
            else:
                self._store_to_tail_pool(
                    layer_id, k_3d, v_3d, forward_batch.out_cache_loc
                )

        use_fused_decode = True
        if use_fused_decode:
            # Fused Triton decode kernel
            # Use static CUDA graph buffers if available, otherwise build fresh
            if hasattr(self, "_cg_block_table") and N <= self._cg_max_batch_size:
                block_table = self._cg_block_table[:N]
                seq_lens_t = self._cg_seq_lens[:N]
                cu_seqlens = self._cg_cu_seqlens[: N + 1]
            else:
                block_table, seq_lens_t, cu_seqlens = self._build_block_table(
                    forward_batch
                )

            out = kvarn_decode_attention(
                query=q_3d,
                kv_cache=self.kv_cache_int4[self._li(layer_id)],
                tail_K=self.tail_K[self._li(layer_id)],
                tail_V=self.tail_V[self._li(layer_id)],
                hadamard=H,
                scale=self.scale,
                cfg=self.cfg,
                impl=self,
                block_table=block_table,
                seq_lens=seq_lens_t,
                cu_seqlens=cu_seqlens,
                sliding_window=(
                    getattr(layer, "sliding_window_size", -1)
                    if getattr(layer, "sliding_window_size", -1) > 0
                    else 0
                ),
            )

            return out.view(N, self.num_heads * self.head_dim)
        else:
            # Python gather + SDPA fallback
            q_rot = (q_3d.float() @ H).to(q.dtype)
            out = torch.empty(
                N, self.num_heads, self.head_dim, dtype=q.dtype, device=q.device
            )
            req_pool_indices = forward_batch.req_pool_indices
            seq_lens = forward_batch.seq_lens
            req_to_token = self.model_runner.req_to_token_pool.req_to_token

            for i in range(N):
                req_idx = int(req_pool_indices[i].item())
                seq_len = int(seq_lens[i].item())
                block_ids = self._get_block_ids_for_request(
                    req_idx, seq_len, req_to_token
                )
                K_full, V_full = self._gather_request_kv(layer_id, block_ids, seq_len)

                q_i = q_rot[i : i + 1].transpose(0, 1).unsqueeze(0).float()
                K_t = K_full.transpose(0, 1).unsqueeze(0).float()
                V_t = V_full.transpose(0, 1).unsqueeze(0).float()
                o = F.scaled_dot_product_attention(
                    q_i,
                    K_t,
                    V_t,
                    is_causal=False,
                    scale=self.scale,
                    enable_gqa=self.num_kv_heads < self.num_heads,
                )
                o = o[0, :, 0, :].to(q.dtype)
                out[i] = (o.float() @ H.T).to(q.dtype)

            return out.view(N, self.num_heads * self.head_dim)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        sinks=None,
    ) -> torch.Tensor:
        """Extend forward: prefill, chunked-prefill, or target_verify.

        For target_verify (speculative decode), uses the fused verify kernel
        (kvarn_verify_attention) which reads int4 + fp16 pool directly per
        token with VQ_INDIRECT=True — no fp16 materialization.

        For prefill/extend, uses build_packed_kv Triton kernel + SDPA.
        """
        from sglang.srt.layers.attention.kvarn_ops.triton_decode import (
            _kvarn_build_packed_kv_kernel,
            kvarn_scatter_store,
        )

        layer_id = layer.layer_id
        N = q.shape[0]
        H = self._get_hadamard(self.device)

        q_3d = q.view(N, self.num_heads, self.head_dim)
        k_3d = k.view(N, self.num_kv_heads, self.head_dim)
        v_3d = v.view(N, self.num_kv_heads, self.v_head_dim)

        # Store K/V to tail pool (slots already allocated in init_forward_metadata_out_graph)
        if save_kv_cache:
            use_triton_store = True
            if use_triton_store:
                k_rot = self._k_rot_scratch[:N]
                v_rot = self._v_rot_scratch[:N]
                torch.matmul(k_3d.to(torch.float16), self._H_fp16, out=k_rot)
                torch.matmul(v_3d.to(torch.float16), self._H_fp16, out=v_rot)
                kvarn_scatter_store(
                    k_rot,
                    v_rot,
                    forward_batch.out_cache_loc.to(torch.int32),
                    self._block_to_slot_t,
                    self.tail_K[self._li(layer_id)],
                    self.tail_V[self._li(layer_id)],
                    self.group,
                    self.head_dim,
                    self._block_lookup_size,
                )
            else:
                self._store_to_tail_pool(
                    layer_id, k_3d, v_3d, forward_batch.out_cache_loc
                )

        # Check if this is a target_verify (speculative decode) batch
        forward_mode = getattr(forward_batch, "forward_mode", None)
        is_verify = (
            forward_mode is not None
            and hasattr(forward_mode, "is_target_verify")
            and forward_mode.is_target_verify()
        )

        if is_verify:
            return self._fused_verify_path(q_3d, layer, forward_batch, H)

        # Rotate Q
        q_rot = (q_3d.float() @ H).to(torch.float16)

        # Normal extend path: build_packed_kv + flash_attn_varlen
        # Build block_table, seq_lens, cu_seqlens
        block_table, seq_lens_t, cu_seqlens = self._build_block_table(forward_batch)

        # Materialize K/V via build_packed_kv Triton kernel
        B = forward_batch.batch_size
        max_blocks = block_table.shape[1]
        total_tokens = int(cu_seqlens[-1].item())

        K_packed = torch.empty(
            total_tokens,
            self.num_kv_heads,
            self.head_dim,
            dtype=torch.float16,
            device=self.device,
        )
        V_packed = torch.empty(
            total_tokens,
            self.num_kv_heads,
            self.v_head_dim,
            dtype=torch.float16,
            device=self.device,
        )

        use_triton_extend = True

        if use_triton_extend and total_tokens > 0:
            _kvarn_build_packed_kv_kernel[(B * max_blocks, self.num_kv_heads)](
                block_table,
                seq_lens_t,
                cu_seqlens,
                self._block_to_slot_t,
                self.kv_cache_int4[self._li(layer_id)],
                self.tail_K[self._li(layer_id)],
                self.tail_V[self._li(layer_id)],
                K_packed,
                V_packed,
                block_table.stride(0),
                self.kv_cache_int4[self._li(layer_id)].stride(0),
                self.kv_cache_int4[self._li(layer_id)].stride(1),
                self._tail_K_stride0,
                self._tail_K_stride1,
                self._tail_K_stride2,
                K_packed.stride(0),
                K_packed.stride(1),
                MAX_BLOCKS_PER_REQ=max_blocks,
                D=self.head_dim,
                GROUP=self.group,
                K_BITS=self.cfg.key_bits,
                V_BITS=self.cfg.value_bits,
                NUM_BLOCKS_LOOKUP=self._block_lookup_size,
                K_PACKED_OFFSET=self.cfg.k_packed_offset,
                K_S_COL_OFFSET=self.cfg.k_s_col_offset,
                K_ZP_OFFSET=self.cfg.k_zp_offset,
                K_S_ROW_OFFSET=self.cfg.k_s_row_offset,
                V_PACKED_OFFSET=self.cfg.v_packed_offset,
                V_S_COL_OFFSET=self.cfg.v_s_col_offset,
                V_S_ROW_OFFSET=self.cfg.v_s_row_offset,
                V_ZP_OFFSET=self.cfg.v_zp_offset,
                num_warps=4,
                num_stages=2,
            )
        else:
            # Python gather fallback
            req_pool_indices = forward_batch.req_pool_indices
            req_to_token = self.model_runner.req_to_token_pool.req_to_token
            token_offset = 0
            for i in range(B):
                req_idx = int(req_pool_indices[i].item())
                seq_len = int(seq_lens_t[i].item())
                if seq_len <= 0:
                    continue
                block_ids = self._get_block_ids_for_request(
                    req_idx, seq_len, req_to_token
                )
                K_full, V_full = self._gather_request_kv(layer_id, block_ids, seq_len)
                K_packed[token_offset : token_offset + seq_len] = K_full
                V_packed[token_offset : token_offset + seq_len] = V_full
                token_offset += seq_len

        # Run attention via flash_attn_varlen (batched, no Python loop)
        # Cached multiquery path: one flash_attn_varlen
        # call for the whole batch instead of per-request SDPA.
        out = torch.empty(
            N, self.num_heads, self.head_dim, dtype=q.dtype, device=q.device
        )

        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        extend_prefix_lens = getattr(forward_batch, "extend_prefix_lens", None)

        if extend_seq_lens is not None:
            # Extend mode: each request has prefix (cached) + new tokens
            qlens = extend_seq_lens.to(torch.int32)
            prefix_lens = (
                extend_prefix_lens.to(torch.int32)
                if extend_prefix_lens is not None
                else torch.zeros_like(qlens)
            )
            seq_lens_full = (prefix_lens + qlens).to(torch.int32)
        else:
            # Pure prefill: all tokens are new
            qlens = seq_lens_t[:B].to(torch.int32)
            prefix_lens = torch.zeros_like(qlens)
            seq_lens_full = qlens

        # Build cu_seqlens for flash_attn_varlen
        cu_q = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
        cu_q[1:] = torch.cumsum(qlens, dim=0)
        cu_k = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
        cu_k[1:] = torch.cumsum(seq_lens_full, dim=0)

        try:
            from flash_attn_interface import flash_attn_varlen_func as _fa_varlen

            _has_flash = True
        except ImportError:
            try:
                from sgl_kernel.flash_attn import flash_attn_varlen_func as _fa_varlen

                _has_flash = True
            except ImportError:
                _has_flash = False

        use_flash = _has_flash and total_tokens > 0

        if use_flash:
            # flash_attn_varlen: q [N, Hq, D], K [total_k, Hk, D], V [total_k, Hk, vD]
            # For GQA: Hq must be a multiple of Hk
            # Causal mask is bottom-right aligned: query token t attends to keys <= prefix+t
            # This matches the extend semantics when seqlen_q < seqlen_k
            q_for_fa = q_rot[:N].contiguous()
            K_for_fa = K_packed[:total_tokens].contiguous()
            V_for_fa = V_packed[:total_tokens].contiguous()

            fa_out = _fa_varlen(
                q_for_fa,
                K_for_fa,
                V_for_fa,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=int(qlens.max().item()) if B > 0 else 1,
                max_seqlen_k=int(seq_lens_full.max().item()) if B > 0 else 1,
                softmax_scale=self.scale,
                causal=True,
            )
            # flash_attn_interface returns (output, lse); sgl_kernel returns output
            if isinstance(fa_out, tuple):
                fa_out = fa_out[0]
            # fa_out may be [N, Hq, D] or [N, Hq*D] depending on implementation
            if fa_out.dim() == 2:
                fa_out = fa_out.view(N, self.num_heads, self.head_dim)
            # Un-rotate each head: [N, Hq, D] @ [D, D]^T = [N, Hq, D]
            out[:] = (fa_out.float() @ H.T).to(q.dtype)
        else:
            # Fallback: per-request SDPA
            if extend_seq_lens is not None:
                token_offset = 0
                for i in range(extend_seq_lens.shape[0]):
                    ext_len = int(extend_seq_lens[i].item())
                    prefix_len = (
                        int(extend_prefix_lens[i].item())
                        if extend_prefix_lens is not None
                        else 0
                    )
                    seq_len = prefix_len + ext_len

                    q_start = token_offset
                    q_end = token_offset + ext_len
                    token_offset = q_end

                    if ext_len <= 0:
                        continue

                    kv_start = int(cu_seqlens[i].item())
                    kv_end = kv_start + seq_len

                    q_i = q_rot[q_start:q_end].transpose(0, 1).unsqueeze(0).float()
                    K_t = K_packed[kv_start:kv_end].transpose(0, 1).unsqueeze(0).float()
                    V_t = V_packed[kv_start:kv_end].transpose(0, 1).unsqueeze(0).float()

                    if prefix_len == 0:
                        o = F.scaled_dot_product_attention(
                            q_i,
                            K_t,
                            V_t,
                            is_causal=True,
                            scale=self.scale,
                            enable_gqa=self.num_kv_heads < self.num_heads,
                        )
                    else:
                        q_len = ext_len
                        q_pos = (
                            torch.arange(q_len, device=q.device).unsqueeze(1)
                            + prefix_len
                        )
                        k_pos = torch.arange(seq_len, device=q.device).unsqueeze(0)
                        mask = k_pos <= q_pos
                        o = F.scaled_dot_product_attention(
                            q_i,
                            K_t,
                            V_t,
                            attn_mask=mask,
                            scale=self.scale,
                            enable_gqa=self.num_kv_heads < self.num_heads,
                        )

                    o = o[0].transpose(0, 1)
                    out[q_start:q_end] = (o.float() @ H.T).to(q.dtype)
            else:
                seq_lens = forward_batch.seq_lens
                token_offset = 0
                for i in range(seq_lens.shape[0]):
                    seq_len = int(seq_lens[i].item())
                    q_start = token_offset
                    q_end = token_offset + seq_len
                    token_offset = q_end
                    if seq_len <= 0:
                        continue

                    kv_start = int(cu_seqlens[i].item())
                    kv_end = kv_start + seq_len

                    q_i = q_rot[q_start:q_end].transpose(0, 1).unsqueeze(0).float()
                    K_t = K_packed[kv_start:kv_end].transpose(0, 1).unsqueeze(0).float()
                    V_t = V_packed[kv_start:kv_end].transpose(0, 1).unsqueeze(0).float()

                    o = F.scaled_dot_product_attention(
                        q_i,
                        K_t,
                        V_t,
                        is_causal=True,
                        scale=self.scale,
                        enable_gqa=self.num_kv_heads < self.num_heads,
                    )
                    o = o[0].transpose(0, 1)
                    out[q_start:q_end] = (o.float() @ H.T).to(q.dtype)

        return out.view(N, self.num_heads * self.head_dim)

    def _fused_verify_path(
        self,
        q_3d: torch.Tensor,  # [NQ, Hq, D]
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        H: torch.Tensor,  # [D, D] fp32 Hadamard
    ) -> torch.Tensor:
        """Speculative-decode verify via the fused dual-source kernel.

        Each query token becomes a virtual kernel row with its own
        bottom-right causal length (cached_len + idx + 1) and an indirection
        to its request's block-table row — so the verify step reads int4
        tiles + the fp16 pool directly without materializing the whole
        context to fp16 scratch.

        Graph-capturable: when CUDA graph buffers are available
        (``self._cg_block_table`` exists) and the batch fits, all metadata
        (block_table, vq_req, vq_seqlen) is pre-computed into static buffers
        by ``_fill_cg_verify_buffers`` in ``init_forward_metadata_out_graph``
        (eager, before capture/replay). The captured forward only launches
        ``kvarn_verify_attention`` reading those buffers — no fresh
        allocations inside the graph.
        """
        from sglang.srt.layers.attention.kvarn_ops.triton_decode import (
            kvarn_verify_attention,
        )

        layer_id = layer.layer_id
        NQ = q_3d.shape[0]
        device = q_3d.device

        extend_seq_lens = forward_batch.extend_seq_lens
        if extend_seq_lens is None:
            raise RuntimeError(
                "KVarN fused verify path requires forward_batch.extend_seq_lens "
                "(expected draft_token_num per request). Got None — "
                "TARGET_VERIFY was not set up as an extend forward. "
                f"NQ={NQ}, forward_mode={forward_batch.forward_mode}."
            )

        # Use static CUDA graph buffers when available; fall back to fresh
        # tensors for eager mode.
        use_cg = (
            hasattr(self, "_cg_block_table")
            and hasattr(self, "_cg_vq_req")
            and forward_batch.batch_size <= self._cg_max_batch_size
            and NQ <= self._cg_max_num_tokens
        )

        if use_cg:
            B = forward_batch.batch_size
            block_table = self._cg_block_table[:B]
            max_ctx_blocks = block_table.shape[1]
            vq_req = self._cg_vq_req[:NQ]
            vq_seqlen = self._cg_vq_seqlen[:NQ]
        else:
            # Eager: build fresh tensors
            block_table, seq_lens_t, _ = self._build_block_table(forward_batch)
            B = block_table.shape[0]
            max_ctx_blocks = block_table.shape[1]

            qlens = extend_seq_lens.to(torch.long)
            vq_req_long = torch.repeat_interleave(
                torch.arange(B, device=device),
                qlens,
            )
            vq_req = vq_req_long.to(torch.int32)

            qsl = torch.zeros(B + 1, dtype=torch.long, device=device)
            qsl[1:] = torch.cumsum(qlens, dim=0)
            pos_in_req = torch.arange(NQ, device=device) - qsl[:-1][vq_req_long]
            committed = seq_lens_t[:B].to(torch.long) - qlens
            vq_seqlen = (committed[vq_req_long] + pos_in_req + 1).to(torch.int32)

        sliding_window = getattr(layer, "sliding_window_size", -1)
        if sliding_window is None or sliding_window <= 0:
            sliding_window = 0

        out = kvarn_verify_attention(
            query=q_3d,
            kv_cache=self.kv_cache_int4[self._li(layer_id)],
            tail_K=self.tail_K[self._li(layer_id)],
            tail_V=self.tail_V[self._li(layer_id)],
            hadamard=H,
            scale=self.scale,
            cfg=self.cfg,
            impl=self,
            block_table=block_table,
            vq_req=vq_req,
            vq_seqlen=vq_seqlen,
            max_ctx_blocks=max_ctx_blocks,
            sliding_window=sliding_window,
        )

        return out.view(NQ, self.num_heads * self.head_dim)

    # ── Store path ──────────────────────────────────────────────────────────

    def _ensure_slots_for_tokens(self, out_cache_loc: torch.Tensor):
        """Pre-allocate tail pool slots for any new blocks in this batch.
        Must be called BEFORE the Triton scatter store kernel.

        Note: _block_fill is NOT updated here — it's already set correctly
        by _maybe_flush_blocks from the committed boundary computation.
        Fill tracking lives in the builder, not the store path.
        """
        # Compute block_ids for all tokens (vectorized on GPU)
        block_ids_t = out_cache_loc // self.page_size  # [N] on GPU
        # Find unique block_ids that need slots (CPU)
        block_ids_cpu = block_ids_t.cpu()
        unique_new = set()
        for bid in block_ids_cpu.tolist():
            if bid >= 0 and bid not in self._block_to_slot:
                unique_new.add(bid)

        for bid in unique_new:
            self._alloc_slot(bid)
            if bid < KVaRN_SINK_BLOCKS:
                self._sink_block_ids.add(bid)

    def _update_block_fill(self, out_cache_loc: torch.Tensor):
        """Update block fill tracking after scatter store. Called after
        the Triton scatter store to track which blocks are full."""
        loc_cpu = out_cache_loc.cpu()
        for i in range(loc_cpu.shape[0]):
            slot_idx = int(loc_cpu[i].item())
            if slot_idx < 0:
                continue
            block_id = slot_idx // self.page_size
            # Allocate a tail pool slot for new blocks
            if block_id not in self._block_to_slot:
                self._alloc_slot(block_id)
                if block_id < KVaRN_SINK_BLOCKS:
                    self._sink_block_ids.add(block_id)
            self._block_fill[block_id] = self._block_fill.get(block_id, 0) + 1

    def _store_to_tail_pool(
        self,
        layer_id: int,
        k: torch.Tensor,  # [N, Hk, D] bf16/fp16 (NOT yet rotated)
        v: torch.Tensor,  # [N, Hk, vD] bf16/fp16
        out_cache_loc: torch.Tensor,  # [N] slot indices
    ):
        """Rotate K/V and store to tail pool using block-to-slot mapping.

        Assumes slots have already been allocated by _ensure_slots_for_tokens.
        Uses vectorized index_put_ for the scatter.
        """
        H = self._get_hadamard(self.device)

        # Rotate K/V in fp32 for precision, cast to fp16 for storage
        k_rot = (k.float() @ H).to(torch.float16)  # [N, Hk, D]
        v_rot = (v.float() @ H).to(torch.float16)  # [N, Hk, vD]

        # Compute block_id and pos_in_block for each token
        loc = out_cache_loc  # [N] on GPU
        block_ids = loc // self.page_size  # [N]
        pos_in_block = loc % self.page_size  # [N]

        # Build tail pool slot indices for each token using the CPU dict
        tail_slots = torch.empty_like(loc, dtype=torch.long)
        for i in range(loc.shape[0]):
            bid = int(block_ids[i].item())
            tail_slots[i] = self._block_to_slot[bid]

        # Scatter to tail pool via linear index: slot * group + pos_in_block
        linear_idx = tail_slots * self.group + pos_in_block

        K_flat = self.tail_K[self._li(layer_id)].view(
            self.pool_slots * self.group, self.num_kv_heads, self.head_dim
        )
        V_flat = self.tail_V[self._li(layer_id)].view(
            self.pool_slots * self.group, self.num_kv_heads, self.v_head_dim
        )
        K_flat.index_put_((linear_idx,), k_rot, accumulate=False)
        V_flat.index_put_((linear_idx,), v_rot, accumulate=False)

    # ── Gather path ─────────────────────────────────────────────────────────

    def _get_block_ids_for_request(
        self,
        req_idx: int,
        seq_len: int,
        req_to_token: torch.Tensor,
    ) -> list[int]:
        """Get the list of block_ids for a request's sequence.

        Uses req_to_token_pool to find the first slot of each page,
        then computes block_id = slot // page_size.
        """
        num_blocks = (seq_len + self.page_size - 1) // self.page_size
        block_ids = []
        for b in range(num_blocks):
            token_pos = b * self.page_size
            slot = int(req_to_token[req_idx, token_pos].item())
            if slot < 0:
                block_ids.append(-1)
            else:
                block_id = slot // self.page_size
                block_ids.append(block_id)
        return block_ids

    def _build_block_table(
        self,
        forward_batch: ForwardBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build block_table, seq_lens, cu_seqlens for the fused Triton kernel.

        block_table: [B, max_blocks] int32 — block_id per (request, block)
        seq_lens: [B] int32 — sequence length per request
        cu_seqlens: [B+1] int32 — prefix sum of seq_lens
        """
        B = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = self.model_runner.req_to_token_pool.req_to_token

        max_seq_len = int(seq_lens.max().item()) if B > 0 else 1
        max_blocks = (max_seq_len + self.page_size - 1) // self.page_size
        max_blocks = max(max_blocks, 1)

        # Vectorized block_table construction:
        # For each request, get the first slot of each page from req_to_token
        # block_id = slot // page_size
        block_table = torch.zeros(B, max_blocks, dtype=torch.int32, device=self.device)
        seq_lens_t = seq_lens.to(torch.int32)

        if B > 0:
            # Build column indices: [0, page_size, 2*page_size, ...]
            col_offsets = torch.arange(max_blocks, device=self.device) * self.page_size
            # For each request, get req_idx
            req_indices = req_pool_indices.long()
            # Gather first slot of each page: req_to_token[req_idx, col_offset]
            # Shape: [B, max_blocks]
            token_positions = col_offsets.unsqueeze(0).expand(B, max_blocks)
            # Clamp to valid range to avoid OOB
            max_token_pos = req_to_token.shape[1] - 1
            token_positions = token_positions.clamp(max=max_token_pos)
            slots = req_to_token[req_indices.unsqueeze(1), token_positions]
            # block_id = slot // page_size (negative slots → negative block_id → ignored by kernel)
            block_table = (slots // self.page_size).to(torch.int32)

        cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=self.device)
        if B > 0:
            cu_seqlens[1:] = torch.cumsum(seq_lens_t, dim=0)

        return block_table, seq_lens_t, cu_seqlens

    def _gather_request_kv(
        self,
        layer_id: int,
        block_ids: list[int],
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather full K/V for a request from tail pool (fp16) + int4 cache.

        Returns (K [seq_len, Hk, D] fp16, V [seq_len, Hk, vD] fp16) in the
        ROTATED frame (Hadamard rotation applied, matching Q which is also
        rotated). Attention is computed in the rotated frame; the output
        is un-rotated by the caller.
        """
        group = self.group
        n_full = seq_len // group
        tail_len = seq_len % group
        D = self.head_dim
        vD = self.v_head_dim
        device = self.device

        K_parts: list[torch.Tensor] = []
        V_parts: list[torch.Tensor] = []

        for i in range(n_full):
            block_id = block_ids[i]
            slot = self._block_to_slot.get(block_id)
            if slot is not None:
                # Block is in tail pool (in-progress or sink) — already rotated
                K_parts.append(self.tail_K[self._li(layer_id)][slot])
                V_parts.append(self.tail_V[self._li(layer_id)][slot])
            else:
                # Block is flushed to int4 cache — dequant returns rotated frame
                K_blk, V_blk = self._read_block_dequantized(layer_id, block_id)
                K_parts.append(K_blk)
                V_parts.append(V_blk)

        if tail_len > 0:
            block_id = block_ids[n_full]
            slot = self._block_to_slot.get(block_id)
            if slot is not None:
                K_parts.append(self.tail_K[self._li(layer_id)][slot, :tail_len])
                V_parts.append(self.tail_V[self._li(layer_id)][slot, :tail_len])
            else:
                K_parts.append(
                    torch.zeros(
                        tail_len,
                        self.num_kv_heads,
                        D,
                        dtype=torch.float16,
                        device=device,
                    )
                )
                V_parts.append(
                    torch.zeros(
                        tail_len,
                        self.num_kv_heads,
                        vD,
                        dtype=torch.float16,
                        device=device,
                    )
                )

        K = (
            torch.cat(K_parts, dim=0)
            if K_parts
            else torch.empty(
                0,
                self.num_kv_heads,
                D,
                dtype=torch.float16,
                device=device,
            )
        )
        V = torch.cat(V_parts, dim=0) if V_parts else torch.empty_like(K)
        return K, V

    def _read_block_dequantized(
        self,
        layer_id: int,
        block_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read a block from the int4 compressed cache and dequantize to fp16.

        Returns (K [group, Hk, D] fp16, V [group, Hk, vD] fp16) in the
        ROTATED frame (as stored in the compressed cache — no un-rotation).
        The caller uses this in the rotated-frame attention computation.
        """
        K_blk, V_blk = self.flush_manager.dequant_block(
            block_id,
            self.kv_cache_int4,
            layer_id,
        )
        return K_blk, V_blk

    # ── Flush path ──────────────────────────────────────────────────────────

    def _maybe_flush_blocks(self, forward_batch: ForwardBatch):
        """Flush full blocks from tail pool to int4 compressed cache.

        Uses the committed-token boundary (seq_len - query_len) to determine
        which blocks are safe to flush — never quantizes speculative tokens
        that may be rejected. Walks backward from the committed boundary
        while blocks still hold pool slots.

        Also reclaims slots from finished requests: complete blocks are
        flushed (so prefix cache hits find valid int4 tiles), partial blocks
        are discarded (scheduler never prefix-caches partial blocks).

        Sink blocks are NEVER flushed — they stay fp16 for the request's
        lifetime, preserving KVarN's fp16-sink accuracy on multi-turn traffic.
        When a request finishes, its sink is RETIRED (kept fp16-resident for
        future prefix-cache hits) and lazily evicted (flushed to int4) only
        when pool slots run dry.
        """
        GROUP = self.group
        B = forward_batch.batch_size
        seq_lens = forward_batch.seq_lens
        req_pool_indices = forward_batch.req_pool_indices
        req_to_token = self.model_runner.req_to_token_pool.req_to_token

        # --- Query lengths (CPU) ---
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
        forward_mode = getattr(forward_batch, "forward_mode", None)

        if (
            forward_mode is not None
            and hasattr(forward_mode, "is_decode")
            and forward_mode.is_decode()
        ):
            query_lens_cpu = [1] * B
        elif extend_seq_lens is not None:
            query_lens_cpu = extend_seq_lens.cpu().tolist()
        else:
            # CUDA graph replay or unknown mode → treat as decode (q_len=1).
            query_lens_cpu = [1] * B

        seq_lens_cpu = seq_lens.cpu().tolist()

        # --- Vectorized block_table construction (one GPU→CPU transfer) ---
        # Replaces the per-block .item() loop that caused O(context) CUDA
        # syncs per decode step (the throughput regression at long context).
        # CPU block table from scheduler metadata.
        if B > 0:
            max_seq = max(seq_lens_cpu) if seq_lens_cpu else 0
            max_blocks = max((max_seq + self.page_size - 1) // self.page_size, 1)
            col_offsets = torch.arange(max_blocks, device=self.device) * self.page_size
            req_indices = req_pool_indices.long()
            token_positions = col_offsets.unsqueeze(0).expand(B, max_blocks)
            max_token_pos = req_to_token.shape[1] - 1
            token_positions = token_positions.clamp(max=max_token_pos)
            slots = req_to_token[req_indices.unsqueeze(1), token_positions]
            block_table_rows = (slots // self.page_size).cpu().tolist()
        else:
            block_table_rows = []

        # --- blocks_needed: blocks written this step ---
        # committed = tokens already in pool before this step = sl - q_len.
        # For the first chunk of a fresh prefill: committed=0 → range starts
        # at k=0 → ALL blocks (including block 0) are covered, block 0 gets
        # marked as a sink and _block_fill set correctly.  For decode:
        # committed = sl-1 → only the current boundary block.
        blocks_needed: set[int] = set()
        for b in range(B):
            sl = seq_lens_cpu[b]
            if sl <= 0 or b >= len(block_table_rows):
                continue
            row = block_table_rows[b]
            q_len = query_lens_cpu[b] if b < len(query_lens_cpu) else 1
            committed = max(sl - q_len, 0)
            for k in range(
                committed // GROUP, min((sl - 1) // GROUP, len(row) - 1) + 1
            ):
                bid = row[k] if k < len(row) else -1
                if bid >= 0:
                    blocks_needed.add(bid)
                    self._block_fill[bid] = min(sl, (k + 1) * GROUP) - k * GROUP

        # Safety superset: every block receiving a write this step is needed
        # (slot_mapping superset).
        # Prevents the reclaim loop from discarding a block that was just
        # written to but not covered by the committed-boundary range above.
        out_cache_loc = getattr(forward_batch, "out_cache_loc", None)
        if out_cache_loc is not None:
            for s in out_cache_loc.cpu().tolist():
                if s >= 0:
                    blocks_needed.add(s // self.page_size)

        # --- Sink marking (block_table[r][0]) ---
        # A sink is an fp16-resident block kept for the request's lifetime.
        # LIVE sinks (already marked) MUST be re-added to blocks_needed every
        # step so the reclaim loop never touches them — without this lifeline,
        # a sink whose _block_fill=0 (block 0 of any >128-token prompt, never
        # updated by the committed-boundary range) would be discarded on the
        # first decode step, freeing its tail slot without writing int4.
        # The decode kernel would then read block 0 from the zero-initialized
        # int4 cache → tokens 0-127 (system prompt + tool schemas) vanish →
        # the model cannot see its tools. (Fixed
        # by the same lifeline; the port had dropped it.)
        row0_set: set[int] = set()
        for b in range(B):
            if b >= len(block_table_rows) or not block_table_rows[b]:
                continue
            s0 = block_table_rows[b][0]
            if s0 < 0:
                continue
            row0_set.add(s0)
            if s0 in self._sink_block_ids:
                blocks_needed.add(s0)  # lifeline: keep slot for request's lifetime
            elif s0 in blocks_needed:  # written this step → fresh sink
                self._sink_block_ids.add(s0)

        # Un-retire any retired sink named this step
        for bid in [b for b in self._retired_sinks if b in blocks_needed]:
            self._retired_sinks.pop(bid, None)
            if bid not in row0_set and bid in self._sink_block_ids:
                self._sink_block_ids.discard(bid)

        # --- Flush walk: full-but-unflushed blocks below committed boundary ---
        flush_block_ids: list[int] = []
        flush_seen: set[int] = set()
        for b in range(B):
            if b >= len(block_table_rows):
                continue
            sl = seq_lens_cpu[b]
            row = block_table_rows[b]
            if sl <= 0:
                continue
            q_len = query_lens_cpu[b] if b < len(query_lens_cpu) else 1
            committed_len = max(sl - q_len, 0)
            k = min(committed_len // GROUP - 1, len(row) - 1)
            while 1 <= k:
                bid = row[k] if k < len(row) else -1
                if (
                    bid < 0
                    or bid in flush_seen
                    or bid in self._sink_block_ids
                    or bid not in self._block_to_slot
                ):
                    break
                flush_seen.add(bid)
                flush_block_ids.append(bid)
                k -= 1

        # --- Reclaim: slot-holding blocks of finished/preempted requests ---
        discard_ids: list[int] = []
        for bid in [
            b
            for b in self._block_to_slot
            if b not in blocks_needed and b not in flush_seen
        ]:
            full = self._block_fill.get(bid, 0) >= GROUP
            if full and bid in self._sink_block_ids:
                # Retire the sink — keep fp16-resident for future prefix-cache hits
                self._retired_sinks[bid] = None
                continue
            if full:
                flush_seen.add(bid)
                flush_block_ids.append(bid)
            else:
                discard_ids.append(bid)
            if bid in self._sink_block_ids:
                self._sink_block_ids.discard(bid)

        # Execute flushes — batch all blocks across all layers in one Sinkhorn+RTN
        if flush_block_ids:
            slots = [self._block_to_slot[bid] for bid in flush_block_ids]
            self.flush_manager.flush_batched_fast(
                block_ids=flush_block_ids,
                tail_K=self.tail_K,
                tail_V=self.tail_V,
                slots=slots,
                compressed_cache=self.kv_cache_int4,
            )
            # Free the flushed blocks' slots
            for bid in flush_block_ids:
                self._free_slot(bid)

        # Free discarded partial blocks
        for bid in discard_ids:
            self._free_slot(bid)

    def _flush_block(self, block_id: int):
        """Compress a block from tail pool to int4 cache and free the slot."""
        slot = self._block_to_slot.get(block_id)
        if slot is None:
            return  # Already flushed

        self.flush_manager.flush_block(
            block_id=block_id,
            tail_K=self.tail_K,
            tail_V=self.tail_V,
            slot=slot,
            compressed_cache=self.kv_cache_int4,
        )

        # Free the tail pool slot
        self._free_slot(block_id)
        logger.debug(f"Flushed block {block_id} to int4 cache")

    # ── Rotation helpers ────────────────────────────────────────────────────

    def _rotate(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Apply Hadamard rotation: x_rot = x @ H"""
        return (x.float() @ H).to(x.dtype)

    def _unrotate(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Apply inverse Hadamard rotation: x = x_rot @ H^T"""
        return (x.float() @ H.T).to(x.dtype)

    # ── Compatibility stubs ─────────────────────────────────────────────────

    def get_forward_metadata(self):
        return None

    def get_kv_cache_shape(self):
        """Return the KV cache shape for this backend."""
        # Used by model_runner to size the pool. For KVarN, the pool is NoOp
        # and the real storage is in the backend's tail pool + int4 cache.
        return self.num_blocks, self.num_kv_heads, self.head_dim

    def get_q_scale(self):
        return self.scale

    def set_kv_buffer(self, layer, loc, k, v, *args, **kwargs):
        """Direct set_kv_buffer — used by some codepaths. Redirect to tail pool."""
        self._store_to_tail_pool(layer.layer_id, k, v, loc)

    def get_kv_buffer(self, layer_id: int):
        """Get KV buffer — returns tail pool for this layer."""
        return self.tail_K[self._li(layer_id)], self.tail_V[self._li(layer_id)]

    # ── HiCache support: gather/scatter for CPU offload ────────────────────

    # ── HiCache support: raw int4 tile copy ────────────────────────────────

    def flush_block_for_hicache(self, block_id: int):
        """Flush a block from tail pool to int4 cache if not already flushed.

        Called by KVarNHostKVCache.backup_from_device_all_layer before
        copying tiles to CPU, ensuring the CPU copy always gets compact
        int4 tiles (not fp16). Sink blocks are also flushed — HiCache
        eviction means the request is done, so the sink no longer needs
        fp16 residency.
        """
        if block_id in self._block_to_slot:
            self._flush_block(block_id)
            self._sink_block_ids.discard(block_id)
