"""Known MTP-prefix producer -> native FA3 paged attention, exact request buckets."""

from copy import copy

import torch
import triton
import triton.language as tl


@triton.jit
def _pages(
    mapping,
    reqs,
    prefix,
    append,
    table,
    lengths,
    MAP_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    TILE: tl.constexpr,
    SLOTS: tl.constexpr,
):
    b, tile = tl.program_id(0), tl.program_id(1)
    c = tile * TILE + tl.arange(0, TILE)
    n = tl.load(prefix + b)
    kept = tl.minimum(n, 4096)
    start = n - kept
    req = tl.load(reqs + b)
    old = tl.load(mapping + req * MAP_STRIDE + start + c, c < kept, 0)
    new = tl.load(append + b * SLOTS + c - kept, (c >= kept) & (c < kept + SLOTS), 0)
    tl.store(table + b * WIDTH + c, tl.where(c < kept, old, new), c < WIDTH)
    if tile == 0:
        tl.store(lengths + b, kept + SLOTS)


class AnchorFA3Forward:
    def __init__(self, backend):
        assert backend.fa_impl_ver == 3 and backend.page_size == 1
        assert backend.topk <= 1, "Anchor pages require native unsplit-query metadata"
        assert not backend.use_mla and not backend.use_sliding_window_kv_pool
        self.backend = backend
        self.original = backend.forward_extend
        self.capture_calls = self.eager_calls = 0
        device = backend.req_to_token.device
        self.slots = backend.speculative_num_draft_tokens
        assert self.slots in (7, 8)
        self.width = 4096 + self.slots
        self.table = torch.empty((32, self.width), dtype=torch.int32, device=device)
        self.lengths = torch.empty(32, dtype=torch.int32, device=device)
        self.query_ptr = torch.arange(33, dtype=torch.int32, device=device) * self.slots

    def __call__(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
        q_rope=None,
        k_rope=None,
        sinks=None,
        **kwargs
    ):
        batch = forward_batch
        # Installed only by the known MTP-prefix controller, not a generic mask adapter.
        if not batch.forward_mode.is_target_verify():
            return self.original(
                q, k, v, layer, batch, save_kv_cache, q_rope, k_rope, sinks, **kwargs
            )
        bs = batch.batch_size
        assert 1 <= bs <= 32 and q.shape[0] == bs * self.slots
        assert batch.out_cache_loc.numel() == bs * self.slots
        assert q_rope is None and k_rope is None and sinks is None
        if any(value is not None and value is not False for value in kwargs.values()):
            raise ValueError(
                "DMTP anchor FA3 does not support descales/score modifiers/LSE"
            )
        assert layer.sliding_window_size in (None, -1)
        if torch.cuda.is_current_stream_capturing():
            self.capture_calls += 1
        else:
            self.eager_calls += 1
        _pages[(bs, triton.cdiv(self.width, 256))](
            self.backend.req_to_token,
            batch.req_pool_indices,
            batch.seq_lens,
            batch.out_cache_loc,
            self.table,
            self.lengths,
            MAP_STRIDE=self.backend.req_to_token.stride(0),
            WIDTH=self.width,
            TILE=256,
            SLOTS=self.slots,
        )
        old = self.backend.forward_metadata
        metadata = copy(old)
        metadata.page_table = self.table[:bs]
        metadata.cache_seqlens_int32 = self.lengths[:bs]
        metadata.cu_seqlens_q = self.query_ptr[: bs + 1]
        metadata.cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(self.lengths[:bs], 0, dtype=torch.int32), (1, 0)
        )
        metadata.max_seq_len_q = self.slots
        metadata.max_seq_len_k = self.width
        metadata.scheduler_metadata = None
        self.backend.forward_metadata = metadata
        try:
            # Existing model is ENCODER_ONLY/full backend: causal=False, window=(-1,-1).
            # Exact last4096 prefix pages + all8 block pages express its original mask.
            return self.original(
                q, k, v, layer, batch, save_kv_cache, q_rope, k_rope, sinks, **kwargs
            )
        finally:
            self.backend.forward_metadata = old


class Fixed8FA3Metadata:
    """Fixed8 private mapping adapter, retaining the native FA3 attention operator."""

    def __init__(self, backend):
        from sglang.srt.layers.attention.flashattention_backend import (
            FlashAttentionMetadata,
        )

        assert backend.fa_impl_ver == 3 and backend.page_size == 1
        self.backend = backend
        self.original = backend.init_forward_metadata_out_graph
        self.metadata_class = FlashAttentionMetadata
        self.buffers = {}

    def __call__(self, batch, in_capture=False):
        if not batch.forward_mode.is_draft_extend_v2():
            return self.original(batch, in_capture=in_capture)
        bs = batch.batch_size
        assert batch.input_ids.numel() == bs * 8
        mapping = self.backend.req_to_token
        if bs not in self.buffers:
            device = mapping.device
            metadata = self.metadata_class()
            metadata.page_table = torch.empty(
                (bs, mapping.shape[1]), dtype=torch.int32, device=device
            )
            metadata.cache_seqlens_int32 = torch.empty(
                bs, dtype=torch.int32, device=device
            )
            metadata.cu_seqlens_q = (
                torch.arange(bs + 1, dtype=torch.int32, device=device) * 8
            )
            metadata.cu_seqlens_k = torch.zeros(
                bs + 1, dtype=torch.int32, device=device
            )
            metadata.max_seq_len_q = 8
            metadata.max_seq_len_k = mapping.shape[1]
            self.buffers[bs] = metadata
        metadata = self.buffers[bs]
        # Graph inputs remain device-resident, unlike native packed decode's CPU length hint.
        torch.index_select(
            mapping, 0, batch.req_pool_indices[:bs].long(), out=metadata.page_table
        )
        metadata.cache_seqlens_int32.copy_(batch.seq_lens[:bs])
        torch.cumsum(
            metadata.cache_seqlens_int32,
            0,
            dtype=torch.int32,
            out=metadata.cu_seqlens_k[1:],
        )
        self.backend.forward_metadata = metadata
