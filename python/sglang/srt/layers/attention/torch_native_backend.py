from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class TorchNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        # Pool refs — captured at construction so they survive deletion of the
        # corresponding ForwardBatch fields.
        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.use_sliding_window_kv_pool = (
            isinstance(self.token_to_kv_pool, SWAKVPool)
            and self.token_to_kv_pool.swa_layer_nums > 0
        )
        # full->SWA translated out_cache_loc, computed once per forward
        self.swa_out_cache_loc = None

        # ---- Embedding KV-cache-skip fast path (opt-in) ---------------------
        # For a pure embedding model (prefill only, no decode) with no cached
        # prefix, the paged-KV write-then-gather-back round trip is pure
        # redundancy: the freshly-computed local k/v ARE each sequence's entire
        # K/V context. When every guard in forward_extend holds we skip both the
        # set_kv_buffer write and the pool gather and run SDPA directly on the
        # local k/v (bit-exact). Opt-in via SGLANG_SKIP_EMBED_KV_CACHE=1; OFF by
        # default -> forward_extend runs the original path byte-for-byte.
        self._skip_embed_kv_cache = os.environ.get("SGLANG_SKIP_EMBED_KV_CACHE") == "1"
        # is_generation == False <=> served purely as an embedding model: a
        # one-shot prefill with no decode step that could read this KV back.
        self._embed_no_decode = not model_runner.model_config.is_generation
        # Chunked-prefill guard: the skip is only bit-exact when a request
        # finishes in ONE forward. If chunking is possible (chunked_prefill_size
        # in (0, context_len)), chunk-1 would skip a KV write that chunk-2 later
        # reads back -> wrong embedding. Cache whether chunking is even possible;
        # if not, extend_prefix_lens==0 alone already makes the fast path safe.
        _cps = getattr(model_runner.server_args, "chunked_prefill_size", None)
        _ctx = model_runner.model_config.context_len
        self._kvskip_chunk_impossible = _cps is None or _cps <= 0 or _cps >= _ctx

    @staticmethod
    def _make_sliding_window_mask(
        *,
        q_len: int,
        kv_len: int,
        sliding_window_size: int,
        device: torch.device,
        query_offset: int = 0,
    ) -> torch.Tensor:
        q_pos = torch.arange(
            query_offset, query_offset + q_len, device=device
        ).unsqueeze(1)
        k_pos = torch.arange(kv_len, device=device).unsqueeze(0)
        return (k_pos <= q_pos) & (k_pos >= q_pos - sliding_window_size)

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            self.swa_out_cache_loc = (
                self.token_to_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )
        else:
            self.swa_out_cache_loc = None

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor] = None,
        scaling=None,
        enable_gqa=False,
        causal=False,
        is_cross_attn=False,
        sliding_window_size: Optional[int] = None,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            encoder_lens: [num_seqs] or None
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool
            is_cross_attn: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
            if encoder_lens is not None:
                if is_cross_attn:
                    start_kv = 0
                    end_kv = encoder_lens[seq_idx]
                else:
                    start_kv = encoder_lens[seq_idx]
                    end_kv = start_kv + seq_len_kv
            else:
                start_kv = 0
                end_kv = start_kv + seq_len_kv
            per_req_query = query[:, start_q:end_q, :]
            per_req_query_redudant = torch.empty(
                (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                dtype=per_req_query.dtype,
                device=per_req_query.device,
            )

            per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, start_kv:end_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            attn_mask = None
            is_causal = causal
            if sliding_window_size is not None and sliding_window_size > -1:
                attn_mask = self._make_sliding_window_mask(
                    q_len=seq_len_kv,
                    kv_len=seq_len_kv,
                    sliding_window_size=sliding_window_size,
                    device=per_req_query.device,
                )
                is_causal = False

            per_req_out_redudant = (
                scaled_dot_product_attention(
                    per_req_query_redudant.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    attn_mask=attn_mask,
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=is_causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_extend_local(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        """Local-k/v variant of ``_run_sdpa_forward_extend`` for the embedding
        KV-cache-skip fast path.

        Identical per-sequence-loop structure to ``_run_sdpa_forward_extend``,
        but reads K/V directly from the local (just-computed, not-yet-cached)
        tensors instead of gathering them back from the paged KV pool via
        ``req_to_token`` indices. Valid only when every sequence's full KV
        context IS exactly its own freshly-computed K/V (``extend_prefix_lens ==
        0`` for the whole batch) -- enforced by the caller. No dummy-query
        padding is needed (unlike the original) because query length == kv
        length exactly when prefix_len == 0.

        Args:
            query:  [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            key:    [num_tokens, num_kv_heads, head_size]
            value:  [num_tokens, num_kv_heads, head_size]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """
        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)
        key = key.movedim(0, key.dim() - 2)
        value = value.movedim(0, value.dim() - 2)

        if not (query.dtype == key.dtype == value.dtype):
            # scaled_dot_product_attention() expects q, k, v to share a dtype
            key = key.to(query.dtype)
            value = value.to(query.dtype)

        start = 0
        for seq_idx in range(extend_seq_lens.shape[0]):
            seq_len = int(extend_seq_lens[seq_idx])
            end = start + seq_len

            per_req_query = query[:, start:end, :]
            per_req_key = key[:, start:end, :]
            per_req_value = value[:, start:end, :]

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start:end, :, :] = per_req_out
            start = end
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor] = None,
        scaling=None,
        enable_gqa=False,
        causal=False,
        is_cross_attn=False,
        sliding_window_size: Optional[int] = None,
    ):
        """Run the decode forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            encoder_lens: [num_seqs] or None
            scaling: float or None
            enable_gqa: bool
            causal: bool
            is_cross_attn: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            seq_len_q = 1
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + seq_len_q
            if encoder_lens is not None:
                if is_cross_attn:
                    start_kv = 0
                    end_kv = encoder_lens[seq_idx]
                else:
                    start_kv = encoder_lens[seq_idx]
                    end_kv = start_kv + seq_len_kv
            else:
                start_kv = 0
                end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.

            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, start_kv:end_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            attn_mask = None
            is_causal = causal
            if sliding_window_size is not None and sliding_window_size > -1:
                attn_mask = self._make_sliding_window_mask(
                    q_len=seq_len_q,
                    kv_len=seq_len_kv,
                    sliding_window_size=sliding_window_size,
                    device=per_req_query.device,
                    query_offset=seq_len_kv - seq_len_q,
                )
                is_causal = False

            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    attn_mask=attn_mask,
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=is_causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def _forward_extend_kv_skip(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
    ):
        """Embedding KV-cache-skip fast path (all guards checked by
        ``forward_extend`` before this is called).

        Runs per-sequence SDPA on the freshly-computed local k/v WITHOUT writing
        them to (``set_kv_buffer``) or gathering them back from the paged KV
        pool. Valid only when every sequence's full K/V context IS exactly its
        own just-computed k/v (``extend_prefix_lens == 0`` everywhere) and no
        sliding window / pool quantization is in play -- all enforced by the
        caller, so this is bit-exact vs. the original write-then-gather path.
        """
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        k_ = k.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        v_ = v.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        self._run_sdpa_forward_extend_local(
            q_,
            o_,
            k_,
            v_,
            forward_batch.extend_seq_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
        )
        return o

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # ---- Embedding KV-cache-skip fast path (opt-in) --------------------
        # Bit-exact iff EVERY guard below holds; any failure => the original
        # path (unchanged, below this block) runs. With SGLANG_SKIP_EMBED_KV_CACHE
        # unset, self._skip_embed_kv_cache is False, so can_skip is always False
        # and this whole block is a no-op -> original behavior, byte-for-byte.
        can_skip = (
            self._skip_embed_kv_cache  # (1) opt-in env switch
            and self._embed_no_decode  # (3) embedding model: no decode reads KV back
            and save_kv_cache  # (2)
            and k is not None
            and v is not None
            and not layer.is_cross_attention  # (4)
            and (  # (8) sliding window not enabled (local path has no SWA mask)
                layer.sliding_window_size is None or layer.sliding_window_size <= -1
            )
            # (5) no cached prefix: every sequence's full KV == its local k/v
            and forward_batch.extend_prefix_lens is not None
            and bool((forward_batch.extend_prefix_lens == 0).all())
        )
        if can_skip and not self._kvskip_chunk_impossible:
            # (6) chunked prefill possible: only skip when THIS forward covers
            # each sequence in full (no later chunk reads the skipped KV back).
            orig = forward_batch.orig_seq_lens
            ext = forward_batch.extend_seq_lens
            can_skip = (
                orig is not None and ext is not None and bool((ext == orig).all())
            )
        if can_skip:
            # (7) pool stores K/V verbatim (no fp8/uint8 repacking) so a gather
            # would return the local k/v byte-for-byte.
            pool = self.token_to_kv_pool
            can_skip = getattr(pool, "store_dtype", pool.dtype) == getattr(
                pool, "dtype", k.dtype
            ) and k.dtype == getattr(pool, "dtype", k.dtype)
        if can_skip:
            return self._forward_extend_kv_skip(q, k, v, layer, forward_batch)
        # ---- Original path (unchanged) -------------------------------------

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache and k is not None and v is not None:
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(cache_loc, self.swa_out_cache_loc), k, v
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if layer.is_cross_attention or layer.attn_type == AttentionType.ENCODER_ONLY:
            causal = False

        self._run_sdpa_forward_extend(
            q_,
            o_,
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            forward_batch.encoder_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
            is_cross_attn=layer.is_cross_attention,
            sliding_window_size=(
                layer.sliding_window_size
                if causal
                and not layer.is_cross_attention
                and layer.sliding_window_size is not None
                and layer.sliding_window_size > -1
                else None
            ),
        )
        return o

    def forward_decode(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        # During torch.compile, there is a bug in rotary_emb that causes the
        # output value to have a 3D tensor shape. This reshapes the output correctly.
        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache and k is not None and v is not None:
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(cache_loc, self.swa_out_cache_loc), k, v
            )

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        self._run_sdpa_forward_decode(
            q_,
            o_,
            self.token_to_kv_pool.get_key_buffer(layer.layer_id),
            self.token_to_kv_pool.get_value_buffer(layer.layer_id),
            self.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.encoder_lens,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
            is_cross_attn=layer.is_cross_attention,
            sliding_window_size=(
                layer.sliding_window_size
                if not layer.is_cross_attention
                and layer.sliding_window_size is not None
                and layer.sliding_window_size > -1
                else None
            ),
        )

        return o

    def support_triton(self):
        return False
