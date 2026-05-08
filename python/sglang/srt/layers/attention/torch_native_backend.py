from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class TorchNativeAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device

    @staticmethod
    def _scaled_dot_product_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        enable_gqa: bool,
        scale,
        is_causal: bool,
        attn_mask=None,
    ) -> torch.Tensor:
        return scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            enable_gqa=enable_gqa,
            scale=scale,
            is_causal=is_causal,
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        pass

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
        forward_batch: ForwardBatch,
        scaling=None,
        enable_gqa=False,
        causal=False,
        custom_mask=None,
    ):
        """Run the extend forward by using torch native sdpa op.

        Args:
            query: [num_tokens, num_heads, head_size]
            output: [num_tokens, num_heads, head_size]
            k_cache: [max_total_num_tokens, num_heads, head_size]
            v_cache: [max_total_num_tokens, num_heads, head_size]
            req_to_token: [max_num_reqs, max_context_len]
            req_pool_indices: [num_seqs]
            seq_lens: [num_seqs]
            extend_prefix_lens: [num_seqs]
            extend_seq_lens: [num_seqs]
            scaling: float or None
            enable_gqa: bool
            causal: bool

        Returns:
            output: [num_tokens, num_heads, head_size]
        """

        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv, start_mask = 0, 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            extend_seq_len_q = int(extend_seq_lens[seq_idx].item())
            prefill_seq_len_q = int(extend_prefix_lens[seq_idx].item())

            seq_len_kv = int(seq_lens[seq_idx].item())
            end_q = start_q + extend_seq_len_q
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_custom_mask = None
            if custom_mask is not None:
                mask_len = extend_seq_len_q * seq_len_kv
                per_req_custom_mask = custom_mask[start_mask : start_mask + mask_len]
                per_req_custom_mask = per_req_custom_mask.view(
                    extend_seq_len_q,
                    seq_len_kv,
                ).to(device=per_req_query.device, dtype=torch.bool)
                start_mask += mask_len
                if bool(per_req_custom_mask.all().item()):
                    per_req_custom_mask = None

            if per_req_custom_mask is not None:
                per_req_custom_mask = torch.zeros(
                    per_req_custom_mask.shape,
                    dtype=torch.float32,
                    device=per_req_query.device,
                ).masked_fill_(~per_req_custom_mask, float("-inf"))
                per_req_out = (
                    self._scaled_dot_product_attention(
                        per_req_query.unsqueeze(0),
                        per_req_key.unsqueeze(0),
                        per_req_value.unsqueeze(0),
                        attn_mask=per_req_custom_mask.unsqueeze(0).unsqueeze(0),
                        enable_gqa=enable_gqa,
                        scale=scaling,
                        is_causal=False,
                    )
                    .squeeze(0)
                    .movedim(query.dim() - 2, 0)
                )
                output[start_q:end_q, :, :] = per_req_out
            else:
                per_req_query_redudant = torch.empty(
                    (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
                    dtype=per_req_query.dtype,
                    device=per_req_query.device,
                )
                per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query
                per_req_out_redudant = (
                    self._scaled_dot_product_attention(
                        per_req_query_redudant.unsqueeze(0),
                        per_req_key.unsqueeze(0),
                        per_req_value.unsqueeze(0),
                        enable_gqa=enable_gqa,
                        scale=scaling,
                        is_causal=causal,
                    )
                    .squeeze(0)
                    .movedim(query.dim() - 2, 0)
                )
                output[start_q:end_q, :, :] = per_req_out_redudant[
                    prefill_seq_len_q:, :, :
                ]
            start_q, start_kv = end_q, end_kv
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
        forward_batch: ForwardBatch,
        scaling=None,
        enable_gqa=False,
        causal=False,
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
            scaling: float or None
            enable_gqa: bool
            causal: bool

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
            end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            # get key and value from cache. per_req_tokens contains the kv cache
            # index for each token in the sequence.
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                # scaled_dot_product_attention() expects query, key, and value to have the same dtype
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_out = (
                self._scaled_dot_product_attention(
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
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def _run_eager_forward_extend(
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
        forward_batch: ForwardBatch,
        scaling=None,
        enable_gqa=False,
        causal=False,
        custom_mask=None,
    ):
        """Run U1 UG U attention with the same eager math as the reference."""

        query = query.movedim(0, query.dim() - 2)
        scale = 1.0 if scaling is None else scaling

        start_q, start_mask = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            extend_seq_len_q = int(extend_seq_lens[seq_idx].item())
            prefill_seq_len_q = int(extend_prefix_lens[seq_idx].item())
            seq_len_kv = int(seq_lens[seq_idx].item())
            end_q = start_q + extend_seq_len_q

            per_req_query = query[:, start_q:end_q, :]
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if enable_gqa:
                repeat = per_req_query.shape[0] // per_req_key.shape[0]
                per_req_key = per_req_key.repeat_interleave(repeat, dim=0)
                per_req_value = per_req_value.repeat_interleave(repeat, dim=0)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            additive_mask = None
            if custom_mask is not None:
                mask_len = extend_seq_len_q * seq_len_kv
                per_req_custom_mask = custom_mask[start_mask : start_mask + mask_len]
                per_req_custom_mask = per_req_custom_mask.view(
                    extend_seq_len_q,
                    seq_len_kv,
                ).to(device=per_req_query.device, dtype=torch.bool)
                start_mask += mask_len
                if not bool(per_req_custom_mask.all().item()):
                    additive_mask = torch.zeros(
                        per_req_custom_mask.shape,
                        dtype=torch.float32,
                        device=per_req_query.device,
                    ).masked_fill_(~per_req_custom_mask, float("-inf"))
            elif causal:
                query_pos = torch.arange(
                    prefill_seq_len_q,
                    seq_len_kv,
                    device=per_req_query.device,
                )
                key_pos = torch.arange(seq_len_kv, device=per_req_query.device)
                keep = key_pos.unsqueeze(0) <= query_pos.unsqueeze(1)
                additive_mask = torch.zeros(
                    keep.shape,
                    dtype=torch.float32,
                    device=per_req_query.device,
                ).masked_fill_(~keep, float("-inf"))

            per_req_out = self._eager_attention(
                per_req_query,
                per_req_key,
                per_req_value,
                additive_mask,
                scale,
            ).movedim(query.dim() - 2, 0)
            output[start_q:end_q, :, :] = per_req_out
            start_q = end_q

        return output

    def _run_eager_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        forward_batch: ForwardBatch,
        scaling=None,
        enable_gqa=False,
        causal=False,
    ):
        del forward_batch, causal
        query = query.movedim(0, query.dim() - 2)
        scale = 1.0 if scaling is None else scaling

        start_q = 0
        for seq_idx in range(seq_lens.shape[0]):
            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + 1

            per_req_query = query[:, start_q:end_q, :]
            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if enable_gqa:
                repeat = per_req_query.shape[0] // per_req_key.shape[0]
                per_req_key = per_req_key.repeat_interleave(repeat, dim=0)
                per_req_value = per_req_value.repeat_interleave(repeat, dim=0)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_out = self._eager_attention(
                per_req_query,
                per_req_key,
                per_req_value,
                None,
                scale,
            ).movedim(query.dim() - 2, 0)
            output[start_q:end_q, :, :] = per_req_out
            start_q = end_q

        return output

    @staticmethod
    def _eager_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        additive_mask: torch.Tensor | None,
        scaling: float,
    ) -> torch.Tensor:
        attn_weights = torch.matmul(query, key.transpose(1, 2)) * scaling
        if additive_mask is not None:
            attn_weights = attn_weights + additive_mask
        attn_weights = torch.nn.functional.softmax(
            attn_weights,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)
        return torch.matmul(attn_weights, value)

    @staticmethod
    def _should_use_u1_reference_eager(forward_batch: ForwardBatch) -> bool:
        metadata = getattr(forward_batch, "session_forward_metadata", None)
        if not metadata:
            return False
        for item in metadata:
            if not isinstance(item, dict):
                continue
            adapter_metadata = item.get("adapter_metadata") or {}
            if isinstance(adapter_metadata, dict) and "u1" in adapter_metadata:
                return True
        return False

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        causal = True
        if (
            layer.is_cross_attention
            or layer.attn_type == AttentionType.ENCODER_ONLY
            or getattr(forward_batch, "cross_attention_custom_mask", None) is not None
        ):
            causal = False

        run_extend = (
            self._run_eager_forward_extend
            if self._should_use_u1_reference_eager(forward_batch)
            else self._run_sdpa_forward_extend
        )
        run_extend(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            forward_batch,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=causal,
            custom_mask=getattr(forward_batch, "cross_attention_custom_mask", None),
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

        if layer.is_cross_attention:
            cache_loc = forward_batch.encoder_out_cache_loc
        else:
            cache_loc = forward_batch.out_cache_loc

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(layer, cache_loc, k, v)

        use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

        q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        run_decode = (
            self._run_eager_forward_decode
            if self._should_use_u1_reference_eager(forward_batch)
            else self._run_sdpa_forward_decode
        )
        run_decode(
            q_,
            o_,
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch,
            scaling=layer.scaling,
            enable_gqa=use_gqa,
            causal=False,
        )

        return o

    def support_triton(self):
        return False
