from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from torch.nn.functional import scaled_dot_product_attention
from sglang.srt.layers.attention.utils import create_torch_native_kv_indices
from sglang.srt.speculative.eagle_utils import EagleDraftInput, EagleVerifyInput
if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class IntelAMXAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        import sgl_kernel

        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device

        self.num_head = (
            model_runner.model_config.num_attention_heads // model_runner.tp_size
        )

        self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]

        self.decode_attention_fwd = torch.ops.sgl_kernel.decode_attention_cpu
        self.extend_attention_fwd = torch.ops.sgl_kernel.extend_attention_cpu
        max_bs = model_runner.req_to_token_pool.size
        self.req_to_token = model_runner.req_to_token_pool.req_to_token.clone()
        self.kv_indptr = [
            torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            ), # encoder
            torch.zeros(
                (max_bs + 1,), dtype=torch.int32, device=model_runner.device
            ), # decoder
        ]

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""

        bs = forward_batch.batch_size
        attn_logits = torch.zeros(
            (
                bs,
                self.num_head,
                8,  # self.num_kv_splits,
                self.v_head_dim + 1,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        if forward_batch.forward_mode.is_decode_or_idle():
            max_extend_len = None
        else:
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()
        self.forward_metadata = (attn_logits, max_extend_len)

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
        scaling=None,
        enable_gqa=False,
        causal=False,
        kv_indices=None,
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

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            # TODO: this loop process a sequence per iter, this is inefficient.
            # Need optimize the performance later.

            extend_seq_len_q = extend_seq_lens[seq_idx]
            prefill_seq_len_q = extend_prefix_lens[seq_idx]

            seq_len_kv = seq_lens[seq_idx]
            end_q = start_q + extend_seq_len_q
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
            if kv_indices is not None:
                per_req_tokens = kv_indices
            else:
                req_pool_idx = req_pool_indices[seq_idx]
                per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            per_req_out_redudant = (
                scaled_dot_product_attention(
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
            output[start_q:end_q, :, :] = per_req_out_redudant[prefill_seq_len_q:, :, :]
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
        scaling=None,
        enable_gqa=False,
        causal=False,
        kv_indices=None,
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
            if kv_indices is not None:
                per_req_tokens = kv_indices
            else:
                req_pool_idx = req_pool_indices[seq_idx]
                per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)
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
            output[start_q:end_q, :, :] = per_req_out
            start_q, start_kv = end_q, end_kv

        return output

    def update_kv_indices_encode(
        self,
        forward_batch: ForwardBatch,
        is_cross_attn=False,
    ):
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        seq_lens_sum = forward_batch.seq_lens_sum
        encoder_lens = forward_batch.encoder_lens
        spec_info = forward_batch.spec_info
        kv_indptr = self.kv_indptr[0]
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        if is_cross_attn:
            paged_kernel_lens = encoder_lens
            kv_start_idx = torch.zeros_like(encoder_lens)
            paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            kv_start_idx = encoder_lens
            paged_kernel_lens_sum = seq_lens_sum
        bs = len(seq_lens)
        if spec_info is None:
            assert len(seq_lens) == len(req_pool_indices)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_torch_native_kv_indices(
                req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_indices,
                kv_start_idx,
            )
        else:
            assert isinstance(spec_info, EagleDraftInput) or isinstance(
                spec_info, EagleVerifyInput
            )
            kv_indices, kv_indptr, _, _ = (
                spec_info.generate_attn_arg_prefill(
                    req_pool_indices,
                    paged_kernel_lens,
                    paged_kernel_lens_sum,
                    req_to_token,
                )
            )
        for i in range(bs):
            req_pool_index = req_pool_indices[i]
            self.req_to_token[req_pool_index, :kv_indptr[-1]] = kv_indices[:kv_indptr[-1]]
        return kv_indices[:kv_indptr[-1]]


    def update_kv_indices_decode(
        self,
        forward_batch: ForwardBatch,
        is_cross_attn=False,
    ):
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        seq_lens_sum = forward_batch.seq_lens_sum
        encoder_lens = forward_batch.encoder_lens
        spec_info = forward_batch.spec_info
        kv_indptr = self.kv_indptr[1]
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        if is_cross_attn:
            paged_kernel_lens = encoder_lens
            kv_start_idx = torch.zeros_like(encoder_lens)
            seq_lens_sum = encoder_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            kv_start_idx = encoder_lens
        if spec_info is None:
            bs = len(req_pool_indices)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(seq_lens_sum, dtype=torch.int32)
            create_torch_native_kv_indices(
                req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_indices,
                kv_start_idx,
            )
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
            bs = kv_indptr.shape[0] - 1
        for i in range(bs):
            req_pool_index = req_pool_indices[i].item()
            self.req_to_token[req_pool_index, :kv_indptr[-1]] = kv_indices[:kv_indptr[-1]]
        return kv_indices[:kv_indptr[-1]]

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        kv_indices = self.update_kv_indices_encode(forward_batch, layer.is_cross_attention)
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )
        if save_kv_cache:
            if k is not None:
                assert v is not None
                forward_batch.token_to_kv_pool.set_kv_buffer(
                    layer, cache_loc, k, v
                )

        _, max_extend_len = self.forward_metadata
        if k is not None:
            assert v is not None
            self.extend_attention_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k,
                v,
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.extend_seq_lens,
                forward_batch.extend_start_loc,
                max_extend_len,
                layer.scaling,
                layer.logit_cap,
            )
        else:
            use_gqa = layer.tp_q_head_num != layer.tp_k_head_num
            q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

            self._run_sdpa_forward_extend(
                q_,
                o_,
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.extend_prefix_lens,
                forward_batch.extend_seq_lens,
                scaling=layer.scaling,
                enable_gqa=use_gqa,
                causal=not layer.is_cross_attention,
                kv_indices=kv_indices,
            )
        return o

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        kv_indices = self.update_kv_indices_decode(forward_batch, layer.is_cross_attention)
        attn_logits, _ = self.forward_metadata

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
        if k is not None:
            assert v is not None
            self.decode_attention_fwd(
                q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                k,
                v,
                cache_loc,
                attn_logits,
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                layer.scaling,
                layer.logit_cap,
            )
        else:
            if save_kv_cache:
                if k is not None:
                    assert v is not None
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v
                    )

            use_gqa = layer.tp_q_head_num != layer.tp_k_head_num

            q_ = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            o_ = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

            self._run_sdpa_forward_decode(
                q_,
                o_,
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                self.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                scaling=layer.scaling,
                enable_gqa=use_gqa,
                causal=False,
                kv_indices=kv_indices,
            )

        return o

    def support_triton(self):
        return False