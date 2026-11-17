"""Eager attention for the two private DMTP draft runners on Ascend."""

import torch
from torch.nn.functional import scaled_dot_product_attention


def dmtp_attention(query, keys, values, *, scale, causal):
    """Attend over one request's gathered token rows in logical order."""
    query = query.transpose(0, 1).unsqueeze(0)
    groups = query.shape[1] // keys.shape[1]
    keys = keys.transpose(0, 1).unsqueeze(0).repeat_interleave(groups, dim=1)
    values = values.transpose(0, 1).unsqueeze(0).repeat_interleave(groups, dim=1)
    mask = None
    if causal:
        # The ragged native MTP append occupies the final query-sized KV suffix.
        kv_len, q_len = keys.shape[-2], query.shape[-2]
        rows = torch.arange(kv_len - q_len, kv_len, device=query.device)
        cols = torch.arange(kv_len, device=query.device)
        mask = cols[None, :] <= rows[:, None]
    output = scaled_dot_product_attention(
        query, keys, values, attn_mask=mask, scale=scale
    )
    return output.squeeze(0).transpose(0, 1).flatten(1)


class AscendDMTPAttention:
    def __init__(self, backend, *, draft_block):
        self.backend = backend
        self.draft_block = draft_block

    def __call__(self, q, k, v, layer, batch, save_kv_cache=True, **kwargs):
        if self.draft_block:
            # The DFlash span uses exact, device-side prefix lengths. Its CPU
            # mirror is an allocator bound and already includes the block.
            prefix_lens = batch.seq_lens.tolist()
            query_lens = [batch.spec_info.draft_token_num] * batch.batch_size
        else:
            prefix_lens = batch.extend_prefix_lens_cpu
            query_lens = batch.extend_seq_lens_cpu

        pool = self.backend.token_to_kv_pool
        if save_kv_cache:
            pool.set_kv_buffer(layer, batch.out_cache_loc, k, v)
        keys, values = pool.get_kv_buffer(layer.layer_id)
        keys = keys.view(-1, layer.tp_k_head_num, layer.qk_head_dim)
        values = values.view(-1, layer.tp_v_head_num, layer.v_head_dim)
        q = q.reshape(-1, layer.tp_q_head_num, layer.qk_head_dim)
        mapping = self.backend.req_to_token_pool.req_to_token
        outputs = []
        start = 0
        for req, prefix, width in zip(
            batch.req_pool_indices.tolist(), prefix_lens, query_lens, strict=True
        ):
            end = start + width
            if self.draft_block:
                # All eight block rows see the same last 4096 committed rows
                # and each other. The block locations are the live write slots.
                cached = mapping[req, max(0, prefix - 4096) : prefix]
                slots = torch.cat((cached, batch.out_cache_loc[start:end])).long()
            else:
                # Private MTP slots are shifted by one token and may be
                # non-contiguous inside a page. Read the per-token mapping.
                slots = mapping[req, : prefix + width].long()
            outputs.append(
                dmtp_attention(
                    q[start:end],
                    keys.index_select(0, slots),
                    values.index_select(0, slots),
                    scale=layer.scaling,
                    causal=not self.draft_block,
                )
            )
            start = end
        return torch.cat(outputs, dim=0)


def install_dmtp_attention(backend, *, draft_block):
    from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
        AscendAttnBackend,
    )

    full_attention = getattr(backend, "full_attn_backend", backend)
    if not isinstance(full_attention, AscendAttnBackend) or full_attention.use_mla:
        raise ValueError("Ascend DMTP requires the dense Ascend attention backend")
    full_attention.forward_extend = AscendDMTPAttention(
        full_attention, draft_block=draft_block
    )
