import torch

from sglang.kernels.ops.attention.dsv4.candidate_blocks import (
    candidate_block_mask,
    select_candidate_block_ids,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.dsv4.dsv41_sparse import token_req_indices
from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore


class DeepseekV4CPUBackend(AttentionBackend):
    def __init__(self, model_runner):
        config = model_runner.model_config.hf_text_config
        if config.model_type != "deepseek_v41":
            raise ValueError("CPU dsv4 attention currently requires DeepSeek-V4.1")
        if not model_runner.spec_algorithm.is_none():
            raise ValueError("CPU DeepSeek-V4.1 does not support speculative decoding")
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.window = config.sliding_window
        self.topk = config.index_topk
        self.selected = {}
        self.candidates = None

    def init_forward_metadata(self, forward_batch):
        self.selected = {}
        self.candidates = None

    def get_swa_out_cache_loc(self, forward_batch):
        return self.token_to_kv_pool.translate_loc_from_full_to_swa(
            forward_batch.out_cache_loc
        )

    def store_cache(self, layer_id, swa_k, forward_batch):
        self.token_to_kv_pool.set_swa_key_buffer_radix_fused(
            layer_id, self.get_swa_out_cache_loc(forward_batch), swa_k
        )

    def forward_low_ratio_sources(self, *, layer, x, q_lora, positions, forward_batch):
        req = token_req_indices(forward_batch, num_tokens=x.shape[0])
        if layer.compressor is not None:
            self._compress(layer, x, req, positions, forward_batch)
        if layer.indexer is not None:
            self._index(layer, x, q_lora, req, positions)

    def _compress(self, layer, x, req, pos, batch):
        pool = self.token_to_kv_pool
        ratio = layer.compress_ratio
        kv, score = layer.compressor.project(x)
        if ratio == 2:
            state = pool.get_attention_compress_states(layer.layer_id)
            carried = state.get_state_by_state_loc(
                state.translate_from_req_position_to_state_loc(req, pos - 1)
            )
            adjacent = torch.zeros(pos.numel(), dtype=torch.bool)
            adjacent[1:] = (req[1:] == req[:-1]) & (pos[1:] == pos[:-1] + 1)
            partner_kv = torch.where(adjacent[:, None], kv.roll(1, 0), carried.kv)
            partner_score = torch.where(
                adjacent[:, None], score.roll(1, 0), carried.score
            )
            pooled = layer.compressor.pool_pairs(
                torch.stack([partner_kv, kv], 1),
                torch.stack([partner_score, score], 1),
            )
            keep = torch.ones(pos.numel(), dtype=torch.bool)
            ring = state.ring_size
            if pos.numel() > ring:
                keep[:-ring] = req[:-ring] != req[ring:]
            state.set_state_by_state_loc(
                state.translate_from_req_position_to_state_loc(req[keep], pos[keep]),
                KVAndScore.from_kv_score(kv=kv[keep], score=score[keep]),
            )
            valid = pos % 2 == 1
            pooled = pooled[valid]
        else:
            valid = torch.ones(pos.numel(), dtype=torch.bool)
            pooled = kv
        group_pos = pos[valid] - (ratio - 1)
        slots = batch.out_cache_loc[valid] // ratio
        latent = layer.compressor.finish(pooled)
        freqs = layer.freqs_cis[group_pos]
        if layer.indexer is not None and layer.indexer.owns_k:
            pool.set_index_k_fp4(
                layer.layer_id, slots, layer.indexer.index_keys(latent, freqs)
            )
        pool.set_extra_key_buffer_fused(layer.layer_id, slots, latent, freqs_cis=freqs)

    def _compressed_slots(self, req, length, ratio):
        return (
            self.req_to_token[req, ratio - 1 : length * ratio : ratio].long() // ratio
        )

    def _index(self, layer, x, q_lora, req, pos):
        indexer = layer.indexer
        ratio = layer.compress_ratio
        query = indexer.queries(q_lora, layer.freqs_cis, positions=pos)
        weights = indexer.head_weights(x)
        selected = torch.full((pos.numel(), self.topk), -1, dtype=torch.long)
        published = []
        for row, (request, position) in enumerate(zip(req.tolist(), pos.tolist())):
            length = (position + 1) // ratio
            if not length:
                published.append(torch.empty(0, dtype=torch.long))
                continue
            slots = self._compressed_slots(request, length, ratio)
            keys = self.token_to_kv_pool.get_low_ratio_index_k_dequant(
                layer.layer_id, slots
            )
            scores = indexer.scores(query[row : row + 1], keys, weights[row : row + 1])[
                0
            ]
            if indexer.is_candidate_source:
                published.append(
                    select_candidate_block_ids(
                        scores[None],
                        length,
                        topk_blocks=indexer.candidate_topk_blocks,
                        block_size=indexer.candidate_block_size,
                    )[0]
                )
            elif indexer.uses_candidates:
                if self.candidates is None:
                    raise RuntimeError("V4.1 candidate source has not run")
                allowed = candidate_block_mask(
                    self.candidates[row],
                    width=length,
                    block_size=indexer.candidate_block_size,
                )
                scores = scores.masked_fill(~allowed, -torch.inf)
            count = min(self.topk, length)
            indices = scores.topk(count, sorted=False).indices
            chosen = slots[indices].masked_fill(~scores[indices].isfinite(), -1)
            selected[row, :count] = chosen
        self.selected[ratio] = selected
        if indexer.is_candidate_source:
            self.candidates = published

    def forward(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        *,
        compress_ratio,
        attn_sink,
        save_kv_cache=True,
        **kwargs,
    ):
        if forward_batch.forward_mode.is_idle():
            return torch.empty_like(q)
        if save_kv_cache:
            self.store_cache(layer.layer_id, k, forward_batch)
        pool = self.token_to_kv_pool
        req = token_req_indices(forward_batch, num_tokens=q.shape[0])
        swa_cache = pool.get_swa_key_buffer_radix(layer.layer_id)
        positions = (
            forward_batch.positions[:, None]
            - self.window
            + 1
            + torch.arange(self.window)
        )
        slots = self.req_to_token[req[:, None], positions.clamp_min(0)].long()
        swa_slots = pool.translate_loc_from_full_to_swa(slots).long()
        swa_slots = swa_slots.masked_fill(positions < 0, -1)
        extra_cache = (
            pool.get_extra_key_buffer(layer.layer_id) if compress_ratio else swa_cache
        )
        extra_slots = (
            self.selected[compress_ratio]
            if compress_ratio
            else torch.empty((q.shape[0], 0), dtype=torch.long)
        )
        return torch.ops.sgl_kernel.dsv41_sparse_attention_cpu(
            q.contiguous(),
            swa_cache,
            extra_cache,
            swa_slots,
            extra_slots,
            attn_sink.float().contiguous(),
            layer.scaling,
        )
