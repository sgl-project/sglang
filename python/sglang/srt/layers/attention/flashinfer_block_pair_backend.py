"""Ordered FlashInfer attention for SDAR refresh sharing.

The caller supplies [completed block, next masked block] per request.
Each layer refreshes the completed block before next-block attention reads it.
The caller owns scratch KV allocation and carries selected token IDs forward.
"""

from copy import copy

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


def bind_owned_cache_view(batch, translator):
    """Bind this view to its own write slots and current read metadata."""
    if translator.is_translating:
        raise ValueError("Qualify paired views with a static KV pool first")
    plan = translator.plan(
        req_pool_indices=batch.req_pool_indices,
        seq_lens=batch.seq_lens,
        seq_lens_cpu=batch.seq_lens_cpu,
        write_virtual=None,
        write_slots=batch.out_cache_loc,
    )
    plan.bind(batch, translator)


def set_pair_token_extent(batch):
    """Keep real-token metadata consistent with an owned, unsharded view."""
    if getattr(batch, "attn_tp_sequence_sharded", False):
        raise ValueError("Refresh pairing requires unsharded token rows")
    count = batch.input_ids.numel()
    batch.global_num_token_non_padded_cpu = count
    for name in ("global_num_token_non_padded", "num_token_non_padded"):
        value = getattr(batch, name, None)
        if value is not None:
            setattr(batch, name, torch.full_like(value, count))
    for name in (
        "original_global_num_tokens_cpu",
        "global_num_tokens_cpu",
        "global_num_tokens_for_logprob_cpu",
        "global_num_tokens_padded_cpu",
    ):
        value = getattr(batch, name, None)
        if value is not None:
            if len(value) != 1:
                raise ValueError("Refresh pairing requires one data-parallel rank")
            setattr(batch, name, [count])
    for name in ("global_num_tokens_gpu", "global_num_tokens_for_logprob_gpu"):
        value = getattr(batch, name, None)
        if value is not None:
            if value.numel() != 1:
                raise ValueError("Refresh pairing requires one data-parallel rank")
            setattr(batch, name, torch.full_like(value, count))


def block_slice(tensor, batch_size, block_size, index):
    """Select one block per request without mixing request rows."""
    if tensor is None:
        raise ValueError("Block-pair attention requires explicit Q, K, and V")
    if index not in (0, 1) or tensor.shape[0] != batch_size * 2 * block_size:
        raise ValueError("Expected two equal blocks per request")
    shape = tensor.shape[1:]
    return (
        tensor.reshape(batch_size, 2, block_size, *shape)[:, index]
        .contiguous()
        .view(batch_size * block_size, *shape)
    )


def pair_part(tensor, batch, block_size, index):
    """Select one block without changing token order."""
    return block_slice(tensor, batch.batch_size, block_size, index)


def join_pair_parts(parts, batch, block_size):
    """Restore the model row layout after the two attention views."""
    return torch.cat(
        [part.view(batch.batch_size, block_size, -1) for part in parts], dim=1
    ).reshape(batch.batch_size * 2 * block_size, -1)


def split_pair_batch(batch, block_size):
    """Create the two attention views of a combined model forward."""
    bs = batch.batch_size
    if not batch.forward_mode.is_dllm_extend():
        raise ValueError("Block pairing requires DLLM_EXTEND")
    if batch.spec_info is not None or batch.encoder_lens is not None:
        raise ValueError("Speculative and cross-attention batches are unsupported")
    if batch.extend_seq_lens_cpu != [2 * block_size] * bs:
        raise ValueError("Each request must contain exactly two complete blocks")
    if any(p < 0 or p % block_size for p in batch.extend_prefix_lens_cpu):
        raise ValueError("The cached prefix must end at a block boundary")
    parts = []
    for index in (0, 1):
        part = copy(batch)
        for name in ("input_ids", "positions", "out_cache_loc"):
            setattr(
                part, name, block_slice(getattr(batch, name), bs, block_size, index)
            )
        if batch.dllm_prompt_mask is not None:
            part.dllm_prompt_mask = block_slice(
                batch.dllm_prompt_mask.reshape(-1), bs, block_size, index
            ).view(bs, block_size)
        part.seq_lens = batch.seq_lens - (1 - index) * block_size
        part.seq_lens_cpu = batch.seq_lens_cpu - (1 - index) * block_size
        part.seq_lens_sum = batch.seq_lens_sum - bs * (1 - index) * block_size
        part.extend_prefix_lens = batch.extend_prefix_lens + index * block_size
        part.extend_prefix_lens_cpu = [
            p + index * block_size for p in batch.extend_prefix_lens_cpu
        ]
        part.extend_seq_lens = torch.full_like(batch.extend_seq_lens, block_size)
        part.extend_seq_lens_cpu = [block_size] * bs
        part.extend_num_tokens = bs * block_size
        part.extend_start_loc = (
            torch.arange(bs, dtype=torch.int32, device=batch.input_ids.device)
            * block_size
        )
        set_pair_token_extent(part)
        parts.append(part)
    return parts


class FlashInferBlockPairBackend(AttentionBackend):
    """Reuse stock FlashInfer attention while sharing selected linear-layer calls."""

    def __init__(self, model_runner):
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
        from sglang.srt.runtime_context import get_parallel

        if get_parallel().tp_size != 1 or get_parallel().pp_size != 1:
            raise ValueError("Refresh sharing requires TP=PP=1")
        if (
            model_runner.dtype != torch.bfloat16
            or model_runner.kv_cache_dtype != torch.bfloat16
        ):
            raise ValueError("Refresh sharing requires BF16 model and KV tensors")
        if model_runner.model_config.quantization is not None:
            raise ValueError("Refresh pairing requires unquantized weights")
        if model_runner.model_config.hf_config.architectures != ["SDARForCausalLM"]:
            raise ValueError("Refresh sharing supports dense SDAR only")
        if model_runner.sliding_window_size is not None or model_runner.page_size != 1:
            raise ValueError(
                "Refresh sharing requires full attention and page size one"
            )
        # Both attention views execute sequentially on the model stream.
        # Each wrapper retains its own integer planning state.
        self.attn_backend_list = [
            FlashInferAttnBackend(model_runner, init_new_workspace=False)
            for _ in range(2)
        ]
        first = self.attn_backend_list[0]
        if first.dllm_config is None or first.dllm_config.algorithm != "LowConfidence":
            raise ValueError("Refresh sharing requires LowConfidence decoding")
        if first.use_paged or first.prefill_uses_dequant_workspace:
            raise ValueError("Refresh sharing requires the stock ragged BF16 KV path")
        self.block_size = first.dllm_config.block_size
        self.req_to_token_pool = first.req_to_token_pool
        self.token_to_kv_pool = first.token_to_kv_pool
        self.kv_index_translator = first.kv_index_translator
        if self.kv_index_translator.is_translating:
            raise ValueError("Refresh sharing requires a static KV pool")
        self.parts = None

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        for backend in self.attn_backend_list:
            backend.init_cuda_graph_state(max_bs, max_bs * self.block_size)

    def init_forward_metadata(self, forward_batch):
        forward_batch.dllm_block_pair_size = self.block_size
        self.parts = split_pair_batch(forward_batch, self.block_size)
        for backend, part in zip(self.attn_backend_list, self.parts):
            bind_owned_cache_view(part, self.kv_index_translator)
            backend.init_forward_metadata(part)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        forward_batch.dllm_block_pair_size = self.block_size
        self.parts = split_pair_batch(forward_batch, self.block_size)
        for backend, part in zip(self.attn_backend_list, self.parts):
            bind_owned_cache_view(part, self.kv_index_translator)
            backend.init_forward_metadata_out_graph(part, in_capture=in_capture)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if self.parts is None:
            raise RuntimeError("Initialize block-pair metadata before the forward")
        block = self.block_size
        outputs = []
        for index, (backend, part) in enumerate(
            zip(self.attn_backend_list, self.parts)
        ):
            # Derive write locations inside capture so replay reads current locations.
            part.out_cache_loc = pair_part(
                forward_batch.out_cache_loc, forward_batch, block, index
            )
            output = backend.forward_extend(
                pair_part(q, forward_batch, block, index),
                pair_part(k, forward_batch, block, index),
                pair_part(v, forward_batch, block, index),
                layer,
                part,
                save_kv_cache=save_kv_cache,
            )
            outputs.append(output)
        # The first attention call writes refreshed KV before the second reads it.
        return join_pair_parts(outputs, forward_batch, block)
