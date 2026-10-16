"""Experimental SDAR refresh/denoise pairing for model-level benchmarks.

The caller supplies [completed block, next masked block] for each request.
Linear layers process both blocks together. Attention processes the completed
block first, then reads its refreshed KV while processing the next block.
The caller owns allocation, selection, and the final standalone refresh.
This backend does not change the production scheduler.
"""

from copy import copy

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend


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


def split_pair_batch(batch, block_size):
    """Create the two attention views of a combined model forward."""
    bs = batch.batch_size
    if not batch.forward_mode.is_dllm_extend():
        raise ValueError("Block pairing requires DLLM_EXTEND")
    if batch.extend_seq_lens_cpu != [2 * block_size] * bs:
        raise ValueError("Each request must contain exactly two complete blocks")
    if any(p < 0 or p % block_size for p in batch.extend_prefix_lens_cpu):
        raise ValueError("The cached prefix must end at a block boundary")
    if batch.spec_info is not None or batch.encoder_lens is not None:
        raise ValueError("Speculative and cross-attention batches are unsupported")
    parts = []
    for index in (0, 1):
        part = copy(batch)
        for name in ("input_ids", "positions", "out_cache_loc"):
            setattr(
                part, name, block_slice(getattr(batch, name), bs, block_size, index)
            )
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
        parts.append(part)
    return parts


class FlashInferBlockPairBackend(AttentionBackend):
    """Reuse stock FlashInfer attention while sharing all linear-layer calls."""

    def __init__(self, model_runner):
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
        from sglang.srt.runtime_context import get_parallel

        if get_parallel().tp_size != 1 or get_parallel().pp_size != 1:
            raise ValueError("Qualify the prototype with one GPU first")
        if (
            model_runner.dtype != torch.bfloat16
            or model_runner.kv_cache_dtype != torch.bfloat16
        ):
            raise ValueError("The prototype requires BF16 model and KV tensors")
        if model_runner.model_config.hf_config.architectures != ["SDARForCausalLM"]:
            raise ValueError("The block-pair prototype supports dense SDAR only")
        if model_runner.sliding_window_size is not None or model_runner.page_size != 1:
            raise ValueError("The prototype requires full attention and page size one")
        self.attn_backend_list = [
            FlashInferAttnBackend(model_runner, init_new_workspace=True)
            for _ in range(2)
        ]
        first = self.attn_backend_list[0]
        if first.dllm_config is None or first.dllm_config.algorithm != "LowConfidence":
            raise ValueError("The prototype requires LowConfidence decoding")
        if first.use_paged or first.prefill_uses_dequant_workspace:
            raise ValueError("The prototype requires the stock ragged BF16 KV path")
        self.block_size = first.dllm_config.block_size
        self.req_to_token_pool = first.req_to_token_pool
        self.token_to_kv_pool = first.token_to_kv_pool
        self.kv_index_translator = first.kv_index_translator
        self.parts = None

    def init_cuda_graph_state(self, max_bs, max_num_tokens):
        for backend in self.attn_backend_list:
            backend.init_cuda_graph_state(max_bs, max_num_tokens // 2)

    def init_forward_metadata(self, forward_batch):
        self.parts = split_pair_batch(forward_batch, self.block_size)
        for backend, part in zip(self.attn_backend_list, self.parts):
            backend.init_forward_metadata(part)

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        self.parts = split_pair_batch(forward_batch, self.block_size)
        for backend, part in zip(self.attn_backend_list, self.parts):
            backend.init_forward_metadata_out_graph(part, in_capture=in_capture)

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True):
        if self.parts is None:
            raise RuntimeError("Initialize block-pair metadata before the forward")
        bs, block = forward_batch.batch_size, self.block_size
        outputs = []
        for index, (backend, part) in enumerate(
            zip(self.attn_backend_list, self.parts)
        ):
            # Derive write locations inside capture so replay reads current locations.
            part.out_cache_loc = block_slice(
                forward_batch.out_cache_loc, bs, block, index
            )
            output = backend.forward_extend(
                block_slice(q, bs, block, index),
                block_slice(k, bs, block, index),
                block_slice(v, bs, block, index),
                layer,
                part,
                save_kv_cache=save_kv_cache,
            )
            outputs.append(output.view(bs, block, -1))
        # The first attention call writes refreshed KV before the second reads it.
        return torch.cat(outputs, dim=1).reshape(bs * 2 * block, -1)
