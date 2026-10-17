"""Embedding adapters preserving native checkpoint names and TP partitioning."""

import torch
from torch import nn

from sglang.srt.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    attn_tp_all_reduce,
    get_attn_tp_context,
    get_masked_input_and_mask,
    is_allocation_symmetric,
    tensor_model_parallel_all_reduce,
    use_symmetric_memory,
)
from sglang.srt.mem_cache.engram.storage import ByteRows
from sglang.srt.runtime_context import get_parallel


class MooncakeVocabParallelEmbedding(VocabParallelEmbedding):
    def __init__(self, *args, **kwargs):
        with torch.device("cpu"):
            super().__init__(*args, **kwargs)
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise ValueError("Mooncake Engram requires an uncompressed row tensor")
        # This method has no post-load work. Avoid the loader staging the CPU
        # table to CUDA for quantization postprocessing.
        self.quant_method = None
        self._rows = ByteRows()

    def _embed_local_shard(self, input_):
        if self._rows.store is None:
            self._rows.initialize(self.weight.detach().view(torch.uint8))
        if self.tp_size > 1:
            ids, mask = get_masked_input_and_mask(
                input_,
                self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index,
            )
            ids = ids.masked_fill(mask, -1)
        else:
            ids = input_
        with use_symmetric_memory(
            get_parallel().tp_group, disabled=not is_allocation_symmetric()
        ):
            out = self._rows.lookup(ids).view(self.weight.dtype)
            if self.output_dtype is not None:
                out = out.to(self.output_dtype)
        return out

    def allocate_output(self, shape, device):
        with (
            use_symmetric_memory(
                get_parallel().tp_group, disabled=not is_allocation_symmetric()
            ),
            torch.inference_mode(False),
        ):
            return torch.empty(
                shape, dtype=self.output_dtype or self.weight.dtype, device=device
            )

    def gather(self, ids, out=None):
        values = self._embed_local_shard(ids)
        if out is not None:
            out.copy_(values)
            return out
        return values

    def reduce(self, output):
        if (
            output.numel()
            and self.tp_size > 1
            and not get_attn_tp_context().input_scattered
        ):
            if self.use_attn_tp_group:
                return attn_tp_all_reduce(output)
            return tensor_model_parallel_all_reduce(output)
        return output


class MooncakeScaledEmbedding(nn.Embedding):
    def __init__(self, *args, embed_scale=1.0, **kwargs):
        with torch.device("cpu"):
            super().__init__(*args, **kwargs)
        self.embed_scale = embed_scale
        self._rows = ByteRows()

    def forward(self, input_ids):
        if self._rows.store is None:
            self._rows.initialize(self.weight.detach().view(torch.uint8))
        return self._rows.lookup(input_ids).view(self.weight.dtype) * self.embed_scale
