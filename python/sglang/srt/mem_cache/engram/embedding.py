"""Embedding adapters preserving native checkpoint names and TP partitioning."""

import torch

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
from sglang.srt.mem_cache.engram import create_storage
from sglang.srt.runtime_context import get_memory, get_parallel


class HiCacheVocabParallelEmbedding(VocabParallelEmbedding):
    """Preserve embedding weight loading and TP semantics over byte-row storage."""

    def __init__(self, *args, **kwargs):
        config = get_memory()
        bounds = kwargs.pop("engram_head_bounds", None)
        namespace = kwargs.pop("engram_namespace", None)
        rows = create_storage(
            config.enable_hi_engram,
            config.hi_engram_config,
            head_bounds=bounds,
            namespace=namespace,
        )
        with torch.device("meta" if config.enable_hi_engram == "remote" else "cpu"):
            super().__init__(*args, **kwargs)
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise ValueError("HiCache Engram requires an uncompressed row tensor")
        # This method has no post-load work. Avoid the loader staging the CPU
        # table to CUDA for quantization postprocessing.
        self.quant_method = None
        self._rows = rows
        self._rows_initialized = False
        if config.enable_hi_engram == "remote":
            old_weight = self.weight
            self.weight = torch.nn.Parameter(
                torch.empty(
                    (rows.row_end - rows.row_start, self.embedding_dim),
                    dtype=old_weight.dtype,
                    device="cpu",
                ),
                requires_grad=False,
            )
            self.weight.__dict__.update(old_weight.__dict__)
            self.storage_row_start, self.storage_row_end = rows.row_start, rows.row_end

    def weight_loader(self, param, loaded_weight):
        if self._rows.remote is not None:
            param.data.copy_(
                loaded_weight[self.storage_row_start : self.storage_row_end]
            )
            return
        return super().weight_loader(param, loaded_weight)

    def _embed_local_shard(self, input_):
        if not self._rows_initialized:
            self._rows.initialize(self.weight.detach().view(torch.uint8))
            # Storage owns the rows now; keep only the parameter's dtype/metadata.
            self.weight.data = self.weight.new_empty(0)
            self._rows_initialized = True
        if self._rows.remote is not None:
            # Keep computational TP ownership exactly as before. Only the
            # physical owner of each selected row is resolved by EngramStore.
            mask = (input_ < self.shard_indices.org_vocab_start_index) | (
                input_ >= self.shard_indices.org_vocab_end_index
            )
            ids = input_.masked_fill(mask, -1)
            with use_symmetric_memory(
                get_parallel().tp_group, disabled=not is_allocation_symmetric()
            ):
                out = self._rows.lookup(ids).view(self.weight.dtype)
                return (
                    out.to(self.output_dtype) if self.output_dtype is not None else out
                )
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
        if out is not None and (
            out.shape != (*ids.shape, self.embedding_dim)
            or out.dtype != (self.output_dtype or self.weight.dtype)
            or out.device != ids.device
        ):
            raise ValueError("Invalid Engram prefetch output shape, dtype or device")
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
