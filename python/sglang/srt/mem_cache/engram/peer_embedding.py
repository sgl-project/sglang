"""Embedding row lookup from pinned host memory or peer GPU shards."""

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.srt.layers.utils.common import set_weight_attrs
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding


@triton.jit
def gather_ple_embedding_kernel(
    weight_ptr,
    ids_ptr,
    output_ptr,
    embedding_dim,
    tp_vocab_start,
    tp_vocab_end,
    is_fp8: tl.constexpr,
    BLOCK_D: tl.constexpr,
    PEER_ROWS: tl.constexpr = 0,
):
    row_id = tl.program_id(0)
    global_idx = tl.load(ids_ptr + row_id)
    in_range = (global_idx >= tp_vocab_start) & (global_idx < tp_vocab_end)
    local_idx = tl.where(in_range, global_idx - tp_vocab_start, 0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < embedding_dim
    if PEER_ROWS:
        row_id = row_id.to(tl.int64)
        local_idx = local_idx.to(tl.int64)
        owner = local_idx // PEER_ROWS
        weight_ptr = tl.load(weight_ptr + owner)
        local_idx = local_idx - owner * PEER_ROWS
    if is_fp8:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.float8e4nv))
    else:
        weight_ptr = weight_ptr.to(tl.int64).to(tl.pointer_type(tl.bfloat16))
    values = tl.load(
        weight_ptr + local_idx * embedding_dim + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    tl.store(
        output_ptr + row_id * embedding_dim + offsets,
        tl.where(in_range, values, 0.0),
        mask=mask,
    )


class PeerVocabParallelEmbedding(VocabParallelEmbedding):
    """Read global row IDs directly from contiguous TP shards in peer HBM.

    The table stores BF16 or FP8 E4M3 rows and emits BF16. Call finish_loading
    on every rank after loading the checkpoint, before the first lookup.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        params_dtype: torch.dtype,
        output_dtype: torch.dtype = torch.bfloat16,
        use_attn_tp_group: bool = False,
    ) -> None:
        import torch.distributed._symmetric_memory as symm_mem

        if (
            params_dtype not in (torch.bfloat16, torch.float8_e4m3fn)
            or output_dtype != torch.bfloat16
        ):
            raise ValueError(
                "Peer lookup requires BF16 or FP8 E4M3 weights and BF16 output"
            )
        with torch.device("meta"):
            super().__init__(
                num_embeddings,
                embedding_dim,
                params_dtype=params_dtype,
                output_dtype=output_dtype,
                use_attn_tp_group=use_attn_tp_group,
            )
        weight = self.weight
        device = torch.device("cuda", torch.cuda.current_device())
        self._shard = symm_mem.empty(weight.shape, dtype=weight.dtype, device=device)
        self.weight = nn.Parameter(self._shard, requires_grad=False)
        set_weight_attrs(self.weight, vars(weight))
        self._block_d = triton.next_power_of_2(self.embedding_dim)

    def finish_loading(self) -> None:
        # Keep the pointer table stable across in-place weight reloads.
        if hasattr(self, "_peer_ptrs"):
            return

        import torch.distributed._symmetric_memory as symm_mem

        # Publish the immutable shards once their checkpoint copies are complete.
        torch.cuda.synchronize(self.weight.device)
        self._peer_handle = symm_mem.rendezvous(
            self._shard, group=self.tp_group.device_group
        )
        self._peer_ptrs = torch.tensor(
            self._peer_handle.buffer_ptrs, dtype=torch.long, device=self.weight.device
        )

    def allocate_output(
        self, shape: tuple[int, ...], device: torch.device
    ) -> torch.Tensor:
        with torch.inference_mode(False):
            return torch.empty(shape, dtype=torch.bfloat16, device=device)

    def gather(
        self, input_ids: torch.Tensor, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        shape = (*input_ids.shape, self.embedding_dim)
        if out is None:
            out = self.allocate_output(shape, input_ids.device)
        elif (
            out.shape != shape
            or out.dtype != torch.bfloat16
            or out.device != input_ids.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "Peer lookup requires a contiguous BF16 output matching the row IDs"
            )

        ids = input_ids.contiguous()
        if ids.numel():
            gather_ple_embedding_kernel[(ids.numel(),)](
                self._peer_ptrs,
                ids,
                out,
                embedding_dim=self.embedding_dim,
                tp_vocab_start=0,
                tp_vocab_end=self.org_vocab_size,
                is_fp8=self.weight.dtype == torch.float8_e4m3fn,
                BLOCK_D=self._block_d,
                PEER_ROWS=self.num_embeddings_per_partition,
            )
        return out

    def reduce(self, output: torch.Tensor) -> torch.Tensor:
        # Peer lookup already returns complete rows on every rank.
        return output

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.gather(input_ids)
