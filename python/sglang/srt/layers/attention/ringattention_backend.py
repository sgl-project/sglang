from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from sglang.srt.distributed.communication_op import AttnContextParallelCommunicate
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@dataclass
class RingAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

    # CP Ring Communication
    cp_ring_comm: AttnContextParallelCommunicate = None


class RingAttentionBackend(AttentionBackend):
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""
        metadata = RingAttentionMetadata()

        metadata.cp_ring_comm = AttnContextParallelCommunicate()

        self.forward_metadata = metadata

    # TODO (Support MLA, see flashattention_backend)
    def forward_extend(
        self,
        attention_fn,
        args,
    ):
        # Use precomputed metadata across all layers
        metadata = self.forward_metadata
        causal = args["causal"]
        k = args["k_cache"]
        v = args["v_cache"]

        cp_ring_comm = metadata.cp_ring_comm
        cp_world_size = cp_ring_comm.cp_world_size

        out, lse = None, None

        for step in range(cp_world_size):
            if step + 1 != cp_world_size:
                next_k = cp_ring_comm.send_recv_kvcache(k)
                next_v = cp_ring_comm.send_recv_kvcache(v)
                cp_ring_comm.commit()

            if not causal or step <= cp_ring_comm.cp_rank:
                block_out, block_softmax_lse, *rest = attention_fn(
                    return_softmax_lse=True,
                    *args,
                )
                out, lse = update_out_and_lse(out, lse, block_out, block_softmax_lse)

            if step + 1 != cp_ring_comm.world_size:
                cp_ring_comm.wait()
                k = next_k
                v = next_v

        return out


def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    def _update(current_out, current_lse):
        # new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
        # out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
        # For additional context and discussion, please refer to:
        # https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
        current_out = current_out - torch.nn.functional.sigmoid(
            block_lse - current_lse
        ) * (current_out - block_out)
        current_lse = current_lse - torch.nn.functional.logsigmoid(
            current_lse - block_lse
        )
        return current_out, current_lse

    block_out = block_out.to(torch.float32)
    block_lse = block_lse.unsqueeze(dim=-1)

    if out is None:
        return block_out, block_lse

    out, lse = _update(out, lse)

    return out, lse
