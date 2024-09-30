from abc import ABC, abstractmethod

from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class AttentionBackend(ABC):
    """The base class of attention backends"""

    @abstractmethod
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        raise NotImplementedError()

    def init_cuda_graph_state(self, max_bs: int):
        """Init the global shared states for cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_capture_cuda_graph(
        self, bs: int, req_pool_indices, seq_lens
    ):
        """Init the metadata for a forward pass for capturing a cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_replay_cuda_graph(
        self, bs: int, req_pool_indices, seq_lens
    ):
        """Init the metadata for a forward pass for replying a cuda graph."""
        raise NotImplementedError()

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for padded seq lens. Typically, it is 0 or 1."""
        raise NotImplementedError()

    def forward(self, q, k, v, layer: nn.Module, forward_batch: ForwardBatch):
        """Run forward on an attention layer."""
        if forward_batch.forward_mode.is_decode():
            return self.forward_decode(q, k, v, layer, forward_batch)
        else:
            return self.forward_extend(q, k, v, layer, forward_batch)

    def forward_decode(self, q, k, v, layer: nn.Module, forward_batch: ForwardBatch):
        """Run a forward for decode."""
        raise NotImplementedError()

    def forward_extend(self, q, k, v, layer: nn.Module, forward_batch: ForwardBatch):
        """Run a forward for extend."""
        raise NotImplementedError()
