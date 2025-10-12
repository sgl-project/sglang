from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.nsa.nsa_indexer import BaseIndexerMetadata
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.speculative.spec_info import SpecInput


class AttentionBackend(ABC):
    """The base class of attention backends"""

    @abstractmethod
    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        raise NotImplementedError()

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Init the global shared states for cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ):
        """Init the metadata for a forward pass for capturing a cuda graph."""
        raise NotImplementedError()

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        """Init the metadata for a forward pass for replaying a cuda graph."""
        raise NotImplementedError()

    def get_cuda_graph_seq_len_fill_value(self):
        """
        Return the integer used to fill padded sequence lengths in CUDA graph captures (typically 0 or 1).
        
        Returns:
            int: Fill value used for padded sequence lengths (e.g., 0 or 1).
        """
        raise NotImplementedError()

    def get_verify_buffers_to_fill_after_draft(self):
        """
        Identify buffers for verify-attention kernels that must be populated after a draft step.
        
        By default returns a two-element list with no mandated buffers; subclasses may override to provide the actual buffer objects (for example, tree mask and position buffers).
        
        Returns:
            list: Two-element list of buffers to fill after draft; each element is a buffer object or `None`.
        """
        return [None, None]

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info: SpecInput, cuda_graph_bs: Optional[int]
    ):
        """
        Update any backend metadata that depends on tree mask or position buffers after a draft step.
        
        This method should recompute or refresh internal buffers and metadata (for example, any verify-related buffers) that were returned by get_verify_buffers_to_fill_after_draft and which require new values once draft-stage buffers are available.
        
        Parameters:
            spec_info (SpecInput): Specification object providing input shapes, masks, or other per-request metadata required to recompute buffers.
            cuda_graph_bs (Optional[int]): CUDA-graph batch size used when capturing/replaying graphs, or None if not using CUDA graphs.
        
        Raises:
            NotImplementedError: If the backend does not implement this update behavior.
        """
        raise NotImplementedError()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """
        Dispatches the appropriate attention forward path based on the forward batch mode.
        
        If forward_batch.forward_mode is idle, returns an empty tensor with shape (batch_size, layer.tp_q_head_num * layer.v_head_dim).
        If forward_batch.forward_mode is decode, delegates to forward_decode; otherwise delegates to forward_extend.
        
        Returns:
            torch.Tensor: Attention output from the chosen path. For idle mode, an empty tensor with the shape described above.
        """
        if forward_batch.forward_mode.is_idle():
            return q.new_empty(q.shape[0], layer.tp_q_head_num * layer.v_head_dim)
        elif forward_batch.forward_mode.is_decode():
            return self.forward_decode(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )
        else:
            return self.forward_extend(
                q,
                k,
                v,
                layer,
                forward_batch,
                save_kv_cache=save_kv_cache,
                **kwargs,
            )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        """Run a forward for decode."""
        raise NotImplementedError()

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ):
        """Run a forward for extend."""
        raise NotImplementedError()

    def support_triton(self):
        """Check if the current backend supports triton."""
        return True

    def get_indexer_metadata(
        self,
        layer_id: int,
        forward_batch: ForwardBatch,
    ) -> Optional[BaseIndexerMetadata]:
        """Get the indexer metadata. None means don't support indexer."""
        return None