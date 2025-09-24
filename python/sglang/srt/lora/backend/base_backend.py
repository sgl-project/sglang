from typing import Optional, Tuple, Union

import torch

from sglang.srt.lora.utils import LoRABatchInfo
from sglang.srt.model_executor.forward_batch_info import ForwardBatch


class BaseLoRABackend:
    """Base class for different Lora backends.
       Each backend has its own implementation of Lora kernels.

    Args:
        max_loras_per_batch: maximum number of different lora weights
                             that can be applied in a single forward batch.
        device: the device where the backend runs.
    """

    def __init__(self, max_loras_per_batch: int, device: torch.device):
        self.max_loras_per_batch = max_loras_per_batch
        self.device = device

    def run_lora_a_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Run segment Gemm of lora a modules with current backend.
        The definition of segment Gemm can be referred to https://docs.flashinfer.ai/api/gemm.html.

        Args:
             x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
             weights: a set of lora weights with shape (num_lora, c * r, input_dim),
                      here r is lora rank, c is a multiplier for stacked modules (e.g., c=3 for qkv_proj, c=2 for gate_up_proj)
                      usually input_dim is much larger than r
        Returns:
             result with shape (s, c * r)
        """
        pass

    def run_lora_b_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Run segment Gemm of lora b modules with current backend.
        The definition of segment Gemm can be referred to https://docs.flashinfer.ai/api/gemm.html.

        Args:
             x: input matrix with shape (s, r), here s is the sum of all sequence lengths, r is lora rank
             weights: a set of lora weights with shape (num_lora, output_dim, r)
                      usually output_dim is much larger than r
        Returns:
             result with shape (s, output_dim)
        """
        pass

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        qkv_lora_b: Union[torch.Tensor, Tuple[torch.Tensor]],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run the lora pass for QKV Layer.

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            qkv_lora_a: lora_a module for qkv, with shape (num_lora, 3 * r, input_dim)
            qkv_lora_b: lora_b module for qkv.
                        If passed in as a tensor, its shape should be (num_lora,output_dim_q + 2 * output_dim_kv, r)
                        If passed in as a tuple of two tensors, it should contain:
                           a lora_b module for q, with shape (1, num_lora, output_dim_q, r)
                           and a combined lora_b module for kv, with shape (2, num_lora, output_dim_kv, r)
        Returns:
            result with shape (s, output_dim_q + 2 * output_dim_kv)
        """
        pass

    def run_gate_up_lora(
        self,
        x: torch.Tensor,
        gate_up_lora_a: torch.Tensor,
        gate_up_lora_b: Union[torch.Tensor, Tuple[torch.Tensor]],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Run the lora pass for gate_up_proj, usually attached to MergedColumnParallelLayer.

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            gate_up_lora_a: lora_a module for gate_up_proj, with shape (num_lora, 2 * r, input_dim)
            gate_up_lora_b: lora_b module for qkv.
                        If passed in as a tensor, its shape should be (num_lora, 2 * output_dim, r)
                        If passed in as a tuple, it should contain two tensors with shape (num_lora, output_dim, r)
        Returns:
            result with shape (s, 2 * output_dim)
        """
        pass

    def init_cuda_graph_batch_info(
        self,
        cuda_graph_batch_info: LoRABatchInfo,
        max_bs_in_cuda_graph: int,
    ):
        """Initialize the batch info for CUDA Graph mode.

        This method provides a hook for each backend to conduct its own initialization
        logic for CUDA Graph mode.

        Args:
            cuda_graph_batch_info: the LoRABatchInfo object created in LoraManager
            max_bs_in_cuda_graph: maximum batch size for CUDA Graph mode
        """
        pass

    def prepare_lora_batch(
        self,
        forward_batch: ForwardBatch,
        weight_indices: list[int],
        lora_ranks: list[int],
        scalings: list[float],
        batch_info: Optional[LoRABatchInfo] = None,
    ):
        """Prepare the lora weights and batch info for current forward batch.

        This method provides a hook for each backend to conduct its own preparation
        logic for each forward batch.

        Args:
            forward_batch: the ForwardBatch object for current forward pass
            weight_indices: list of indices of lora weights to be applied for current batch
            lora_ranks: list of lora ranks corresponding to weight_indices
            scalings: list of scaling factors corresponding to weight_indices
            batch_info: optional LoRABatchInfo object, if not provided, the backend should use its own
                        internal batch info (e.g., self.cuda_graph_batch_info for CUDA Graph mode)
        """
        pass


def get_backend_from_name(name: str) -> BaseLoRABackend:
    """
    Get corresponding backend class from backend's name
    """
    if name == "triton":
        from sglang.srt.lora.backend.triton_backend import TritonLoRABackend

        return TritonLoRABackend
    elif name == "csgmv":
        from sglang.srt.lora.backend.chunked_backend import ChunkedSgmvLoRABackend

        return ChunkedSgmvLoRABackend
    elif name == "flashinfer":
        raise ValueError(
            "FlashInfer LoRA backend has been deprecated, please use `triton` instead."
        )
    else:
        raise ValueError(f"Invalid backend: {name}")
