from typing import Optional

import torch

from sglang.srt.lora.utils import LoRABatchInfo

from .graph_lora_ops import (
    sgemm_lora_a_embedding_graph_fwd,
    sgemm_lora_a_graph_fwd,
    sgemm_lora_b_graph_fwd,
)
from .lora_ops import sgemm_lora_a_embedding_fwd as sgemm_lora_a_embedding_control_fwd
from .lora_ops import sgemm_lora_a_fwd as sgemm_lora_a_control_fwd
from .lora_ops import sgemm_lora_b_fwd as sgemm_lora_b_control_fwd


def sgemm_lora_a_embedding_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    vocab_size: int,
) -> torch.Tensor:
    output: torch.Tensor
    if batch_info.use_cuda_graph:
        output = sgemm_lora_a_embedding_graph_fwd(
            inputs,
            weights,
            batch_info.weight_indices,
            batch_info.seg_lens,
            batch_info.scalings,
            vocab_size,
        )
    else:
        output = sgemm_lora_a_embedding_control_fwd(
            inputs,
            weights,
            batch_info.weight_indices_cpu,
            batch_info.seg_lens_cpu,
            batch_info.lora_ranks_cpu,
            batch_info.scalings_cpu,
            vocab_size,
        )
    return output


def sgemm_lora_a_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    num_slices: int = 1,
) -> torch.Tensor:
    output: torch.Tensor
    if batch_info.use_cuda_graph:
        output = sgemm_lora_a_graph_fwd(
            inputs,
            weights,
            batch_info.weight_indices,
            batch_info.seg_lens,
            batch_info.scalings,
            num_slices,
        )
    else:
        output = sgemm_lora_a_control_fwd(
            inputs,
            weights,
            batch_info.weight_indices_cpu,
            batch_info.seg_lens_cpu,
            batch_info.lora_ranks_cpu,
            batch_info.scalings_cpu,
            num_slices,
        )
    return output


def sgemm_lora_b_fwd(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    batch_info: LoRABatchInfo,
    slice_offsets: torch.Tensor,
    base_output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    output: torch.Tensor
    if batch_info.use_cuda_graph:
        output = sgemm_lora_b_graph_fwd(
            inputs,
            weights,
            batch_info.weight_indices,
            batch_info.seg_lens,
            slice_offsets,
            base_output,
        )
    else:
        output = sgemm_lora_b_control_fwd(
            inputs,
            weights,
            batch_info.weight_indices_cpu,
            batch_info.seg_lens_cpu,
            batch_info.lora_ranks_cpu,
            slice_offsets,
            base_output,
        )
    return output


__all__ = [
    "sgemm_lora_a_embedding_fwd",
    "sgemm_lora_a_fwd",
    "sgemm_lora_b_fwd",
]
