from typing import Tuple, Union

import torch

from sglang.srt.lora.utils import LoRABatchInfo


def get_fuse_output_scaling_add_from_name(name: str) -> bool:
    mapping = {
        "triton": True,
        "flashinfer": False,
    }
    return mapping.get(name, False)


def get_fuse_qkv_lora_b_from_name(name: str) -> bool:
    mapping = {
        "triton": True,
        "flashinfer": False,
    }
    return mapping.get(name, False)


class BaseLoRABackend:
    """Base class for different Lora backends.
       Each backend has its own implementation of Lora kernels.

    Args:
        name: name of backend
        batch_info: information of current batch for use
        fuse_output_scaling_add: if set to True, the output buffer for storing result will be passed in when doing lora_b forward,
                                 and the operation of scaling and adding will be fused into kernel
    """

    def __init__(self, name: str, batch_info: LoRABatchInfo = None):
        self.name = name
        self.batch_info = batch_info
        self.fuse_output_scaling_add = get_fuse_output_scaling_add_from_name(name)
        self.fuse_qkv_lora_b = get_fuse_qkv_lora_b_from_name(name)

    def run_lora_a_sgemm(
        self, x: torch.Tensor, weights: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """Run segment Gemm of lora a modules with current backend.
        The definition of segment Gemm can be referred to https://docs.flashinfer.ai/api/gemm.html.

        Args:
             x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
             weights: a set of lora weights with shape (num_lora, r, input_dim), here r is lora rank
                      usually input_dim is much larger than r
        Returns:
             result with shape (s, r)
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
        **kwargs
    ) -> torch.Tensor:
        """Run the lora pass for QKV Layer.

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            qkv_lora_a: lora_a module for qkv, with shape (num_lora, 3 * r, input_dim)
            qkv_lora_b: lora_b module for qkv.
                        If passed in as a tensor, its shape should be (num_lora,output_dim_q + 2 * output_dim_kv, r)
                        If passed in as a tuple of two tensors containing:
                           a lora_b module for q, with shape (1, num_lora, output_dim_q, r)
                           and a combined lora_b module for kv, with shape (2, num_lora, output_dim_kv, r)
        Returns:
            result with shape (s, output_dim_q + 2 * output_dim_kv)
        """
        pass

    def set_batch_info(self, batch_info: LoRABatchInfo):
        self.batch_info = batch_info
