import torch

from sglang.srt.lora.lora import LoraBatchInfo


class BaseLoraBackend:
    """Base class for different Lora backends.
       Each backend has its own implementation of Lora kernels.

    Args:
        name: name of backend
        batch_info: information of current batch for use
    """

    def __init__(self, name: str, batch_info: LoraBatchInfo = None):
        self.name = name
        self.batch_info = batch_info

    def run_sgemm(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Run segment Gemm with current backend.
        The definition of segment Gemm can be referred to https://docs.flashinfer.ai/api/gemm.html.

        Args:
             x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
             weights: a set of lora weights with shape (num_lora, output_dim, input_dim)
        Returns:
             result with shape (s, output_dim)
        """
        pass

    def run_qkv_lora(
        self,
        x: torch.Tensor,
        qkv_lora_a: torch.Tensor,
        q_lora_b: torch.Tensor,
        kv_lora_b: torch.Tensor,
    ) -> torch.Tensor:
        """Run the lora pass for QKV Layer.

        Args:
            x: input matrix with shape (s, input_dim), here s is the sum of all sequence lengths
            qkv_lora_a: lora_a module for qkv, with shape (num_lora, 3 * r, input_dim)
            q_lora_b: lora_b module for q, with shape (1, num_lora, output_dim_q, r)
            kv_lora_b: lora_b module for kv, with shape (2, num_lora, output_dim_kv, r)

        Returns:
            result with shape (s, output_dim_q + 2 * output_dim_kv)
        """
        pass

    def set_batch_info(self, batch_info: LoraBatchInfo):
        self.batch_info = batch_info
