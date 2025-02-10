import torch
from torch import nn

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    split_tensor_along_last_dim,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.lora.backend import BaseLoRABackend


class BaseLayerWithLoRA(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        lora_rank: int,
        scaling: float,
        lora_backend: BaseLoRABackend,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer
        self.lora_rank: int = lora_rank
        self.scaling: float = scaling
        self.set_lora: bool = False
        self.lora_backend: BaseLoRABackend = lora_backend

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_lora_info(self, *args):
        pass


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: VocabParallelEmbedding,
        lora_rank: int,
        scaling: float,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)
        self.weight = base_layer.weight


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: ColumnParallelLinear,
        lora_rank: int,
        scaling: float,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output, "scaling": self.scaling}
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            lora_a_output,
            self.B_buffer[0],
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_scaling_add
            else base_output + lora_output * self.scaling
        )

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in ColumnParallelLinear
        bias = self.base_layer.bias if not self.base_layer.skip_bias_add else None
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_, bias
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_)

        if self.base_layer.gather_output:
            output = tensor_model_parallel_all_gather(output_parallel)
        else:
            output = output_parallel
        output_bias = self.base_layer.bias if self.base_layer.skip_bias_add else None
        return output, output_bias


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: MergedColumnParallelLinear,
        lora_rank: int,
        scaling: float,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_gate_up = A_buffer
        if self.lora_backend.fuse_stacked_lora_b:
            # B_buffer_gate_up: (num_lora, 2 * output_dim, r)
            self.B_buffer_gate_up = torch.cat(
                (B_buffer[0], B_buffer[1]), dim=-2
            ).contiguous()
        else:
            self.B_buffer_gate_up = (B_buffer[0], B_buffer[1])

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output, "scaling": self.scaling}

        lora_output = self.lora_backend.run_gate_up_lora(
            x,
            self.A_buffer_gate_up,
            self.B_buffer_gate_up,
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_scaling_add
            else base_output + lora_output * self.scaling
        )


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def init__(
        self,
        base_layer: QKVParallelLinear,
        lora_rank: int,
        scaling: float,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)

    def set_lora_info(
        self,
        A_buffer_qkv: torch.Tensor,
        B_buffer_q: torch.Tensor,
        B_buffer_kv: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv

        if self.lora_backend.fuse_stacked_lora_b:
            assert (
                B_buffer_q.shape[-1] == B_buffer_kv.shape[-1]
            ), "The lora rank of q and kv should be the same when enabling fusion of qkv lora_b"
            output_dim_q, output_dim_kv = B_buffer_q.shape[-2], B_buffer_kv.shape[-2]

            # B_buffer_qkv: (num_lora, output_dim_q + 2 * output_dim_kv, r)
            self.B_buffer_qkv = torch.cat(
                (B_buffer_q[0], B_buffer_kv[0], B_buffer_kv[1]), dim=-2
            ).contiguous()

            # Offsets of q/k/v in output dimension
            self.output_offset = torch.tensor(
                [
                    0,
                    output_dim_q,
                    output_dim_q + output_dim_kv,
                    output_dim_q + 2 * output_dim_kv,
                ],
                dtype=torch.int32,
                device=B_buffer_q.device,
            )
            # For computing number of launched blocks
            self.max_qkv_out_dim = max(output_dim_q, output_dim_kv)
        else:
            self.B_buffer_qkv = (
                B_buffer_q,
                B_buffer_kv,
            )

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output, "scaling": self.scaling}
        if self.lora_backend.fuse_stacked_lora_b:
            backend_kwargs["output_offset"] = self.output_offset
            backend_kwargs["max_qkv_out_dim"] = self.max_qkv_out_dim

        lora_output = self.lora_backend.run_qkv_lora(
            x,
            self.A_buffer_qkv,
            self.B_buffer_qkv,
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_scaling_add
            else base_output + lora_output * self.scaling
        )


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: RowParallelLinear,
        lora_rank: int,
        scaling: float,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_rank, scaling, lora_backend)

    def set_lora_info(self, A_buffer: torch.Tensor, B_buffer: torch.Tensor):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output, "scaling": self.scaling}
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            lora_a_output,
            self.B_buffer[0],
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_scaling_add
            else base_output + lora_output * self.scaling
        )

    def forward(self, input_: torch.Tensor):
        # duplicate the logic in RowParallelLinear
        if self.base_layer.input_is_parallel:
            input_parallel = input_
        else:
            tp_rank = get_tensor_model_parallel_rank()
            splitted_input = split_tensor_along_last_dim(
                input_, num_partitions=self.base_layer.tp_size
            )
            input_parallel = splitted_input[tp_rank].contiguous()
        output_parallel = self.base_layer.quant_method.apply(
            self.base_layer, input_parallel
        )

        if self.set_lora:
            output_parallel = self.apply_lora(output_parallel, input_parallel)

        if self.base_layer.reduce_results and self.base_layer.tp_size > 1:
            output_ = tensor_model_parallel_all_reduce(output_parallel)
        else:
            output_ = output_parallel

        if not self.base_layer.skip_bias_add:
            output = (
                output_ + self.base_layer.bias
                if self.base_layer.bias is not None
                else output_
            )
            output_bias = None
        else:
            output = output_
            output_bias = self.base_layer.bias
        return output, output_bias


def get_lora_layer(
    layer: nn.Module, lora_rank: int, scaling: int, lora_backend: BaseLoRABackend
) -> BaseLayerWithLoRA:
    supported_layer_types = {
        # the order matters
        VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
        QKVParallelLinear: QKVParallelLinearWithLoRA,
        MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
        ColumnParallelLinear: ColumnParallelLinearWithLoRA,
        RowParallelLinear: RowParallelLinearWithLoRA,
    }
    for src_layer_type, lora_layer_type in supported_layer_types.items():
        if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
            ret = lora_layer_type(layer, lora_rank, scaling, lora_backend)
            return ret
    raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")
