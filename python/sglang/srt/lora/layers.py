from typing import List, Tuple, Optional

import torch
from torch import nn
import torch.nn.functional as F

from sglang.srt.distributed import (
    divide,
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
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.utils import LoRABatchInfo


class BaseLayerWithLoRA(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        lora_backend: BaseLoRABackend,
    ):
        super().__init__()
        self.base_layer: nn.Module = base_layer
        self.set_lora: bool = False
        self.lora_backend: BaseLoRABackend = lora_backend

    def forward(self, x: torch.Tensor):
        return self.base_layer.forward(x)

    def set_lora_info(self, *args):
        pass

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        pass

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        pass


class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
    """
    Vocab parallel embedding layer with support for LoRA (Low-Rank Adaptation).

    Note: The current version does not yet implement the LoRA functionality.
    This class behaves exactly the same as the base VocabParallelEmbedding.
    Future versions will integrate LoRA functionality to support efficient parameter fine-tuning.
    """

    def __init__(
        self,
        base_layer: VocabParallelEmbedding,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        self.weight = base_layer.weight
        self.embed_dim = base_layer.embedding_dim
        self.vocab_size = base_layer.org_vocab_size

    def set_lora_info(
        self,
        new_embeddings_buffer: Optional[torch.Tensor],
        embedding_A_buffer: torch.Tensor,
        embedding_B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.new_embeddings_buffer = new_embeddings_buffer
        self.embedding_A_buffer = embedding_A_buffer
        self.embedding_B_buffer = embedding_B_buffer

    def apply_lora(self, base_output: torch.Tensor, input_: torch.Tensor, batch_info: LoRABatchInfo) -> torch.Tensor:
        """
        Apply LoRA to base embedding output.
        Formula: output = base_output + lora_a_embedding(input_) @ lora_b_weights
        """
        token_weight_indices = self._get_token_weight_indices(input_, batch_info)
        
        lora_a_output = self._run_lora_a_embedding(input_, token_weight_indices)
        
        lora_b_output = self.lora_backend.run_lora_b_sgemm(
            lora_a_output,
            self.embedding_B_buffer,
            base_output=base_output if self.lora_backend.fuse_output_add else None,
        )
        
        return (
            lora_b_output
            if self.lora_backend.fuse_output_add
            else base_output + lora_b_output
        )

    def _get_token_weight_indices(self, input_: torch.Tensor, batch_info: LoRABatchInfo) -> torch.Tensor:
        """Map each token position to its corresponding LoRA adapter index."""
        token_weight_indices = torch.zeros(input_.shape[0], dtype=torch.int32, device=input_.device)
        
        current_pos = 0
        for i in range(batch_info.bs):
            seg_len = batch_info.seg_lens[i]
            weight_idx = batch_info.weight_indices[i]
            token_weight_indices[current_pos:current_pos + seg_len] = weight_idx
            current_pos += seg_len
            
        return token_weight_indices

    def _run_lora_a_embedding(self, input_: torch.Tensor, token_weight_indices: torch.Tensor) -> torch.Tensor:
        lora_a_output = torch.zeros(
            (input_.shape[0], self.embedding_A_buffer.shape[1]), 
            dtype=self.embedding_A_buffer.dtype, 
            device=input_.device
        )

        unique_weight_indices = torch.unique(token_weight_indices)

        for idx in unique_weight_indices:
            token_mask = token_weight_indices == idx
            lora_a_weights = self.embedding_A_buffer[idx]
            lora_a_output[token_mask] = F.embedding(input_[token_mask], lora_a_weights.t())
            
        return lora_a_output

    def forward(self, input_: torch.Tensor):
        batch_info = self.lora_backend.batch_info
        added_tokens_mask = torch.where(input_ > self.vocab_size - 1, 1, 0)
        base_output = self.base_layer.forward(input_ * (1-added_tokens_mask))

        if added_tokens_mask.sum() > 0:
            base_output = self._forward(input_, added_tokens_mask, batch_info, base_output)
        
        if self.set_lora:
            output = self.apply_lora(base_output, input_, batch_info)
        else:
            output = base_output

        return output

    def _forward(self, input_: torch.Tensor, added_tokens_mask: torch.Tensor, batch_info: LoRABatchInfo, base_output: torch.Tensor) -> torch.Tensor:        
        token_weight_indices = self._get_token_weight_indices(input_, batch_info)
        added_weight_indices = token_weight_indices[added_tokens_mask.bool()]
        unique_added_weight_indices = torch.unique(added_weight_indices)

        for idx in unique_added_weight_indices:
            lora_mask = added_weight_indices == idx
            added_token_positions = torch.where(added_tokens_mask)[0][lora_mask]
            x = input_[added_token_positions] - self.vocab_size
            new_embeddings = F.embedding(
                x,
                self.new_embeddings_buffer[idx]
            )
            base_output[added_token_positions] = new_embeddings
        
        return base_output

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        shard_size = divide(self.base_layer.embedding_dim, self.base_layer.tp_size)
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        B = B[start_idx:end_idx, :]
        return B


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: ColumnParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output}
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            lora_a_output,
            self.B_buffer[0],
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_add
            else base_output + lora_output
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

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        B = B[start_idx:end_idx, :]
        return B


class MergedColumnParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: MergedColumnParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_gate_up = A_buffer
        if self.lora_backend.fuse_stacked_lora_b:
            # B_buffer_gate_up: (num_lora, 2 * output_dim, r)
            if getattr(self, "B_buffer_gate_up", None) is None:
                self.B_buffer_gate_up = torch.empty(
                    (
                        B_buffer[0].shape[0],
                        2 * B_buffer[0].shape[1],
                        B_buffer[0].shape[2],
                    ),
                    dtype=B_buffer[0].dtype,
                    device=B_buffer[0].device,
                )
            self.B_buffer_gate_up[:, : B_buffer[0].shape[1], :].copy_(B_buffer[0])
            self.B_buffer_gate_up[:, B_buffer[0].shape[1] :, :].copy_(B_buffer[1])
        else:
            self.B_buffer_gate_up = (B_buffer[0], B_buffer[1])

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output}

        lora_output = self.lora_backend.run_gate_up_lora(
            x,
            self.A_buffer_gate_up,
            self.B_buffer_gate_up,
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_add
            else base_output + lora_output
        )

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # Since the outputs for both gate and up are identical, we use a random one.
        shard_size = self.base_layer.output_partition_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return B[:, start_idx:end_idx, :]


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: QKVParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)

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
            if getattr(self, "B_buffer_qkv", None) is None:
                self.B_buffer_qkv = torch.empty(
                    (
                        B_buffer_q[0].shape[0],
                        output_dim_q + 2 * output_dim_kv,
                        B_buffer_q[0].shape[2],
                    ),
                    dtype=B_buffer_q[0].dtype,
                    device=B_buffer_q[0].device,
                )
            self.B_buffer_qkv[:, :output_dim_q, :].copy_(B_buffer_q[0])
            self.B_buffer_qkv[:, output_dim_q : output_dim_q + output_dim_kv, :].copy_(
                B_buffer_kv[0]
            )
            self.B_buffer_qkv[:, output_dim_q + output_dim_kv :, :].copy_(
                B_buffer_kv[1]
            )

            # Offsets of q/k/v in output dimension
            if getattr(self, "output_offset", None) is None:
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
        backend_kwargs = {"base_output": base_output}
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
            if self.lora_backend.fuse_output_add
            else base_output + lora_output
        )

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(
        self, B: List[torch.Tensor], tp_rank: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B_q, B_kv = B
        base_layer = self.base_layer
        q_proj_shard_size = base_layer.q_proj_shard_size
        kv_proj_shard_size = base_layer.kv_proj_shard_size
        num_kv_head_replicas = base_layer.num_kv_head_replicas

        q_start_idx = q_proj_shard_size * tp_rank
        q_end_idx = q_start_idx + q_proj_shard_size

        kv_shard_id = tp_rank // num_kv_head_replicas
        kv_start_idx = kv_proj_shard_size * kv_shard_id
        kv_end_idx = kv_start_idx + kv_proj_shard_size

        return B_q[q_start_idx:q_end_idx, :], B_kv[:, kv_start_idx:kv_end_idx, :]


class RowParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: RowParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)

    def set_lora_info(self, A_buffer: torch.Tensor, B_buffer: torch.Tensor):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        backend_kwargs = {"base_output": base_output}
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            lora_a_output,
            self.B_buffer[0],
            **backend_kwargs,
        )
        return (
            lora_output
            if self.lora_backend.fuse_output_add
            else base_output + lora_output
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

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        shard_size = self.base_layer.input_size_per_partition
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        A = A[:, start_idx:end_idx].contiguous()
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        return B


def get_lora_layer(
    layer: nn.Module, lora_backend: BaseLoRABackend
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
            ret = lora_layer_type(layer, lora_backend)
            return ret
    raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")
