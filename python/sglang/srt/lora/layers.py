from typing import Optional

import torch
import torch.nn.functional as F
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
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
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
        if hasattr(self.base_layer, "weight"):
            self.weight = self.base_layer.weight

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
    Vocab parallel embedding layer with LoRA support (simplified for TP=1, no extra tokens).

    For embedding layers: output = base_embedding(x) + lora_B @ lora_A[x]
    where lora_A[x] is direct embedding lookup from lora_A weights.
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

        self.output_offset = torch.tensor(
            [0, self.embed_dim],
            dtype=torch.int32,
            device=next(base_layer.parameters()).device,
        )

    def set_lora_info(
        self,
        new_embeddings_buffer: Optional[torch.Tensor],  # For extra tokens
        embedding_A_buffer: torch.Tensor,
        embedding_B_buffer: torch.Tensor,
    ):
        """Set LoRA buffers for embedding layer."""
        self.set_lora = True
        self.new_embeddings_buffer = new_embeddings_buffer
        self.embedding_A_buffer = embedding_A_buffer  # (num_loras, rank, vocab_size)
        self.embedding_B_buffer = embedding_B_buffer  # (num_loras, embed_dim, rank)

    def apply_lora(
        self, base_output: torch.Tensor, input_: torch.Tensor, batch_info
    ) -> torch.Tensor:
        """
        Apply LoRA to base embedding output.
        Formula: output = base_output + lora_B @ lora_A_embedding(input_)
        """

        # Efficient embedding lookup for LoRA A (already support extra token embedding process)
        lora_a_output = self.run_lora_a_embedding(input_, batch_info)

        # Apply LoRA B weights using backend
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.embedding_B_buffer,
            output_offset=self.output_offset,
            base_output=base_output,
        )
        return lora_output

    def run_lora_a_embedding(
        self, input_: torch.Tensor, batch_info: LoRABatchInfo
    ) -> torch.Tensor:
        """
        Apply LoRA A weights using efficient embedding lookup with CUDA graph support.
        Maps tokens to their corresponding LoRA adapters internally.
        It also includes added/extra token processing.
        """
        # Efficient embedding lookup for LoRA A (already support extra token embedding process)
        lora_a_output = self.lora_backend.run_lora_a_embedding(
            input_ids=input_,
            weights=self.embedding_A_buffer,
            vocab_size=self.vocab_size,
            extra_embeddings=(
                self.new_embeddings_buffer
                if hasattr(self, "new_embeddings_buffer")
                and self.new_embeddings_buffer is not None
                else None
            ),
        )

        return lora_a_output

    def extra_token_embedding(
        self, input_: torch.Tensor, base_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Need to impl:

        Process extra tokens (tokens >= vocab_size) by looking up their embeddings
        from the new_embeddings_buffer and replacing them in base_output.

        Args:
            input_: (s,) token IDs
            base_output: (s, embed_dim) base embedding output to be modified in-place

        Returns:
            base_output: (s, embed_dim) modified input base_output (tensor[0,0,0,...]) with extra token embeddings
        """
        # return base_output
        raise NotImplementedError(
            "Error in sglang/python/sglang/srt/lora/layers.py - VocabParallelEmbeddingWithLoRA \n"
            "Current SGLang codebase did not support tuned lora with extra/added tokens. \n"
            "[TODO]: \n"
            "1. Refer to this commit: https://github.com/yushengsu-thu/sglang/commit/90415211eee8a28a316de262583d4d33fa615d10#diff-191177438bcc223837963de63c005850371f8c8a860acb153b26744b66ecc623 to complete \n"
            "2. And then you need to modified the en/decoder tokenizer - tokenizer_manager.py to support extra_token_embedding in-place. \n"
        )

    def forward(self, input_: torch.Tensor):
        """
        Forward pass with LoRA support and CUDA graph compatibility.

        Extra tokens (tokens >= vocab_size) are now handled efficiently
        in the backend's run_lora_a_embedding method.
        """
        batch_info = self.lora_backend.batch_info

        # Get base embedding output
        # For tokens >= vocab_size, base_layer will clamp or handle them
        # We mask them to 0 to avoid out-of-bounds access
        added_tokens_mask = input_ > self.vocab_size - 1
        base_output = self.base_layer.forward(input_.masked_fill(added_tokens_mask, 0))

        # [TODO] SGLang did not support extra/added token process; thus, self.extra_token_embedding only return original input_ now
        # Extra tokens - It will replace extra token embedding with self.new_embeddings_buffer's emb (Default is 0)
        if (
            hasattr(self, "new_embeddings_buffer")
            and self.new_embeddings_buffer is not None
        ):
            base_output = self.extra_token_embedding(input_, base_output)

        # Apply LoRA if configured
        if self.set_lora:
            # The backend's run_lora_a_embedding now handles both regular
            # and extra tokens efficiently with CUDA graph support
            base_output = self.apply_lora(base_output, input_, batch_info)

        return base_output

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        # For TP=1, no slicing needed
        # LoRA A weights (rank, vocab_size) are not sliced for embedding
        # For TP>1, Need to modify code in: sglang/python/sglang/srt/lora/mem_pool.py
        # return A
        if tp_rank > 1:
            raise NotImplementedError(
                f"VocabParallelEmbeddingWithLoRA does not support tensor parallelism > 1. "
                f"Got tp_size={tp_rank}"
            )

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # For TP=1, no slicing needed
        # LoRA B weights (embedding_dim, rank) would be sliced along embedding dimension for TP>1
        # For TP>1, Need to modify code in: sglang/python/sglang/srt/lora/mem_pool.py
        # return B
        if tp_rank > 1:
            raise NotImplementedError(
                f"VocabParallelEmbeddingWithLoRA does not support tensor parallelism > 1. "
                f"Got tp_size={tp_rank}"
            )


class ParallelLMHeadWithLoRA(BaseLayerWithLoRA):
    """
    Parallel LM Head layer with LoRA support (simplified for TP=1).

    The LM head computes logits = hidden_states @ (W + B @ A)^T
    """

    def __init__(
        self,
        base_layer: ParallelLMHead,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        self.weight = base_layer.weight
        self.embed_dim = base_layer.embedding_dim
        self.vocab_size = base_layer.org_vocab_size
        self.output_offset = torch.tensor(
            [0, self.vocab_size],
            dtype=torch.int32,
            device=next(base_layer.parameters()).device,
        )

    def set_lora_info(
        self,
        lm_head_A_buffer: torch.Tensor,
        lm_head_B_buffer: torch.Tensor,
    ):
        """Set LoRA buffers for LM head layer."""
        self.set_lora = True
        self.lm_head_A_buffer = lm_head_A_buffer  # (num_loras, rank, hidden_dim)
        self.lm_head_B_buffer = lm_head_B_buffer  # (num_loras, vocab_size, rank)

    def apply_lora(
        self, base_output: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply LoRA to LM head layer.

        For LM head: output = hidden @ (W + B @ A)^T
                           = hidden @ W^T + hidden @ A^T @ B^T
                           = base_output + (hidden @ A^T) @ B^T
        """
        # Apply lora_A^T: hidden_states @ A^T
        lora_a_output = self.lora_backend.run_lora_a_sgemm(
            hidden_states, self.lm_head_A_buffer
        )

        # Apply lora_B^T: lora_a_output @ B^T
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.lm_head_B_buffer,
            output_offset=self.output_offset,
            base_output=base_output,
        )

        return lora_output

    def forward(self, hidden_states: torch.Tensor):
        # Apply base linear transformation
        base_output = F.linear(
            hidden_states, self.weight, bias=getattr(self.base_layer, "bias", None)
        )

        # Apply LoRA if set
        if self.set_lora:
            base_output = self.apply_lora(base_output, hidden_states)

        return base_output

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        # For TP=1, no slicing needed
        # For TP>1, need to modify code in: sglang/python/sglang/srt/lora/mem_pool.py
        # return A
        if tp_rank > 1:
            raise NotImplementedError(
                f"ParallelLMHeadWithLoRA does not support tensor parallelism > 1. "
                f"Got tp_size={tp_rank}"
            )

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # For TP=1, no slicing needed
        # For TP>1, would slice along vocab dimension, need to modify code in: sglang/python/sglang/srt/lora/mem_pool.py
        # return B
        if tp_rank > 1:
            raise NotImplementedError(
                f"ParallelLMHeadWithLoRA does not support tensor parallelism > 1. "
                f"Got tp_size={tp_rank}"
            )


class ColumnParallelLinearWithLoRA(BaseLayerWithLoRA):
    def __init__(
        self,
        base_layer: ColumnParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        shard_size = self.base_layer.output_partition_sizes[0]
        self.output_offset = torch.tensor(
            [
                0,
                shard_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )

    def set_lora_info(
        self,
        A_buffer: torch.Tensor,
        B_buffer: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer = A_buffer
        self.B_buffer = B_buffer

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.B_buffer,
            output_offset=self.output_offset,
            base_output=base_output,
        )
        return lora_output

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
        self.B_buffer_gate_up = B_buffer

        shard_size = self.base_layer.output_partition_sizes[0]
        self.output_offset = torch.tensor(
            [
                0,
                shard_size,
                2 * shard_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_output = self.lora_backend.run_gate_up_lora(
            x=x,
            gate_up_lora_a=self.A_buffer_gate_up,
            gate_up_lora_b=self.B_buffer_gate_up,
            output_offset=self.output_offset,
            base_output=base_output,
        )
        return lora_output

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # Since the outputs for both gate and up are identical, we use a random one.
        shard_size = self.base_layer.output_partition_sizes[0]
        gate_size = self.base_layer.output_sizes[0]
        start_idx = tp_rank * shard_size
        end_idx = (tp_rank + 1) * shard_size
        return torch.concat(
            (
                B[start_idx:end_idx, :],
                B[gate_size + start_idx : gate_size + end_idx],
            ),
            dim=0,
        )


class QKVParallelLinearWithLoRA(ColumnParallelLinearWithLoRA):
    def __init__(
        self,
        base_layer: QKVParallelLinear,
        lora_backend: BaseLoRABackend,
    ) -> None:
        super().__init__(base_layer, lora_backend)
        q_proj_shard_size = self.base_layer.q_proj_shard_size
        kv_proj_shard_size = self.base_layer.kv_proj_shard_size
        self.output_offset = torch.tensor(
            [
                0,
                q_proj_shard_size,
                q_proj_shard_size + kv_proj_shard_size,
                q_proj_shard_size + 2 * kv_proj_shard_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )
        self.output_offset_cpu = self.output_offset.cpu()

        # For computing number of launched blocks
        self.max_qkv_out_dim = max(q_proj_shard_size, kv_proj_shard_size)

    def set_lora_info(
        self,
        A_buffer_qkv: torch.Tensor,
        B_buffer_qkv: torch.Tensor,
    ):
        self.set_lora = True
        self.A_buffer_qkv = A_buffer_qkv
        self.B_buffer_qkv = B_buffer_qkv

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_output = self.lora_backend.run_qkv_lora(
            x=x,
            qkv_lora_a=self.A_buffer_qkv,
            qkv_lora_b=self.B_buffer_qkv,
            base_output=base_output,
            output_offset=self.output_offset,
            output_offset_cpu=self.output_offset_cpu,
            max_qkv_out_dim=self.max_qkv_out_dim,
        )

        return lora_output

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int) -> torch.Tensor:
        base_layer = self.base_layer
        q_proj_shard_size = base_layer.q_proj_shard_size
        kv_proj_shard_size = base_layer.kv_proj_shard_size
        num_kv_head_replicas = base_layer.num_kv_head_replicas

        q_start_idx = q_proj_shard_size * tp_rank
        q_end_idx = q_start_idx + q_proj_shard_size

        kv_shard_id = tp_rank // num_kv_head_replicas
        kv_start_idx = kv_proj_shard_size * kv_shard_id
        kv_end_idx = kv_start_idx + kv_proj_shard_size

        q_size, k_size, _ = base_layer.output_sizes
        B_q_shard = B[q_start_idx:q_end_idx, :]
        B_k_shard = B[q_size + kv_start_idx : q_size + kv_end_idx, :]
        B_v_shard = B[q_size + k_size + kv_start_idx : q_size + k_size + kv_end_idx, :]

        return torch.concat(
            (
                B_q_shard,
                B_k_shard,
                B_v_shard,
            ),
            dim=0,
        )


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
        output_size = self.base_layer.output_size
        self.output_offset = torch.tensor(
            [
                0,
                output_size,
            ],
            dtype=torch.int32,
            device=next(self.base_layer.parameters()).device,
        )

    def apply_lora(self, base_output: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        lora_a_output = self.lora_backend.run_lora_a_sgemm(x, self.A_buffer)
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.B_buffer,
            output_offset=self.output_offset,
            base_output=base_output,
        )
        return lora_output

    def forward(self, input_: torch.Tensor, skip_all_reduce=False):
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

        if (
            self.base_layer.reduce_results
            and self.base_layer.tp_size > 1
            and not skip_all_reduce
        ):
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
        ParallelLMHead: ParallelLMHeadWithLoRA,
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
