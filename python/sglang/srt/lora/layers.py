import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional

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
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
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

#############################
#########emb lora############
#############################
# class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
#     """
#     Vocab parallel embedding layer with support for LoRA (Low-Rank Adaptation).

#     Note: The current version does not yet implement the LoRA functionality.
#     This class behaves exactly the same as the base VocabParallelEmbedding.
#     Future versions will integrate LoRA functionality to support efficient parameter fine-tuning.
#     """

#     def __init__(
#         self,
#         base_layer: VocabParallelEmbedding,
#         lora_backend: BaseLoRABackend,
#     ) -> None:
#         super().__init__(base_layer, lora_backend)
#         self.weight = base_layer.weight


##### -----
##### -----
##### -----
# class VocabParallelEmbeddingWithLoRA(BaseLayerWithLoRA):
#     """
#     Vocab parallel embedding layer with support for LoRA (Low-Rank Adaptation).
    
#     This layer supports LoRA adapters on embedding layers, including handling
#     of extra tokens added by LoRA adapters. The implementation uses efficient
#     embedding lookup instead of one-hot encoding.
    
#     For embedding layers: output = base_embedding(x) + lora_B @ lora_A[x]
#     where lora_A[x] is direct embedding lookup from lora_A weights.
#     """

#     def __init__(
#         self,
#         base_layer: VocabParallelEmbedding,
#         lora_backend: BaseLoRABackend,
#     ) -> None:
#         super().__init__(base_layer, lora_backend)
#         self.weight = base_layer.weight
#         self.embed_dim = base_layer.embedding_dim
#         self.vocab_size = base_layer.org_vocab_size

#     def set_lora_info(
#         self,
#         new_embeddings_buffer: Optional[torch.Tensor],
#         embedding_A_buffer: torch.Tensor,
#         embedding_B_buffer: torch.Tensor,
#     ):
#         self.set_lora = True
#         self.new_embeddings_buffer = new_embeddings_buffer  # For extra tokens
#         self.embedding_A_buffer = embedding_A_buffer
#         self.embedding_B_buffer = embedding_B_buffer

#     def _get_token_weight_indices(
#         self, input_: torch.Tensor, batch_info
#     ) -> torch.Tensor:
#         """Map each token position to its corresponding LoRA adapter index."""
#         token_weight_indices = torch.zeros(
#             input_.shape[0], dtype=torch.int32, device=input_.device
#         )

#         # current_pos = 0
#         # for i in range(batch_info.bs):
#         #     seg_len = int(batch_info.seg_lens[i])
#         #     weight_idx = int(batch_info.weight_indices[i])
#         #     token_weight_indices[current_pos : current_pos+seg_len] = weight_idx
#         #     current_pos += seg_len

#         # Use cumsum for positions - avoid Python loops
#         seg_lens = batch_info.seg_lens[:batch_info.bs]  # (bs,)
#         cum_lens = torch.cumsum(seg_lens, dim=0)  # cumulative positions
#         start_positions = torch.cat([torch.zeros(1, dtype=cum_lens.dtype, device=cum_lens.device), cum_lens[:-1]])
        
#         # Vectorized assignment using tensor operations - allow enable cuda-graph
#         for i in range(batch_info.bs):
#             start = start_positions[i]
#             end = cum_lens[i]
#             weight_idx = batch_info.weight_indices[i]
#             token_weight_indices[start:end] = weight_idx
        
#         return token_weight_indices

#     def _run_lora_a_embedding(
#         self, input_: torch.Tensor, token_weight_indices: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Apply LoRA A weights using efficient embedding lookup.
#         This avoids creating one-hot vectors.
#         """
#         lora_a_output = torch.zeros(
#             (input_.shape[0], self.embedding_A_buffer.shape[1]),
#             dtype=self.embedding_A_buffer.dtype,
#             device=input_.device,
#         )

#         unique_weight_indices = torch.unique(token_weight_indices)

#         for idx in unique_weight_indices:
#             token_mask = token_weight_indices == idx
#             lora_a_weights = self.embedding_A_buffer[idx]
#             # Use F.embedding for efficient lookup instead of one-hot @ weights
#             # lora_a_weights shape: (rank, vocab_size)
#             # We need (vocab_size, rank) for embedding lookup
#             lora_a_output[token_mask] = F.embedding(
#                 input_[token_mask], lora_a_weights.t()
#             )

#         return lora_a_output

#     def apply_lora(
#         self, base_output: torch.Tensor, input_: torch.Tensor, batch_info
#     ) -> torch.Tensor:
#         """
#         Apply LoRA to base embedding output.
#         Formula: output = base_output + lora_B @ lora_A_embedding(input_)
#         """
#         token_weight_indices = self._get_token_weight_indices(input_, batch_info)

#         # Efficient embedding lookup for LoRA A
#         lora_a_output = self._run_lora_a_embedding(input_, token_weight_indices)

#         # Apply LoRA B weights
#         lora_output = self.lora_backend.run_lora_b_sgemm(
#             x=lora_a_output,
#             weights=self.embedding_B_buffer,
#             base_output=base_output,
#         )
#         return lora_output

#     def _forward(
#         self,
#         input_: torch.Tensor,
#         added_tokens_mask: torch.Tensor,
#         batch_info,
#         base_output: torch.Tensor,
#     ) -> torch.Tensor:
#         """Handle extra tokens that are beyond the base vocabulary."""
#         token_weight_indices = self._get_token_weight_indices(input_, batch_info)
#         added_weight_indices = token_weight_indices[added_tokens_mask]
#         unique_added_weight_indices = torch.unique(added_weight_indices)

#         for idx in unique_added_weight_indices:
#             lora_mask = added_weight_indices == idx
#             added_token_positions = torch.where(added_tokens_mask)[0][lora_mask]
#             # Remap to extra token range
#             x = input_[added_token_positions] - self.vocab_size
#             new_embeddings = F.embedding(x, self.new_embeddings_buffer[idx])
#             base_output[added_token_positions] = new_embeddings

#         return base_output

#     def forward(self, input_: torch.Tensor):
#         batch_info = self.lora_backend.batch_info
        
#         # Mask tokens that are beyond base vocabulary (extra tokens)
#         added_tokens_mask = input_ > self.vocab_size - 1
        
#         # Get base embedding, masking extra tokens temporarily
#         base_output = self.base_layer.forward(input_.masked_fill(added_tokens_mask, 0))

#         # Handle extra tokens
#         if added_tokens_mask.any():
#             base_output = self._forward(
#                 input_, added_tokens_mask, batch_info, base_output
#             )

#         # Apply LoRA if configured
#         if self.set_lora:
#             output = self.apply_lora(base_output, input_, batch_info)
#         else:
#             output = base_output

#         return output

#     def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
#         # LoRA A weights (rank, vocab_size) are not sliced for embedding
#         # because each token needs access to full vocabulary
#         return A

#     def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
#         ## LoRA B weights (embedding_dim, rank) are sliced along embedding dimension
#         # from sglang.srt.distributed import divide
#         # shard_size = divide(self.base_layer.embedding_dim, self.base_layer.tp_size)
#         # start_idx = tp_rank * shard_size
#         # end_idx = (tp_rank + 1) * shard_size
#         # B = B[start_idx:end_idx, :]

#         # TP = 1
#         return B


        
# class ParallelLMHeadWithLoRA(BaseLayerWithLoRA):
#     """
#     Parallel LM Head layer with support for LoRA.
    
#     The LM head computes logits = hidden_states @ (W + B @ A)^T
#     This is different from embedding which uses lookup operations.
    
#     Note: This class is NOT in the official SGLang implementation.
#     You may need to verify if LM head LoRA is needed for your use case.
#     """

#     def __init__(
#         self,
#         base_layer,  # ParallelLMHead
#         lora_backend: BaseLoRABackend,
#     ) -> None:
#         super().__init__(base_layer, lora_backend)
#         self.weight = base_layer.weight
#         self.embed_dim = base_layer.embedding_dim
#         self.vocab_size = base_layer.org_vocab_size

#     def set_lora_info(
#         self,
#         lm_head_A_buffer: torch.Tensor,
#         lm_head_B_buffer: torch.Tensor,
#     ):
#         self.set_lora = True
#         self.lm_head_A_buffer = lm_head_A_buffer
#         self.lm_head_B_buffer = lm_head_B_buffer

#     def apply_lora(
#         self, base_output: torch.Tensor, hidden_states: torch.Tensor
#     ) -> torch.Tensor:
#         """
#         Apply LoRA to LM head layer.
        
#         Args:
#             base_output: Base logits, shape (batch_size, vocab_size)
#             hidden_states: Hidden states, shape (batch_size, hidden_dim)
            
#         Returns:
#             Logits with LoRA applied
#         """
#         # For LM head: output = hidden @ (W + B @ A)^T
#         # = hidden @ W^T + hidden @ A^T @ B^T
#         # = base_output + (hidden @ A^T) @ B^T
        
#         # Apply lora_A^T: hidden_states @ A^T
#         # lm_head_A_buffer shape: (num_loras, rank, hidden_dim)
#         lora_a_output = self.lora_backend.run_lora_a_sgemm(
#             hidden_states, self.lm_head_A_buffer
#         )
        
#         # Apply lora_B^T: lora_a_output @ B^T
#         # lm_head_B_buffer shape: (num_loras, vocab_size, rank)
#         lora_output = self.lora_backend.run_lora_b_sgemm(
#             x=lora_a_output,
#             weights=self.lm_head_B_buffer,
#             base_output=base_output,
#         )
        
#         return lora_output

#     def forward(self, hidden_states: torch.Tensor):
#         # Apply base linear transformation
#         base_output = F.linear(
#             hidden_states, 
#             self.weight, 
#             bias=getattr(self.base_layer, 'bias', None)
#         )
        
#         # Apply LoRA if set
#         if self.set_lora:
#             base_output = self.apply_lora(base_output, hidden_states)
            
#         return base_output

#     def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
#         # LoRA A is not sliced (similar to ColumnParallelLinear)
#         return A

#     def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
#         # LoRA B is sliced along vocab dimension (output dimension)
#         # Similar to ColumnParallelLinear slicing
#         # from sglang.srt.distributed import divide
#         # shard_size = divide(self.vocab_size, self.base_layer.tp_size)
#         # start_idx = tp_rank * shard_size
#         # end_idx = (tp_rank + 1) * shard_size
#         # B = B[start_idx:end_idx, :]
#         # TP=1
#         return B
##### -----
##### -----
##### -----


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

    def set_lora_info(
        self,
        ##############################
        ##########emb lora############
        ##############################
        new_embeddings_buffer: Optional[torch.Tensor],  # For extra tokens 
        ##############################
        ##############################
        ##############################
        embedding_A_buffer: torch.Tensor,
        embedding_B_buffer: torch.Tensor,
    ):
        """Set LoRA buffers for embedding layer."""
        self.set_lora = True
        ##############################
        ##########emb lora############
        ##############################
        self.new_embeddings_buffer = new_embeddings_buffer
        ##############################
        ##############################
        ##############################
        self.embedding_A_buffer = embedding_A_buffer  # (num_loras, rank, vocab_size)
        self.embedding_B_buffer = embedding_B_buffer  # (num_loras, embed_dim, rank)

    def apply_lora(
        self, base_output: torch.Tensor, input_: torch.Tensor, batch_info
    ) -> torch.Tensor:
        """
        Apply LoRA to base embedding output.
        Formula: output = base_output + lora_B @ lora_A_embedding(input_)
        """

        # Efficient embedding lookup for LoRA A (cannot call run_lora_a_sgemm since needing index  lookup)
        lora_a_output = self.run_lora_a_embedding(input_, batch_info)
        
        # Apply LoRA B weights using backend
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.embedding_B_buffer,
            base_output=base_output,
        )
        return lora_output

    def run_lora_a_embedding(
        self, input_: torch.Tensor, batch_info: LoRABatchInfo
    ) -> torch.Tensor:
        """
        Apply LoRA A weights using efficient embedding lookup.
        Maps tokens to their corresponding LoRA adapters internally.
        """
        token_weight_indices = self._get_token_weight_indices(input_, batch_info)
        lora_a_output = self._run_lora_a_embedding(input_, token_weight_indices)

        return lora_a_output 
        
    def _get_token_weight_indices(
        self, input_: torch.Tensor, batch_info: LoRABatchInfo
    ) -> torch.Tensor:
        # (Step1) Get token-to-lora mapping
        token_weight_indices = torch.zeros(
            input_.shape[0], dtype=torch.int32, device=input_.device
        )
        
        current_pos = 0
        for i in range(batch_info.bs):
            seg_len = int(batch_info.seg_lens[i])
            weight_idx = int(batch_info.weight_indices[i])
            token_weight_indices[current_pos : current_pos+seg_len] = weight_idx
            current_pos += seg_len
        
        return token_weight_indices
    
    def _run_lora_a_embedding(
        self, input_: torch.Tensor, token_weight_indices: torch.Tensor
    ) -> torch.Tensor:
        # (Step2) Apply embedding lookup for each LoRA adapter
        lora_a_output = torch.zeros(
            (input_.shape[0], self.embedding_A_buffer.shape[1]),
            dtype=self.embedding_A_buffer.dtype,
            device=input_.device,
        )
        
        unique_weight_indices = torch.unique(token_weight_indices)
        
        for idx in unique_weight_indices:
            token_mask = token_weight_indices == idx
            lora_a_weights = self.embedding_A_buffer[idx]  # (rank, vocab_size)
            lora_a_output[token_mask] = F.embedding(
                input_[token_mask], lora_a_weights.t()
            )

        return lora_a_output

    def forward(self, input_: torch.Tensor):
        ##############################
        ##########emb lora############
        ##############################
        # # Get base embedding output ( do not consider extra tokens)
        # base_output = self.base_layer.forward(input_)
        
        # # Apply LoRA if configured
        # if self.set_lora:
        #     batch_info = self.lora_backend.batch_info
        #     output = self.apply_lora(base_output, input_, batch_info)
        # else:
        #     output = base_output
        
        # return output

        ###############
        ############### consider both non-extra and extra tokens
        ###############
        
        batch_info = self.lora_backend.batch_info
    
        # Handle added tokens (tokens beyond base vocabulary)
        added_tokens_mask = input_ > self.vocab_size - 1
        base_output = self.base_layer.forward(input_.masked_fill(added_tokens_mask, 0))

        # Process extra tokens if they exist
        if added_tokens_mask.any():
            token_weight_indices = self._get_token_weight_indices(input_, batch_info)
            added_weight_indices = token_weight_indices[added_tokens_mask]
            unique_added_weight_indices = torch.unique(added_weight_indices)

            for idx in unique_added_weight_indices:
                lora_mask = added_weight_indices == idx
                added_token_positions = torch.where(added_tokens_mask)[0][lora_mask]
                x = input_[added_token_positions] - self.vocab_size
                new_embeddings = F.embedding(x, self.new_embeddings_buffer[idx])
                base_output[added_token_positions] = new_embeddings

        # Apply LoRA if set
        if self.set_lora:
            output = self.apply_lora(base_output, input_, batch_info)
        else:
            output = base_output

        return output
        ##############################
        ##############################
        ##############################

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        # For TP=1, no slicing needed
        # LoRA A weights (rank, vocab_size) are not sliced for embedding
        # For TP>1, Need to modify code in: sglang/python/sglang/srt/lora/mem_pool.py
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # For TP=1, no slicing needed
        # LoRA B weights (embedding_dim, rank) would be sliced along embedding dimension for TP>1
        # For TP>1, Need to modify code in: sglang/python/sglang/srt/lora/mem_pool.py
        return B


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
            base_output=base_output,
        )
        
        return lora_output

    def forward(self, hidden_states: torch.Tensor):
        # Apply base linear transformation
        base_output = F.linear(
            hidden_states, 
            self.weight, 
            bias=getattr(self.base_layer, 'bias', None)
        )
        
        # Apply LoRA if set
        if self.set_lora:
            base_output = self.apply_lora(base_output, hidden_states)
        
        return base_output

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        # For TP=1, no slicing needed
        # For TP>1, need to modify code in: sglang/python/sglang/srt/lora/mem_pool.py
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # For TP=1, no slicing needed
        # For TP>1, would slice along vocab dimension, eed to modify code in: sglang/python/sglang/srt/lora/mem_pool.py
        return B


##############################
##############################
##############################


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

##############################
##########emb lora############
##############################
# def get_lora_layer(
#     layer: nn.Module, lora_backend: BaseLoRABackend
# ) -> BaseLayerWithLoRA:
#     supported_layer_types = {
#         # the order matters
#         VocabParallelEmbedding: VocabParallelEmbeddingWithLoRA,
#         QKVParallelLinear: QKVParallelLinearWithLoRA,
#         MergedColumnParallelLinear: MergedColumnParallelLinearWithLoRA,
#         ColumnParallelLinear: ColumnParallelLinearWithLoRA,
#         RowParallelLinear: RowParallelLinearWithLoRA,
#     }
#     for src_layer_type, lora_layer_type in supported_layer_types.items():
#         if isinstance(layer, src_layer_type):  # pylint: disable=unidiomatic-typecheck
#             ret = lora_layer_type(layer, lora_backend)
#             return ret
#     raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")



# def get_lora_layer(
#     layer: nn.Module, lora_backend: BaseLoRABackend, lora_extra_vocab_size: int = 0
# ) -> BaseLayerWithLoRA:
    
#     supported_layer_types = {
#         # the order matters - check ParallelLMHead before VocabParallelEmbedding
#         # since ParallelLMHead is a subclass of VocabParallelEmbedding
#         ParallelLMHead: lambda l, b: ParallelLMHeadWithLoRA(l, b, lora_extra_vocab_size),
#         VocabParallelEmbedding: lambda l, b: VocabParallelEmbeddingWithLoRA(l, b, lora_extra_vocab_size),
#         QKVParallelLinear: lambda l, b: QKVParallelLinearWithLoRA(l, b),
#         MergedColumnParallelLinear: lambda l, b: MergedColumnParallelLinearWithLoRA(l, b),
#         ColumnParallelLinear: lambda l, b: ColumnParallelLinearWithLoRA(l, b),
#         RowParallelLinear: lambda l, b: RowParallelLinearWithLoRA(l, b),
#     }
#     for src_layer_type, lora_layer_factory in supported_layer_types.items():
#         if isinstance(layer, src_layer_type):
#             ret = lora_layer_factory(layer, lora_backend)
#             return ret
#     raise Exception(f"No corresponding LoRA layer supported for {type(layer)}.")



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
##############################
##############################
##############################
