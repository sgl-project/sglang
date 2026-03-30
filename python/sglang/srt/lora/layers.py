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
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.moe.topk import TopKOutput
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.lora.backend.base_backend import BaseLoRABackend
from sglang.srt.lora.utils import LoRABatchInfo, get_lm_head_lora_b_shard_size


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

        # Embedding LoRA with TP > 1 keeps weights fully replicated
        # (unsharded) on every rank.  This works correctly because the
        # base VocabParallelEmbedding all-reduces its output before the
        # LoRA delta is added, but it means each rank holds the full
        # LoRA A (rank, vocab_size) and LoRA B (embed_dim, rank) tensors,
        # which may cause OOM on large vocabularies or high LoRA ranks.
        #
        # input_scattered mode (DeepSeek-v2 MLA) skips the base
        # all-reduce, making the unsharded LoRA approach mathematically
        # incorrect — a sharded LoRA kernel would be needed.
        if hasattr(base_layer, "tp_size") and base_layer.tp_size > 1:
            from sglang.srt.layers.communicator import get_attn_tp_context

            assert (
                not get_attn_tp_context().allow_input_scattered
            ), "VocabParallelEmbeddingWithLoRA with TP > 1 under input_scattered mode (e.g., DeepSeek-v2 MLA with --enable-attn-tp-input-scattered) is not fully supported and may produce incorrect results. Consider disabling input_scattered or removing embed_tokens from LoRA target modules."

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
        # LoRA A weights (rank, vocab_size) are kept unsharded.
        # Each rank does a full embedding lookup; the result is complete
        # on every rank and added to the already all-reduced base output.
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # LoRA B weights (embedding_dim, rank) are kept unsharded.
        # The base embedding output is all-reduced (full embedding_dim),
        # so LoRA B must also produce full embedding_dim.
        return B


class ParallelLMHeadWithLoRA(BaseLayerWithLoRA):
    """
    Parallel LM Head layer with LoRA support.

    The LM head computes logits = hidden_states @ (W + B @ A)^T

    With TP > 1, lm_head is column-parallel: each rank holds
    weight (vocab_size/tp_size, hidden_size) and produces a shard
    of logits.  LoRA A is kept unsharded (rank, hidden_size) while
    LoRA B is sliced along the vocab dimension to (vocab_size/tp_size, rank).
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

        tp_size = base_layer.tp_size if hasattr(base_layer, "tp_size") else 1

        # lm_head LoRA keeps A unsharded and shards B along the vocab
        # dimension, matching the column-parallel base output.  This is
        # incompatible with input_scattered mode where the all-reduce is
        # skipped.
        if tp_size > 1:
            from sglang.srt.layers.communicator import get_attn_tp_context

            if get_attn_tp_context().allow_input_scattered:
                raise ValueError(
                    "ParallelLMHeadWithLoRA is not compatible with "
                    "input_scattered mode (e.g., DeepSeek-v2 MLA with "
                    "--enable-attn-tp-input-scattered). Please disable "
                    "input_scattered or remove lm_head from LoRA "
                    "target modules."
                )

            self.shard_vocab_size = get_lm_head_lora_b_shard_size(
                self.vocab_size,
                shard_indices=base_layer.shard_indices,
            )
            self.output_offset = torch.tensor(
                [0, self.shard_vocab_size],
                dtype=torch.int32,
                device=next(base_layer.parameters()).device,
            )
        else:
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

    def _get_lm_head_batch_info(self, num_tokens: int):
        """Resolve and validate the active lm_head batch_info.

        When the logits processor calls lm_head in multiple passes
        (chunked logprobs), _lm_head_pass_idx selects a precomputed
        per-pass batch_info.  Otherwise the full-pruned batch_info is used.

        Returns None when no lm_head pruning applies (decode, no LoRA, etc.).
        """
        pass_idx = self.lora_backend._lm_head_pass_idx
        if (
            pass_idx is not None
            and self.lora_backend.lm_head_pass_batch_infos is not None
        ):
            batch_info = self.lora_backend.lm_head_pass_batch_infos[pass_idx]
        else:
            batch_info = self.lora_backend.lm_head_batch_info

        if batch_info is not None:
            if batch_info.use_cuda_graph:
                raise RuntimeError(
                    "lm_head LoRA with pruned batch info is not supported "
                    "under CUDA graph. lm_head pruning should only occur "
                    "during extend, which does not use CUDA graph."
                )
            if num_tokens != batch_info.expected_tokens:
                raise RuntimeError(
                    f"lm_head LoRA input token count mismatch: got "
                    f"{num_tokens} tokens but lm_head_batch_info expects "
                    f"{batch_info.expected_tokens}. This likely means "
                    f"a pruning step in LogitsProcessor._get_pruned_states is "
                    f"not reflected in get_lm_head_pruned_lens()."
                )

        return batch_info

    def apply_lora(
        self,
        base_output: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply LoRA to LM head layer.

        For LM head: output = hidden @ (W + B @ A)^T
                           = hidden @ W^T + hidden @ A^T @ B^T
                           = base_output + (hidden @ A^T) @ B^T
        """
        lm_head_batch_info = self._get_lm_head_batch_info(hidden_states.shape[0])

        # Apply lora_A^T: hidden_states @ A^T
        lora_a_output = self.lora_backend.run_lora_a_sgemm(
            hidden_states,
            self.lm_head_A_buffer,
            pruned_batch_info=lm_head_batch_info,
        )

        # Apply lora_B^T: lora_a_output @ B^T
        lora_output = self.lora_backend.run_lora_b_sgemm(
            x=lora_a_output,
            weights=self.lm_head_B_buffer,
            output_offset=self.output_offset,
            base_output=base_output,
            pruned_batch_info=lm_head_batch_info,
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

    # ------------------------------------------------------------------
    # Multi-pass lm_head support (chunked logprobs)
    # ------------------------------------------------------------------

    def set_lm_head_pass(self, pass_idx: int):
        """Set the active lm_head pass index before a logprobs chunk.

        Called by LogitsProcessor.process_input_logprobs_by_chunk() before
        each chunk's _get_logits call.  _get_lm_head_batch_info() will
        resolve to lm_head_pass_batch_infos[pass_idx].
        """
        self.lora_backend._lm_head_pass_idx = pass_idx

    def reset_lm_head_pass(self):
        """Reset the lm_head pass index after all passes are done."""
        self.lora_backend._lm_head_pass_idx = None

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        # LoRA A weights (rank, hidden_size) are kept unsharded.
        # Each rank receives full hidden_states, so A operates on full input.
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        # lm_head is column-parallel: each rank produces vocab_size/tp_size (shard_vocab_size)
        # logits.  LoRA B (vocab_size, rank) must be sliced along the vocab
        # dimension to match the sharded base output.
        # Uses the base layer's shard_indices for the actual vocab range on
        # this rank, staying consistent with base model weight sharding.
        tp_size = self.base_layer.tp_size if hasattr(self.base_layer, "tp_size") else 1
        if tp_size <= 1:
            return B
        start_idx = self.base_layer.shard_indices.org_vocab_start_index
        end_idx = self.base_layer.shard_indices.org_vocab_end_index
        return B[start_idx:end_idx, :]


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

        should_reduce = (
            self.base_layer.reduce_results
            and self.base_layer.tp_size > 1
            and not skip_all_reduce
        )

        if self.set_lora and should_reduce:
            lora_a_output = self.lora_backend.run_lora_a_sgemm(
                input_parallel, self.A_buffer
            )
            output_ = tensor_model_parallel_all_reduce(output_parallel)
            lora_a_output = tensor_model_parallel_all_reduce(lora_a_output)
            output_ = self.lora_backend.run_lora_b_sgemm(
                x=lora_a_output,
                weights=self.B_buffer,
                output_offset=self.output_offset,
                base_output=output_,
            )
        else:
            if self.set_lora:
                output_parallel = self.apply_lora(output_parallel, input_parallel)
            if should_reduce:
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


class FusedMoEWithLoRA(BaseLayerWithLoRA):
    """
    Wrapper around FusedMoE that integrates LoRA into the MoE computation.

    Design: LoRA deltas are added at specific points in the MoE forward pass:
    1. After gate_up projection, BEFORE activation (halfway through)
    2. After down projection, BEFORE final reduction

    This follows the vLLM/HF approach where LoRA is fused into the computation
    rather than computed independently and added at the end.
    """

    def __init__(
        self,
        base_layer: FusedMoE,
        lora_backend: BaseLoRABackend,
    ):
        # initializes FusedMoE with its own moe_runner for base path
        super().__init__(base_layer, lora_backend)

        self.tp_size = getattr(base_layer, "moe_tp_size", 1)
        self.tp_rank = getattr(base_layer, "moe_tp_rank", 0)
        self.intermediate_size_per_partition = getattr(
            base_layer, "intermediate_size_per_partition", None
        )

        # initialize triton_lora moe runner for batches with lora enabled
        from sglang.srt.layers.moe.moe_runner.runner import MoeRunner
        from sglang.srt.layers.moe.moe_runner.triton import TritonMoeQuantInfo

        self._lora_runner = MoeRunner(
            base_layer.quant_method.runner.runner_backend,
            base_layer.moe_runner_config,
            lora_enabled=True,
        )

        # Pre-compute quant info for efficiency (weights don't change during inference)
        self._quant_info = TritonMoeQuantInfo(
            w13_weight=base_layer.w13_weight,
            w2_weight=base_layer.w2_weight,
            b13=getattr(base_layer, "w13_weight_bias", None),
            b2=getattr(base_layer, "w2_weight_bias", None),
        )

    def set_lora_info(
        self,
        gate_up_lora_a_weights: torch.Tensor,
        gate_up_lora_b_weights: torch.Tensor,
        down_lora_a_weights: torch.Tensor = None,
        down_lora_b_weights: torch.Tensor = None,
    ):
        """Set LoRA weight tensors from memory pool."""
        self.set_lora = True
        self.gate_up_lora_a_weights = gate_up_lora_a_weights
        self.gate_up_lora_b_weights = gate_up_lora_b_weights
        self.down_lora_a_weights = down_lora_a_weights
        self.down_lora_b_weights = down_lora_b_weights

    def _get_lora_info(self):
        """
        Build LoRAInfo for the current batch.

        Returns None if LoRA is not enabled or weights are not set.
        """
        from sglang.srt.lora.lora_moe_runners import LoRAInfo

        # Get LoRA batch info from backend
        batch_info = self.lora_backend.batch_info
        lora_ranks = batch_info.lora_ranks  # [num_loras]

        max_lora_rank = self.down_lora_a_weights.shape[2]

        # Create adapter_enabled tensor for the current batch
        # Only enable LoRA adapters that are actually used in this batch
        # TODO: Jonahbernard: check that this doesn't slow down inference for this batch
        adapter_enabled = torch.zeros(
            len(lora_ranks), dtype=torch.int32, device=lora_ranks.device
        )
        adapter_enabled.index_fill_(0, batch_info.weight_indices.long(), 1)

        return LoRAInfo(
            gate_up_lora_a_weights=self.gate_up_lora_a_weights,
            gate_up_lora_b_weights=self.gate_up_lora_b_weights,
            down_lora_a_weights=self.down_lora_a_weights,
            down_lora_b_weights=self.down_lora_b_weights,
            seg_indptr=batch_info.seg_indptr,
            req_to_lora=batch_info.weight_indices,
            lora_ranks=lora_ranks,
            adapter_enabled=adapter_enabled,
            max_lora_rank=max_lora_rank,
            num_experts=self.base_layer.num_experts,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            hidden_size=getattr(self.base_layer, "hidden_size", 0),
        )

    def forward(self, hidden_states: torch.Tensor, topk_output: TopKOutput, **kwargs):
        """
        Forward pass with integrated LoRA computation.

        LoRA deltas are added at the correct points inside the MoE computation:
        1. After gate_up projection, before activation
        2. After down projection, before final reduction
        """

        # Build LoRA info for this batch
        lora_info = self._get_lora_info()

        # run lora moe_runner
        return self._forward_with_lora(hidden_states, topk_output, lora_info, **kwargs)

    def _forward_with_lora(
        self,
        hidden_states: torch.Tensor,
        topk_output: TopKOutput,
        lora_info,
        **kwargs,
    ):
        """
        Run MoE forward with LoRA integration at the correct points.
        """
        # Get the base layer's dispatch and combine logic
        base_layer = self.base_layer

        # Dispatch tokens (doesn't do much in the LoRA case)
        dispatch_output = base_layer.dispatcher.dispatch(
            hidden_states=hidden_states, topk_output=topk_output
        )

        # Use pre-computed quant info (doesn't change so not sure why we need to pass it in every time)
        quant_info = self._quant_info

        # Run the only lora moe runner (Triton)
        combine_input = self._lora_runner.run(
            dispatch_output, quant_info, lora_info=lora_info
        )

        final_hidden_states = base_layer.dispatcher.combine(combine_input=combine_input)

        return final_hidden_states

    def slice_lora_a_weights(self, A: torch.Tensor, tp_rank: int):
        return A

    def slice_lora_b_weights(self, B: torch.Tensor, tp_rank: int):
        return B

    def slice_moe_lora_a_weights(
        self, A: torch.Tensor, tp_rank: int, target_module: str
    ) -> torch.Tensor:
        """Slice LoRA A weights for MoE with TP.

        Per-expert weight shapes:
          gate_up_proj_moe A: [rank, hidden_size]  — input is full hidden_states, no slice
          down_proj_moe A:    [rank, intermediate_size] — input is sharded intermediate
        """
        if self.tp_size <= 1:
            return A
        if target_module == "down_proj_moe":
            shard_size = self.intermediate_size_per_partition
            start = tp_rank * shard_size
            end = start + shard_size
            return A[:, start:end].contiguous()
        return A

    def slice_moe_lora_b_weights(
        self, B: torch.Tensor, tp_rank: int, target_module: str
    ) -> torch.Tensor:
        """Slice LoRA B weights for MoE with TP.

        Per-expert weight shapes:
          gate_up_proj_moe B: [intermediate_size*2, rank] — output matches sharded base w13
          down_proj_moe B:    [hidden_size, rank] — output is all-reduced, no slice
        """
        if self.tp_size <= 1:
            return B
        if target_module == "gate_up_proj_moe":
            shard_size = self.intermediate_size_per_partition
            start = tp_rank * shard_size
            end = start + shard_size
            full_inter = B.shape[0] // 2
            gate_b = B[start:end, :]
            up_b = B[full_inter + start : full_inter + end, :]
            return torch.cat([gate_b, up_b], dim=0).contiguous()
        return B


def get_lora_layer(
    layer: nn.Module, lora_backend: BaseLoRABackend
) -> BaseLayerWithLoRA:
    supported_layer_types = {
        # the order matters
        FusedMoE: FusedMoEWithLoRA,
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
