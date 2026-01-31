from bisect import bisect_right
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, List, Optional, Set, Tuple

import torch

from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.hf_transformers_utils import AutoConfig


@dataclass
class LoRABatchInfo:
    # The forward mode is using CUDA Graph.
    use_cuda_graph: bool

    # Batch size
    bs: int

    # Number of segments. For triton backend, it is equal to batch size.
    num_segments: int

    # Indice pointers of each segment in shape (num_segments + 1, )
    seg_indptr: torch.Tensor

    # The index of lora adapter used by each segment, in shape (num_segments,)
    weight_indices: torch.Tensor

    # ranks of each lora adapter, in shape (lora_num,)
    lora_ranks: torch.Tensor

    # scaling of each lora adapter, in shape (lora_num,)
    scalings: torch.Tensor

    # Maximum segment length of current batch
    max_len: Optional[int]

    # Lengths of each segments in shape (num_segments,)
    seg_lens: Optional[torch.Tensor]

    # The logical (re)ordering of input rows (tokens), in shape (num_tokens,)
    permutation: Optional[torch.Tensor]

    # Total number of tokens in the batch (sum of all segment lengths)
    total_tokens: Optional[int] = None

    # Per-sequence adapter indices on CPU, mapping each sequence to its LoRA adapter index.
    # Used by layers to build pruned batch_info when handling pruned inputs.
    adapter_indices: Optional[List[int]] = None

    # Original sequence lengths on CPU, used to compute token-to-sequence mapping
    # when handling pruned inputs.
    seq_lens_cpu: Optional[List[int]] = None

    def is_pruned_input(self, num_tokens: int) -> bool:
        """
        Check if the input tensor has been pruned from the expected size.

        Args:
            num_tokens: Number of tokens in the input tensor

        Returns:
            True if input token count doesn't match expected total_tokens

        Raises:
            ValueError: If total_tokens is None (should always be set)
        """
        if self.total_tokens is None:
            raise ValueError(
                "total_tokens must be set in LoRABatchInfo before calling is_pruned_input"
            )
        return num_tokens != self.total_tokens


class LoRAType(Enum):
    LORA_A = 0
    LORA_B = 1


def get_hidden_dim(
    module_name: str,
    config: AutoConfig,
    base_model: torch.nn.Module,
    layer_idx: int,
    lora_added_vocab_size: int = 0,
) -> Tuple[int]:
    """
    Given a module_name (might be a stacked name), return the hidden dims of modules' input and output.
    """

    if hasattr(base_model, "get_hidden_dim"):
        return base_model.get_hidden_dim(module_name, layer_idx)
    else:
        """
        WARNING: get_hidden_dim() is not defined,
        which is used to get the hidden dim for different lora modules
        Use the default one, but please check if it is correct for your model.
        Please implement the function in the model class if it is not.
        You can reference this function in llama.py.
        """
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        if module_name == "qkv_proj":
            return config.hidden_size, head_dim * (
                config.num_attention_heads + config.num_key_value_heads * 2
            )
        elif module_name == "o_proj":
            return (
                head_dim * config.num_attention_heads,
                config.hidden_size,
            )
        elif module_name == "gate_up_proj":
            return config.hidden_size, config.intermediate_size * 2
        elif module_name == "down_proj":
            return config.intermediate_size, config.hidden_size
        elif module_name == "embed_tokens":
            # For embedding: input is vocab_size (as embedding lookup), output is hidden_size
            # if contain extra tokens will be added; otherwise is 0.
            return config.vocab_size + lora_added_vocab_size, config.hidden_size
        elif module_name == "lm_head":
            # For lm_head: input is hidden_size, output is vocab_size
            # if contain extra tokens will be added; otherwise is 0.
            return config.hidden_size, config.vocab_size + lora_added_vocab_size
        else:
            raise NotImplementedError(
                "get_hidden_dim not implemented for " + module_name
            )


def get_normalized_target_modules(
    target_modules: Iterable[str],
) -> set[str]:
    """
    Mapping a list of target module name to names of the normalized LoRA weights.
    Handles both base module names (e.g., "gate_proj") and prefixed module names (e.g., "feed_forward.gate_proj").
    """
    params_mapping = {
        "q_proj": "qkv_proj",
        "k_proj": "qkv_proj",
        "v_proj": "qkv_proj",
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
        "embed_tokens": "embed_tokens",
        "vocab_emb": "embed_tokens",
        "embeddings": "embed_tokens",
        "word_embeddings": "embed_tokens",
        "lm_head": "lm_head",
        "output": "lm_head",
    }

    result = set()
    for name in target_modules:
        base_name = name.split(".")[-1]
        normalized_name = params_mapping.get(base_name, base_name)
        result.add(normalized_name)
    return result


def get_stacked_multiply(module_name: str) -> int:
    """
    Mapping a lora module name to its magnification at output dimension
    """
    stacked_rank = {
        "qkv_proj": 3,
        "gate_up_proj": 2,
    }
    return stacked_rank[module_name] if module_name in stacked_rank else 1


def get_target_module_name(full_module_name: str, target_modules: Set[str]) -> str:
    """
    Get the target module name in target_modules that can match full_module_name.

    If there is a target module name in target_modules that can match full_module_name, return this name
    Else raise ValueError.
    """
    for target_module in target_modules:
        if target_module in full_module_name:
            return target_module
    raise ValueError(
        f"Cannot find target module name for {full_module_name} in {target_modules}"
    )


EMBEDDING_NAMES = ["embed_tokens", "lm_head"]
ROW_PARALLELISM_LINEAR_LORA_NAMES = ["o_proj", "down_proj"]

# Minimum chunk size for chunked SGMV backend
MIN_CHUNK_SIZE = 16


def get_chunk_size_for_tokens(num_tokens: int, max_chunk_size: int) -> int:
    """
    Determine the chunk size based on token count.

    This function heuristically selects a chunk size for the chunked SGMV
    backend based on the number of tokens in the batch. Larger batches use
    larger chunks for better parallelism.

    Args:
        num_tokens: Number of tokens in the batch
        max_chunk_size: Maximum allowed chunk size

    Returns:
        The determined chunk size, clamped to [MIN_CHUNK_SIZE, max_chunk_size]
    """
    if max_chunk_size <= MIN_CHUNK_SIZE:
        return MIN_CHUNK_SIZE

    if num_tokens >= 256:
        chunk_size = 128
    elif num_tokens >= 64:
        chunk_size = 32
    else:  # num_tokens < 64
        chunk_size = 16

    return min(max_chunk_size, chunk_size)


def generate_sequence_lengths(
    forward_batch: ForwardBatch, device: Optional[torch.device] = None
) -> torch.Tensor:

    device = torch.get_default_device() if device is None else device
    with torch.device(device):
        if forward_batch.forward_mode.is_decode():
            seg_lens = torch.ones(forward_batch.batch_size, dtype=torch.int32)
        elif forward_batch.forward_mode.is_target_verify():
            seg_lens = torch.full(
                size=(forward_batch.batch_size,),
                fill_value=forward_batch.spec_info.draft_token_num,
                dtype=torch.int32,
            )
        elif forward_batch.forward_mode.is_extend():
            seg_lens = (
                forward_batch.extend_seq_lens
                if forward_batch.extend_seq_lens.device == device
                else torch.tensor(
                    forward_batch.extend_seq_lens_cpu,
                    dtype=torch.int32,
                )
            )
        else:
            raise ValueError(f"Unsupported forward mode: {forward_batch.forward_mode}")
    return seg_lens


def build_chunked_segments(
    pruning_token_indices: List[int],
    boundaries: List[int],
    adapter_indices: List[int],
    chunk_size: int,
) -> Tuple[List[int], List[int]]:
    """
    Build chunked segments from pruned token indices.

    Groups consecutive tokens that use the same adapter into segments,
    splitting at chunk boundaries when a segment exceeds chunk_size.

    Args:
        pruning_token_indices: Indices of tokens in the pruned input
        boundaries: Cumulative sequence boundaries (from accumulate(seq_lens))
        adapter_indices: LoRA adapter index for each sequence
        chunk_size: Maximum segment length before splitting

    Returns:
        Tuple of (seg_weight_indices, seg_lens) where:
            - seg_weight_indices: List of adapter indices, one per segment
            - seg_lens: List of segment lengths
    """
    seg_weight_indices: List[int] = []
    seg_lens: List[int] = []
    current_adapter = None
    current_count = 0

    for token_idx in pruning_token_indices:
        # Find which sequence this token belongs to using binary search
        seq_idx = bisect_right(boundaries, token_idx)
        adapter = adapter_indices[seq_idx]

        if adapter != current_adapter:
            # Adapter changed, emit previous segment if any
            if current_count > 0:
                seg_weight_indices.append(current_adapter)
                seg_lens.append(current_count)
            current_adapter = adapter
            current_count = 1
        elif current_count == chunk_size:
            # Same adapter but reached chunk limit, emit and start new chunk
            seg_weight_indices.append(current_adapter)
            seg_lens.append(chunk_size)
            current_count = 1
        else:
            # Same adapter, still room in chunk
            current_count += 1

    # Emit final segment
    if current_count > 0:
        seg_weight_indices.append(current_adapter)
        seg_lens.append(current_count)

    return seg_weight_indices, seg_lens
