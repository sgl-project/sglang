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

    # Total number of tokens this batch info expects (host-side int).
    # Used by lm_head LoRA to validate input shape without GPU sync.
    expected_tokens: Optional[int] = None


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


def get_lm_head_pruned_lens(
    forward_batch: ForwardBatch,
) -> Optional[List[int]]:
    """
    Compute per-sequence pruned lengths for lm_head LoRA.

    Returns a list of pruned lengths (one per sequence) if pruning applies,
    or None if lm_head pruning is not applicable for this batch.

    Pruning rules:
    - Extend without logprobs: 1 token per sequence
    - Extend with logprobs: max(extend_len - logprob_start_len, 1) per sequence
    - Decode / target_verify / draft_extend_v2: no pruning

    IMPORTANT: This must stay in sync with LogitsProcessor._get_pruned_states()
    in sglang/srt/layers/logits_processor.py, which determines how many tokens
    per sequence are passed to lm_head. If the pruning conditions or lengths
    there change, this function must be updated to match, otherwise the
    lm_head LoRA will operate on incorrectly shaped inputs.
    """
    lm_head_pruning = (
        forward_batch.forward_mode.is_extend()
        and not forward_batch.forward_mode.is_target_verify()
        and not forward_batch.forward_mode.is_draft_extend_v2()
    )

    if not lm_head_pruning:
        return None

    if forward_batch.return_logprob:
        pruned_lens = []
        for ext_len, start_len in zip(
            forward_batch.extend_seq_lens_cpu,
            forward_batch.extend_logprob_start_lens_cpu,
        ):
            pruned_lens.append(1 if ext_len == start_len else ext_len - start_len)
    else:
        pruned_lens = [1] * forward_batch.batch_size

    return pruned_lens


def merge_and_chunk_segments(
    weight_indices: list[int],
    pruned_lens: List[int],
    chunk_size: int,
) -> Tuple[List[int], List[int]]:
    """
    Merge consecutive same-adapter sequences and chunk at chunk_size boundaries.

    Merges consecutive sequences that use the same adapter into single
    segments, splitting any segment that exceeds chunk_size.

    Args:
        weight_indices: Per-sequence adapter indices.
        pruned_lens: Per-sequence pruned token counts.
        chunk_size: Maximum segment length before splitting.

    Returns:
        (seg_weight_indices, seg_lens): Merged and chunked segments.
    """
    seg_weight_indices: List[int] = []
    seg_lens: List[int] = []
    for wi, pl in zip(weight_indices, pruned_lens):
        if seg_weight_indices and seg_weight_indices[-1] == wi:
            seg_lens[-1] += pl
        else:
            seg_weight_indices.append(wi)
            seg_lens.append(pl)
        # Split the last segment if it exceeds chunk_size
        while seg_lens[-1] > chunk_size:
            remainder = seg_lens[-1] - chunk_size
            seg_lens[-1] = chunk_size
            seg_weight_indices.append(wi)
            seg_lens.append(remainder)

    return seg_weight_indices, seg_lens


def build_lm_head_pass_segments(
    weight_indices: List[int],
    pruned_lens: List[int],
    logprobs_chunk_size: int,
) -> List[Tuple[List[int], List[int]]]:
    """
    Precompute per-pass segment info for lm_head LoRA logprobs processing.

    When LogitsProcessor uses chunked logprobs processing
    (process_input_logprobs_by_chunk), pruned hidden states are split into
    fixed-size passes.  Each pass needs its own segmentation
    (weight_indices, seg_lens) so that lm_head LoRA operates on the
    correct adapter assignments per pass.

    Args:
        weight_indices: Per-sequence adapter indices.
        pruned_lens: Per-sequence pruned token counts.
        logprobs_chunk_size: Fixed pass size used by LogitsProcessor.

    Returns:
        List of (seg_weight_indices, seg_lens) tuples, one per pass.
    """
    # Expand to per-token weight index
    token_wi: List[int] = []
    for wi, pl in zip(weight_indices, pruned_lens):
        token_wi.extend([wi] * pl)
    total = len(token_wi)
    num_passes = (total + logprobs_chunk_size - 1) // logprobs_chunk_size

    result: List[Tuple[List[int], List[int]]] = []
    for i in range(num_passes):
        start = i * logprobs_chunk_size
        end = min((i + 1) * logprobs_chunk_size, total)

        # Run-length encode the pass's adapter indices
        seg_wi: List[int] = []
        seg_lens: List[int] = []
        for t in range(start, end):
            if seg_wi and seg_wi[-1] == token_wi[t]:
                seg_lens[-1] += 1
            else:
                seg_wi.append(token_wi[t])
                seg_lens.append(1)
        result.append((seg_wi, seg_lens))

    return result
