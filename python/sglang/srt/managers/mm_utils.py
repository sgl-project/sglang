"""
Multi-modality utils
"""

import copy
import hashlib
from abc import abstractmethod
from collections import defaultdict
from multiprocessing import shared_memory
from typing import Callable, Literal

import numpy as np
import torch
from torch import nn

from sglang.srt.environ import envs
from sglang.srt.layers.multimodal import gpu_tensor_hash
from sglang.srt.managers.schedule_batch import (
    CudaIpcTensorTransportProxy,
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.multimodal.evs import EVSEmbeddingResult
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import flatten_nested_list, is_npu, print_warning_once
from sglang.utils import logger

_is_npu = is_npu()

# TODO(mick): nccl
# cuda_ipc: for intranode tensor sharing
TensorTransportMode = Literal["cuda_ipc", "auto", "default"]


_GPU_FEATURE_BUFFER: torch.Tensor | None = None
_BUFFER_OFFSET = 0

_is_default_tensor_transport = None


def init_feature_buffer(device):
    global _GPU_FEATURE_BUFFER, _BUFFER_OFFSET
    if (
        device == "cpu"
        or envs.SGLANG_MM_BUFFER_SIZE_MB.get() == 0
        or _GPU_FEATURE_BUFFER is not None
    ):
        return
    try:
        size_mb = envs.SGLANG_MM_BUFFER_SIZE_MB.get()
        num_elements = int(size_mb * 1024 * 1024 / 4)
        _GPU_FEATURE_BUFFER = torch.empty(
            num_elements, dtype=torch.float32, device=device
        )
        logger.info(f"Preallocated {size_mb}MB GPU buffer")
    except RuntimeError as e:
        logger.warning(f"Failed to preallocate GPU feature buffer: {e}")
        _GPU_FEATURE_BUFFER = None


def reset_buffer_offset():
    global _BUFFER_OFFSET
    _BUFFER_OFFSET = 0


def is_feature_buffer_initialized():
    return _GPU_FEATURE_BUFFER is not None


def try_add_to_buffer(tensor: torch.Tensor) -> torch.Tensor | None:
    global _BUFFER_OFFSET

    if _GPU_FEATURE_BUFFER is None:
        return tensor

    tensor_size = tensor.numel()

    if _BUFFER_OFFSET + tensor_size <= _GPU_FEATURE_BUFFER.numel():
        buffer_view = _GPU_FEATURE_BUFFER[_BUFFER_OFFSET : _BUFFER_OFFSET + tensor_size]
        buffer_view.copy_(tensor.flatten(), non_blocking=True)
        result = buffer_view.view(tensor.shape)
        _BUFFER_OFFSET += tensor_size
        return result
    else:
        return tensor


class MultiModalityDataPaddingPattern:
    """
    Data tokens (like image tokens) often need special handling during padding
    to maintain model compatibility. This class provides the interface for
    implementing different padding strategies for data tokens
    """

    @abstractmethod
    def pad_input_tokens(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        """
        Pad the input ids sequence containing data tokens, and replace them with pad_values
        """
        pass


class MultiModalityDataPaddingPatternTokenPairs(MultiModalityDataPaddingPattern):
    """In this pattern, data tokens should be enclosed by special token pairs (e.g. <image>...</image>, data_token_pairs)

    The padded value in a region enclosed by a token pair with be the same one, as the MultimodalDataItem's pad value

    This strategy should be applied when data content is marked by start/end token pairs in the input sequence.
    """

    def __init__(
        self,
        data_token_pairs: list[tuple[int, int]] | None,
        data_start_token_ids: list[int] | None = None,
    ) -> None:
        """

        Args:
            data_start_token_ids marks the start of a single multimodal data
            See Minicpmo's slice_start_id for example
        """
        self.data_token_id_pairs = data_token_pairs
        self.data_start_token_ids = data_start_token_ids or [
            s for s, _e in data_token_pairs
        ]

    def pad_input_tokens(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        """
        This function will replace the data-tokens in between with pad_values accordingly
        """
        pad_values = [item.pad_value for item in mm_inputs.mm_items]
        data_token_pairs = self.data_token_id_pairs
        mm_inputs.data_offsets = []
        if data_token_pairs is None:
            data_token_pairs = [mm_inputs.im_start_id, mm_inputs.im_end_id]
        if data_token_pairs is None:
            print_warning_once(
                "No data_token_pairs provided, RadixAttention might be influenced."
            )
            return input_ids
        start_token_ids = {s for s, _e in data_token_pairs}
        end_tokens_ids = {e for _s, e in data_token_pairs}

        padded_ids = []
        last_idx = 0
        data_idx = -1

        start_indices = [i for i, x in enumerate(input_ids) if x in start_token_ids]
        end_indices = [i for i, x in enumerate(input_ids) if x in end_tokens_ids]

        if len(start_indices) != len(end_indices):
            return input_ids

        for start_idx, end_idx in zip(start_indices, end_indices):
            padded_ids.extend(input_ids[last_idx : start_idx + 1])

            if input_ids[start_idx] in self.data_start_token_ids:
                data_idx += 1
                mm_inputs.data_offsets += [start_idx]

            if data_idx >= len(pad_values):
                data_idx = len(pad_values) - 1

            num_tokens = end_idx - start_idx - 1
            pad_value = pad_values[data_idx]
            padded_ids.extend([pad_value] * num_tokens)

            last_idx = end_idx

        padded_ids.extend(input_ids[last_idx:])

        assert len(input_ids) == len(padded_ids), "Length validation fails"
        return padded_ids


class MultiModalityDataPaddingPatternMultimodalTokens(MultiModalityDataPaddingPattern):
    """In this pattern, data tokens should be represented as repetitions of a single token
    e.g. <image><image>....<image>, or <audio><audio>...<audio>
    """

    def pad_input_tokens(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        """
        Replaces multimodal tokens in input_ids with corresponding pad_values from mm_items.
        Each modality (image, audio, video) is handled separately based on its token_id.
        """
        if not input_ids or not mm_inputs.mm_items:
            return input_ids

        input_ids_tensor = torch.as_tensor(input_ids)

        # Check if MM splitting is enabled
        if envs.SGLANG_ENABLE_MM_SPLITTING.get():
            items_by_modality = defaultdict(list)
            for item in mm_inputs.mm_items:
                items_by_modality[item.modality].append(item)

            token_id_map = {
                Modality.IMAGE: mm_inputs.im_token_id,
                Modality.MULTI_IMAGES: mm_inputs.im_token_id,
                Modality.AUDIO: mm_inputs.audio_token_id,
                Modality.VIDEO: mm_inputs.video_token_id,
            }

            for modality, items in items_by_modality.items():
                token_id = token_id_map.get(modality)

                if not items or token_id is None:
                    continue

                for item in items:
                    for offset in item.offsets:
                        input_ids_tensor[offset[0] : offset[1] + 1] = item.pad_value
        else:
            # Create mapping of token_ids to pad_values for each modality
            token_to_pad_mapping = {}
            for item in mm_inputs.mm_items:
                if item.is_image() and mm_inputs.im_token_id is not None:
                    token_to_pad_mapping[mm_inputs.im_token_id] = item.pad_value
                elif item.is_audio() and mm_inputs.audio_token_id is not None:
                    token_to_pad_mapping[mm_inputs.audio_token_id] = item.pad_value
                elif item.is_video() and mm_inputs.video_token_id is not None:
                    token_to_pad_mapping[mm_inputs.video_token_id] = item.pad_value
                else:
                    raise ValueError(
                        f"No multimodal token id provided for {item.modality}"
                    )

            # Apply replacements for all tokens at once
            for token_id, pad_value in token_to_pad_mapping.items():
                input_ids_tensor[input_ids_tensor == token_id] = pad_value

        ret_input_ids = input_ids_tensor.tolist()
        return ret_input_ids


embedding_cache: MultiModalStaticCache | None = None


def init_mm_embedding_cache(max_size: int = 0):
    global embedding_cache
    embedding_cache = MultiModalStaticCache(max_size)


def get_embedding_chunk(
    embedding: torch.Tensor,
    extend_prefix_len: int,
    extend_seq_len: int,
    items_offset: list[tuple[int, int]],
) -> tuple[torch.Tensor, int, int]:
    """
    Extract a chunk of embeddings based on the specified prefix length, sequence length, and offset ranges.

    Args:
        embedding: The full embedding tensor to extract a chunk from
        extend_prefix_len: The starting position (prefix length) for extraction
        extend_seq_len: The number of tokens to extract
        items_offset: List of [start, end] offset ranges for multimodal items in the input sequence

    Returns:
        A tuple containing:
        - The extracted embedding chunk as a tensor
        - The start index used for extraction
        - The end index used for extraction

    Note:
        If there's no overlap between the requested range and the offset ranges,
        an empty tensor is returned with zeros for start and end indices.
    """
    start_index, end_index = 0, 0
    extend_start_index = extend_prefix_len
    extend_end_index = extend_prefix_len + extend_seq_len - 1

    for start, end in items_offset:
        if extend_start_index >= start and extend_start_index <= end:
            start_index += extend_start_index - start
        elif extend_start_index > end:
            start_index += end - start + 1

        if extend_end_index >= start and extend_end_index <= end:
            end_index += extend_end_index - start + 1
        elif extend_end_index > end:
            end_index += end - start + 1
    # some models' embedding is 3-dim, reshape it to 2-dim
    embedding = embedding.reshape(-1, embedding.shape[-1])
    embedding_chunk = embedding[start_index:end_index]
    return embedding_chunk, start_index, end_index


def _get_precomputed_embedding(
    items: list[MultimodalDataItem],
    prefix_length: list[int],
    extend_length: list[int],
    items_offset_list: list[list[tuple[int, int]]],
) -> torch.Tensor | None:
    """
    If all items have precomputed_embeddings, return their concatenation.
    If some but not all have precomputed_embeddings, raise NotImplementedError.
    If none have precomputed_embeddings, return None.
    """
    precomputed_embeddings = []
    for idx, item in enumerate(items):
        if item.precomputed_embeddings is None:
            precomputed_embeddings.append(None)
            continue
        seq_start_idx = prefix_length[idx]
        seq_end_idx = seq_start_idx + extend_length[idx] - 1
        prefix_embedding_length = []
        extend_embedding_length = []
        for mm_start_idx, mm_end_idx in items_offset_list[idx]:
            if mm_start_idx > seq_end_idx:
                break
            if seq_start_idx > mm_start_idx:
                prefix_embedding_length.append(
                    min(seq_start_idx - mm_start_idx, mm_end_idx - mm_start_idx + 1)
                )
            if mm_end_idx >= seq_start_idx:
                extend_embedding_length.append(
                    min(
                        mm_end_idx - seq_start_idx + 1,
                        seq_end_idx - mm_start_idx + 1,
                        mm_end_idx - mm_start_idx + 1,
                        seq_end_idx - seq_start_idx + 1,
                    )
                )
        prefix_embedding_length = int(np.sum(prefix_embedding_length))
        extend_embedding_length = int(np.sum(extend_embedding_length))
        precomputed_embeddings.append(
            item.precomputed_embeddings[
                prefix_embedding_length : prefix_embedding_length
                + extend_embedding_length
            ]
        )

    if any(feature is not None for feature in precomputed_embeddings):
        if not all(feature is not None for feature in precomputed_embeddings):
            raise NotImplementedError(
                "MM inputs where only some items are precomputed."
            )
        result = torch.concat(precomputed_embeddings)
        # some models embedding is 3-dim, reshape it to 2-dim (similar to get_embedding_chunk)
        result = result.reshape(-1, result.shape[-1])
        return result
    return None


DataEmbeddingFunc = Callable[
    [list[MultimodalDataItem]], torch.Tensor | EVSEmbeddingResult
]


# TODO: To be obsoleted.
def _get_chunked_prefill_embedding(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: list[MultimodalDataItem],
    items_size: list[int],
    prefix_length: list[int],
    extend_length: list[int],
    items_offset_list: list[list[tuple[int, int]]],
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    # Calculate embedding for each request, try to get it from cache to avoid repeated calculation
    embedding_list = []
    # FIXME(Xinyuan): temporary workaround for eagle3, which may have len(items_size) > len(prefix_length)
    max_iterations = min(len(items_size) - 1, len(prefix_length))
    for i in range(max_iterations):
        if items_size[i] == items_size[i + 1]:
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        assert items_offset is not None, items_offset
        # if all items has been prefixed, we do not need to calculate embedding
        if all(offset_end < prefix_length[i] for _, offset_end in items_offset):
            continue
        item_hashes = [item.hash for item in embedding_items_per_req]
        embedding_items_hash = MultiModalStaticCache.combine_hashes(item_hashes)
        embedding_per_req = embedding_cache.get(item_hashes)
        if embedding_per_req is None:
            embedding = data_embedding_func(embedding_items_per_req)
            embedding_per_req = (
                EmbeddingResult(embedding=embedding)
                if isinstance(embedding, torch.Tensor)
                else embedding
            )
            if not embedding_cache.set(embedding_items_hash, embedding_per_req):
                print_warning_once(
                    "Multimodal embedding cache is full. This typically occurs when a single "
                    "embedding exceeds the cache size limit. Consider increasing the "
                    "`SGLANG_VLM_CACHE_SIZE_MB` environment variable or reducing the input "
                    "embedding size."
                )

        extend_prefix_len = prefix_length[i]
        extend_seq_len = extend_length[i] if i < len(extend_length) else 0

        if isinstance(embedding_per_req, EVSEmbeddingResult):
            item = embedding_items_per_req[0]
            input_ids, items_offset = (
                embedding_per_req.redistribute_pruned_frames_placeholders(
                    input_ids,
                    items_offset,
                    item=item,
                    extend_prefix_len=extend_prefix_len,
                    extend_seq_len=extend_seq_len,
                )
            )

        embedding_per_req_chunk, _, _ = get_embedding_chunk(
            embedding=embedding_per_req.embedding,
            extend_prefix_len=extend_prefix_len,
            extend_seq_len=extend_seq_len,
            items_offset=items_offset,
        )
        embedding_list.append(embedding_per_req_chunk)
    if len(embedding_list) == 0:
        return None, input_ids
    return torch.concat(embedding_list, dim=0), input_ids


def _get_multimodal_mask(
    input_ids: torch.Tensor, placeholder_tensor: torch.Tensor
) -> torch.Tensor:
    return torch.isin(input_ids, placeholder_tensor).unsqueeze(-1)


def _adjust_embedding_length(
    embedding: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    num_mm_tokens_in_embedding = embedding.shape[0]
    num_mm_tokens_in_input_ids = mask.sum().item()
    if num_mm_tokens_in_input_ids != num_mm_tokens_in_embedding:
        logger.warning(
            f"Number of tokens in multimodal embedding does not match those in the input text. "
            f"Got {num_mm_tokens_in_input_ids} tokens in the text but {num_mm_tokens_in_embedding} "
            f"tokens from multimodal embeddings."
        )
        if num_mm_tokens_in_input_ids < num_mm_tokens_in_embedding:
            chunked_prefill_size = get_global_server_args().chunked_prefill_size
            if chunked_prefill_size != -1:
                logger.warning(
                    "You may want to avoid this issue by raising `chunked_prefill_size`, or disabling chunked prefill"
                )
            # extract from the end: this is a compromise
            if embedding.dim() == 2:
                embedding = embedding[-num_mm_tokens_in_input_ids:, :]
            else:
                num_multimodal = num_mm_tokens_in_input_ids // embedding.shape[0]
                embedding = embedding[-num_multimodal:, :]
        else:
            raise RuntimeError(
                f"Insufficient multimodal embedding length: {num_mm_tokens_in_input_ids=} vs {num_mm_tokens_in_embedding=}. This is an internal error"
            )
    return embedding


def get_embedding_and_mask(
    data_embedding_func: DataEmbeddingFunc,
    embedding_items: list[MultimodalDataItem],
    placeholder_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    items_size: list[int],
    prefix_length: list[int],
    extend_length: list[int],
    items_offset_list: list[list[tuple[int, int]]],
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """
    Generate multimodal embeddings and create a mask for identifying their positions in the input sequence.

    Args:
        data_embedding_func: Function that generates embeddings for multimodal items
        embedding_items: List of multimodal items to embed
        placeholder_tensor: Tensor containing token IDs that serve as placeholders for multimodal content
        input_ids: The input token IDs tensor
        items_size: Cumulative sizes of multimodal items per request
        prefix_length: Prefix lengths for each request
        extend_length: Sequence lengths for each request
        items_offset_list: List of offset ranges for multimodal items in each request

    Returns:
        A tuple containing:
        - The generated embeddings tensor
        - A boolean mask tensor indicating where these embeddings should be placed
        - If EVS is used, the pruned input ids tensor; otherwise, the original input ids tensor
    """
    # 1. Get embedding
    embedding = _get_precomputed_embedding(
        embedding_items, prefix_length, extend_length, items_offset_list
    )
    if embedding is None:
        embedding, input_ids = _get_chunked_prefill_embedding(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
            input_ids,
        )
        if embedding is None:
            return None, None, input_ids
    # 2. Get mask
    if _is_npu:
        torch.npu.current_stream().synchronize()
    special_multimodal_mask = _get_multimodal_mask(input_ids, placeholder_tensor)
    # 3. Adjust embedding length if needed
    embedding = _adjust_embedding_length(embedding, special_multimodal_mask)
    return embedding, special_multimodal_mask, input_ids


def embed_mm_inputs(
    mm_inputs_list: list[MultimodalInputs],
    extend_prefix_lens: list[int],
    extend_seq_lens: list[int],
    input_ids: torch.Tensor,
    input_embedding: nn.Embedding,
    multimodal_model: nn.Module = None,
    data_embedding_func_mapping: dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: dict[Modality, list[int]] = None,
    use_deepstack: dict[Modality, bool] = {},
) -> torch.Tensor | None:
    """
    Embed multimodal inputs and integrate them with text token embeddings.

    Args:
        mm_inputs_list: List of multimodal inputs to process
        extend_prefix_lens: Prefix lengths for each request
        extend_seq_lens: Sequence lengths for each request
        input_ids: Input token IDs tensor
        input_embedding: Embedding layer for text tokens
        placeholder_tokens: Token IDs for multimodal placeholders (uses pad_values if None)

    Returns:
        Combined embedding tensor with multimodal content integrated
    """
    other_info = {}
    if mm_inputs_list is None:
        return None

    # 1. Calculate the multimodal data which exists in input_ids, with the help of pad_values
    # we assume that multimodal data are represented with its pad_values in input_ids
    item_flatten_list = []
    for mm_inputs in mm_inputs_list:
        item_flatten_list += [item for item in mm_inputs.mm_items if item is not None]

    # deepstack_embeddings: per-modality
    modalities, embeddings, masks, deepstack_embeddings = [], [], [], []

    # 2. Get multimodal embedding separately
    # Try get mm embedding if any
    for modality in Modality.all():
        items = [
            item for item in item_flatten_list if item.is_modality(modality=modality)
        ]
        embedder = (
            None
            if data_embedding_func_mapping is None
            else data_embedding_func_mapping.get(modality, None)
        )
        if embedder is None:
            # "image", "video", etc
            modality_id = modality.name.lower()
            embedder = getattr(multimodal_model, f"get_{modality_id}_feature", None)
        if len(items) != 0:
            assert embedder is not None, f"no embedding method found for {modality}"
            placeholder_tensor = torch.as_tensor(
                [item.pad_value for item in items],
                device=input_ids.device,
            )
            # calculate per request items length offset
            items_size = torch.zeros(len(mm_inputs_list) + 1, dtype=int)
            items_offsets = []
            for i, mm_inputs in enumerate(mm_inputs_list):
                mm_items = [
                    item
                    for item in mm_inputs.mm_items
                    if item.is_modality(modality=modality)
                ]
                items_size[i + 1] = len(mm_items)
                items_offsets.append(
                    flatten_nested_list([item.offsets for item in mm_items])
                )
            items_size = torch.cumsum(items_size, dim=0).tolist()

            embedding, mask, input_ids = get_embedding_and_mask(
                data_embedding_func=embedder,
                embedding_items=items,
                placeholder_tensor=placeholder_tensor,
                input_ids=input_ids,
                items_size=items_size,
                prefix_length=extend_prefix_lens,
                extend_length=extend_seq_lens,
                items_offset_list=items_offsets,
            )

            if use_deepstack.get(modality, None) and embedding is not None:
                embedding, deepstack_embedding = (
                    multimodal_model.separate_deepstack_embeds(embedding)
                )
                deepstack_embeddings += [deepstack_embedding]
            modalities += [modality]
            embeddings += [embedding]
            masks += [mask]

    # 3. Get input embeddings
    vocab_size = input_embedding.num_embeddings
    # Important: clamp after getting original multimodal regions
    # Clamp input ids. This is because the input_ids for the multimodal tokens are
    # filled with the hash values of the multimodal for the prefix matching in the radix attention.
    # There values are useless because their embeddings will be replaced by vision embeddings anyway.
    input_ids.clamp_(min=0, max=vocab_size - 1)
    input_embeds = input_embedding(input_ids)

    # deepstack embedding
    if use_deepstack:
        num_deepstack_embeddings = len(multimodal_model.deepstack_visual_indexes)

        deepstack_embedding_shape = input_embeds.shape[:-1] + (
            input_embeds.shape[-1] * num_deepstack_embeddings,
        )
        # a zero-filled embedding, with the same length of input_embeds, but different hidden_size
        input_deepstack_embeds = torch.zeros(
            deepstack_embedding_shape,
            device=input_embeds.device,
            dtype=input_embeds.dtype,
        )

        other_info["input_deepstack_embeds"] = input_deepstack_embeds

    # 4. scatter embeddings into input embedding
    for i, modality, embedding, mask in zip(
        range(len(embeddings)), modalities, embeddings, masks
    ):
        if embedding is None or mask is None:
            continue
        # in-place update
        indices = torch.where(mask.squeeze(dim=-1))[0]
        input_embeds[indices] = embedding.to(input_embeds.device, input_embeds.dtype)
        if use_deepstack.get(modality, None):
            input_deepstack_embeds[indices] = deepstack_embeddings[i].to(
                input_embeds.device, input_embeds.dtype
            )

    return input_embeds, other_info


def general_mm_embed_routine(
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    language_model: nn.Module,
    multimodal_model: nn.Module | None = None,
    data_embedding_funcs: dict[Modality, DataEmbeddingFunc] = None,
    placeholder_tokens: dict[Modality, list[int]] | None = None,
    use_deepstack: dict[Modality, bool] = {},
    **kwargs,
) -> torch.Tensor:
    """
    Process multimodal inputs and forward through language model.

    Args:
        input_ids: Input token IDs tensor
        forward_batch: Batch information for model forward pass
        language_model: Base language model to use
        data_embedding_funcs: A dictionary mapping from modality type to the corresponding embedding function.
        placeholder_tokens: Token IDs for multimodal placeholders
        use_deepstack: Whether to use deepstack embeddings for each modality, default False
        **kwargs: Additional arguments passed to language model

    Returns:
        Hidden states from language model forward pass
    """
    assert hasattr(language_model, "get_input_embeddings")
    embed_tokens = language_model.get_input_embeddings()
    if not hasattr(language_model, "pp_group") or language_model.pp_group.is_first_rank:
        if (
            not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.contains_mm_inputs()
        ):
            mm_inputs_list = [
                mm_input for mm_input in forward_batch.mm_inputs if mm_input is not None
            ]
            extend_prefix_lens = [
                prefix_len
                for i, prefix_len in enumerate(forward_batch.extend_prefix_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            extend_seq_lens = [
                seq_len
                for i, seq_len in enumerate(forward_batch.extend_seq_lens_cpu)
                if forward_batch.mm_inputs[i] is not None
            ]
            input_embeds, other_info = embed_mm_inputs(
                mm_inputs_list=mm_inputs_list,
                extend_prefix_lens=extend_prefix_lens,
                extend_seq_lens=extend_seq_lens,
                input_ids=input_ids,
                multimodal_model=multimodal_model,
                input_embedding=embed_tokens,
                data_embedding_func_mapping=data_embedding_funcs,
                placeholder_tokens=placeholder_tokens,
                use_deepstack=use_deepstack,
            )
            # add for qwen3_vl deepstack
            if use_deepstack:
                kwargs["input_deepstack_embeds"] = other_info["input_deepstack_embeds"]
            # Offload GPU features to CPU instead of discarding them to balance memory
            # efficiency and data persistence.
            # In chunked-prefill, a request is processed across multiple batches, and
            # the original multimodal data must remain accessible until the entire
            # prefill phase is complete. Since the multimodal embedding cache is
            # best-effort, offloading to CPU ensures we have a reliable fallback
            # if a cache miss occurs in subsequent chunks, while still freeing up
            # critical GPU memory.
            if mm_inputs_list:
                for mm_input_obj in mm_inputs_list:
                    if mm_input_obj and hasattr(mm_input_obj, "mm_items"):
                        for mm_item in mm_input_obj.mm_items:
                            feature = getattr(mm_item, "feature", None)
                            if isinstance(feature, torch.Tensor) and feature.is_cuda:
                                mm_item.feature = feature.to("cpu", non_blocking=True)
            forward_batch.mm_inputs = None
            forward_batch.mm_input_embeds = input_embeds
        else:
            input_embeds = embed_tokens(input_ids)
        # Copy to pre-allocated buffer if available (for CUDA graph address stability)
        if forward_batch.input_embeds is not None:
            forward_batch.input_embeds.copy_(input_embeds)
            input_embeds = forward_batch.input_embeds
    else:
        input_embeds = None

    hidden_states = language_model(
        input_ids=None,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        **kwargs,
    )
    return hidden_states


def data_hash(data) -> int:
    hash_bytes = hashlib.sha256(data).digest()[:8]
    return int.from_bytes(hash_bytes, byteorder="big", signed=False)


def tensor_hash(tensor_list) -> int:
    """
    hash a tensor or a tensor list
    """
    tensor = tensor_list
    if isinstance(tensor_list, list):
        tensor_list = flatten_nested_list(tensor_list)
        tensor_list = [
            x.flatten() if isinstance(x, torch.Tensor) else x for x in tensor_list
        ]
        tensor = torch.concat(tensor_list)
    if tensor.is_cuda:
        return gpu_tensor_hash(tensor.cuda())
    tensor = tensor.detach().contiguous()

    if tensor.dtype == torch.bfloat16:
        # memoryview() doesn't support PyTorch's BFloat16 dtype
        tensor = tensor.float()

    assert isinstance(tensor, torch.Tensor)
    tensor_cpu = tensor.cpu()

    mv = memoryview(tensor_cpu.numpy())
    return data_hash(mv.tobytes())


def hash_feature(f):
    if isinstance(f, list):
        if isinstance(f[0], torch.Tensor):
            return tensor_hash(f)
        return data_hash(tuple(flatten_nested_list(f)))
    elif isinstance(f, np.ndarray):
        arr = np.ascontiguousarray(f)
        arr_bytes = arr.tobytes()
        return data_hash(arr_bytes)
    elif isinstance(f, torch.Tensor):
        return tensor_hash([f])
    elif isinstance(f, CudaIpcTensorTransportProxy):
        reconstruct_t = f.reconstruct_on_target_device(torch.cuda.current_device())
        return tensor_hash([reconstruct_t])
    return data_hash(f)


def extend_mrope_positions_for_retracted_request(
    mrope_positions: torch.Tensor, output_ids_len: int
) -> torch.Tensor:
    """
    Extend mrope_positions for retracted requests by appending positions for output_ids.

    When a request is retracted and has multimodal inputs with mrope_positions,
    we need to extend the positions to cover the output_ids that were already generated.
    For pure text tokens, all three dimensions use the same incremental sequence.

    Args:
        mrope_positions: The original mrope positions tensor, shape (3, origin_input_ids_len)
        output_ids_len: The number of output tokens to generate positions for

    Returns:
        Extended mrope_positions tensor with shape (3, origin_input_ids_len + output_ids_len)
    """
    if output_ids_len <= 0:
        return mrope_positions

    # Get the last position value corresponding to origin_input_ids
    # mrope_positions shape: (3, origin_input_ids_len)
    last_position = mrope_positions[:, -1]  # shape: (3,)

    # Generate pure text mrope positions for output_ids
    # All three dimensions for pure text are the same incremental sequence
    start_pos = last_position[0] + 1  # Start from last position + 1
    output_positions = (
        torch.arange(
            start_pos,
            start_pos + output_ids_len,
            dtype=torch.int64,
            device=mrope_positions.device,
        )
        .unsqueeze(0)
        .expand(3, -1)
    )  # shape: (3, output_ids_len)

    # Concatenate to the original mrope_positions
    return torch.cat([mrope_positions, output_positions], dim=1)


def _get_length(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.shape[0] if value.ndim > 0 else None
    if isinstance(value, np.ndarray):
        return value.shape[0] if value.ndim > 0 else None
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _slice_value(value, start, end):
    if isinstance(value, torch.Tensor):
        return value[start:end]
    if isinstance(value, np.ndarray):
        return value[start:end]
    if isinstance(value, list):
        return value[start:end]
    if isinstance(value, tuple):
        return value[start:end]
    try:
        return value[start:end]
    except Exception as e:
        logger.debug(
            f"Cannot slice {type(value).__name__}[{start}:{end}], returning as-is: {e}"
        )
        return value


def _slice_model_data(
    data: dict,
    index: int,
    start: int,
    end: int,
    num_items: int,
    total_feature_len: int | None,
):
    sliced = {}
    for key, value in data.items():
        length = _get_length(value)
        if length == num_items:
            sliced[key] = _slice_value(value, index, index + 1)
        elif total_feature_len is not None and length == total_feature_len:
            sliced[key] = _slice_value(value, start, end)
        else:
            sliced[key] = value
    return sliced


def get_new_expanded_mm_items(original_mm_items):
    expanded_mm_items = []
    for item in original_mm_items:
        is_bundled = item.offsets is not None and len(item.offsets) > 1

        if is_bundled:
            num_items = len(item.offsets)

            if item.is_image():
                image_grid_thw = item.model_specific_data.get("image_grid_thw")
                grid_len = _get_length(image_grid_thw)
                if image_grid_thw is None or grid_len != num_items:
                    expanded_mm_items.append(item)
                    continue

                patches_per_item = []
                for grid in image_grid_thw:
                    grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                    patches_per_item.append(int(torch.prod(grid_tensor).item()))

                cumulative = torch.cumsum(
                    torch.tensor(patches_per_item, dtype=torch.long), dim=0
                )
                slice_indices = [0] + cumulative.tolist()

                feature_len = _get_length(item.feature)
                if feature_len is None:
                    feature_len = _get_length(item.precomputed_embeddings)
                if feature_len is None or slice_indices[-1] != feature_len:
                    expanded_mm_items.append(item)
                    continue

                total_feature_len = feature_len
                for i in range(num_items):
                    start, end = slice_indices[i], slice_indices[i + 1]
                    new_item = copy.deepcopy(item)
                    if item.feature is not None:
                        new_item.feature = _slice_value(item.feature, start, end)
                    if item.precomputed_embeddings is not None:
                        new_item.precomputed_embeddings = _slice_value(
                            item.precomputed_embeddings, start, end
                        )
                    new_item.offsets = [item.offsets[i]]
                    new_item.model_specific_data = _slice_model_data(
                        item.model_specific_data,
                        index=i,
                        start=start,
                        end=end,
                        num_items=num_items,
                        total_feature_len=total_feature_len,
                    )
                    new_item.hash = None
                    expanded_mm_items.append(new_item)

            elif item.is_video():
                video_grid_thw = item.model_specific_data.get("video_grid_thw")
                if video_grid_thw is None:
                    expanded_mm_items.append(item)
                    continue

                # video_grid_thw shape: [num_videos, 3] where each row is [T, H, W]
                # When T > 1, item.offsets contains frames (num_items = total frames)
                # grid_len = num_videos, num_items = sum(T for each video) = total frames
                grid_len = _get_length(video_grid_thw)
                num_videos = grid_len

                # Calculate total frames and frames per video
                frames_per_video = []
                total_frames = 0
                for i in range(num_videos):
                    grid = video_grid_thw[i]
                    if isinstance(grid, torch.Tensor):
                        T = int(grid[0].item())  # T is the first element [T, H, W]
                    else:
                        grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                        T = int(grid_tensor[0].item())
                    frames_per_video.append(T)
                    total_frames += T

                # num_items should equal total_frames when T > 1
                if num_items != total_frames:
                    expanded_mm_items.append(item)
                    continue

                # Calculate patches per video: T * H * W for each video
                patches_per_video = []
                for i in range(num_videos):
                    grid = video_grid_thw[i]
                    if isinstance(grid, torch.Tensor):
                        patches_per_video.append(int(torch.prod(grid).item()))
                    else:
                        grid_tensor = torch.as_tensor(grid, dtype=torch.long)
                        patches_per_video.append(int(torch.prod(grid_tensor).item()))

                # Calculate cumulative patches to get slice indices for each video
                cumulative = torch.cumsum(
                    torch.tensor(patches_per_video, dtype=torch.long), dim=0
                )
                slice_indices = [0] + cumulative.tolist()

                feature_len = _get_length(item.feature)
                if feature_len is None:
                    feature_len = _get_length(item.precomputed_embeddings)
                if feature_len is None or slice_indices[-1] != feature_len:
                    expanded_mm_items.append(item)
                    continue

                total_feature_len = feature_len
                # Group frames by video: calculate frame indices for each video
                frame_start_indices = [0]
                for i in range(num_videos):
                    frame_start_indices.append(
                        frame_start_indices[-1] + frames_per_video[i]
                    )

                # Expand each video into a separate item
                for video_idx in range(num_videos):
                    start, end = (
                        slice_indices[video_idx],
                        slice_indices[video_idx + 1],
                    )
                    frame_start, frame_end = (
                        frame_start_indices[video_idx],
                        frame_start_indices[video_idx + 1],
                    )

                    new_item = copy.deepcopy(item)
                    if item.feature is not None:
                        new_item.feature = _slice_value(item.feature, start, end)
                    if item.precomputed_embeddings is not None:
                        new_item.precomputed_embeddings = _slice_value(
                            item.precomputed_embeddings, start, end
                        )
                    # Group offsets for this video (all frames of this video)
                    new_item.offsets = item.offsets[frame_start:frame_end]
                    # For video_grid_thw, slice the corresponding row [T, H, W] for this video
                    new_item.model_specific_data = _slice_model_data(
                        item.model_specific_data,
                        index=video_idx,
                        start=start,
                        end=end,
                        num_items=num_videos,
                        total_feature_len=total_feature_len,
                    )
                    new_item.hash = None
                    expanded_mm_items.append(new_item)
            else:
                expanded_mm_items.append(item)

        else:
            expanded_mm_items.append(item)
    return expanded_mm_items


class ShmPointerMMData:
    """
    Wraps a tensor to be sent via a shared memory handle.
    This acts as a "pointer" to the tensor data across process boundaries.
    """

    def __init__(self, tensor: torch.Tensor):
        self.cpu_tensor = tensor.cpu().contiguous()
        self.shape = self.cpu_tensor.shape
        self.dtype = self.cpu_tensor.dtype

        nbytes = self.cpu_tensor.numel() * self.cpu_tensor.element_size()

        self.shm = shared_memory.SharedMemory(create=True, size=nbytes)

        try:
            shm_view = np.ndarray((nbytes,), dtype=np.uint8, buffer=self.shm.buf)

            shm_view[:] = self.cpu_tensor.view(torch.uint8).numpy().flatten()
        finally:
            self.shm.close()

    def __getstate__(self):
        if not hasattr(self, "shm") or self.shm is None:
            tensor = getattr(self, "cpu_tensor", None)
            if tensor is None:
                tensor = getattr(self, "tensor", None)
            if tensor is None:
                raise RuntimeError(
                    "ShmPointerMMData cannot recreate shared memory without tensor"
                )

            cpu_tensor = tensor.cpu().contiguous()
            self.shape = cpu_tensor.shape
            self.dtype = cpu_tensor.dtype

            nbytes = cpu_tensor.numel() * cpu_tensor.element_size()
            self.shm = shared_memory.SharedMemory(create=True, size=nbytes)
            try:
                shm_view = np.ndarray((nbytes,), dtype=np.uint8, buffer=self.shm.buf)
                shm_view[:] = cpu_tensor.view(torch.uint8).numpy().flatten()
            finally:
                self.shm.close()

        return {
            "shm_name": self.shm.name,
            "shape": self.shape,
            "dtype": self.dtype,
        }

    def __setstate__(self, state):
        self.shm_name = state["shm_name"]
        self.shape = state["shape"]
        self.dtype = state["dtype"]
        self.shm = None

        shm_handle = shared_memory.SharedMemory(name=self.shm_name)
        try:
            self.tensor = (
                torch.frombuffer(shm_handle.buf, dtype=self.dtype)
                .reshape(self.shape)
                .clone()
            )
        finally:
            shm_handle.close()
            shm_handle.unlink()


def _get_is_default_transport():
    global _is_default_tensor_transport
    if _is_default_tensor_transport is None:
        from sglang.srt.managers.tokenizer_manager import (
            _determine_tensor_transport_mode,
        )

        _is_default_tensor_transport = (
            _determine_tensor_transport_mode(get_global_server_args()) == "default"
        )
    return _is_default_tensor_transport


def wrap_shm_features(obj):
    """
    Scan the object for multimodal tensors and wrap them in SHM pointers.
    """
    if _get_is_default_transport() or get_global_server_args().skip_tokenizer_init:
        return obj

    if hasattr(obj, "mm_inputs") and obj.mm_inputs:
        mm_items = obj.mm_inputs.get("mm_items", [])
        for item in mm_items:
            if (
                hasattr(item, "feature")
                and isinstance(item.feature, torch.Tensor)
                and item.feature.is_cpu
            ):
                item.feature = ShmPointerMMData(item.feature)
    return obj


def unwrap_shm_features(obj):
    """
    Restore ShmPointerMMData wrappers back into standard torch.Tensors.
    """
    if _get_is_default_transport() or get_global_server_args().skip_tokenizer_init:
        return obj
    if hasattr(obj, "mm_inputs") and obj.mm_inputs:
        mm_items = obj.mm_inputs.get("mm_items", [])
        for item in mm_items:
            if isinstance(item.feature, ShmPointerMMData):
                item.feature = item.feature.tensor
    return obj
