"""
Multi-modality utils
"""

import dataclasses
import logging
from abc import abstractmethod
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    global_server_args_dict,
)
from sglang.srt.mem_cache.multimodal_cache import MultiModalCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import flatten_nested_list, print_warning_once
from sglang.utils import logger

# NOTE: Using the shared logger from sglang.utils instead of creating a module-specific logger
# to ensure consistent logging behavior across the codebase. This prevents issues with log
# propagation that can cause some log messages (like 'server is fired up') to not appear
# in the console when multimodal support is enabled.


class MultiModalityDataPaddingPattern:
    """
    Data tokens (like image tokens) often need special handling during padding
    to maintain model compatibility. This class provides the interface for
    implementing different padding strategies for data tokens
    """

    @abstractmethod
    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
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
        data_token_pairs: Optional[List[Tuple[int, int]]],
        data_start_token_ids: Optional[List[int]] = None,
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
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
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
        start_token_ids = [s for s, _e in data_token_pairs]
        end_tokens_ids = [e for _s, e in data_token_pairs]

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

    def __init__(self, token_ids: List[int]) -> None:
        self.token_ids = token_ids

    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """
        Finds contiguous regions of tokens matching `self.token_ids` in `input_ids`
        and replaces each region with the corresponding `pad_value` from `mm_inputs.mm_items`.
        """
        pad_values = [item.pad_value for item in mm_inputs.mm_items]
        if not pad_values:
            # No multimodal items, return original input_ids
            return input_ids
        if not input_ids:
            return []

        input_ids_tensor = torch.tensor(input_ids)
        device = input_ids_tensor.device
        token_ids_tensor = torch.tensor(self.token_ids, device=device)
        mask = torch.isin(input_ids_tensor, token_ids_tensor)

        if not mask.any():
            # No tokens match token_ids, return original input_ids
            return input_ids

        # Find contiguous regions
        padded_mask = torch.cat(
            (
                torch.tensor([False], device=device),
                mask,
                torch.tensor([False], device=device),
            )
        )
        # Find indices where the mask value changes
        diff_indices = torch.where(padded_mask[1:] != padded_mask[:-1])[0]

        # Start indices are where False changes to True
        starts = diff_indices[::2]
        # End indices are where True changes to False (exclusive index)
        ends = diff_indices[1::2]

        # Check if the number of regions matches the number of pad values
        if len(starts) != len(pad_values):
            # Maybe log a warning here?
            num_regions = len(starts)
            num_pad_values = len(pad_values)
            if num_regions > 0 and num_pad_values > 0:
                pad_values = (pad_values * (num_regions // num_pad_values + 1))[
                    :num_regions
                ]
            else:  # If no regions or no pad_values, this loop won't run anyway.
                pad_values = []  # Ensure pad_values is empty if starts is empty

        # Create a copy to modify
        output_ids_tensor = input_ids_tensor.clone()

        # Replace tokens in each region with the corresponding pad value
        # Ensure we don't iterate if pad_values became empty due to mismatch and num_regions=0
        for i in range(min(len(starts), len(pad_values))):
            start_idx = starts[i]
            end_idx = ends[i]
            pad_value = pad_values[i]
            if pad_value is not None:  # Ensure pad_value is not None before assignment
                output_ids_tensor[start_idx:end_idx] = pad_value
            else:
                logger.warning(f"Skipping region {i} due to None pad_value.")
        return output_ids_tensor.tolist()


embedding_cache = None


def init_embedding_cache(max_size: int):
    global embedding_cache
    embedding_cache = MultiModalCache(max_size)


def get_embedding_hash(embedding_items: List[MultimodalDataItem]) -> int:
    hash_list = [item.hash for item in embedding_items]
    return hash(tuple(hash_list))


def get_embedding_chunk(
    embedding: torch.Tensor,
    extend_prefix_len: int,
    extend_seq_len: int,
    items_offset: List[Tuple[int, int]],
) -> Tuple[torch.Tensor, int, int]:
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
    # some models embedding is 3-dim, reshape it to 2-dim
    embedding = embedding.reshape(-1, embedding.shape[-1])
    embedding_chunk = embedding[start_index:end_index]
    return embedding_chunk, start_index, end_index


def _get_precomputed_embedding(
    items: List[MultimodalDataItem],
) -> Optional[torch.Tensor]:
    """
    If all items have precomputed_features, return their concatenation.
    If some but not all have precomputed_features, raise NotImplementedError.
    If none have precomputed_features, return None.
    """
    precomputed_features = [item.precomputed_features for item in items]
    if any(feature is not None for feature in precomputed_features):
        if not all(feature is not None for feature in precomputed_features):
            raise NotImplementedError(
                "MM inputs where only some items are precomputed."
            )
        result = torch.concat(precomputed_features)
        # some models embedding is 3-dim, reshape it to 2-dim (similar to get_embedding_chunk)
        result = result.reshape(-1, result.shape[-1])
        return result
    return None


def _get_chunked_prefill_embedding(
    data_embedding_func: Callable[[List[MultimodalDataItem]], torch.Tensor],
    embedding_items: List[MultimodalDataItem],
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Optional[torch.Tensor]:
    # Calculate embedding for each request, try to get it from cache to avoid repeated calculation
    embedding_list = []
    for i in range(len(items_size) - 1):
        if items_size[i] == items_size[i + 1]:
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        embedding_items_hash = get_embedding_hash(embedding_items_per_req)
        # if all items has been prefixed, we do not need to calculate embedding
        if all([offset_end < prefix_length[i] for _, offset_end in items_offset]):
            continue
        embedding_per_req = embedding_cache.get(embedding_items_hash)
        if embedding_per_req is None:
            embedding_per_req = data_embedding_func(embedding_items_per_req)
            if not embedding_cache.put(embedding_items_hash, embedding_per_req):
                print_warning_once(
                    "Multimodal embedding cache is full. Consider increasing the "
                    "`SGLANG_VLM_CACHE_SIZE_MB` environment variable."
                )

        embedding_per_req_chunk, _, end_index = get_embedding_chunk(
            embedding=embedding_per_req,
            extend_prefix_len=prefix_length[i],
            extend_seq_len=extend_length[i],
            items_offset=items_offset,
        )
        # remove this item from cache if chunk reaches to the end
        embedding_per_req_length = (
            embedding_per_req.shape[0]
            if embedding_per_req.dim() == 2
            else embedding_per_req.shape[0] * embedding_per_req.shape[1]
        )
        if end_index == embedding_per_req_length:
            embedding_cache.free(embedding_items_hash)
        embedding_list.append(embedding_per_req_chunk)
    if len(embedding_list) == 0:
        return None
    return torch.concat(embedding_list, dim=0)


def _get_multimodal_mask(
    input_ids: torch.Tensor, placeholder_tensor: torch.Tensor
) -> torch.Tensor:
    return torch.isin(input_ids, placeholder_tensor).unsqueeze(-1)


def _adjust_embedding_length(
    embedding: torch.Tensor,
    mask: torch.Tensor,
    logger,
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
            chunked_prefill_size = global_server_args_dict["chunked_prefill_size"]
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
    data_embedding_func: Callable[[List[MultimodalDataItem]], torch.Tensor],
    embedding_items: List[MultimodalDataItem],
    placeholder_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    """
    # 1. Get embedding
    embedding = _get_precomputed_embedding(embedding_items)
    if embedding is None:
        embedding = _get_chunked_prefill_embedding(
            data_embedding_func,
            embedding_items,
            items_size,
            prefix_length,
            extend_length,
            items_offset_list,
        )
        if embedding is None:
            return None, None
    # 2. Get mask
    special_multimodal_mask = _get_multimodal_mask(input_ids, placeholder_tensor)
    # 3. Adjust embedding length if needed
    embedding = _adjust_embedding_length(embedding, special_multimodal_mask, logger)
    return embedding, special_multimodal_mask


def embed_mm_inputs(
    mm_inputs_list: List[MultimodalInputs],
    extend_prefix_lens: List[int],
    extend_seq_lens: List[int],
    input_ids: torch.Tensor,
    input_embedding: nn.Embedding,
    image_data_embedding_func: Callable[
        [List[MultimodalDataItem]], torch.Tensor
    ] = None,
    audio_data_embedding_func: Callable[
        [List[MultimodalDataItem]], torch.Tensor
    ] = None,
    placeholder_tokens: dict[Modality, List[int]] = None,
) -> Optional[torch.Tensor]:
    """
    Embed multimodal inputs and integrate them with text token embeddings.

    Args:
        mm_inputs_list: List of multimodal inputs to process
        extend_prefix_lens: Prefix lengths for each request
        extend_seq_lens: Sequence lengths for each request
        input_ids: Input token IDs tensor
        input_embedding: Embedding layer for text tokens
        image_data_embedding_func: Function to embed image data
        audio_data_embedding_func: Function to embed audio data
        placeholder_tokens: Token IDs for multimodal placeholders (uses pad_values if None)

    Returns:
        Combined embedding tensor with multimodal content integrated
    """

    if mm_inputs_list is None:
        return None

    # 1. Calculate the multimodal data which exists in input_ids, with the help of pad_values
    # we assume that multimodal data are represented with its pad_values in input_ids
    item_flatten_list = []
    for mm_inputs in mm_inputs_list:
        item_flatten_list += [item for item in mm_inputs.mm_items if item is not None]

    embeddings, masks = [], []

    # 2. Get multimodal embedding separately
    # TODO: make this more generic
    # Try get image embedding if any
    if (
        any(True for item in item_flatten_list if item.is_image())
        and image_data_embedding_func
    ):
        items = [item for item in item_flatten_list if item.is_image()]
        placeholder_tensor = torch.tensor(
            [item.pad_value for item in items],
            device=input_ids.device,
        )
        # calculate per request items length offset
        items_size = torch.zeros(len(mm_inputs_list) + 1, dtype=int)
        items_offsets = []
        for i, mm_inputs in enumerate(mm_inputs_list):
            image_items = [item for item in mm_inputs.mm_items if item.is_image()]
            items_size[i + 1] = len(image_items)
            items_offsets.append(
                flatten_nested_list(
                    [
                        item.image_offsets
                        for item in mm_inputs.mm_items
                        if item.is_image()
                    ]
                )
            )
        items_size = torch.cumsum(items_size, dim=0).tolist()

        embedding, mask = get_embedding_and_mask(
            data_embedding_func=image_data_embedding_func,
            embedding_items=items,
            placeholder_tensor=placeholder_tensor,
            input_ids=input_ids,
            items_size=items_size,
            prefix_length=extend_prefix_lens,
            extend_length=extend_seq_lens,
            items_offset_list=items_offsets,
        )
        embeddings += [embedding]
        masks += [mask]

    # Try get audio embedding if any
    if (
        any(True for item in item_flatten_list if item.is_audio())
        and audio_data_embedding_func
    ):
        items = [item for item in item_flatten_list if item.is_audio()]
        placeholder_tensor = torch.tensor(
            [item.pad_value for item in items],
            device=input_ids.device,
        )
        items_offsets = []
        # calculate per request items length offset
        items_size = torch.zeros(len(mm_inputs_list) + 1, dtype=int)
        for i, mm_inputs in enumerate(mm_inputs_list):
            audio_items = [item for item in mm_inputs.mm_items if item.is_audio()]
            items_size[i + 1] = len(audio_items)
            items_offsets.append(
                flatten_nested_list(
                    [
                        item.audio_offsets
                        for item in mm_inputs.mm_items
                        if item.is_audio()
                    ]
                )
            )
        items_size = torch.cumsum(items_size, dim=0)

        embedding, mask = get_embedding_and_mask(
            data_embedding_func=audio_data_embedding_func,
            embedding_items=items,
            placeholder_tensor=placeholder_tensor,
            input_ids=input_ids,
            items_size=items_size,
            prefix_length=extend_prefix_lens,
            extend_length=extend_seq_lens,
            items_offset_list=items_offsets,
        )
        embeddings += [embedding]
        masks += [mask]

    # 3. Get input embeddings
    vocab_size = input_embedding.num_embeddings
    # Important: clamp after getting original multimodal regions
    # Clamp input ids. This is because the input_ids for the multimodal tokens are
    # filled with the hash values of the multimodal for the prefix matching in the radix attention.
    # There values are useless because their embeddings will be replaced by vision embeddings anyway.
    input_ids.clamp_(min=0, max=vocab_size - 1)
    inputs_embeds = input_embedding(input_ids)

    # 4. scatter embeddings into input embedding
    for embedding, mask in zip(embeddings, masks):
        if embedding is None or mask is None:
            continue
        mask = mask.expand_as(inputs_embeds).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.masked_scatter(
            mask,
            embedding.to(inputs_embeds.device, inputs_embeds.dtype),
        )
    return inputs_embeds


def general_mm_embed_routine(
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
    language_model: nn.Module,
    image_data_embedding_func: Optional[
        Callable[[List[MultimodalDataItem]], torch.Tensor]
    ] = None,
    audio_data_embedding_func: Optional[
        Callable[[List[MultimodalDataItem]], torch.Tensor]
    ] = None,
    placeholder_tokens: Optional[dict[Modality, List[int]]] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Process multimodal inputs and forward through language model.

    Args:
        input_ids: Input token IDs tensor
        forward_batch: Batch information for model forward pass
        language_model: Base language model to use
        image_data_embedding_func: Function to embed image data
        audio_data_embedding_func: Function to embed audio data
        placeholder_tokens: Token IDs for multimodal placeholders
        **kwargs: Additional arguments passed to language model

    Returns:
        Hidden states from language model forward pass
    """
    assert hasattr(language_model, "get_input_embeddings")
    embed_tokens = language_model.get_input_embeddings()
    if (
        not forward_batch.forward_mode.is_decode()
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
        inputs_embeds = embed_mm_inputs(
            mm_inputs_list=mm_inputs_list,
            extend_prefix_lens=extend_prefix_lens,
            extend_seq_lens=extend_seq_lens,
            input_ids=input_ids,
            input_embedding=embed_tokens,
            image_data_embedding_func=image_data_embedding_func,
            audio_data_embedding_func=audio_data_embedding_func,
            placeholder_tokens=placeholder_tokens,
        )
        # once used, mm_inputs is useless, considering chunked-prefill is disabled for multimodal models
        # just being defensive here
        forward_batch.mm_inputs = None
    else:
        inputs_embeds = embed_tokens(input_ids)

    hidden_states = language_model(
        input_ids=None,
        forward_batch=forward_batch,
        input_embeds=inputs_embeds,
        **kwargs,
    )
    return hidden_states


def get_multimodal_data_bounds(
    input_ids: torch.Tensor, pad_values: List[int], token_pairs: List[Tuple[int, int]]
) -> torch.Tensor:
    """
    Returns a tensor indicating the bounds of multimodal data (images, video, audio, etc.)

    Returns:
        [bounds_count, 2]
    """
    # All the multimodal data in the batch should share the same special bound token ids.
    start_tokens = [s for s, _e in token_pairs]
    end_tokens = [e for _s, e in token_pairs]

    assert all(isinstance(t, int) for t in start_tokens)
    assert all(isinstance(t, int) for t in end_tokens)

    start_cond = torch.isin(
        input_ids, torch.tensor(start_tokens, device=input_ids.device)
    )
    end_cond = torch.isin(input_ids, torch.tensor(end_tokens, device=input_ids.device))

    (data_start_tokens,) = torch.where(start_cond)
    (data_end_tokens,) = torch.where(end_cond)

    # the im_start_id sometimes can be cached as prefix, but it is needed for the embedding of the multimodal data
    if len(data_start_tokens) != len(data_end_tokens):
        if (
            len(data_start_tokens) + 1 == len(data_end_tokens)
            and input_ids[0] in pad_values
            and data_end_tokens[0] < data_start_tokens[0]
        ):
            data_start_tokens = torch.cat(
                [
                    torch.tensor([0], device=data_start_tokens.device),
                    data_start_tokens,
                ]
            )
    valid_mm_data_nums = min(len(data_start_tokens), len(data_end_tokens))

    if valid_mm_data_nums == 0:
        return torch.zeros((0, 2), device=input_ids.device)

    # Filter out pairs where start_token >= end_token
    valid_pairs = []
    for i in range(valid_mm_data_nums):
        start_token = data_start_tokens[i]
        end_token = data_end_tokens[i]
        if start_token < end_token:
            valid_pairs.append((start_token + 1, end_token - 1))

    if not valid_pairs:
        return torch.zeros((0, 2), device=input_ids.device)

    # Convert valid pairs to tensor
    valid_pairs_tensor = torch.tensor(valid_pairs, device=input_ids.device)
    return valid_pairs_tensor
