"""
    Multi-modality utils
"""

from abc import abstractmethod
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.managers.schedule_batch import (
    MultimodalDataItem,
    MultimodalInputs,
    global_server_args_dict,
    logger,
)
from sglang.srt.mem_cache.multimodal_cache import MultiModalCache
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import flatten_nested_list, print_warning_once
from sglang.utils import logger


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

    This strategy should be applied when data content is marked by start/end token pairs in the input sequence.
    """

    def __init__(self, data_token_pairs: Optional[List[Tuple[int, int]]]) -> None:
        self.data_token_id_pairs = data_token_pairs

    def pad_input_tokens(
        self, input_ids: List[int], mm_inputs: MultimodalInputs
    ) -> List[int]:
        """
        This function will replace the data-tokens inbetween with pad_values accordingly
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

            if input_ids[start_idx] in start_token_ids:
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


class MultiModalityDataPaddingPatternImageTokens(MultiModalityDataPaddingPattern):
    """In this pattern, data tokens should be represented as repetitions of a single token
    e.g. <image><image>....<image>, or <audio><audio>...<audio>
    """

    def __init__(self, image_token_id: torch.Tensor) -> None:
        self.image_token_id = image_token_id

    def pad_input_tokens(self, input_ids: List[int], mm_inputs) -> List[int]:
        """
        This function will replace the data-tokens in between with pad_values accordingly
        """
        pad_values = [item.pad_value for item in mm_inputs.mm_items]
        assert len(pad_values) != 0

        input_ids_tensor = torch.tensor(input_ids)
        mask = torch.isin(input_ids_tensor, self.image_token_id)

        num_image_tokens = mask.sum().item()
        repeated_pad_values = torch.tensor(pad_values).repeat(
            num_image_tokens // len(pad_values) + 1
        )[:num_image_tokens]

        input_ids_tensor[mask] = repeated_pad_values
        return input_ids_tensor.tolist()


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
) -> torch.Tensor:
    """
    Extract embedding chunk according to [extend_prefix_len, extend_prefix_len + extend_seq_len - 1]
    and items_offset, items_offset records list of each items [start, end] offset in origin_input_ids in a request
    it is used for chunk prefill for multimodal items
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
    embedding_chunk = embedding[start_index:end_index]
    return embedding_chunk, start_index, end_index


def get_embedding_and_mask(
    data_embedding_func: Callable[[List[MultimodalDataItem]], torch.Tensor],
    embedding_items: List[MultimodalDataItem],
    placeholder_tensor: torch.Tensor,
    input_ids: torch.Tensor,
    items_size: List[int],
    prefix_length: List[int],
    extend_length: List[int],
    items_offset_list: List[List[Tuple[int, int]]],
):
    """
    Get the multimodal embedding and its mask from input_ids
        Args:
            items_size: the number of items for each request
            items_offset_list: list of [start, end] offset for each item in a request
    """
    # 1. Get the embedding
    #    Calculate embedding for each request, try to get it from cache to avoid repeated calculation
    embedding_list = []
    for i in range(len(items_size) - 1):
        if items_size[i] == items_size[i + 1]:
            continue
        embedding_items_per_req = embedding_items[items_size[i] : items_size[i + 1]]
        items_offset = items_offset_list[i]
        embedding_items_hash = get_embedding_hash(embedding_items_per_req)
        embedding_per_req = embedding_cache.get(embedding_items_hash)
        if embedding_per_req is None:
            embedding_per_req = data_embedding_func(embedding_items_per_req)
            embedding_cache.put(embedding_items_hash, embedding_per_req)

        embedding_per_req_chunk, _, end_index = get_embedding_chunk(
            embedding=embedding_per_req,
            extend_prefix_len=prefix_length[i],
            extend_seq_len=extend_length[i],
            items_offset=items_offset,
        )
        # remove this item from cache if chunk reaches to the end
        if end_index == embedding_per_req.shape[0]:
            embedding_cache.free(embedding_items_hash)
        embedding_list.append(embedding_per_req_chunk)
    embedding = torch.concat(embedding_list, dim=0)
    # 2. Check the embedding
    if embedding.dim() == 2:
        num_mm_tokens_in_embedding = embedding.shape[0]
    else:
        num_mm_tokens_in_embedding = embedding.shape[0] * embedding.shape[1]

    # the mask of multimodal tokens from input_ids
    special_multimodal_mask = torch.isin(
        input_ids,
        placeholder_tensor,
    ).unsqueeze(-1)

    num_mm_tokens_in_input_ids = special_multimodal_mask.sum()
    assert (
        num_mm_tokens_in_input_ids == num_mm_tokens_in_embedding
    ), f"expected {num_mm_tokens_in_input_ids}, got {num_mm_tokens_in_embedding}"
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
    placeholder_token_ids: List[int] = None,
) -> Optional[torch.Tensor]:
    """
    Calculate the multimodal embeddings if necessary, then scatter the result with the help of a boolean mask denoting the embed locations

        Args:
            placeholder_token_ids: denoting the token of multimodal data in input_ids.
                If none, the pad_values of multimodal items are used

        Returns:
            final embedding: Optional[torch.Tensor]
    """

    if mm_inputs_list is None:
        return None

    # 1. Calculate the multimodal data which exists in input_ids, with the help of pad_values
    # we assume that multimodal data are represented with its pad_values in input_ids
    item_flatten_list = []
    for mm_inputs in mm_inputs_list:
        item_flatten_list += [item for item in mm_inputs.mm_items if item is not None]

    placeholder_token_ids = placeholder_token_ids or [
        item.pad_value for item in item_flatten_list
    ]

    placeholder_tensor = torch.tensor(placeholder_token_ids, device=input_ids.device)

    embeddings, masks = [], []

    # 2. Get multimodal embedding separately
    # TODO: make this more generic
    # Try get image embedding if any
    if (
        any(True for item in item_flatten_list if item.is_image())
        and image_data_embedding_func
    ):
        items = [item for item in item_flatten_list if item.is_image()]
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
        items_offsets = []
        # calculate per request items length offset
        items_size = torch.zeros(len(mm_inputs_list + 1))
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
    image_data_embedding_func: Callable[
        [List[MultimodalDataItem]], torch.Tensor
    ] = None,
    audio_data_embedding_func: Callable[
        [List[MultimodalDataItem]], torch.Tensor
    ] = None,
    placeholder_token_ids: List[int] = None,
    **kwargs,
) -> torch.Tensor:
    """
    A general wrapper function to get final input embeds from multimodal models with a language model as causal model

        Args:
            placeholder_token_ids (List[int]): the ids of mm data placeholder tokens
            image_data_embedding_func : the function returning the image embedding
            audio_data_embedding_func : the function returning the image embedding

        Returns:
            inputs_embedding
            forwarded hidden states

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
            placeholder_token_ids=placeholder_token_ids,
        )
        # once used, mm_inputs is useless
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
