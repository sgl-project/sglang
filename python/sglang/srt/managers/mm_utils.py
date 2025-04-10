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
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import print_warning_once
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


def get_embedding_and_mask(
    data_embedding_func: Callable[[List[MultimodalDataItem]], torch.Tensor],
    embedding_items: List[MultimodalDataItem],
    placeholder_tensor: torch.Tensor,
    input_ids: torch.Tensor,
):
    """
    Get the multimodal embedding and its mask from input_ids

    """
    # 1. Get the embedding
    embedding = data_embedding_func(embedding_items)

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

    num_mm_tokens_in_input_ids = special_multimodal_mask.sum().item()

    if num_mm_tokens_in_input_ids != num_mm_tokens_in_embedding:
        logger.warning(
            f"Number of tokens in multimodal embedding does not match those in the input text."
            f"Got {num_mm_tokens_in_input_ids} tokens in the text but {num_mm_tokens_in_embedding} "
            "tokens from multimodal embeddings."
        )
        if num_mm_tokens_in_input_ids < num_mm_tokens_in_embedding:
            # TODO: chunked prefill will split special tokens from input_ids into several passes, failing the embedding
            # a fix may be cache the unfinished multimodal embedding for future reuse, determine the tokens to embed with
            # extend_start_loc and extend_seq_lens
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

    return embedding, special_multimodal_mask


def embed_mm_inputs(
    mm_inputs: MultimodalInputs,
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

    if mm_inputs is None:
        return None

    # 1. Calculate the multimodal data which exists in input_ids, with the help of pad_values
    # we assume that multimodal data are represented with its pad_values in input_ids
    placeholder_token_ids = placeholder_token_ids or [
        item.pad_value for item in mm_inputs.mm_items
    ]

    placeholder_tensor = torch.tensor(placeholder_token_ids, device=input_ids.device)

    placeholder_masks = torch.isin(input_ids, placeholder_tensor)

    appearing_pad_values = torch.unique(
        input_ids[placeholder_masks], return_counts=False
    )

    if appearing_pad_values.numel() == 0:
        # all been prefixed
        inputs_embeds = input_embedding(input_ids)
    else:
        appearing_items = [
            item
            for item in mm_inputs.mm_items
            if item.pad_value is not None and item.pad_value in appearing_pad_values
        ]

        using_all_items = False
        if len(appearing_items) == 0:
            # This happens mostly when arg placeholder_token_ids is passed
            logger.warning_once(
                "No multimodal data item's pad value exist in placeholder ids. Using all items"
            )
            using_all_items = True
            appearing_items = mm_inputs.mm_items

        embeddings, masks = [], []

        # 2. Get multimodal embedding separately
        # TODO: make this more generic
        # Try get image embedding if any
        if (
            any(True for item in appearing_items if item.is_image())
            and image_data_embedding_func
        ):
            items = [item for item in appearing_items if item.is_image()]
            embedding, mask = get_embedding_and_mask(
                data_embedding_func=image_data_embedding_func,
                embedding_items=items,
                placeholder_tensor=(
                    placeholder_tensor
                    if using_all_items
                    else torch.tensor(
                        [item.pad_value for item in items],
                        device=input_ids.device,
                    )
                ),
                input_ids=input_ids,
            )
            embeddings += [embedding]
            masks += [mask]

        # Try get audio embedding if any
        if (
            any(True for item in appearing_items if item.is_audio())
            and audio_data_embedding_func
        ):
            items = [item for item in appearing_items if item.is_audio()]
            embedding, mask = get_embedding_and_mask(
                data_embedding_func=audio_data_embedding_func,
                embedding_items=items,
                placeholder_tensor=(
                    placeholder_tensor
                    if using_all_items
                    else torch.tensor(
                        [item.pad_value for item in items],
                        device=input_ids.device,
                    )
                ),
                input_ids=input_ids,
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
        mm_input = forward_batch.merge_mm_inputs()
        inputs_embeds = embed_mm_inputs(
            mm_inputs=mm_input,
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
