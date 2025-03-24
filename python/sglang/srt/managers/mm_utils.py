"""
    Multimodality utils
"""

from abc import abstractmethod
from typing import Callable, List, Optional, Tuple

import torch
from torch import nn

from sglang.srt.managers.schedule_batch import (
    ImageInputs,
    global_server_args_dict,
    logger,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.utils import logger


class MultiModalityDataPaddingPattern:
    """
    Data tokens (like image tokens) often need special handling during padding
    to maintain model compatibility. This class provides the interface for
    implementing different padding strategies for data tokens
    """

    @abstractmethod
    def pad_input_tokens(
        self, input_ids: List[int], image_inputs: ImageInputs
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
        self, input_ids: List[int], image_inputs: ImageInputs
    ) -> List[int]:
        """
        This function will replace the data-tokens inbetween with pad_values accordingly
        """
        pad_values = image_inputs.pad_values
        data_token_pairs = self.data_token_id_pairs
        image_inputs.image_offsets = []
        if data_token_pairs is None:
            data_token_pairs = [image_inputs.im_start_id, image_inputs.im_end_id]
        if data_token_pairs is None:
            logger.warning(
                "No data_token_pairs provided, RadixAttention might be influenced."
            )
            return input_ids
        start_token_ids = [s for s, _e in data_token_pairs]
        end_tokens_ids = [e for _s, e in data_token_pairs]
        # First start token marks new data
        data_start_token = start_token_ids[0]

        padded_ids = []
        last_idx = 0
        data_idx = -1

        start_indices = [i for i, x in enumerate(input_ids) if x in start_token_ids]
        end_indices = [i for i, x in enumerate(input_ids) if x in end_tokens_ids]

        if len(start_indices) != len(end_indices):
            return input_ids

        for start_idx, end_idx in zip(start_indices, end_indices):
            padded_ids.extend(input_ids[last_idx : start_idx + 1])

            if input_ids[start_idx] == data_start_token:
                data_idx += 1
                image_inputs.image_offsets += [start_idx]

            num_tokens = end_idx - start_idx - 1
            pad_value = pad_values[data_idx]
            padded_ids.extend([pad_value] * num_tokens)

            last_idx = end_idx

        padded_ids.extend(input_ids[last_idx:])

        assert len(input_ids) == len(padded_ids)
        return padded_ids


class MultModalityDataPaddingPatternSingleToken(MultiModalityDataPaddingPattern):
    """In this pattern, data is represented with a special token_id ( image_inputs.im_token_id ),
         which needs first to be expanded to multiple tokens, then replaced with their padding values

    This strategy should be used when a single data token represents content that should
    be expanded to multiple tokens during processing.
    """

    def __init__(
        self, num_data_token_calc_func: Callable[[Tuple[int, int, int]], int]
    ) -> None:
        self.num_data_token_calc_func = num_data_token_calc_func

    def pad_input_tokens(
        self, input_ids: List[int], image_inputs: ImageInputs
    ) -> List[int]:
        """
        This function will follow the procedure of:
            1. the data token will be expanded, of which the final number will be calculated by `num_data_token_calc_func`
            2. the padded data tokens will be replaced with their pad_values
        """
        image_grid_thws = image_inputs.image_grid_thws
        pad_values = image_inputs.pad_values

        image_indices = [
            idx
            for idx, token in enumerate(input_ids)
            if token == image_inputs.im_token_id
        ]

        image_inputs.image_offsets = []

        input_ids_with_image = []
        for image_cnt, _ in enumerate(image_grid_thws):
            num_image_tokens = self.num_data_token_calc_func(image_grid_thws[image_cnt])
            if image_cnt == 0:
                non_image_tokens = input_ids[: image_indices[image_cnt]]
            else:
                non_image_tokens = input_ids[
                    image_indices[image_cnt - 1] + 1 : image_indices[image_cnt]
                ]
            input_ids_with_image.extend(non_image_tokens)
            image_inputs.image_offsets.append(len(input_ids_with_image))
            pad_ids = pad_values * (
                (num_image_tokens + len(pad_values)) // len(pad_values)
            )
            input_ids_with_image.extend(pad_ids[:num_image_tokens])
        input_ids_with_image.extend(input_ids[image_indices[-1] + 1 :])

        return input_ids_with_image


class MultiModalityDataPaddingPatternImageTokens(MultiModalityDataPaddingPattern):
    """In this pattern, data tokens should be represented as image tokens (e.g. <image><image>....<image>)"""

    def __init__(self, image_token_id: torch.Tensor) -> None:
        self.image_token_id = image_token_id

    def pad_input_tokens(self, input_ids: List[int], image_inputs) -> List[int]:
        """
        This function will replace the data-tokens in between with pad_values accordingly
        """
        pad_values = image_inputs.pad_values
        assert len(pad_values) != 0

        input_ids_tensor = torch.tensor(input_ids)
        mask = torch.isin(input_ids_tensor, self.image_token_id)

        num_image_tokens = mask.sum().item()
        repeated_pad_values = torch.tensor(pad_values).repeat(
            num_image_tokens // len(pad_values) + 1
        )[:num_image_tokens]

        input_ids_tensor[mask] = repeated_pad_values
        return input_ids_tensor.tolist()


def embed_image_inputs(
    image_input: ImageInputs,
    input_ids: torch.Tensor,
    input_embedding: nn.Embedding,
    image_embedding_func,
    placeholder_token_ids: List[int] = None,
) -> Optional[torch.Tensor]:
    """
    Calculate the image embeddings if necessary, then scatter the result with
    the help of a boolean mask denoting the embed locations

    Returns:
        final embedding: Optional[torch.Tensor]
    """
    if image_input is None:
        return None

    placeholder_token_ids = placeholder_token_ids or image_input.pad_values

    # boolean masking the special tokens
    special_image_mask = torch.isin(
        input_ids,
        torch.tensor(placeholder_token_ids, device=input_ids.device),
    ).unsqueeze(-1)

    num_image_tokens_in_input_ids = special_image_mask.sum()

    if num_image_tokens_in_input_ids == 0:
        # unexpected
        inputs_embeds = input_embedding(input_ids)
    else:
        image_embedding = image_embedding_func(image_input)

        if image_embedding.dim() == 2:
            num_image_tokens_in_embedding = image_embedding.shape[0]
        else:
            num_image_tokens_in_embedding = (
                image_embedding.shape[0] * image_embedding.shape[1]
            )
        if num_image_tokens_in_input_ids != num_image_tokens_in_embedding:
            num_image = num_image_tokens_in_input_ids // image_embedding.shape[1]
            image_embedding = image_embedding[:num_image, :]
            logger.warning(
                f"Number of images does not match number of special image tokens in the input text. "
                f"Got {num_image_tokens_in_input_ids} image tokens in the text but {num_image_tokens_in_embedding} "
                "tokens from image embeddings."
            )

            # TODO: chunked prefill will split special tokens from input_ids into several passes, failing the embedding
            # a fix may be cache the unfinished image embedding for future reuse, determine the tokens to embed with
            # extend_start_loc and extend_seq_lens
            if num_image_tokens_in_input_ids > num_image_tokens_in_embedding:
                chunked_prefill_size = global_server_args_dict["chunked_prefill_size"]
                if chunked_prefill_size != -1:
                    logger.warning(
                        "You may want to avoid this issue by raising `chunked_prefill_size`, or disabling chunked_prefill"
                    )

        vocab_size = input_embedding.num_embeddings
        # Important: clamp after getting original image regions
        # Clamp input ids. This is because the input_ids for the image tokens are
        # filled with the hash values of the image for the prefix matching in the radix attention.
        # There values are useless because their embeddings will be replaced by vision embeddings anyway.
        input_ids.clamp_(min=0, max=vocab_size - 1)
        inputs_embeds = input_embedding(input_ids)

        special_image_mask = special_image_mask.expand_as(inputs_embeds).to(
            inputs_embeds.device
        )

        inputs_embeds = inputs_embeds.masked_scatter(
            special_image_mask,
            image_embedding.to(inputs_embeds.device, inputs_embeds.dtype),
        )
    return inputs_embeds


def embed_image_embedding(
    inputs_embeds: torch.Tensor,
    image_embedding: torch.Tensor,
    image_bounds: torch.Tensor,
) -> torch.Tensor:
    """
    scatter image_embedding into inputs_embeds according to image_bounds
    """
    if len(image_bounds) > 0:
        image_indices = torch.stack(
            [
                torch.arange(start, end, dtype=torch.long)
                for start, end in image_bounds.tolist()
            ]
        ).to(inputs_embeds.device)

        inputs_embeds.scatter_(
            0,
            image_indices.view(-1, 1).repeat(1, inputs_embeds.shape[-1]),
            image_embedding.view(-1, image_embedding.shape[-1]),
        )
    return inputs_embeds


def general_mm_embed_routine(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    embed_tokens: nn.Embedding,
    image_embedding_func: Callable[[ImageInputs], torch.Tensor],
    placeholder_token_ids: List[int] = None,
):
    """
    a general wrapper function to get final input embeds from multimodal models
    with a language model as causal model
    """
    if (
        forward_batch.forward_mode.is_decode()
        or not forward_batch.contains_image_inputs()
    ):
        inputs_embeds = embed_tokens(input_ids)
    else:
        image = forward_batch.merge_image_inputs()
        inputs_embeds = embed_image_inputs(
            image_input=image,
            input_ids=input_ids,
            input_embedding=embed_tokens,
            image_embedding_func=image_embedding_func,
            placeholder_token_ids=placeholder_token_ids,
        )
        # once used, image_inputs is useless
        # just being defensive here
        forward_batch.image_inputs = None
    return inputs_embeds
