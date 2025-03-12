from abc import abstractmethod
from typing import Callable, List, Optional, Tuple

from sglang.srt.managers.schedule_batch import ImageInputs
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
            print(f"image_cnt {image_cnt}")
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
