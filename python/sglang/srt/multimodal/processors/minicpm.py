from typing import List, Union

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.minicpmo import MiniCPMO
from sglang.srt.models.minicpmv import MiniCPMV
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

def get_image_token_regex():
    import re
    """
    Generates a regular expression pattern to match two types of image tokens.

    The pattern supports:
    1. prec placeholder pattern: Tokens with <image_id>,
        followed by <image> content,
        and optionally 0+ <slice> elements (each may be followed by newlines).
    2. raw image pattern: Standalone parenthesized image tokens like (<image>./</image>).

    Returns:
        re.Pattern: Compiled regular expression object.
    """
    # Core components of the patterns
    image_id = r'<image_id>\d+</image_id>'          # Matches <image_id> with digits (e.g., <image_id>0</image_id>)
    image_content = r'<image>(?:<unk>)+</image>'    # Matches <image> with one or more <unk> (e.g., <image><unk><unk></image>)
    slice_content = r'<slice>(?:<unk>)+</slice>'    # Matches <slice> with one or more <unk> (e.g., <slice><unk></slice>)
    paren_image = r'\(<image>\./</image>\)'         # Matches parenthesized image (e.g., (<image>./</image>))

    # prec pattern: <image_id> + <image> + optional 0+ <slice> (with optional newlines after each slice)
    prec_pattern = (
        f'{image_id}{image_content}'                # Mandatory part: image_id + image content
        f'(?:{slice_content}\\n*)*'                 # Optional part: 0+ slices (each may have 0+ newlines after)
    )
    # Combine prec and raw patterns, return compiled regex
    return re.compile(f'(?:{prec_pattern}|{paren_image})')

# Compatible with both 'O' and 'V'
class MiniCPMMultimodalProcessor(BaseMultimodalProcessor):
    models = [MiniCPMV, MiniCPMO]

    def __init__(self, hf_config, server_args, _processor):
        super().__init__(hf_config, server_args, _processor)
        self.image_token = "(<image>./</image>)"
        self.image_token_regex = get_image_token_regex()
        self.audio_token = "(<audio>./</audio>)"
        self.video_token = "(<video>./</video>)"

        # Set token IDs for process_and_combine_mm_data
        tokenizer = _processor.tokenizer
        self.IM_TOKEN_ID = tokenizer.unk_id
        self.AUDIO_TOKEN_ID = getattr(tokenizer, "audio_token_id", None)


    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        ret = super().process_mm_data(input_text, images, videos, audios, **kwargs)
        ret['tgt_size'] = ret['tgt_sizes']
        if ( "audio_features" in ret
            and (ret["audio_features"] is None
            or len(ret["audio_features"]) == 0)
        ):
            ret.pop('audio_features')
            ret.pop('audio_feature_lens')
            ret.pop('audio_bounds')
            ret.pop('spk_bounds')
        return ret

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes]],
        audio_data: List[Union[str, bytes]],
        input_text,
        request_obj,
        max_req_input_len,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            max_req_input_len=max_req_input_len,
            audio_data=audio_data,
            image_data=image_data,
            multimodal_tokens=MultimodalSpecialTokens(
                image_token=self.image_token,
                video_token=self.video_token,
                image_token_regex=self.image_token_regex,
                audio_token=self.audio_token,
            ),
        )
        if base_output is None:
            return None

        # Collect special token ids
        tokenizer = self._processor.tokenizer
        slice_start_id, slice_end_id, audio_start_id, audio_end_id = (
            None,
            None,
            None,
            None,
        )
        if tokenizer.slice_start_id:
            slice_start_id = tokenizer.slice_start_id
            slice_end_id = tokenizer.slice_end_id
        if hasattr(tokenizer, "audio_start_id"):
            audio_start_id = tokenizer.audio_start_id
            audio_end_id = tokenizer.audio_end_id

        im_start_id = tokenizer.im_start_id
        im_end_id = tokenizer.im_end_id
        im_token_id = tokenizer.unk_id
        mm_items, input_ids = self.process_and_combine_mm_data(base_output)

        for item in mm_items:
            if item.modality in [Modality.IMAGE, Modality.MULTI_IMAGES]:
                pixel_values = item.pixel_values
                tgt_sizes = item.tgt_size
                if not isinstance(pixel_values, (torch.Tensor, list)):
                    raise ValueError(
                        "Incorrect type of pixel values. " f"Got type: {type(pixel_values)}"
                    )

                if not isinstance(tgt_sizes, (torch.Tensor, list)):
                    raise ValueError(
                        "Incorrect type of target sizes. " f"Got type: {type(tgt_sizes)}"
                    )

                if len(pixel_values) != len(tgt_sizes):
                    raise ValueError(
                        "Inconsistent batch lengths, found: "
                        f"{len(pixel_values)} vs. {len(tgt_sizes)}"
                    )

                pixel_values_flat: List[torch.Tensor] = []
                tgt_sizes_flat: List[torch.Tensor] = []
                for pixel_b, tgt_b in zip(pixel_values, tgt_sizes):
                    # per image
                    if len(pixel_b) != len(tgt_b):
                        raise ValueError(
                            "Inconsistent N lengths, found: " f"{len(pixel_b)} vs {len(tgt_b)}"
                        )
                    for pixel_n, tgt_n in zip(pixel_b, tgt_b):
                        pixel_values_flat += [pixel_n]
                        tgt_sizes_flat += [tgt_n]

                item.pixel_values = pixel_values_flat
                item.tgt_size = tgt_sizes_flat

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": audio_start_id,
            "audio_end_id": audio_end_id,
            "im_token_id": im_token_id,
            "im_start_id": im_start_id,
            "im_end_id": im_end_id,
            "slice_start_id": slice_start_id,
            "slice_end_id": slice_end_id,
        }
