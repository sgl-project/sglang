# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SGLang Team
"""SGLang multimodal processor for Cohere2Vision (Command-A-Vision)."""

import json
import logging
import os
from typing import Dict, List, Union

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.cohere2_vision import Cohere2VisionForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens

logger = logging.getLogger(__name__)


def _maybe_unwrap_nested_image_processor_config(
    image_processor, model_path: str
) -> None:
    """Some Cohere2Vision checkpoints (notably ``command-a-plus-05-2026-bf16``)
    ship a ``preprocessor_config.json`` where the image-processor settings are
    wrapped under an ``"image_processor"`` key instead of being at the root::

        {
          "image_processor": { "image_mean": [0.5,0.5,0.5], ... },
          "processor_class": "Cohere2VisionProcessor"
        }

    HuggingFace's ``AutoImageProcessor.from_pretrained`` only reads root-level
    fields, so it silently falls back to the class's ``OPENAI_CLIP_*`` defaults
    (``mean = (0.481, 0.458, 0.408)``, ``std = (0.269, 0.261, 0.276)``) — the
    wrong normalization for a Siglip-trained model. This drops MMMU by a couple
    of points vs the FP8 / re-packaged sibling checkpoints whose configs use
    the flat schema.

    We detect this case after the HF loader has run and overlay the nested
    fields onto the loaded image processor in-place. Affects both vLLM and
    sglang equally upstream, but fixing it locally here lets sglang produce
    the correct vision tower input without re-packing the checkpoint.

    NOTE: This is a temporary bypass to paper over divergent preprocessor_config
    schemas across the published Cohere2Vision model variants (the bf16 release
    uses the nested layout above, while the fp8 / w4a4 releases ship the flat
    HF-canonical layout). Remove this helper once Cohere republishes a
    consistent flat schema on every variant.
    """
    if not model_path:
        return
    cfg_path = None
    if os.path.isdir(model_path):
        candidate = os.path.join(model_path, "preprocessor_config.json")
        if os.path.exists(candidate):
            cfg_path = candidate
    else:
        # Repo-id form. Resolve through the HF hub cache.
        try:
            from huggingface_hub import try_to_load_from_cache

            cached = try_to_load_from_cache(model_path, "preprocessor_config.json")
            if cached and os.path.exists(cached):
                cfg_path = cached
        except Exception as e:
            logger.debug(f"preprocessor_config cache lookup failed: {e}")
    if cfg_path is None:
        return
    try:
        with open(cfg_path, "r") as fh:
            raw = json.load(fh)
    except Exception as e:
        logger.debug(f"preprocessor_config sniff failed: {e}")
        return
    nested = raw.get("image_processor")
    if not isinstance(nested, dict):
        return
    # Heuristic: only apply when the on-disk nested config actually disagrees
    # with what the HF loader produced. Compare a couple of small fields.
    fixed = False
    for attr in ("image_mean", "image_std"):
        if attr not in nested:
            continue
        nested_val = tuple(nested[attr])
        loaded_val = tuple(getattr(image_processor, attr, ()))
        if nested_val != loaded_val:
            setattr(image_processor, attr, nested_val)
            fixed = True
    # ``size`` / ``patch_size`` / ``crop_to_patches`` / etc. would also live
    # under the same nested block; in practice the HF loader picks them up
    # from root fallbacks correctly. We only overlay normalization here.
    if fixed:
        logger.warning(
            "Cohere2Vision: preprocessor_config.json uses nested "
            '"image_processor" wrapping; overlaid image_mean/image_std from '
            f"the nested block ({nested.get('image_mean')}, "
            f"{nested.get('image_std')}) to override the CLIP fallbacks the HF "
            "AutoImageProcessor would otherwise apply."
        )


class Cohere2VisionSGLangImageProcessor(SGLangBaseProcessor):
    models = [Cohere2VisionForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        # Cohere2Vision uses three special tokens around an image:
        #   <|START_OF_IMG|> [<|IMG_PATCH|> * P^2 + <|IMG_LINE_BREAK|>] * N <|END_OF_IMG|>
        # where N = number of image patches and P = patch_size.  The Hugging
        # Face processor expands the placeholder ``<|IMG_PATCH|>`` to the full
        # patch sequence; we just need to recognise the expanded form here so
        # SGLang can pad the input ids properly.
        proc = _processor

        # Fix up the malformed nested preprocessor_config.json shipped with
        # some checkpoints before any image goes through the tower.
        _maybe_unwrap_nested_image_processor_config(
            proc.image_processor, getattr(server_args, "model_path", None)
        )
        # Resolve token strings + ids from the HF processor's tokenizer.
        boi_token = proc.boi_token
        eoi_token = proc.eoi_token
        image_token = proc.image_token  # "<|IMG_PATCH|>"
        line_break_token = proc.img_line_break_token

        self.image_token_id = proc.image_token_id
        self.boi_token_id = proc.tokenizer.convert_tokens_to_ids(boi_token)
        self.eoi_token_id = proc.tokenizer.convert_tokens_to_ids(eoi_token)
        self.img_line_break_token_id = proc.tokenizer.convert_tokens_to_ids(
            line_break_token
        )

        # The chat template emits a single ``<|IMG_PATCH|>`` placeholder per
        # image; the HF Cohere2VisionProcessor expands it into the full
        # ``<|START_OF_IMG|> [<|IMG_PATCH|> * P^2 + <|IMG_LINE_BREAK|>] * N
        # <|END_OF_IMG|>`` block once it knows each image's patch count. Match
        # the unexpanded placeholder here so SGLang correctly pairs each
        # placeholder with its image_data entry before processing.
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=image_token,
            image_token_id=self.image_token_id,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: List[Union[str, bytes, Dict]],
        input_text,
        request_obj,
        *args,
        **kwargs,
    ):
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
            discard_alpha_channel=True,
        )

        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.image_token_id,
            im_start_id=self.boi_token_id,
            im_end_id=self.eoi_token_id,
        )
