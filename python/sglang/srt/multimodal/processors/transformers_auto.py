from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalProcessorOutput,
)
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)
from sglang.srt.utils import load_image


def _first_attr(obj, names: tuple[str, ...], default=None):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _uses_mrope(hf_config) -> bool:
    text_config = getattr(hf_config, "text_config", hf_config)
    rope_scaling = getattr(text_config, "rope_scaling", None) or {}
    if isinstance(rope_scaling, dict) and "mrope_section" in rope_scaling:
        return True
    rope_type = str(getattr(text_config, "rope_type", "")).lower()
    return "mrope" in rope_type


class TransformersAutoMultimodalProcessor(BaseMultimodalProcessor):
    """Generic multimodal processor for the Transformers backend.

    Unlike model-specific processors that rely on regex-based token matching
    in the raw prompt, this processor applies the HF processor directly to
    the prompt text + raw media.  This handles models like Gemma3 where the
    chat template uses a marker (``<start_of_image>``) that the HF processor
    internally expands into placeholder tokens.
    """

    models = []

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.mm_tokens = MultimodalSpecialTokens(
            image_token=getattr(_processor, "image_token", None),
            video_token=getattr(_processor, "video_token", None),
            audio_token=getattr(_processor, "audio_token", None),
            image_token_id=_first_attr(
                hf_config,
                ("image_token_id", "image_token_index", "im_token_id"),
            ),
            video_token_id=_first_attr(
                hf_config,
                ("video_token_id",),
            ),
            audio_token_id=_first_attr(
                hf_config,
                ("audio_token_id", "audio_token_index"),
            ),
        ).build(_processor)

        self._is_mrope = _uses_mrope(hf_config)
        if self._is_mrope:
            vision_config = getattr(hf_config, "vision_config", None)
            self._spatial_merge_size = getattr(vision_config, "spatial_merge_size", 2)
            self._tokens_per_second = getattr(vision_config, "tokens_per_second", None)
            self._vision_start_token_id = _first_attr(
                hf_config, ("vision_start_token_id",)
            )
            self._model_type = getattr(hf_config, "model_type", "")

    def _compute_mrope_positions(
        self,
        input_ids: list[int],
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
    ):
        from sglang.srt.layers.rotary_embedding import MRotaryEmbedding

        input_ids_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
        mrope_positions, mrope_position_delta = MRotaryEmbedding.get_rope_index(
            spatial_merge_size=self._spatial_merge_size,
            image_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id or -1,
            vision_start_token_id=self._vision_start_token_id,
            model_type=self._model_type,
            input_ids=input_ids_tensor,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            tokens_per_second=self._tokens_per_second,
        )
        return mrope_positions.squeeze(1), mrope_position_delta

    def _load_images(self, image_data) -> list:
        """Download / decode images from URLs, file paths, or base64."""
        if not image_data:
            return []
        images = []
        for data in image_data:
            img, _ = load_image(data)
            if img.mode != "RGB":
                img = img.convert("RGB")
            images.append(img)
        return images

    def _apply_hf_processor(self, text: str, images=None, videos=None):
        """Run the HF processor on text + media and return the full output.

        This is the key method that makes the generic processor work for
        models with non-trivial token expansion (Gemma3, PaliGemma, etc.).
        The HF processor handles chat-template expansion, image token
        insertion, and tokenization in one shot.
        """
        kwargs = {}
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
        return self._processor(text=text, return_tensors="pt", **kwargs)

    def _build_mm_items(
        self, processor_output: dict, input_ids: torch.Tensor
    ) -> list[MultimodalDataItem]:
        """Extract MultimodalDataItem objects from the HF processor output."""
        items = self.collect_mm_items_from_processor_output(processor_output)

        modality_to_token_id = {
            Modality.IMAGE: self.mm_tokens.image_token_id,
            Modality.VIDEO: self.mm_tokens.video_token_id,
            Modality.AUDIO: self.mm_tokens.audio_token_id,
        }

        for item in items:
            token_id = modality_to_token_id.get(item.modality)
            if token_id is not None:
                item.offsets = self.get_mm_items_offset(input_ids, token_id)

        return items

    def _build_audio_prompt(self, input_text) -> str:
        """Ensure the prompt carries the model's audio placeholder token.

        Audio-only entrypoints (``/v1/audio/transcriptions``) send empty text,
        which would otherwise make the HF processor crash (empty ``text`` list)
        and leave no ``<|audio|>`` placeholder for the encoder features to fill.
        Build a minimal chat prompt with one audio token per clip.
        """
        if isinstance(input_text, list):
            input_text = (
                self._processor.tokenizer.decode(input_text) if input_text else ""
            )
        if (
            input_text
            and input_text.strip()
            and self.mm_tokens.audio_token in input_text
        ):
            return input_text

        audio_token = self.mm_tokens.audio_token
        content = (
            f"{audio_token}Transcribe the speech into written text. "
            f"Output only the exact transcription, with no preamble, commentary, "
            f"quotation marks, or explanation."
        )
        tokenizer = self._processor.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return content

    async def process_mm_data_async(
        self,
        image_data,
        audio_data,
        input_text,
        request_obj,
        **kwargs,
    ):
        video_data = getattr(request_obj, "video_data", None)
        if video_data is not None and not isinstance(video_data, list):
            video_data = [video_data]

        # Audio path: load clips and run the shared token-expansion machinery,
        # which passes ``audio=`` to the HF processor and packs audio features
        # into MultimodalDataItem objects with the right token offsets.
        if audio_data:
            prompt = self._build_audio_prompt(input_text)
            base_output = await self.load_mm_data(
                prompt=prompt,
                audio_data=audio_data,
                multimodal_tokens=self.mm_tokens,
            )
            if base_output is None:
                return None
            mm_items, input_ids, _ = self.process_and_combine_mm_data(
                base_output, self.mm_tokens
            )
            ret = MultimodalProcessorOutput(
                input_ids=input_ids.tolist(),
                mm_items=mm_items,
            )
            if self.mm_tokens.audio_token_id is not None:
                ret.audio_token_id = self.mm_tokens.audio_token_id
            return ret

        # Load raw media
        images = self._load_images(image_data)
        # TODO: video loading when needed

        # Apply HF processor — handles token expansion internally
        processor_output = self._apply_hf_processor(
            text=input_text,
            images=images or None,
            videos=video_data or None,
        )

        input_ids = processor_output["input_ids"].flatten()

        # Build mm_items from processor output
        mm_items = self._build_mm_items(processor_output, input_ids)

        ret = MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
        )

        # Propagate token_type_ids for models that need it (Gemma3, PaliGemma)
        token_type_key = (
            "mm_token_type_ids"
            if "mm_token_type_ids" in processor_output
            else "token_type_ids"
        )
        if token_type_key in processor_output:
            ret.token_type_ids = processor_output[token_type_key].flatten()

        if self.mm_tokens.image_token_id is not None:
            ret.im_token_id = self.mm_tokens.image_token_id
        if self.mm_tokens.video_token_id is not None:
            ret.video_token_id = self.mm_tokens.video_token_id
        if self.mm_tokens.audio_token_id is not None:
            ret.audio_token_id = self.mm_tokens.audio_token_id

        image_start_id = _first_attr(
            self.hf_config,
            ("image_start_token_id", "vision_start_token_id", "im_start_id"),
        )
        image_end_id = _first_attr(
            self.hf_config,
            ("image_end_token_id", "vision_end_token_id", "im_end_id"),
        )
        if image_start_id is not None:
            ret.im_start_id = image_start_id
        if image_end_id is not None:
            ret.im_end_id = image_end_id

        # M-RoPE positions (Qwen2.5-VL, Qwen3-VL)
        if self._is_mrope:
            image_grid_thw = processor_output.get("image_grid_thw")
            video_grid_thw = processor_output.get("video_grid_thw")
            mrope_positions, mrope_position_delta = self._compute_mrope_positions(
                ret.input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
            ret.mrope_positions = mrope_positions
            ret.mrope_position_delta = mrope_position_delta

        return ret
