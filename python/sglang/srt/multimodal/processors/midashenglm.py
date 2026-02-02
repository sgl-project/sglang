import logging
import re

import torch

from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.midashenglm import MiDashengLMModel
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)

logger = logging.getLogger(__name__)


class MiDashengLMMultimodalProcessor(BaseMultimodalProcessor):
    """Multimodal processor for MiDashengLM audio-language model."""

    models = [MiDashengLMModel]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)

        self.AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>"
        )

        tokenizer = self._processor.tokenizer
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        self.ATTR_NAME_TO_MODALITY.update(
            {
                "input_values": Modality.AUDIO,
                "audio_length": Modality.AUDIO,
            }
        )

        if "input_values" not in self.FEATURE_NAMES:
            self.FEATURE_NAMES.append("input_values")

    def process_mm_data(
        self, input_text, images=None, videos=None, audios=None, **kwargs
    ):
        """Override to use correct audio parameter name for MiDashengLM processor."""
        if images:
            kwargs["images"] = images
        if videos:
            kwargs["videos"] = videos
        if audios:
            kwargs["audio"] = audios
            kwargs["audio_kwargs"] = {}
            kwargs["audio_kwargs"].setdefault("truncation", False)

        processor = self._processor
        result = processor.__call__(
            text=[input_text],
            padding=True,
            return_tensors="pt",
            **kwargs,
        )

        if not getattr(self.server_args, "keep_mm_feature_on_device", False):
            for feature_name in ["input_values"]:
                if feature_name in result:
                    result[feature_name] = result[feature_name].cpu()

        return result

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ):
        """Process audio data for MiDashengLM model.

        Args:
            audio_data: Audio input data
            input_text: Text prompt
            **kwargs: Additional arguments

        Returns:
            Dictionary containing processed multimodal data
        """
        logger.info("=" * 80)
        logger.info("process_mm_data_async called")
        logger.info(f"audio_data is not None: {audio_data is not None}")
        logger.info(f"input_text: {input_text}")
        logger.info("=" * 80)

        if audio_data and not self.AUDIO_TOKEN_REGEX.search(input_text):
            input_text = f"{self.AUDIO_TOKEN}{input_text}"
            logger.info("Auto-prepended audio token")

        base_output = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            logger.info("base_output is None")
            return None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        logger.info(f"mm_items count: {len(mm_items)}")
        logger.info(f"ret keys: {list(ret.keys())}")
        logger.info(f"input_ids shape: {input_ids.shape}")
        logger.info(
            f"audio_token_id={self.audio_token_id}, audio_start_id={self.audio_start_id}, audio_end_id={self.audio_end_id}"
        )
        logger.info(
            f"Count of audio_token_id in input_ids: {(input_ids == self.audio_token_id).sum().item()}"
        )
        for i, item in enumerate(mm_items):
            logger.info(f"mm_item[{i}] modality: {item.modality}")
            logger.info(
                f"mm_item[{i}] pad_value: {getattr(item, 'pad_value', 'NOT SET')}"
            )
            logger.info(f"mm_item[{i}] offsets: {getattr(item, 'offsets', 'NOT SET')}")
            logger.info(f"mm_item[{i}] has feature: {hasattr(item, 'feature')}")
            if hasattr(item, "feature") and item.feature is not None:
                logger.info(f"mm_item[{i}] feature shape: {item.feature.shape}")

        if "audio_length" in ret and len(mm_items) > 0:
            audio_length = ret["audio_length"]
            if isinstance(audio_length, torch.Tensor):
                audio_length = (
                    audio_length.item()
                    if audio_length.numel() == 1
                    else audio_length[0].item()
                )
            mm_items[0].audio_length = audio_length
            logger.info(
                f"Set audio_length={audio_length} (from processor, mel frame count)"
            )
        elif "input_values" in ret and len(mm_items) > 0:
            input_values = ret["input_values"]
            audio_length = (
                input_values.shape[-1]
                if input_values.ndim >= 2
                else input_values.shape[0]
            )
            mm_items[0].audio_length = audio_length
            logger.info(f"Set audio_length={audio_length} (fallback, waveform length)")

        result = {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
        }
        logger.info(f"Returning {len(result['mm_items'])} mm_items")
        return result
