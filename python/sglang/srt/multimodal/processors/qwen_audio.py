import re

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.qwen2_audio import Qwen2AudioForConditionalGeneration
from sglang.srt.multimodal.processors.base_processor import (
    BaseMultimodalProcessor,
    MultimodalSpecialTokens,
)


class Qwen2AudioMultimodalProcessor(BaseMultimodalProcessor):
    models = [Qwen2AudioForConditionalGeneration]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(hf_config, server_args, _processor, *args, **kwargs)
        self.AUDIO_TOKEN = "<|audio_bos|><|AUDIO|><|audio_eos|>"
        self.AUDIO_TOKEN_REGEX = re.compile(
            r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>"
        )
        # Collect special token ids
        tokenizer = self._processor.tokenizer
        self.audio_start_id = tokenizer.convert_tokens_to_ids("<|audio_bos|>")
        self.audio_token_id = tokenizer.convert_tokens_to_ids("<|AUDIO|>")
        self.audio_end_id = tokenizer.convert_tokens_to_ids("<|audio_eos|>")

        self.mm_tokens = MultimodalSpecialTokens(
            audio_token=self.AUDIO_TOKEN,
            audio_token_regex=self.AUDIO_TOKEN_REGEX,
            audio_token_id=self.audio_token_id,
        ).build(_processor)

        self.ATTR_NAME_TO_MODALITY.update({"feature_attention_mask": Modality.AUDIO})

    def get_mm_data(self, prompt, embeddings, **kwargs):
        audio_feature_lens = kwargs.get("audio_feature_lens", None)

        # Convert audio_feature_lens to token counts for build_input_ids
        output_lengths = None
        input_lengths = None
        if audio_feature_lens is not None:
            if audio_feature_lens.dim() > 1:
                audio_feature_lens = audio_feature_lens.flatten()
            input_lengths = (audio_feature_lens - 1) // 2 + 1
            output_lengths = (input_lengths - 2) // 2 + 1

        input_ids, offsets, modality_list = self.build_input_ids(
            prompt,
            audio_seq_lens=output_lengths,
        )

        mm_items = []
        embedding_index = 0
        for modality, offset in zip(modality_list, offsets):
            start_idx, end_idx = offset
            num_tokens = end_idx - start_idx + 1
            embedding_slice = embeddings[embedding_index : embedding_index + num_tokens]
            embedding_index += num_tokens
            mm_items.append(
                MultimodalDataItem(
                    modality=modality,
                    offsets=offset,
                    precomputed_embeddings=embedding_slice,
                )
            )

        mm_items[0].audio_feature_lens = output_lengths

        return {
            "mm_items": mm_items,
            "input_ids": input_ids,
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
        }

    async def process_mm_data_async(
        self,
        audio_data,
        input_text,
        **kwargs,
    ):
        base_output = self.load_mm_data(
            prompt=input_text,
            audio_data=audio_data,
            multimodal_tokens=self.mm_tokens,
        )
        if base_output is None:
            return None

        mm_items, input_ids, ret = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )

        assert (
            "feature_attention_mask" in ret
        ), "feature_attention_mask not found in processor output"
        input_lengths = ret["feature_attention_mask"].sum(dim=-1)
        input_lengths = (input_lengths - 1) // 2 + 1
        output_lengths = (input_lengths - 2) // 2 + 1

        mm_items[0].audio_feature_lens = output_lengths

        return {
            "mm_items": mm_items,
            "input_ids": input_ids.tolist(),
            "audio_start_id": self.audio_start_id,
            "audio_token_id": self.audio_token_id,
            "audio_end_id": self.audio_end_id,
        }
