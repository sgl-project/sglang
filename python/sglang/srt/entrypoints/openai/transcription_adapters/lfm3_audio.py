from sglang.srt.entrypoints.openai.transcription_adapters.base import (
    register_transcription_adapter,
)
from sglang.srt.entrypoints.openai.transcription_adapters.qwen2_audio import (
    Qwen2AudioAdapter,
)


@register_transcription_adapter("Lfm3AudioForConditionalGeneration")
class Lfm3AudioAdapter(Qwen2AudioAdapter):
    """LFM3 transcription prompting with decoder-only sampling and responses."""

    def build_prompt(self, request, tokenizer) -> str:
        if request.timestamp_granularities:
            raise ValueError("LFM3-Audio does not support timestamp_granularities.")
        system_prompt = (
            f"Perform ASR in {request.language}."
            if request.language is not None
            else "Perform ASR."
        )
        return tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "<|audio|>"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    @property
    def supports_chunked_streaming(self) -> bool:
        return True

    @property
    def chunked_streaming_config(self) -> dict:
        return {
            "chunk_size_sec": 0.5,
            "unfixed_chunk_num": 2,
            "unfixed_token_num": 5,
        }

    @property
    def prompt_template(self) -> str:
        return (
            "<|startoftext|><|im_start|>system\nPerform ASR.<|im_end|>\n"
            "<|im_start|>user\n<|audio|><|im_end|>\n"
            "<|im_start|>assistant\n"
        )
