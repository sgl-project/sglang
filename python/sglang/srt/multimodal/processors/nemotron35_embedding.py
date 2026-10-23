from sglang.srt.models.nemotron35_embedding import EmbeddingNemotron35Model
from sglang.srt.multimodal.processors.pixtral import PixtralProcessor


class Nemotron35EmbeddingProcessor(PixtralProcessor):
    models = [EmbeddingNemotron35Model]

    def process_mm_data(self, input_text, **kwargs) -> dict:
        result = super().process_mm_data(input_text, **kwargs)
        # The tokenizer pads to a multiple of 8; there is no attention mask
        # downstream, so trailing pads would join the mean pool.
        keep = result["attention_mask"][0].bool()
        result["input_ids"] = result["input_ids"][:, keep]
        result["attention_mask"] = result["attention_mask"][:, keep]
        return result
