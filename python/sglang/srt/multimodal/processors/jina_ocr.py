from sglang.srt.models.jina_ocr import JinaOCRForCausalLM
from sglang.srt.multimodal.processors.deepseek_ocr import DeepseekOCRProcessor


class JinaOCRProcessor(DeepseekOCRProcessor):
    models = [JinaOCRForCausalLM]
