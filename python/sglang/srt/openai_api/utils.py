import logging
import re

logger = logging.getLogger(__name__)

# MAPPING OF REGEX PATTERNS
# Extend and maintain this mapping as needed...
# see sglang/srt/conversation.py
VISION_MODEL_TEMPLATE_MAP = {
    r"llama": "llama_3_vision",
    r"qwen": "qwen2-vl",
    r"gemma": "gemma-it",
    r"minicpm": "minicpmv",
    r"deepseek": "deepseek-vl2",
    r"llava_llama": "llava_llama_3",
    r"llava": "chatml-llava",
}


def auto_select_chat_template(model_type: str) -> str:
    """
    Returns the chat template name based on the model_type.
    If no match is found, returns an empty string.
    """
    lower_model_type = model_type.lower()
    logger.info(f"auto_select_chat_template: Received model_type '{lower_model_type}'.")
    for pattern, template in VISION_MODEL_TEMPLATE_MAP.items():
        if re.search(pattern, lower_model_type):
            logger.info(
                f"auto_select_chat_template: Matched pattern '{pattern}', returning '{template}'."
            )
            return template
    logger.warning(
        f"auto_select_chat_template: No match found for model_type '{lower_model_type}'."
    )
    return ""
