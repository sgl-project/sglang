# Test fixture for external model loading tests
from .custom_qwen2_vl import (
    CustomProcessor,
    EntryClass,
    Qwen2VLForConditionalGeneration,
)

__all__ = ["Qwen2VLForConditionalGeneration", "CustomProcessor", "EntryClass"]
