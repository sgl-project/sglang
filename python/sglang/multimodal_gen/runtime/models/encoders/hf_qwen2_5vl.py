from transformers import Qwen2_5_VLForConditionalGeneration

from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
    LayerwiseOffloadableModuleMixin,
)


class LayerwiseQwen2_5VLForConditionalGeneration(
    Qwen2_5_VLForConditionalGeneration,
    LayerwiseOffloadableModuleMixin,
):
    """Hugging Face Qwen2.5-VL with explicit layerwise residency support."""

    layerwise_offload_dit_group_enabled = False
    layer_names = ["model.language_model.layers"]
