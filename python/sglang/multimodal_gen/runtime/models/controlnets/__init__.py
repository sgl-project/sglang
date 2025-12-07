# SPDX-License-Identifier: Apache-2.0

# Use InstantX implementation (matches pretrained weights)
from sglang.multimodal_gen.runtime.models.controlnets.controlnet_qwenimage_instantx import (
    QwenImageControlNetModel,
)

# Keep our custom implementation available for reference/future use
from sglang.multimodal_gen.runtime.models.controlnets.qwen_image_controlnet import (
    QwenImageControlNetModel as QwenImageControlNetModelCustom,
)

__all__ = ["QwenImageControlNetModel", "QwenImageControlNetModelCustom"]
