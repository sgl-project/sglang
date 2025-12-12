# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""
Fast-dLLM v2 model for SGLang.

This is a minimal wrapper that:
1. Registers the model architecture so SGLang can load it
2. Marks the model as Fast_dLLM so we can use special handling

The actual generation logic uses the model's native generate() method
from the HuggingFace trust_remote_code implementation.
"""

import logging
from typing import Optional

from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.qwen2 import Qwen2ForCausalLM

logger = logging.getLogger(__name__)


class FastDLLMForCausalLM(Qwen2ForCausalLM):
    """
    Fast-dLLM v2 model wrapper for SGLang.

    This model is loaded as Qwen2 but marked as Fast_dLLM.
    Generation should use the model's native generate() method.
    """

    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config=quant_config, prefix=prefix)

        # Fast_dLLM specific config
        self.bd_size = getattr(config, "bd_size", 32)
        self.mask_token_id = getattr(config, "mask_token_id", 151665)

        logger.info(
            f"FastDLLM initialized: bd_size={self.bd_size}, "
            f"mask_token_id={self.mask_token_id}"
        )

    @staticmethod
    def is_fast_dllm() -> bool:
        """Marker to identify Fast_dLLM models."""
        return True


# Register with HuggingFace architecture name
FastDLLMForCausalLM.__name__ = "Fast_dLLM_QwenForCausalLM"

EntryClass = FastDLLMForCausalLM
