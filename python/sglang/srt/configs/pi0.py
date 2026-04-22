# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""HuggingFace-compatible config for the π0 VLA model.

Registered with SGLang's ``_CONFIG_REGISTRY`` so that ``model_type: "pi0"``
in ``config.json`` is recognized before ``sglang.srt.models.pi0`` is imported.
"""

from typing import Dict, Optional, Tuple

from transformers import PretrainedConfig


class Pi0Config(PretrainedConfig):
    """Config surface for the π0 VLA model.

    Only the parameters that actually shape runtime behaviour are stored here;
    the transformer dimensions (hidden size, head count, etc.) are derived
    from ``paligemma_variant`` / ``action_expert_variant`` inside the model
    via ``get_gemma_config``. See ``sglang.srt.models.pi0`` for details.
    """

    model_type = "pi0"

    def __init__(
        self,
        # Backbone variants — the model maps these to Gemma dimensions via
        # ``get_gemma_config``. ``gemma_2b`` + ``gemma_300m`` is the stock
        # π0 pairing; ``_7b`` / ``_27b`` would change the VLM at the cost of
        # compute.
        paligemma_variant: str = "gemma_2b",
        action_expert_variant: str = "gemma_300m",
        # Action chunk shape.
        chunk_size: int = 50,
        max_action_dim: int = 32,
        max_state_dim: int = 32,
        # Flow-matching denoising schedule.
        num_inference_steps: int = 10,
        # Image preprocessing. Consumed by sglang.srt.multimodal.processors.pi0.
        image_resolution: Tuple[int, int] = (224, 224),
        # Per-dataset normalization stats, optional. Shape matches LeRobot's
        # ``NormalizerProcessorStep``; see ``_build_norm_buffers`` in
        # ``sglang.srt.models.pi0`` for the expected schema.
        norm_stats: Optional[Dict] = None,
        # Weight dtype the checkpoint was saved in; parity tests pass this
        # through to match LeRobot's ``PI0Config.dtype``.
        dtype: str = "float32",
        # Plumbing-only: ModelConfig._derive_model_shapes expects these.
        # They point to the PaliGemma LM (Gemma 2B) dimensions by default.
        # The model itself builds configs from get_gemma_config(variant), so
        # these are not used at inference time.
        hidden_size: int = 2048,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 18,
        num_key_value_heads: int = 1,
        vocab_size: int = 257152,  # PaliGemma vocab (256k tokens + image token)
        **kwargs,
    ):
        if "architectures" not in kwargs:
            kwargs["architectures"] = ["Pi0ForActionPrediction"]
        super().__init__(**kwargs)
        self.paligemma_variant = paligemma_variant
        self.action_expert_variant = action_expert_variant
        self.chunk_size = chunk_size
        self.max_action_dim = max_action_dim
        self.max_state_dim = max_state_dim
        self.num_inference_steps = num_inference_steps
        self.norm_stats = norm_stats
        self.dtype = dtype
        # π0 / SigLIP only support square inputs; fail fast rather than
        # silently dropping the width downstream.
        if (
            not isinstance(image_resolution, (tuple, list))
            or len(image_resolution) != 2
            or image_resolution[0] != image_resolution[1]
        ):
            raise ValueError(
                f"π0 expects a square image_resolution (H == W); got "
                f"{image_resolution!r}."
            )
        self.image_resolution = tuple(image_resolution)
        # Satisfy ModelConfig._derive_model_shapes (not used by model).
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
