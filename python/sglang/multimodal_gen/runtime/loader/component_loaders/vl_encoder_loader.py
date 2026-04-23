from typing import Any

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import get_hf_config


class VisionLanguageEncoderLoader(ComponentLoader):
    """Loader for vision language encoder (typically Causal LM or Vision2Seq)."""

    component_names = ["vision_language_encoder"]
    expected_library = "transformers"

    def should_offload(self, server_args, model_config=None) -> bool:
        """Check if the vision language encoder should be offloaded to CPU."""
        return bool(server_args.vision_language_encoder_cpu_offload)

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        transformers_or_diffusers: str = "vision_language_encoder",
    ) -> Any:
        if transformers_or_diffusers == "vision_language_encoder":
            from transformers import GlmImageForConditionalGeneration

            config = get_hf_config(
                component_model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )

            # Determine target device based on offload setting
            should_offload = self.should_offload(server_args)
            target_device = self.target_device(should_offload)

            model = GlmImageForConditionalGeneration.from_pretrained(
                component_model_path,
                config=config,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            ).to(target_device)
            return model
        else:
            raise ValueError(
                f"Unsupported library for VisionLanguageEncoder: {transformers_or_diffusers}"
            )
