import logging
from typing import Any

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

logger = logging.getLogger(__name__)


class VisionLanguageEncoderLoader(ComponentLoader):
    """Loader for vision language encoder via SGLang Engine."""

    component_names = ["vision_language_encoder"]
    expected_library = "transformers"
    engine = None

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        transformers_or_diffusers: str = "vision_language_encoder",
    ) -> Any:
        if transformers_or_diffusers == "vision_language_encoder":

            if server_args.srt_encoder_url is not None:
                return server_args.srt_encoder_url

            # from sglang.srt.models.glm_image import GlmImageForConditionalGeneration
            # model_root = os.path.dirname(component_model_path)
            # processor_path = os.path.join(model_root, "processor")
            #
            # model = GlmImageForConditionalGeneration(
            #    config,
            # )
            #
            # return model

            raise ValueError("Unsupported yet")

        else:
            raise ValueError(
                f"Unsupported library for VisionLanguageEncoder: {transformers_or_diffusers}"
            )
