import logging
from typing import Any

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import get_hf_config

logger = logging.getLogger(__name__)


class VisionLanguageEncoderLoader(ComponentLoader):
    """Loader for vision language encoder (typically Causal LM or Vision2Seq)."""

    component_names = ["vision_language_encoder"]
    expected_library = "transformers"

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        transformers_or_diffusers: str = "vision_language_encoder",
    ) -> Any:
        if transformers_or_diffusers == "vision_language_encoder":

            if server_args.srt_encoder_url is not None:
                logger.info(
                    f"Use {server_args.srt_encoder_url} as a backend for AR model"
                    "make sure that a server with the appropriate model is running at this address."
                )
                return server_args.srt_encoder_url

            from transformers import GlmImageForConditionalGeneration

            config = get_hf_config(
                component_model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
            model = GlmImageForConditionalGeneration.from_pretrained(
                component_model_path,
                config=config,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            ).to(get_local_torch_device())
            return model
        else:
            raise ValueError(
                f"Unsupported library for VisionLanguageEncoder: {transformers_or_diffusers}"
            )
