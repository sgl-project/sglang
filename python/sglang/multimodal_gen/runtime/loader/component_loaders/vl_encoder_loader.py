import logging
from typing import Any

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import get_hf_config

logger = logging.getLogger(__name__)
import requests


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
                health_url = server_args.srt_encoder_url.rstrip("/") + "/health"
                try:
                    logger.info(f"Checking AR encoder server health at: {health_url}")
                    response = requests.get(
                        health_url, timeout=server_args.srt_encoder_connect_timeout
                    )

                    if response.status_code != 200:
                        error_msg = (
                            f"AR encoder server returned unhealthy status code: {response.status_code}. "
                            f"Please ensure the server at {server_args.srt_encoder_url} is fully initialized and compatible."
                        )
                        logger.error(error_msg)
                        raise RuntimeError(error_msg)
                    logger.info("Successfully connected to AR encoder server.")
                except requests.RequestException as e:
                    error_msg = (
                        f"Failed to reach AR encoder server at {server_args.srt_encoder_url}. "
                        f"Error: {e}."
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e
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
