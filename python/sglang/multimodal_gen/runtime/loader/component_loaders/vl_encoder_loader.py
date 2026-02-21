import logging
import multiprocessing
import os
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

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        transformers_or_diffusers: str = "vision_language_encoder",
    ) -> Any:
        if transformers_or_diffusers == "vision_language_encoder":
            from sglang import Engine

            model_root = os.path.dirname(component_model_path)
            processor_path = os.path.join(model_root, "processor")

            # sglang.Engine spawns scheduler/detokenizer subprocesses, but
            # multimodal_gen GPU workers run as daemon processes which cannot
            # create children. Temporarily clear the daemon flag to allow
            # subprocess creation, then restore it.
            current = multiprocessing.current_process()
            was_daemon = current._config.get("daemon", False)
            if was_daemon:
                logger.info(
                    "Temporarily clearing daemon flag to allow "
                    "sglang.Engine subprocess creation"
                )
                current._config["daemon"] = False
            try:
                engine = Engine(
                    model_path=component_model_path,
                    tokenizer_path=processor_path,
                    mem_fraction_static=0.8,
                    enable_multimodal=True,
                    disable_cuda_graph=True,
                    tp_size=server_args.tp_size if server_args.tp_size > 0 else 1,
                )
            finally:
                current._config["daemon"] = was_daemon
            return engine
        else:
            raise ValueError(
                f"Unsupported library for VisionLanguageEncoder: {transformers_or_diffusers}"
            )
