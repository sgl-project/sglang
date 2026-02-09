import json
import os

from sglang.multimodal_gen.configs.models import ModelConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import _clean_hf_config_inplace
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ImageEncoderLoader(TextEncoderLoader):

    component_names = ["image_encoder"]
    expected_library = "transformers"

    def should_offload(self, server_args, model_config: ModelConfig | None = None):
        should_offload = server_args.image_encoder_cpu_offload
        if not should_offload:
            return False
        # _fsdp_shard_conditions is in arch_config, not directly on model_config
        arch_config = (
            getattr(model_config, "arch_config", model_config) if model_config else None
        )
        fsdp_shard_conditions = (
            getattr(arch_config, "_fsdp_shard_conditions", []) if arch_config else []
        )
        use_cpu_offload = should_offload and len(fsdp_shard_conditions) > 0
        return use_cpu_offload

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
    ):
        """Load the text encoders based on the model path, and inference args."""
        # model_config: PretrainedConfig = get_hf_config(
        #     model=model_path,
        #     trust_remote_code=server_args.trust_remote_code,
        #     revision=server_args.revision,
        #     model_override_args=None,
        # )
        with open(os.path.join(component_model_path, "config.json")) as f:
            model_config = json.load(f)
        _clean_hf_config_inplace(model_config)
        logger.debug("HF model config: %s", model_config)

        encoder_config = server_args.pipeline_config.image_encoder_config
        encoder_config.update_model_arch(model_config)

        # Always start with local device; load_model will adjust for offload if needed
        # TODO(will): add support for other dtypes
        return self.load_model(
            component_model_path,
            encoder_config,
            server_args,
            server_args.pipeline_config.image_encoder_precision,
            cpu_offload_flag=server_args.image_encoder_cpu_offload,
        )
