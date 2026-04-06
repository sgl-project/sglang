from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


class AudioEncoderLoader(ComponentLoader):
    component_names = ["audio_encoder"]
    expected_library = "transformers"

    def should_offload(self, server_args, model_config=None):
        return bool(getattr(server_args, "audio_encoder_cpu_offload", False))

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        config = get_diffusers_component_config(component_path=component_model_path)
        cls_name = config.pop("_class_name", None)
        if cls_name is None:
            raise ValueError(
                f"Audio encoder config at {component_model_path} does not contain _class_name"
            )
        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)
        encoder_config = getattr(server_args.pipeline_config, "audio_encoder_config")
        encoder_config.update_model_arch(config)
        target_device = self.target_device(self.should_offload(server_args))
        return model_cls(
            config=encoder_config,
            component_model_path=component_model_path,
            dtype=PRECISION_TO_TYPE[
                server_args.pipeline_config.audio_encoder_precision
            ],
            target_device=target_device,
        )
