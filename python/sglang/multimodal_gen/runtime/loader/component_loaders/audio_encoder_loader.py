from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.precision_types import PRECISION_TO_TYPE


class AudioEncoderLoader(ComponentLoader):
    component_names = ["audio_encoder"]
    expected_library = "transformers"

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
        # No layerwise hooks: record it before placing, so a layerwise "all" selection
        # resolves this component to its non-layerwise placement instead of the host.
        server_args.record_component_layerwise_capability(
            component_name, supported=False
        )
        target_device = self.target_device(
            server_args.should_start_component_on_cpu(component_name)
        )
        return model_cls(
            config=encoder_config,
            component_model_path=component_model_path,
            dtype=PRECISION_TO_TYPE[
                server_args.pipeline_config.audio_encoder_precision
            ],
            target_device=target_device,
        )
