from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class ImageEncoderLoader(TextEncoderLoader):
    component_names = ["image_encoder"]
    expected_library = "transformers"

    def component_load_precision(
        self, server_args: ServerArgs, component_name: str
    ) -> str | None:
        return server_args.component_precisions.get(
            component_name, server_args.pipeline_config.image_encoder_precision
        )

    def build_model_config(
        self, component_model_path, model_config, server_args, component_name
    ):
        encoder_config = server_args.pipeline_config.image_encoder_config
        encoder_config.update_model_arch(model_config)
        return encoder_config
