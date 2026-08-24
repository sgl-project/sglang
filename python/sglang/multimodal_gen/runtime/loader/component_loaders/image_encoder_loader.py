from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
    _resolve_and_configure_encoder_quantization,
)
from sglang.multimodal_gen.runtime.models.encoders.base import finalize_encoder_folding
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class ImageEncoderLoader(TextEncoderLoader):
    component_names = ["image_encoder"]
    expected_library = "transformers"

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str = "image_encoder",
    ):
        """Load the text encoders based on the model path, and inference args."""
        component_weights_path = self.resolve_model_weights_path(
            component_model_path,
            server_args,
            component_name,
        )
        # model_config: PretrainedConfig = get_hf_config(
        #     model=model_path,
        #     trust_remote_code=server_args.trust_remote_code,
        #     revision=server_args.revision,
        #     model_override_args=None,
        # )
        model_config = get_diffusers_component_config(
            component_path=component_model_path
        )

        encoder_config = server_args.pipeline_config.image_encoder_config
        encoder_config.update_model_arch(model_config)
        _resolve_and_configure_encoder_quantization(
            encoder_config,
            model_config,
            component_model_path,
            component_weights_path,
            component_name,
        )
        # real dims are populated now; resolve fold vs replicate
        finalize_encoder_folding(
            encoder_config,
            server_args.encoder_parallel,
        )

        # Always start with local device; load_model will adjust for offload if needed
        # TODO(will): add support for other dtypes
        return self.load_model(
            component_weights_path,
            encoder_config,
            server_args,
            server_args.pipeline_config.image_encoder_precision,
            component_name=component_name,
        )
