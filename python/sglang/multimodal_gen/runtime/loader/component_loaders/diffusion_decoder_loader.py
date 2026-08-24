# SPDX-License-Identifier: Apache-2.0

from sglang.multimodal_gen.configs.models.decoders.ltx_2_5_diffusion_decoder import (
    LTX25DiffusionDecoderConfig,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    PlainStateDictComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    load_safetensors_state_dict,
    set_default_torch_dtype,
    skip_init_modules,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision


class DiffusionDecoderLoader(PlainStateDictComponentLoader):
    """Loader for the standalone, replicated LTX-2.5 diffusion decoder."""

    component_names = ["diffusion_decoder"]
    expected_library = "diffusers"

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str = "diffusion_decoder",
        *args,
    ):
        config = self.load_component_config(component_model_path, component_name)
        component_weights_path = self.resolve_component_weights_path(
            component_model_path, server_args, component_name
        )
        class_name = config.pop("_class_name", None)
        if class_name is None:
            raise ValueError(
                "Model config does not contain a _class_name attribute. "
                "Only diffusers format is supported."
            )
        config.pop("_diffusers_version", None)
        config.pop("_name_or_path", None)

        server_args.model_paths[component_name] = component_model_path
        model_cls, _ = ModelRegistry.resolve_model_cls(class_name)
        target_device = self.target_device(
            server_args.should_start_component_on_cpu(component_name)
        )
        dtype = resolve_precision(
            server_args, component_name, precision_attr="vae_precision"
        )

        decoder_config = LTX25DiffusionDecoderConfig()
        decoder_config.update_model_arch(config)
        with set_default_torch_dtype(dtype), skip_init_modules():
            model = model_cls(decoder_config).to(device=target_device, dtype=dtype)

        model.load_state_dict(
            load_safetensors_state_dict(component_weights_path), strict=True
        )
        return model
