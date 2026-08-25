import re

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
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_component_precision
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class VocoderLoader(PlainStateDictComponentLoader):
    component_names = ["vocoder"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        config = self.load_component_config(component_model_path, component_name)
        component_weights_path = self.resolve_component_weights_path(
            component_model_path, server_args, component_name
        )
        class_name = config.pop("_class_name", None) or self.component_architecture
        assert (
            class_name is not None
        ), "Vocoder class name must be available from component config or pipeline config."

        server_args.model_paths[component_name] = component_model_path

        from sglang.multimodal_gen.configs.models.vocoder.ltx_vocoder import (
            LTXVocoderConfig,
        )

        vocoder_config = LTXVocoderConfig()
        vocoder_config.update_model_arch(config)

        resolved_vocoder_dtype = resolve_component_precision(server_args, "vocoder")
        vocoder_dtype = (
            resolved_vocoder_dtype
            if resolved_vocoder_dtype is not None
            else PRECISION_TO_TYPE["fp32"]
        )

        component_starts_on_cpu = server_args.should_start_component_on_cpu(
            component_name
        )
        target_device = self.target_device(component_starts_on_cpu)

        with set_default_torch_dtype(vocoder_dtype), skip_init_modules():
            vocoder_cls, _ = ModelRegistry.resolve_model_cls(class_name)
            vocoder = vocoder_cls(vocoder_config).to(target_device)

        loaded = load_safetensors_state_dict(component_weights_path)
        mapping = vocoder_config.arch_config.param_names_mapping
        loaded = {_remap_vocoder_key(k, mapping): v for k, v in loaded.items()}

        missing_keys, unexpected_keys = vocoder.load_state_dict(loaded, strict=False)
        # A half-loaded vocoder produces plausible but wrong audio.
        if missing_keys or unexpected_keys:
            raise ValueError(
                f"Vocoder weights at '{component_weights_path}' do not match the "
                f"instantiated {class_name}. Missing: {sorted(missing_keys)}. "
                f"Unexpected: {sorted(unexpected_keys)}."
            )
        return vocoder


def _remap_vocoder_key(key: str, param_names_mapping: dict[str, str]) -> str:
    # Applied in order, not first-match: one key can need several rules.
    for pattern, replacement in param_names_mapping.items():
        key = re.sub(pattern, replacement, key)
    return key
