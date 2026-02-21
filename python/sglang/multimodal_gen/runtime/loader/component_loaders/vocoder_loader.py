from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen.configs.models import ModelConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    _model_construction_lock,
    set_default_torch_dtype,
    skip_init_modules,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class VocoderLoader(ComponentLoader):
    component_names = ["vocoder"]
    expected_library = "diffusers"

    def should_offload(
        self, server_args: ServerArgs, model_config: ModelConfig | None = None
    ):
        return server_args.vae_cpu_offload

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        config = get_diffusers_component_config(component_path=component_model_path)
        class_name = config.pop("_class_name", None)
        assert (
            class_name is not None
        ), "Model config does not contain a _class_name attribute. Only diffusers format is supported."

        server_args.model_paths[component_name] = component_model_path

        from sglang.multimodal_gen.configs.models.vocoder.ltx_vocoder import (
            LTXVocoderConfig,
        )

        vocoder_config = LTXVocoderConfig()
        vocoder_config.update_model_arch(config)

        try:
            vocoder_precision = server_args.pipeline_config.audio_vae_precision
        except AttributeError:
            vocoder_precision = "fp32"
        vocoder_dtype = PRECISION_TO_TYPE[vocoder_precision]

        should_offload = self.should_offload(server_args)
        target_device = self.target_device(should_offload)

        with _model_construction_lock, set_default_torch_dtype(
            vocoder_dtype
        ), skip_init_modules():
            vocoder_cls, _ = ModelRegistry.resolve_model_cls(class_name)
            vocoder = vocoder_cls(vocoder_config).to(target_device)

        safetensors_list = _list_safetensors_files(component_model_path)
        assert (
            len(safetensors_list) == 1
        ), f"Found {len(safetensors_list)} safetensors files in {component_model_path}"
        loaded = safetensors_load_file(safetensors_list[0])
        incompatible = vocoder.load_state_dict(loaded, strict=False)
        missing_keys = []
        unexpected_keys = []
        try:
            missing_keys = incompatible.missing_keys
            unexpected_keys = incompatible.unexpected_keys
        except AttributeError:
            # Best-effort fallback in case older torch returns a tuple-like.
            try:
                missing_keys = incompatible[0]
                unexpected_keys = incompatible[1]
            except Exception:
                pass

        if missing_keys or unexpected_keys:
            logger.warning(
                "Loaded vocoder with missing_keys=%d unexpected_keys=%d",
                len(missing_keys),
                len(unexpected_keys),
            )
        return vocoder
