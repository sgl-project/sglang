# SPDX-License-Identifier: Apache-2.0
from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen.configs.models import ModelConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
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


class SoundTokenizerLoader(ComponentLoader):
    component_names = ["sound_tokenizer"]
    expected_library = "diffusers"

    def should_offload(
        self, server_args: ServerArgs, model_config: ModelConfig | None = None
    ) -> bool:
        return server_args.vae_cpu_offload

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        config = get_diffusers_component_config(component_path=component_model_path)
        class_name = config.pop("_class_name", None) or self.component_architecture
        assert (
            class_name is not None
        ), "Sound tokenizer class name must be available from component config."

        server_args.model_paths[component_name] = component_model_path

        try:
            precision = server_args.pipeline_config.vae_precision
        except AttributeError:
            precision = "bf16"
        dtype = PRECISION_TO_TYPE[precision]
        target_device = self.target_device(self.should_offload(server_args))

        with set_default_torch_dtype(dtype), skip_init_modules():
            model_cls, _ = ModelRegistry.resolve_model_cls(class_name)
            model = model_cls(config).to(target_device)

        safetensors_list = _list_safetensors_files(component_model_path)
        assert (
            len(safetensors_list) == 1
        ), f"Found {len(safetensors_list)} safetensors files in {component_model_path}"
        loaded = safetensors_load_file(safetensors_list[0])
        incompatible = model.load_state_dict(loaded, strict=False)
        missing = getattr(incompatible, "missing_keys", [])
        # The tokenizer is decoder-only; the checkpoint's encoder weights are
        # expected leftovers, so they're excluded from the load warning.
        unexpected = [
            k
            for k in getattr(incompatible, "unexpected_keys", [])
            if not k.startswith("encoder.")
        ]
        if missing or unexpected:
            logger.warning(
                "Loaded sound_tokenizer with missing_keys=%d unexpected_keys=%d",
                len(missing),
                len(unexpected),
            )
        model.eval()
        return model
