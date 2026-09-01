import re

from sglang.multimodal_gen.configs.models.adapter.ltx_2_connector import (
    LTX2ConnectorConfig,
)
from sglang.multimodal_gen.configs.models.adapter.ltx_2_duration_head import (
    LTX2DurationHeadConfig,
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


class AdapterLoader(PlainStateDictComponentLoader):
    """Loader for small adapter-style modules (e.g., LTX-2 connectors).

    This loader intentionally avoids FSDP sharding and just:
    1) Instantiates the module from `config.json`.
    2) Loads the safetensors state_dict (single-file or sharded).
    """

    component_names = ["connectors", "duration_head"]
    expected_library = "diffusers"

    # `update_model_arch` fills each from the component's `config.json`.
    _CONFIG_CLASSES = {
        "connectors": LTX2ConnectorConfig,
        "duration_head": LTX2DurationHeadConfig,
    }

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str = "connectors",
        *args,
    ):
        config = self.load_component_config(component_model_path, component_name)
        component_weights_path = self.resolve_component_weights_path(
            component_model_path, server_args, component_name
        )

        cls_name = config.pop("_class_name", None)
        if cls_name is None:
            raise ValueError(
                "Model config does not contain a _class_name attribute. "
                "Only diffusers format is supported."
            )

        config.pop("_diffusers_version", None)
        config.pop("_name_or_path", None)

        server_args.model_paths[component_name] = component_model_path

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        # Not a fixed name: connectors follow DiT offload, while the duration
        # head stays resident unless selected explicitly.
        target_device = self.target_device(
            server_args.should_start_component_on_cpu(component_name)
        )
        default_dtype = resolve_precision(
            server_args, component_name, precision_attr="dit_precision"
        )

        component_type = self.structural_component_type(component_name)
        config_cls = self._CONFIG_CLASSES[component_type]
        with set_default_torch_dtype(default_dtype), skip_init_modules():
            adapter_cfg = config_cls()
            adapter_cfg.update_model_arch(config)
            model = model_cls(adapter_cfg).to(device=target_device, dtype=default_dtype)

        loaded = load_safetensors_state_dict(component_weights_path)
        mapping = adapter_cfg.arch_config.param_names_mapping
        loaded = {_remap_connector_key(k, mapping): v for k, v in loaded.items()}

        missing, unexpected = model.load_state_dict(loaded, strict=False)
        # `strict=False` because a checkpoint carries either the shared
        # `text_proj_in` or the per-modality projections, never both. Anything
        # else uninitialized would surface later as garbage embeddings.
        if missing or unexpected:
            raise ValueError(
                f"Adapter weights at '{component_weights_path}' do not match the "
                f"instantiated {cls_name}. Missing: {sorted(missing)}. "
                f"Unexpected: {sorted(unexpected)}. This usually means the "
                "adapter config or its weight-name mapping is wrong."
            )

        return model


def _remap_connector_key(key: str, param_names_mapping: dict[str, str]) -> str:
    for pattern, replacement in param_names_mapping.items():
        key, replaced = re.subn(pattern, replacement, key)
        if replaced:
            break
    return key
