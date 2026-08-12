import json
import os
import re

from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen.configs.models.adapter.ltx_2_connector import (
    LTX2ConnectorConfig,
)
from sglang.multimodal_gen.configs.models.adapter.ltx_2_duration_head import (
    LTX2DurationHeadConfig,
)
from sglang.multimodal_gen.configs.models.vaes.ltx_2_5_diffusion_decoder import (
    LTX25DiffusionDecoderConfig,
)
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
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
from sglang.multimodal_gen.runtime.utils.precision import resolve_precision


class AdapterLoader(ComponentLoader):
    """Loader for small adapter-style modules (e.g., LTX-2 connectors).

    This loader intentionally avoids FSDP sharding and just:
    1) Instantiates the module from `config.json`.
    2) Loads the safetensors state_dict (single-file or sharded).
    """

    component_names = ["connectors", "duration_head", "diffusion_decoder"]
    expected_library = "diffusers"

    # Each adapter carries its own arch config; `update_model_arch` then fills it
    # straight from the component's `config.json`.
    _CONFIG_CLASSES = {
        "connectors": LTX2ConnectorConfig,
        "duration_head": LTX2DurationHeadConfig,
        "diffusion_decoder": LTX25DiffusionDecoderConfig,
    }

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str = "connectors",
        *args,
    ):
        config = get_diffusers_component_config(component_path=component_model_path)

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

        target_device = get_local_torch_device()
        default_dtype = resolve_precision(
            server_args, component_name, precision_attr="dit_precision"
        )

        config_cls = self._CONFIG_CLASSES[component_name]
        with set_default_torch_dtype(default_dtype), skip_init_modules():
            adapter_cfg = config_cls()
            adapter_cfg.update_model_arch(config)
            model = model_cls(adapter_cfg).to(device=target_device, dtype=default_dtype)

        loaded = self._load_connector_state_dict(component_model_path)
        mapping = adapter_cfg.arch_config.param_names_mapping
        loaded = {_remap_connector_key(k, mapping): v for k, v in loaded.items()}

        missing, unexpected = model.load_state_dict(loaded, strict=False)
        # `strict=False` is needed because a checkpoint carries either the shared
        # `text_proj_in` or the per-modality projections, never both. Anything
        # else left uninitialized would surface much later as garbage
        # embeddings, so fail loudly here instead.
        if missing or unexpected:
            raise ValueError(
                f"Adapter weights at '{component_model_path}' do not match the "
                f"instantiated {cls_name}. Missing: {sorted(missing)}. "
                f"Unexpected: {sorted(unexpected)}. This usually means the "
                "adapter config or its weight-name mapping is wrong."
            )

        return model

    @staticmethod
    def _load_connector_state_dict(component_model_path: str) -> dict:
        """Read the connector weights, single-file or sharded.

        LTX-2.0 ships one file; LTX-2.5 ships a 2-shard set *alongside* a
        single-file copy of the same weights, so prefer the index when it is
        present and fall back to the lone file otherwise.
        """
        index_path = os.path.join(
            component_model_path, "diffusion_pytorch_model.safetensors.index.json"
        )
        if os.path.exists(index_path):
            with open(index_path) as f:
                shards = sorted(set(json.load(f)["weight_map"].values()))
            state_dict: dict = {}
            for shard in shards:
                state_dict.update(
                    safetensors_load_file(os.path.join(component_model_path, shard))
                )
            return state_dict

        safetensors_list = _list_safetensors_files(component_model_path)
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {component_model_path}")
        if len(safetensors_list) != 1:
            raise ValueError(
                f"Found {len(safetensors_list)} safetensors files in "
                f"{component_model_path} and no index to disambiguate them."
            )
        return safetensors_load_file(safetensors_list[0])


def _remap_connector_key(key: str, param_names_mapping: dict[str, str]) -> str:
    for pattern, replacement in param_names_mapping.items():
        key, replaced = re.subn(pattern, replacement, key)
        if replaced:
            break
    return key
