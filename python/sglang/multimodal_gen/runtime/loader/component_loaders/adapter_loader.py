from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
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
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


class AdapterLoader(ComponentLoader):
    """Loader for small adapter-style modules (e.g., LTX-2 connectors).

    This loader intentionally avoids FSDP sharding and just:
    1) Instantiates the module from `config.json`.
    2) Loads a single safetensors state_dict.
    """

    component_names = ["connectors"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, *args
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

        server_args.model_paths["connectors"] = component_model_path

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        target_device = get_local_torch_device()
        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]

        from types import SimpleNamespace

        with _model_construction_lock, set_default_torch_dtype(
            default_dtype
        ), skip_init_modules():
            connector_cfg = SimpleNamespace(**config)
            model = model_cls(connector_cfg).to(
                device=target_device, dtype=default_dtype
            )

        safetensors_list = _list_safetensors_files(component_model_path)
        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {component_model_path}")
        if len(safetensors_list) != 1:
            raise ValueError(
                f"Found {len(safetensors_list)} safetensors files in {component_model_path}, expected 1"
            )

        loaded = safetensors_load_file(safetensors_list[0])
        model.load_state_dict(loaded, strict=False)

        return model
