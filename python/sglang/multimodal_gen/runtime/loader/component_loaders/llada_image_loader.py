# SPDX-License-Identifier: Apache-2.0

from typing import ClassVar

from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE


class LLaDAImageAuxiliaryLoader(ComponentLoader):
    """Load the QueryFormer, text projection, and SigVQ modules."""

    component_names: ClassVar[list[str]] = ["queryformer", "text_projection", "sigvq"]
    expected_library = "diffusers"

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
    ):
        config = get_diffusers_component_config(component_path=component_model_path)
        class_name = config.pop("_class_name", None)
        config.pop("_diffusers_version", None)
        if class_name is None:
            raise ValueError(
                f"{component_name} config does not contain a _class_name attribute"
            )

        model_cls, _ = ModelRegistry.resolve_model_cls(class_name)
        dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]
        with set_default_torch_dtype(dtype):
            model = model_cls(**config)

        safetensors_list = _list_safetensors_files(component_model_path)
        if len(safetensors_list) != 1:
            raise ValueError(
                f"Expected one safetensors file for {component_name}, "
                f"found {len(safetensors_list)}"
            )
        state_dict = safetensors_load_file(safetensors_list[0])
        model.load_state_dict(state_dict, strict=True)
        server_args.model_paths[component_name] = component_model_path
        return model.to(device=get_local_torch_device(), dtype=dtype).eval()
