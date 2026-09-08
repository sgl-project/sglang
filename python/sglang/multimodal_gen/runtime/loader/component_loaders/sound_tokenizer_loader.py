# SPDX-License-Identifier: Apache-2.0

import torch

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


class SoundTokenizerLoader(PlainStateDictComponentLoader):
    component_names = ["sound_tokenizer"]
    expected_library = "diffusers"

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        config = self.load_component_config(component_model_path, component_name)
        component_weights_path = self.resolve_component_weights_path(
            component_model_path, server_args, component_name
        )
        class_name = config.pop("_class_name", None) or self.component_architecture
        assert class_name is not None, (
            "Sound tokenizer class name must be available from component config."
        )

        server_args.model_paths[component_name] = component_model_path

        dtype = resolve_component_precision(server_args, component_name)
        if dtype is None:
            try:
                dtype = PRECISION_TO_TYPE[server_args.pipeline_config.vae_precision]
            except AttributeError:
                dtype = torch.bfloat16
        target_device = self.target_device(
            server_args.should_start_component_on_cpu(component_name)
        )

        with set_default_torch_dtype(dtype), skip_init_modules():
            model_cls, _ = ModelRegistry.resolve_model_cls(class_name)
            model = model_cls(config).to(device=target_device, dtype=dtype)

        loaded = load_safetensors_state_dict(component_weights_path)
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
