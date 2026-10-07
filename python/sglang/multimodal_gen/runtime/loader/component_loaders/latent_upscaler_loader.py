# SPDX-License-Identifier: Apache-2.0
"""Loader of the ``latent_upscaler`` component of an official Kandinsky 6 SR repo."""

import torch
from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import _list_safetensors_files
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class LatentUpscalerLoader(ComponentLoader):
    """Builds the bank from ``config.json`` and loads its weights strictly.

    Every failure (unparsable model config, missing or unexpected keys) surfaces as an
    error: this component has no native diffusers fallback, so the base class must not
    swallow the customized-load error and try ``AutoModel``.
    """

    component_names = ["latent_upscaler"]
    # The official ``save_pretrained`` writes "kandinsky6", the Hub repo "diffusers".
    expected_library = ("diffusers", "kandinsky6")

    def should_raise_customized_load_error(
        self, server_args: ServerArgs, component_name: str
    ) -> bool:
        return True

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        config = get_diffusers_component_config(component_path=component_model_path)
        class_name = config.pop("_class_name", None)
        if class_name is None:
            raise ValueError(f"{component_model_path}/config.json has no _class_name")
        bank_cls, _ = ModelRegistry.resolve_model_cls(class_name)
        bank = self._build_bank(bank_cls, config, component_model_path)

        state_dict: dict[str, torch.Tensor] = {}
        for path in _list_safetensors_files(component_model_path):
            state_dict.update(safetensors_load_file(path))
        if not state_dict:
            raise FileNotFoundError(f"no safetensors files in {component_model_path}")
        bank.load_state_dict(state_dict, strict=True)

        target_device = self.target_device(
            server_args.should_start_component_on_cpu(component_name)
        )
        # The reference runs the LU in bf16 (weights and non-persistent buffers).
        return (
            bank.eval()
            .requires_grad_(False)
            .to(device=target_device, dtype=torch.bfloat16)
        )

    @staticmethod
    def _build_bank(bank_cls, config: dict, component_model_path: str):
        for key in ("models", "scaling_factor"):
            if key not in config:
                raise ValueError(f"{component_model_path}/config.json lacks {key!r}")
        try:
            return bank_cls(
                models=config["models"],
                scaling_factor=config["scaling_factor"],
                scales=config.get("scales", (2, 4)),
            )
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(
                f"invalid latent upscaler config in {component_model_path}: "
                f"{type(error).__name__}: {error}"
            ) from error
