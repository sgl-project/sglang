import importlib.util
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file

from sglang.multimodal_gen import envs
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


def _convert_conv3d_weights_to_channels_last_3d(module: nn.Module) -> int:
    """
    Convert Conv3d weights to channels_last_3d (NDHWC) memory format.
    Returns the number of Conv3d modules converted.
    """
    if not hasattr(torch, "channels_last_3d"):
        return 0
    num_converted = 0
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            try:
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)
                num_converted += 1
            except Exception:
                # Best-effort; skip unsupported cases.
                continue
    return num_converted


class VAELoader(ComponentLoader):
    """Shared loader for (video/audio) VAE modules."""

    component_names = ["vae", "audio_vae", "video_vae"]
    expected_library = "diffusers"

    def should_offload(
        self, server_args: ServerArgs, model_config: ModelConfig | None = None
    ):
        return server_args.vae_cpu_offload

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        """Load the VAE based on the model path, and inference args."""
        config = get_diffusers_component_config(component_path=component_model_path)
        class_name = config.pop("_class_name", None)
        assert (
            class_name is not None
        ), "Model config does not contain a _class_name attribute. Only diffusers format is supported."

        server_args.model_paths[component_name] = component_model_path

        logger.debug("HF model config: %s", config)
        if component_name in ("vae", "video_vae"):
            pipeline_vae_config_attr = "vae_config"
            pipeline_vae_precision = "vae_precision"
        elif component_name in ("audio_vae",):
            pipeline_vae_config_attr = "audio_vae_config"
            pipeline_vae_precision = "audio_vae_precision"
        else:
            raise ValueError(
                f"Unsupported module name for VAE loader: {component_name}"
            )
        vae_config = getattr(server_args.pipeline_config, pipeline_vae_config_attr)
        vae_precision = getattr(server_args.pipeline_config, pipeline_vae_precision)
        vae_config.update_model_arch(config)
        if hasattr(vae_config, "post_init"):
            # NOTE: some post init logics are only available after updated with config
            vae_config.post_init()

        should_offload = self.should_offload(server_args)
        target_device = self.target_device(should_offload)

        # Check for auto_map first (custom VAE classes)
        auto_map = config.get("auto_map", {})
        auto_model_map = auto_map.get("AutoModel")
        if auto_model_map:
            module_path, cls_name = auto_model_map.rsplit(".", 1)
            custom_module_file = os.path.join(component_model_path, f"{module_path}.py")
            spec = importlib.util.spec_from_file_location("_custom", custom_module_file)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
            vae_cls = getattr(custom_module, cls_name)
            vae_dtype = PRECISION_TO_TYPE[vae_precision]
            with set_default_torch_dtype(vae_dtype):
                vae = vae_cls.from_pretrained(
                    component_model_path,
                    revision=server_args.revision,
                    trust_remote_code=server_args.trust_remote_code,
                )
            vae = vae.to(device=target_device, dtype=vae_dtype)
            if (
                component_name in ("vae", "video_vae")
                and torch.cuda.is_available()
                and getattr(envs, "SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D", False)
            ):
                n = _convert_conv3d_weights_to_channels_last_3d(vae)
                if n > 0:
                    logger.info(
                        "VAE: converted %d Conv3d weights to channels_last_3d", n
                    )
            return vae

        # Load from ModelRegistry (standard VAE classes)
        with _model_construction_lock, set_default_torch_dtype(
            PRECISION_TO_TYPE[vae_precision]
        ), skip_init_modules():
            vae_cls, _ = ModelRegistry.resolve_model_cls(class_name)
            vae = vae_cls(vae_config).to(target_device)

        safetensors_list = _list_safetensors_files(component_model_path)
        assert (
            len(safetensors_list) == 1
        ), f"Found {len(safetensors_list)} safetensors files in {component_model_path}"
        loaded = safetensors_load_file(safetensors_list[0])
        vae.load_state_dict(loaded, strict=False)

        state_keys = set(vae.state_dict().keys())
        loaded_keys = set(loaded.keys())
        missing_keys = sorted(state_keys - loaded_keys)
        unexpected_keys = sorted(loaded_keys - state_keys)
        if missing_keys:
            logger.warning("VAE missing keys: %s", missing_keys)
        if unexpected_keys:
            logger.warning("VAE unexpected keys: %s", unexpected_keys)

        if (
            component_name in ("vae", "video_vae")
            and torch.cuda.is_available()
            and getattr(envs, "SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D", False)
        ):
            n = _convert_conv3d_weights_to_channels_last_3d(vae)
            if n > 0:
                logger.info("VAE: converted %d Conv3d weights to channels_last_3d", n)

        return vae
