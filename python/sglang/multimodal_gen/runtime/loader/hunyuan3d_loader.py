# SPDX-License-Identifier: Apache-2.0
"""Hunyuan3D specific component loader."""

import glob
import importlib
import os
from typing import Any, ClassVar, Dict

import torch

from sglang.multimodal_gen.runtime.loader.component_loader import ComponentLoader
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Hunyuan3DShapeLoader(ComponentLoader):
    """Loader for Hunyuan3D shape components."""

    # Class-level cache for loaded components
    _cache: ClassVar[Dict[str, Dict[str, Any]]] = {}

    @staticmethod
    def get_obj_from_str(string: str) -> Any:
        """Get an object from a module path string."""
        from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

        # Try to resolve via alias first (for external paths like hy3dshape.*, hy3dgen.*)
        model_cls = ModelRegistry.resolve_by_alias(string)
        if model_cls is not None:
            return model_cls

        # Extract class name from the full path and try direct lookup
        class_name = string.rsplit(".", 1)[-1]
        try:
            model_cls, _ = ModelRegistry.resolve_model_cls(class_name)
            return model_cls
        except Exception:
            pass

        # Try to resolve via hunyuan3d_utils (for ImageProcessorV2, MVImageProcessorV2, etc.)
        from sglang.multimodal_gen.runtime.models.hunyuan3d_utils import (
            resolve_hunyuan3d_tool,
        )

        # First try with the full target string
        tool_cls = resolve_hunyuan3d_tool(string)
        if tool_cls is not None:
            return tool_cls

        # Then try with just the class name
        tool_cls = resolve_hunyuan3d_tool(class_name)
        if tool_cls is not None:
            return tool_cls

        # Fallback to direct import (for non-registered classes)
        module, cls = string.rsplit(".", 1)
        return getattr(importlib.import_module(module, package=None), cls)

    @classmethod
    def instantiate_from_config(cls, config: Dict[str, Any], **kwargs) -> Any:
        """Instantiate an object from config with path remapping."""
        if "target" not in config:
            raise KeyError("Expected key 'target' to instantiate.")
        target_cls = cls.get_obj_from_str(config["target"])
        params = config.get("params", {})
        kwargs.update(params)
        return target_cls(**kwargs)

    def load_customized(self, component_model_path: str, server_args, module_name: str):
        """Load Hunyuan3D shape components."""
        import yaml

        from sglang.multimodal_gen.configs.pipeline_configs.hunyuan3d import (
            Hunyuan3D2PipelineConfig,
        )
        from sglang.multimodal_gen.runtime.distributed import get_local_torch_device

        config = server_args.pipeline_config
        if not isinstance(config, Hunyuan3D2PipelineConfig):
            raise ValueError(
                f"Hunyuan3DShapeLoader expects Hunyuan3D2PipelineConfig, got {type(config)}"
            )

        model_path = config.shape_model_path or server_args.model_path
        cache_key = (
            f"{model_path}:{config.shape_subfolder}:"
            f"{config.shape_use_safetensors}:{config.shape_variant}"
        )

        if cache_key not in self._cache:
            logger.info(f"Loading Hunyuan3D shape models from {model_path}")

            # Use internal smart_load_model implementation
            config_path, ckpt_path = self._smart_load_model(
                model_path,
                subfolder=config.shape_subfolder,
                use_safetensors=config.shape_use_safetensors,
                variant=config.shape_variant,
            )

            with open(config_path, "r") as f:
                model_config = yaml.safe_load(f)

            # Load checkpoint
            if config.shape_use_safetensors:
                import safetensors.torch

                safetensors_ckpt = safetensors.torch.load_file(ckpt_path, device="cpu")
                ckpt = {}
                for key, value in safetensors_ckpt.items():
                    model_name = key.split(".")[0]
                    new_key = key[len(model_name) + 1 :]
                    if model_name not in ckpt:
                        ckpt[model_name] = {}
                    ckpt[model_name][new_key] = value
            else:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

            # Instantiate and load models using internal implementations
            model = self.instantiate_from_config(model_config["model"])
            model.load_state_dict(ckpt["model"], strict=False)

            vae = self.instantiate_from_config(model_config["vae"])
            vae.load_state_dict(ckpt["vae"], strict=False)

            conditioner = self.instantiate_from_config(model_config["conditioner"])

            conditioner_params = sum(p.numel() for p in conditioner.parameters())

            cond_result = None
            conditioner_from_ckpt = "conditioner" in ckpt
            if conditioner_from_ckpt:
                conditioner.load_state_dict(ckpt["conditioner"], strict=False)

            image_processor = self.instantiate_from_config(
                model_config["image_processor"]
            )

            scheduler = self.instantiate_from_config(model_config["scheduler"])

            # Set device and dtype
            dtype = torch.float16
            if config.shape_variant and "bf16" in config.shape_variant:
                dtype = torch.bfloat16
            device = get_local_torch_device()

            for module in (model, vae, conditioner):
                module.to(device=device, dtype=dtype)
                module.eval()

            self._cache[cache_key] = {
                "hy3dshape_model": model,
                "hy3dshape_vae": vae,
                "hy3dshape_conditioner": conditioner,
                "hy3dshape_image_processor": image_processor,
                "hy3dshape_scheduler": scheduler,
            }

            logger.info(f"Hunyuan3D shape models loaded and cached (key: {cache_key})")

        return self._cache[cache_key][module_name]

    def _smart_load_model(
        self,
        model_path: str,
        subfolder: str,
        use_safetensors: bool = False,
        variant: str = "fp16",
    ):
        """Smart model loading with HuggingFace Hub support.

        Args:
            model_path: HuggingFace repo ID or local path
            subfolder: Subfolder within the model path
            use_safetensors: Whether to use safetensors format
            variant: Model variant (e.g., 'fp16', 'bf16')

        Returns:
            Tuple of (config_path, checkpoint_path)
        """
        base_dir = os.environ.get("HY3DGEN_MODELS", "~/.cache/hy3dgen")
        model_fld = os.path.expanduser(os.path.join(base_dir, model_path))
        local_path = os.path.expanduser(os.path.join(base_dir, model_path, subfolder))

        logger.info(f"Attempting to load model from: {local_path}")

        if not os.path.exists(local_path):
            logger.info("Local path not found, downloading from HuggingFace Hub")
            try:
                from huggingface_hub import snapshot_download

                path = snapshot_download(
                    repo_id=model_path,
                    allow_patterns=[f"{subfolder}/*"],
                    local_dir=model_fld,
                )
                local_path = os.path.join(path, subfolder)
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required for downloading models. "
                    "Install with: pip install huggingface_hub"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download model: {e}")

        # Find config and checkpoint files
        config_path = os.path.join(local_path, "config.yaml")
        if not os.path.exists(config_path):
            for alt in ["config.yml", "model_config.yaml"]:
                alt_path = os.path.join(local_path, alt)
                if os.path.exists(alt_path):
                    config_path = alt_path
                    break

        # Find checkpoint
        if use_safetensors:
            ckpt_name = f"model.safetensors"
            if variant:
                # Hunyuan3D uses "model.fp16.safetensors" pattern (with dot, not dash)
                ckpt_name = f"model.{variant}.safetensors"
        else:
            ckpt_name = "model.ckpt"
            if variant:
                ckpt_name = f"model-{variant}.ckpt"

        ckpt_path = os.path.join(local_path, ckpt_name)
        print(f"[DEBUG] Looking for checkpoint: {ckpt_path}")
        print(f"[DEBUG] Checkpoint exists: {os.path.exists(ckpt_path)}")
        if not os.path.exists(ckpt_path):
            ckpt_pattern = "*.safetensors" if use_safetensors else "*.ckpt"
            files = glob.glob(os.path.join(local_path, ckpt_pattern))
            print(f"[DEBUG] Fallback search pattern: {ckpt_pattern}")
            print(f"[DEBUG] Found files: {files}")
            if files:
                ckpt_path = files[0]

        logger.info(f"Config path: {config_path}")
        logger.info(f"Checkpoint path: {ckpt_path}")

        return config_path, ckpt_path
