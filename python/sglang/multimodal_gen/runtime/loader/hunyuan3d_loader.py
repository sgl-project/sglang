# SPDX-License-Identifier: Apache-2.0
"""
Hunyuan3D specific component loader.

This module handles loading Hunyuan3D models which use a different
loading paradigm (YAML config + bundled checkpoint) compared to
standard diffusers models.

Hunyuan3D uses:
- YAML configuration files with 'target' + 'params' pattern
- Single checkpoint file containing all components (model, vae, conditioner, etc.)
- Custom class resolution via aliases

This keeps the main component_loader.py clean and generic.
"""

import glob
import importlib
import os
from typing import Any, ClassVar, Dict

import torch

from sglang.multimodal_gen.runtime.loader.component_loader import ComponentLoader
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Hunyuan3DShapeLoader(ComponentLoader):
    """Loader for Hunyuan3D shape components.

    This loader handles loading of Hunyuan3D models including:
    - DiT model for shape generation
    - VAE for latent encoding/decoding
    - Conditioner for image conditioning
    - Image processor for input preprocessing
    - Scheduler for diffusion process

    Models are loaded from HuggingFace Hub or local paths and cached
    to avoid redundant loading. All components use internal implementations.

    Unlike standard diffusers loaders, Hunyuan3D:
    1. Uses YAML config files with 'target' and 'params' keys
    2. Loads all components from a single bundled checkpoint
    3. Caches all components together for efficiency
    """

    # Class-level cache for loaded components
    _cache: ClassVar[Dict[str, Dict[str, Any]]] = {}

    @staticmethod
    def get_obj_from_str(string: str) -> Any:
        """Get an object from a module path string.

        This method resolves class references from various formats:
        1. Registry aliases (e.g., 'hy3dshape.models.denoisers.Hunyuan3DDiT')
        2. Direct class names via ModelRegistry
        3. Hunyuan3D tool classes (ImageProcessorV2, etc.)
        4. Fallback to direct import

        Args:
            string: A module path string like 'module.submodule.ClassName'

        Returns:
            The resolved class object
        """
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
        """Instantiate an object from config with path remapping.

        This handles the Hunyuan3D YAML config format:
        {
            "target": "module.path.ClassName",
            "params": {"param1": value1, ...}
        }

        Args:
            config: Configuration dict with 'target' and optional 'params' keys
            **kwargs: Additional arguments to pass to the constructor

        Returns:
            Instantiated object

        Raises:
            KeyError: If 'target' key is missing from config
        """
        if "target" not in config:
            raise KeyError("Expected key 'target' to instantiate.")
        target_cls = cls.get_obj_from_str(config["target"])
        params = config.get("params", {})
        kwargs.update(params)
        return target_cls(**kwargs)

    def load_customized(self, component_model_path: str, server_args, module_name: str):
        """Load Hunyuan3D shape components.

        This method loads all components from a bundled checkpoint and caches them.
        Subsequent calls for different components return from cache.

        Args:
            component_model_path: Path to the model (used for cache key)
            server_args: Server arguments containing pipeline config
            module_name: Name of the component to return (e.g., 'hy3dshape_model')

        Returns:
            The requested component
        """
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
            model.load_state_dict(ckpt["model"])

            vae = self.instantiate_from_config(model_config["vae"])

            result = vae.load_state_dict(ckpt["vae"], strict=False)
            print(f"[DEBUG] Missing keys count: {len(result.missing_keys)}")
            print(f"[DEBUG] Missing keys: {result.missing_keys}")
            print(f"[DEBUG] Unexpected keys count: {len(result.unexpected_keys)}")
            print(f"[DEBUG] Unexpected keys: {result.unexpected_keys}")

            # 验证 output_proj 权重是否真正被加载
            ckpt_weight = ckpt["vae"]["geo_decoder.output_proj.weight"]
            model_weight = vae.geo_decoder.output_proj.weight.cpu()
            print(
                f"[DEBUG] Ckpt output_proj.weight sum: {ckpt_weight.sum().item():.6f}"
            )
            print(
                f"[DEBUG] Model output_proj.weight sum: {model_weight.sum().item():.6f}"
            )
            print(
                f"[DEBUG] Weights match: {torch.allclose(ckpt_weight.float(), model_weight.float())}"
            )
            if "vae" in ckpt:
                vae_keys = list(ckpt["vae"].keys())
                print(f"[DEBUG] VAE total keys: {len(vae_keys)}")

                # 检查 geo_decoder 相关的 keys
                geo_decoder_keys = [k for k in vae_keys if k.startswith("geo_decoder")]
                print(f"[DEBUG] geo_decoder keys count: {len(geo_decoder_keys)}")
                print(f"[DEBUG] geo_decoder keys: {geo_decoder_keys[:10]}")  # 前10个

                # 特别检查 output_proj
                output_proj_keys = [k for k in vae_keys if "output_proj" in k]
                print(f"[DEBUG] output_proj keys: {output_proj_keys}")
            conditioner = self.instantiate_from_config(model_config["conditioner"])
            if "conditioner" in ckpt:
                conditioner.load_state_dict(ckpt["conditioner"])

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

            # Enable FlashVDM if configured
            if hasattr(config, "vae_enable_flashvdm") and config.vae_enable_flashvdm:
                if hasattr(vae, "enable_flashvdm_decoder"):
                    vae.enable_flashvdm_decoder(
                        enabled=True,
                        topk_mode=getattr(config, "vae_flashvdm_topk_mode", "mean"),
                    )
                    logger.info("Enabled FlashVDM decoder for VAE")

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
                ckpt_name = f"model-{variant}.safetensors"
        else:
            ckpt_name = "model.ckpt"
            if variant:
                ckpt_name = f"model-{variant}.ckpt"

        ckpt_path = os.path.join(local_path, ckpt_name)
        if not os.path.exists(ckpt_path):
            ckpt_pattern = "*.safetensors" if use_safetensors else "*.ckpt"
            files = glob.glob(os.path.join(local_path, ckpt_pattern))
            if files:
                ckpt_path = files[0]

        logger.info(f"Config path: {config_path}")
        logger.info(f"Checkpoint path: {ckpt_path}")

        return config_path, ckpt_path
