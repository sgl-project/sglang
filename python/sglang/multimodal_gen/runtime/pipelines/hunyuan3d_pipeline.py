"""
Hunyuan3D image-to-mesh pipeline implementation.

Shape pipeline: BeforeDenoising -> Denoising -> Export -> Save
Paint pipeline (optional): Preprocess -> TexGen -> Postprocess
"""

from __future__ import annotations

import glob
import importlib
import os
from itertools import chain
from typing import Any

import torch
import torch.nn as nn

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan3d import (
    Hunyuan3D2PipelineConfig,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    load_model_from_full_model_state_dict,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    Hunyuan3DPaintPostprocessStage,
    Hunyuan3DPaintPreprocessStage,
    Hunyuan3DPaintTexGenStage,
    Hunyuan3DShapeBeforeDenoisingStage,
    Hunyuan3DShapeDenoisingStage,
    Hunyuan3DShapeExportStage,
    Hunyuan3DShapeSaveStage,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


class Hunyuan3D2Pipeline(ComposedPipelineBase):
    """Hunyuan3D 2.0 image-to-mesh pipeline.

    Shape pipeline: BeforeDenoising -> Denoising -> Export -> Save
    Paint pipeline (optional): Preprocess -> TexGen -> Postprocess
    """

    pipeline_name = "Hunyuan3D2Pipeline"
    _required_config_modules = [
        "hy3dshape_model",
        "hy3dshape_vae",
        "hy3dshape_scheduler",
        "hy3dshape_conditioner",
        "hy3dshape_image_processor",
    ]

    def _load_config(self) -> dict[str, Any]:
        return {
            "_class_name": self.pipeline_name,
            "_diffusers_version": "0.0.0",
            "hy3dshape_model": ["diffusers", "Hunyuan3DShapeModel"],
            "hy3dshape_vae": ["diffusers", "Hunyuan3DShapeVAE"],
            "hy3dshape_scheduler": ["diffusers", "Hunyuan3DShapeScheduler"],
            "hy3dshape_conditioner": ["diffusers", "Hunyuan3DShapeConditioner"],
            "hy3dshape_image_processor": ["diffusers", "Hunyuan3DShapeImageProcessor"],
        }

    # Class resolution
    @staticmethod
    def _resolve_class(target: str) -> Any:
        """Resolve a YAML target string to a Python class."""
        from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

        cls = ModelRegistry.resolve_by_alias(target)
        if cls is not None:
            return cls

        class_name = target.rsplit(".", 1)[-1]
        try:
            cls, _ = ModelRegistry.resolve_model_cls(class_name)
            return cls
        except Exception:
            pass

        from sglang.multimodal_gen.runtime.utils.mesh3d_utils import (
            resolve_hunyuan3d_tool,
        )

        for name in (target, class_name):
            tool_cls = resolve_hunyuan3d_tool(name)
            if tool_cls is not None:
                return tool_cls

        module, cls_name = target.rsplit(".", 1)
        return getattr(importlib.import_module(module, package=None), cls_name)

    # Path / checkpoint resolution
    @staticmethod
    def _resolve_shape_dir(
        model_path: str,
        subfolder: str,
        use_safetensors: bool,
        variant: str | None,
    ) -> tuple[str, str]:
        """Locate (or download) the shape subfolder and return (config_path, ckpt_path)."""
        local_path = os.path.join(model_path, subfolder)
        if not os.path.exists(local_path):
            local_path = os.path.expanduser(local_path)

        if not os.path.exists(local_path):
            logger.info(
                "Local path %s not found, downloading from HuggingFace Hub",
                local_path,
            )
            from huggingface_hub import snapshot_download

            downloaded = snapshot_download(
                repo_id=model_path,
                allow_patterns=[f"{subfolder}/*"],
            )
            local_path = os.path.join(downloaded, subfolder)

        config_path = os.path.join(local_path, "config.yaml")
        if not os.path.exists(config_path):
            for alt in ("config.yml", "model_config.yaml"):
                alt_path = os.path.join(local_path, alt)
                if os.path.exists(alt_path):
                    config_path = alt_path
                    break

        if use_safetensors:
            ckpt_name = (
                f"model.{variant}.safetensors" if variant else "model.safetensors"
            )
        else:
            ckpt_name = f"model-{variant}.ckpt" if variant else "model.ckpt"

        ckpt_path = os.path.join(local_path, ckpt_name)
        if not os.path.exists(ckpt_path):
            pattern = "*.safetensors" if use_safetensors else "*.ckpt"
            files = glob.glob(os.path.join(local_path, pattern))
            if files:
                ckpt_path = files[0]

        logger.info("Config path: %s", config_path)
        logger.info("Checkpoint path: %s", ckpt_path)
        return config_path, ckpt_path

    @staticmethod
    def _resolve_paint_dir(model_path: str, subfolder: str) -> str:
        """Locate (or download) the paint subfolder and return its local path."""
        local_path = os.path.join(model_path, subfolder)
        if not os.path.exists(local_path):
            local_path = os.path.expanduser(local_path)

        if not os.path.exists(local_path):
            logger.info(
                "Local path %s not found, downloading from HuggingFace Hub",
                local_path,
            )
            from huggingface_hub import snapshot_download

            downloaded = snapshot_download(
                repo_id=model_path,
                allow_patterns=[f"{subfolder}/*"],
            )
            local_path = os.path.join(downloaded, subfolder)

        for subdir in ("vae", "unet"):
            config_file = os.path.join(local_path, subdir, "config.json")
            if not os.path.exists(config_file):
                raise FileNotFoundError(
                    f"Paint model incomplete: {config_file} not found. "
                    "Download the model or check network connectivity."
                )

        logger.info("Resolved paint model directory: %s", local_path)
        return local_path

    @staticmethod
    def _load_and_split_checkpoint(
        ckpt_path: str, use_safetensors: bool
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Load a bundled checkpoint and split by the first '.' in each key."""
        if use_safetensors:
            import safetensors.torch

            flat = safetensors.torch.load_file(ckpt_path, device="cpu")
            ckpt: dict[str, dict[str, torch.Tensor]] = {}
            for key, value in flat.items():
                component = key.split(".")[0]
                sub_key = key[len(component) + 1 :]
                ckpt.setdefault(component, {})[sub_key] = value
            return ckpt
        else:
            return torch.load(ckpt_path, map_location="cpu", weights_only=True)

    # Component loading helpers
    @classmethod
    def _load_dit_model(
        cls,
        cfg: dict[str, Any],
        weights: dict[str, torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        """Load the DiT model using meta-device instantiation + standard weight loading."""
        if "target" not in cfg:
            raise KeyError("Expected key 'target' in model config.")
        target_cls = cls._resolve_class(cfg["target"])
        params = cfg.get("params", {})

        if hasattr(target_cls, "build_config_from_params"):
            dit_config = target_cls.build_config_from_params(params)
            init_kwargs: dict[str, Any] = {"config": dit_config, "hf_config": {}}
        else:
            init_kwargs = params

        with set_default_torch_dtype(dtype), torch.device("meta"):
            model = target_cls(**init_kwargs)

        weight_iterator = ((k, v) for k, v in weights.items())
        param_names_mapping_fn = get_param_names_mapping(model.param_names_mapping)

        load_model_from_full_model_state_dict(
            model,
            weight_iterator,
            device,
            dtype,
            strict=False,
            param_names_mapping=param_names_mapping_fn,
        )

        for name, p in chain(model.named_parameters(), model.named_buffers()):
            if p.is_meta:
                raise RuntimeError(f"Unexpected param or buffer {name} on meta device.")
            if isinstance(p, nn.Parameter):
                p.requires_grad = False

        return model.eval()

    @classmethod
    def _load_simple_component(
        cls,
        cfg: dict[str, Any],
        weights: dict[str, torch.Tensor] | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Module:
        """Load a component (VAE / conditioner) with direct instantiation + state_dict."""
        if "target" not in cfg:
            raise KeyError("Expected key 'target' in component config.")
        target_cls = cls._resolve_class(cfg["target"])
        params = cfg.get("params", {})

        with set_default_torch_dtype(dtype):
            component = target_cls(**params)

        if weights is not None:
            component.load_state_dict(weights, strict=False)

        component.to(device=device, dtype=dtype)
        return component.eval()

    @classmethod
    def _instantiate_component(cls, cfg: dict[str, Any]) -> Any:
        """Instantiate a lightweight component (scheduler / image_processor) without weights."""
        if "target" not in cfg:
            raise KeyError("Expected key 'target' in component config.")
        target_cls = cls._resolve_class(cfg["target"])
        params = cfg.get("params", {})
        return target_cls(**params)

    # Module loading override
    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """Load all Hunyuan3D shape components from a bundled checkpoint."""
        import yaml

        from sglang.multimodal_gen.runtime.distributed import get_local_torch_device

        config = server_args.pipeline_config
        if not isinstance(config, Hunyuan3D2PipelineConfig):
            raise TypeError(f"Expected Hunyuan3D2PipelineConfig, got {type(config)}")

        model_path = config.shape_model_path or server_args.model_path

        logger.info("Loading Hunyuan3D shape models from %s", model_path)

        config_path, ckpt_path = self._resolve_shape_dir(
            model_path,
            config.shape_subfolder,
            config.shape_use_safetensors,
            config.shape_variant,
        )

        with open(config_path, "r") as f:
            model_config = yaml.safe_load(f)

        ckpt = self._load_and_split_checkpoint(ckpt_path, config.shape_use_safetensors)

        dtype = torch.float16
        if config.shape_variant and "bf16" in config.shape_variant:
            dtype = torch.bfloat16
        device = get_local_torch_device()

        components: dict[str, Any] = {}

        components["hy3dshape_model"] = self._load_dit_model(
            model_config["model"], ckpt["model"], device, dtype
        )

        components["hy3dshape_vae"] = self._load_simple_component(
            model_config["vae"], ckpt.get("vae"), device, dtype
        )

        components["hy3dshape_conditioner"] = self._load_simple_component(
            model_config["conditioner"], ckpt.get("conditioner"), device, dtype
        )

        components["hy3dshape_scheduler"] = self._instantiate_component(
            model_config["scheduler"]
        )
        components["hy3dshape_image_processor"] = self._instantiate_component(
            model_config["image_processor"]
        )

        logger.info("All Hunyuan3D shape components loaded successfully")

        if config.paint_enable:
            try:
                paint_dir = self._resolve_paint_dir(
                    server_args.model_path, config.paint_subfolder
                )
                components["hy3dpaint_dir"] = paint_dir
            except Exception as e:
                logger.warning("Failed to resolve paint model path: %s", e)

        return components

    # Pipeline lifecycle
    def initialize_pipeline(self, server_args: ServerArgs):
        config = server_args.pipeline_config
        if not isinstance(config, Hunyuan3D2PipelineConfig):
            raise TypeError(
                "Hunyuan3D2Pipeline requires Hunyuan3D2PipelineConfig, "
                f"got {type(config)}"
            )

    def create_pipeline_stages(self, server_args: ServerArgs):
        config = server_args.pipeline_config
        assert isinstance(config, Hunyuan3D2PipelineConfig)

        # Shape: 4 stages
        self.add_stage(
            stage_name="shape_before_denoising",
            stage=Hunyuan3DShapeBeforeDenoisingStage(
                image_processor=self.get_module("hy3dshape_image_processor"),
                conditioner=self.get_module("hy3dshape_conditioner"),
                vae=self.get_module("hy3dshape_vae"),
                model=self.get_module("hy3dshape_model"),
                scheduler=self.get_module("hy3dshape_scheduler"),
                config=config,
            ),
        )
        self.add_stage(
            stage_name="shape_denoising",
            stage=Hunyuan3DShapeDenoisingStage(
                transformer=self.get_module("hy3dshape_model"),
                scheduler=self.get_module("hy3dshape_scheduler"),
            ),
        )
        self.add_stage(
            stage_name="shape_export",
            stage=Hunyuan3DShapeExportStage(
                vae=self.get_module("hy3dshape_vae"),
                config=config,
            ),
        )
        self.add_stage(
            stage_name="shape_save",
            stage=Hunyuan3DShapeSaveStage(config=config),
        )

        # Paint: 3 stages (optional)
        if config.paint_enable:
            self.add_stage(
                stage_name="paint_preprocess",
                stage=Hunyuan3DPaintPreprocessStage(config=config),
            )
            self.add_stage(
                stage_name="paint_texgen",
                stage=Hunyuan3DPaintTexGenStage(
                    config=config,
                    paint_dir=self.get_module("hy3dpaint_dir"),
                ),
            )
            self.add_stage(
                stage_name="paint_postprocess",
                stage=Hunyuan3DPaintPostprocessStage(config=config),
            )


EntryClass = Hunyuan3D2Pipeline
