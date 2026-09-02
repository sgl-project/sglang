"""
Hunyuan3D image-to-mesh pipeline implementation.

Shape pipeline: BeforeDenoising -> Denoising -> Export -> Save
Paint pipeline (optional): Preprocess -> TexGen -> Postprocess
"""

from __future__ import annotations

import glob
import importlib
import json
import os
from itertools import chain
from typing import Any

import torch
import torch.nn as nn
import yaml
from diffusers import EulerAncestralDiscreteScheduler, LCMScheduler
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors
from transformers import AutoTokenizer

from sglang.multimodal_gen.configs.models.vaes.stable_diffusion import (
    StableDiffusionVAEConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.hunyuan3d import (
    Hunyuan3D2PipelineConfig,
)
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import (
    load_model_from_full_model_state_dict,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.models.dits.hunyuan3d_paint import (
    Hunyuan3DPaintUNet,
)
from sglang.multimodal_gen.runtime.models.dits.stable_diffusion import (
    StableDiffusionUNet2DConditionModel,
    StableDiffusionUNetConfig,
)
from sglang.multimodal_gen.runtime.models.vaes.autoencoder import (
    AutoencoderKL as StableDiffusionAutoencoderKL,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan3d import (
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
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

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

    def validate_disagg_role(self, role: RoleType) -> None:
        if role == RoleType.MONOLITHIC:
            return
        config = self.server_args.pipeline_config
        if not isinstance(config, Hunyuan3D2PipelineConfig):
            raise TypeError(
                "Hunyuan3D2Pipeline requires Hunyuan3D2PipelineConfig, "
                f"got {type(config)}"
            )
        if config.paint_enable:
            raise ValueError(
                "Hunyuan3D2Pipeline only supports shape-only disaggregation. "
                "Disable paint_enable when launching encoder/denoiser/decoder roles."
            )

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
            downloaded = snapshot_download(
                repo_id=model_path,
                allow_patterns=[f"{subfolder}/**"],
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
    def _resolve_model_subfolder(
        model_path: str,
        subfolder: str,
        required_files: tuple[str, ...],
    ) -> str:
        local_path = os.path.join(model_path, subfolder)
        if not os.path.exists(local_path):
            local_path = os.path.expanduser(local_path)

        if not os.path.exists(local_path):
            logger.info(
                "Local path %s not found, downloading from HuggingFace Hub",
                local_path,
            )
            downloaded = snapshot_download(
                repo_id=model_path,
                allow_patterns=[f"{subfolder}/**"],
            )
            local_path = os.path.join(downloaded, subfolder)

        for relative_path in required_files:
            required_file = os.path.join(local_path, relative_path)
            if not os.path.exists(required_file):
                raise FileNotFoundError(
                    f"Hunyuan3D model incomplete: {required_file} not found. "
                    "Download the model or check network connectivity."
                )

        logger.debug("Resolved Hunyuan3D model directory: %s", local_path)
        return local_path

    @staticmethod
    def _load_and_split_checkpoint(
        ckpt_path: str, use_safetensors: bool
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Load a bundled checkpoint and split by the first '.' in each key."""
        if use_safetensors:
            flat = load_safetensors(ckpt_path, device="cpu")
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

    @staticmethod
    def _read_json(path: str) -> dict[str, Any]:
        with open(path, encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _load_component_weights(component_dir: str) -> dict[str, torch.Tensor]:
        safetensors_path = os.path.join(
            component_dir, "diffusion_pytorch_model.safetensors"
        )
        pytorch_path = os.path.join(component_dir, "diffusion_pytorch_model.bin")
        if os.path.isfile(safetensors_path):
            return load_safetensors(safetensors_path, device="cpu")
        if os.path.isfile(pytorch_path):
            return torch.load(pytorch_path, map_location="cpu", weights_only=True)
        raise FileNotFoundError(
            f"No diffusion_pytorch_model weights found in {component_dir}."
        )

    @staticmethod
    def _component_device(server_args: ServerArgs, component_name: str) -> torch.device:
        if server_args.should_start_component_on_cpu(component_name):
            return torch.device("cpu")
        return get_local_torch_device()

    @staticmethod
    def _freeze(module: nn.Module) -> nn.Module:
        for parameter in module.parameters():
            parameter.requires_grad = False
        return module.eval()

    @staticmethod
    def _maybe_compile_texture_transformer(
        module: nn.Module,
        server_args: ServerArgs,
        config: Hunyuan3D2PipelineConfig,
    ) -> None:
        if not server_args.enable_torch_compile:
            return
        compile_mode = (
            os.environ.get("SGLANG_TORCH_COMPILE_MODE")
            or config.dit_config.torch_compile_mode
        )
        logger.info(
            "Compiling %s with mode: %s",
            module.__class__.__name__,
            compile_mode,
        )
        module.compile(mode=compile_mode, fullgraph=False, dynamic=None)

    @classmethod
    def _load_stable_diffusion_vae(
        cls,
        component_dir: str,
        dtype: torch.dtype,
        device: torch.device,
    ) -> StableDiffusionAutoencoderKL:
        config_data = cls._read_json(os.path.join(component_dir, "config.json"))
        vae_config = StableDiffusionVAEConfig()
        vae_config.update_model_arch(config_data)
        with set_default_torch_dtype(dtype):
            vae = StableDiffusionAutoencoderKL(vae_config)
        weights = cls._load_component_weights(component_dir)
        vae.load_state_dict(weights, strict=True)
        vae.to(device=device, dtype=dtype)
        return cls._freeze(vae)

    @classmethod
    def _load_stable_diffusion_unet(
        cls,
        component_dir: str,
        dtype: torch.dtype,
        device: torch.device,
        *,
        paint: bool,
        is_turbo: bool = False,
    ) -> nn.Module:
        config_data = cls._read_json(os.path.join(component_dir, "config.json"))
        unet_config = StableDiffusionUNetConfig.from_dict(config_data)
        with set_default_torch_dtype(dtype), torch.device("meta"):
            unet: nn.Module
            if paint:
                unet = Hunyuan3DPaintUNet(unet_config, is_turbo=is_turbo)
            else:
                unet = StableDiffusionUNet2DConditionModel(unet_config)
        weights = cls._load_component_weights(component_dir)
        unet.load_state_dict(weights, strict=True, assign=True)
        unet.to(device=device, dtype=dtype)
        return cls._freeze(unet)

    @classmethod
    def _load_texture_components(
        cls,
        server_args: ServerArgs,
        config: Hunyuan3D2PipelineConfig,
    ) -> dict[str, Any]:
        dtype = PRECISION_TO_TYPE[config.dit_precision]
        components: dict[str, Any] = {}

        paint_dir = cls._resolve_model_subfolder(
            server_args.model_path,
            config.paint_subfolder,
            (
                "vae/config.json",
                "unet/config.json",
                "scheduler/scheduler_config.json",
            ),
        )
        components["paint_vae"] = cls._load_stable_diffusion_vae(
            os.path.join(paint_dir, "vae"),
            dtype,
            cls._component_device(server_args, "paint_vae"),
        )
        components["paint_transformer"] = cls._load_stable_diffusion_unet(
            os.path.join(paint_dir, "unet"),
            dtype,
            cls._component_device(server_args, "paint_transformer"),
            paint=True,
            is_turbo=config.paint_turbo_mode,
        )
        cls._maybe_compile_texture_transformer(
            components["paint_transformer"], server_args, config
        )
        paint_scheduler_config = cls._read_json(
            os.path.join(paint_dir, "scheduler", "scheduler_config.json")
        )
        scheduler_class = (
            LCMScheduler if config.paint_turbo_mode else EulerAncestralDiscreteScheduler
        )
        components["paint_scheduler"] = scheduler_class.from_config(
            paint_scheduler_config,
            **({} if config.paint_turbo_mode else {"timestep_spacing": "trailing"}),
        )

        if config.delight_enable:
            delight_dir = cls._resolve_model_subfolder(
                server_args.model_path,
                config.delight_subfolder,
                (
                    "vae/config.json",
                    "unet/config.json",
                    "scheduler/scheduler_config.json",
                    "text_encoder/config.json",
                    "tokenizer/tokenizer_config.json",
                ),
            )
            components["delight_vae"] = cls._load_stable_diffusion_vae(
                os.path.join(delight_dir, "vae"),
                dtype,
                cls._component_device(server_args, "delight_vae"),
            )
            components["delight_transformer"] = cls._load_stable_diffusion_unet(
                os.path.join(delight_dir, "unet"),
                dtype,
                cls._component_device(server_args, "delight_transformer"),
                paint=False,
            )
            cls._maybe_compile_texture_transformer(
                components["delight_transformer"], server_args, config
            )
            delight_text_encoder, _ = TextEncoderLoader().load(
                os.path.join(delight_dir, "text_encoder"),
                server_args,
                "delight_text_encoder",
                "transformers",
            )
            components["delight_text_encoder"] = delight_text_encoder
            components["delight_tokenizer"] = AutoTokenizer.from_pretrained(
                os.path.join(delight_dir, "tokenizer")
            )
            delight_scheduler_config = cls._read_json(
                os.path.join(delight_dir, "scheduler", "scheduler_config.json")
            )
            components["delight_scheduler"] = (
                EulerAncestralDiscreteScheduler.from_config(delight_scheduler_config)
            )

        return components

    # Module loading override
    def load_modules(
        self,
        server_args: ServerArgs,
        loaded_modules: dict[str, torch.nn.Module] | None = None,
    ) -> dict[str, Any]:
        """Load Hunyuan3D shape and optional texture components."""
        del loaded_modules

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
            model_config["model"],
            ckpt["model"],
            (
                torch.device("cpu")
                if server_args.should_start_component_on_cpu("hy3dshape_model")
                else device
            ),
            dtype,
        )

        components["hy3dshape_vae"] = self._load_simple_component(
            model_config["vae"],
            ckpt.get("vae"),
            (
                torch.device("cpu")
                if server_args.should_start_component_on_cpu("hy3dshape_vae")
                else device
            ),
            dtype,
        )

        components["hy3dshape_conditioner"] = self._load_simple_component(
            model_config["conditioner"],
            ckpt.get("conditioner"),
            (
                torch.device("cpu")
                if server_args.should_start_component_on_cpu("hy3dshape_conditioner")
                else device
            ),
            dtype,
        )

        components["hy3dshape_scheduler"] = self._instantiate_component(
            model_config["scheduler"]
        )
        components["hy3dshape_image_processor"] = self._instantiate_component(
            model_config["image_processor"]
        )

        if config.paint_enable:
            components.update(self._load_texture_components(server_args, config))

        logger.info("Loaded Hunyuan3D pipeline components: %s", sorted(components))

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
        latent_shape = tuple(config.vae_config.arch_config.latent_shape)
        guidance_embed = bool(config.dit_config.arch_config.guidance_embed)

        # Shape: 4 stages
        self.add_stage(
            stage_name="shape_before_denoising",
            stage=Hunyuan3DShapeBeforeDenoisingStage(
                image_processor=self.get_module("hy3dshape_image_processor"),
                conditioner=self.get_module("hy3dshape_conditioner"),
                scheduler=self.get_module("hy3dshape_scheduler"),
                config=config,
                latent_shape=latent_shape,
                guidance_embed=guidance_embed,
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
                stage=Hunyuan3DPaintPreprocessStage(
                    config=config,
                    delight_transformer=self.get_module("delight_transformer"),
                    delight_vae=self.get_module("delight_vae"),
                    delight_text_encoder=self.get_module("delight_text_encoder"),
                    delight_tokenizer=self.get_module("delight_tokenizer"),
                    delight_scheduler=self.get_module("delight_scheduler"),
                ),
            )
            self.add_stage(
                stage_name="paint_texgen",
                stage=Hunyuan3DPaintTexGenStage(
                    config=config,
                    transformer=self.get_module("paint_transformer"),
                    scheduler=self.get_module("paint_scheduler"),
                    vae=self.get_module("paint_vae"),
                ),
            )
            self.add_stage(
                stage_name="paint_postprocess",
                stage=Hunyuan3DPaintPostprocessStage(config=config),
            )


EntryClass = Hunyuan3D2Pipeline
