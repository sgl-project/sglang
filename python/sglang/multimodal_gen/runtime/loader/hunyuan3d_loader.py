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

            # Debug: Print loaded config
            print(f"[DEBUG] =============== Model Config ===============")
            print(f"[DEBUG] Config path: {config_path}")
            for component in ["model", "vae", "conditioner", "scheduler"]:
                if component in model_config:
                    comp_cfg = model_config[component]
                    print(f"[DEBUG] {component}:")
                    print(f"[DEBUG]   target: {comp_cfg.get('target', 'N/A')}")
                    if "params" in comp_cfg:
                        params = comp_cfg["params"]
                        # Print key params for model
                        if component == "model":
                            for key in [
                                "in_channels",
                                "hidden_size",
                                "num_heads",
                                "depth",
                                "depth_single_blocks",
                                "context_in_dim",
                            ]:
                                if key in params:
                                    print(f"[DEBUG]   {key}: {params[key]}")

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

            # Debug: Print checkpoint structure
            print(f"[DEBUG] =============== Checkpoint Structure ===============")
            print(f"[DEBUG] Top-level keys in checkpoint: {list(ckpt.keys())}")
            for key in ckpt.keys():
                if isinstance(ckpt[key], dict):
                    print(f"[DEBUG]   {key}: {len(ckpt[key])} keys")
                else:
                    print(f"[DEBUG]   {key}: {type(ckpt[key])}")

            # Debug: Print model architecture summary
            print(f"[DEBUG] =============== DiT Model Architecture ===============")
            print(f"[DEBUG] Model class: {type(model).__name__}")
            model_params = sum(p.numel() for p in model.parameters())
            print(f"[DEBUG] Model total parameters: {model_params:,}")

            # Debug: Compare model state dict with checkpoint
            model_state = model.state_dict()
            ckpt_model_keys = set(ckpt["model"].keys())
            model_keys = set(model_state.keys())

            missing_in_ckpt = model_keys - ckpt_model_keys
            extra_in_ckpt = ckpt_model_keys - model_keys

            print(f"[DEBUG] Model state dict keys: {len(model_keys)}")
            print(f"[DEBUG] Checkpoint 'model' keys: {len(ckpt_model_keys)}")
            print(
                f"[DEBUG] Keys missing in checkpoint (model has, ckpt doesn't): {len(missing_in_ckpt)}"
            )
            if missing_in_ckpt:
                print(f"[DEBUG]   Missing keys: {list(missing_in_ckpt)[:20]}...")
            print(
                f"[DEBUG] Extra keys in checkpoint (ckpt has, model doesn't): {len(extra_in_ckpt)}"
            )
            if extra_in_ckpt:
                print(f"[DEBUG]   Extra keys: {list(extra_in_ckpt)[:20]}...")

            # Load DiT model with strict=False to catch issues
            dit_result = model.load_state_dict(ckpt["model"], strict=False)
            print(f"[DEBUG] =============== DiT Model Loading Result ===============")
            print(f"[DEBUG] DiT missing keys count: {len(dit_result.missing_keys)}")
            if dit_result.missing_keys:
                print(f"[DEBUG] DiT missing keys: {dit_result.missing_keys[:30]}")
            print(
                f"[DEBUG] DiT unexpected keys count: {len(dit_result.unexpected_keys)}"
            )
            if dit_result.unexpected_keys:
                print(f"[DEBUG] DiT unexpected keys: {dit_result.unexpected_keys[:30]}")

            # Verify key weights were loaded correctly
            if "latent_in.weight" in ckpt["model"]:
                ckpt_weight = ckpt["model"]["latent_in.weight"]
                model_weight = model.latent_in.weight.cpu()
                print(
                    f"[DEBUG] DiT latent_in.weight ckpt sum: {ckpt_weight.sum().item():.6f}"
                )
                print(
                    f"[DEBUG] DiT latent_in.weight model sum: {model_weight.sum().item():.6f}"
                )
                print(
                    f"[DEBUG] DiT latent_in weights match: {torch.allclose(ckpt_weight.float(), model_weight.float())}"
                )

            # Debug: VAE Loading
            print(f"[DEBUG] =============== VAE Loading ===============")
            print(
                f"[DEBUG] VAE config target: {model_config['vae'].get('target', 'N/A')}"
            )
            vae = self.instantiate_from_config(model_config["vae"])
            print(f"[DEBUG] VAE class: {type(vae).__name__}")
            vae_params = sum(p.numel() for p in vae.parameters())
            print(f"[DEBUG] VAE total parameters: {vae_params:,}")

            result = vae.load_state_dict(ckpt["vae"], strict=False)
            print(f"[DEBUG] VAE missing keys count: {len(result.missing_keys)}")
            if result.missing_keys:
                # Group missing keys by prefix
                encoder_keys = [
                    k for k in result.missing_keys if k.startswith("encoder.")
                ]
                other_keys = [
                    k for k in result.missing_keys if not k.startswith("encoder.")
                ]
                print(
                    f"[DEBUG]   Encoder missing keys: {len(encoder_keys)} (expected - encoder only used for training)"
                )
                if other_keys:
                    print(f"[DEBUG]   WARNING: Other missing keys: {other_keys}")
            print(f"[DEBUG] VAE unexpected keys count: {len(result.unexpected_keys)}")
            if result.unexpected_keys:
                print(f"[DEBUG]   Unexpected keys: {result.unexpected_keys[:10]}...")

            # Verify key decoder weights
            if "geo_decoder.output_proj.weight" in ckpt["vae"]:
                ckpt_weight = ckpt["vae"]["geo_decoder.output_proj.weight"]
                model_weight = vae.geo_decoder.output_proj.weight.cpu()
                print(
                    f"[DEBUG] VAE geo_decoder.output_proj.weight ckpt sum: {ckpt_weight.sum().item():.6f}"
                )
                print(
                    f"[DEBUG] VAE geo_decoder.output_proj.weight model sum: {model_weight.sum().item():.6f}"
                )
                print(
                    f"[DEBUG] VAE geo_decoder weights match: {torch.allclose(ckpt_weight.float(), model_weight.float())}"
                )
            # Debug: Conditioner loading
            print(f"[DEBUG] =============== Conditioner Loading ===============")
            print(
                f"[DEBUG] Conditioner config target: {model_config['conditioner'].get('target', 'N/A')}"
            )
            conditioner = self.instantiate_from_config(model_config["conditioner"])
            print(f"[DEBUG] Conditioner class: {type(conditioner).__name__}")
            conditioner_params = sum(p.numel() for p in conditioner.parameters())
            print(f"[DEBUG] Conditioner total parameters: {conditioner_params:,}")

            cond_result = None
            conditioner_from_ckpt = "conditioner" in ckpt
            if conditioner_from_ckpt:
                print(f"[DEBUG] Loading conditioner weights from checkpoint...")
                print(
                    f"[DEBUG] Checkpoint 'conditioner' keys count: {len(ckpt['conditioner'])}"
                )
                cond_result = conditioner.load_state_dict(
                    ckpt["conditioner"], strict=False
                )
                print(
                    f"[DEBUG] Conditioner missing keys: {len(cond_result.missing_keys)}"
                )
                if cond_result.missing_keys:
                    print(f"[DEBUG]   {cond_result.missing_keys[:20]}...")
                print(
                    f"[DEBUG] Conditioner unexpected keys: {len(cond_result.unexpected_keys)}"
                )
                if cond_result.unexpected_keys:
                    print(f"[DEBUG]   {cond_result.unexpected_keys[:20]}...")
            else:
                print(f"[DEBUG] WARNING: No 'conditioner' key in checkpoint!")
                print(
                    f"[DEBUG] Conditioner will use pretrained weights from HuggingFace"
                )

            # Debug: Print conditioner internal model info
            if hasattr(conditioner, "main_image_encoder"):
                enc = conditioner.main_image_encoder
                if hasattr(enc, "model"):
                    print(
                        f"[DEBUG] Main encoder model type: {type(enc.model).__name__}"
                    )
                    if hasattr(enc.model, "config"):
                        print(
                            f"[DEBUG] Main encoder hidden_size: {enc.model.config.hidden_size}"
                        )
                        print(
                            f"[DEBUG] Main encoder image_size: {getattr(enc.model.config, 'image_size', 'N/A')}"
                        )

            # Debug: Image Processor
            print(f"[DEBUG] =============== Image Processor ===============")
            print(
                f"[DEBUG] Image processor config target: {model_config['image_processor'].get('target', 'N/A')}"
            )
            image_processor = self.instantiate_from_config(
                model_config["image_processor"]
            )
            print(f"[DEBUG] Image processor class: {type(image_processor).__name__}")

            # Debug: Scheduler
            print(f"[DEBUG] =============== Scheduler ===============")
            print(
                f"[DEBUG] Scheduler config target: {model_config['scheduler'].get('target', 'N/A')}"
            )
            if "params" in model_config["scheduler"]:
                sched_params = model_config["scheduler"]["params"]
                print(f"[DEBUG] Scheduler params: {sched_params}")
            scheduler = self.instantiate_from_config(model_config["scheduler"])
            print(f"[DEBUG] Scheduler class: {type(scheduler).__name__}")

            # Summary of loading issues
            print(f"[DEBUG] =============== Loading Summary ===============")
            has_issues = False
            if dit_result.missing_keys:
                print(
                    f"[DEBUG] WARNING: DiT model has {len(dit_result.missing_keys)} missing keys!"
                )
                has_issues = True
            if dit_result.unexpected_keys:
                print(
                    f"[DEBUG] WARNING: DiT model has {len(dit_result.unexpected_keys)} unexpected keys!"
                )
                has_issues = True
            if result.unexpected_keys:
                print(
                    f"[DEBUG] WARNING: VAE has {len(result.unexpected_keys)} unexpected keys!"
                )
                has_issues = True
            # Check for critical VAE missing keys (excluding encoder)
            critical_vae_missing = [
                k for k in result.missing_keys if not k.startswith("encoder.")
            ]
            if critical_vae_missing:
                print(
                    f"[DEBUG] WARNING: VAE has {len(critical_vae_missing)} critical missing keys (not encoder)!"
                )
                has_issues = True
            # Check conditioner loading
            if not conditioner_from_ckpt:
                print(
                    f"[DEBUG] NOTE: Conditioner using HuggingFace pretrained weights (no ckpt)"
                )
            elif cond_result and (
                cond_result.missing_keys or cond_result.unexpected_keys
            ):
                if cond_result.missing_keys:
                    print(
                        f"[DEBUG] WARNING: Conditioner has {len(cond_result.missing_keys)} missing keys!"
                    )
                    has_issues = True
                if cond_result.unexpected_keys:
                    print(
                        f"[DEBUG] WARNING: Conditioner has {len(cond_result.unexpected_keys)} unexpected keys!"
                    )
                    has_issues = True
            if not has_issues:
                print(
                    f"[DEBUG] All models loaded successfully with no critical issues."
                )
            print(f"[DEBUG] =============================================")

            # Set device and dtype
            dtype = torch.float16
            if config.shape_variant and "bf16" in config.shape_variant:
                dtype = torch.bfloat16
            device = get_local_torch_device()

            for module in (model, vae, conditioner):
                module.to(device=device, dtype=dtype)
                module.eval()

            # ============================================================
            # DEBUG: Print weight checksums for comparison with original
            # ============================================================
            def print_weight_info(name, param):
                p = param.float().cpu()
                print(
                    f"[WEIGHT] {name}: shape={tuple(p.shape)}, sum={p.sum().item():.6f}, mean={p.mean().item():.6f}, std={p.std().item():.6f}"
                )

            print("\n" + "=" * 70)
            print("[SGLANG WEIGHT CHECK] Model weights after loading")
            print("=" * 70)

            # DiT Model weights
            print("\n--- DiT Model ---")
            print_weight_info("time_in.in_layer.weight", model.time_in.in_layer.weight)
            print_weight_info("time_in.in_layer.bias", model.time_in.in_layer.bias)
            print_weight_info(
                "time_in.out_layer.weight", model.time_in.out_layer.weight
            )
            print_weight_info("latent_in.weight", model.latent_in.weight)
            print_weight_info("latent_in.bias", model.latent_in.bias)
            # First and last transformer block
            if hasattr(model, "double_blocks") and len(model.double_blocks) > 0:
                print_weight_info(
                    "double_blocks[0].img_mod.lin.weight",
                    model.double_blocks[0].img_mod.lin.weight,
                )
                print_weight_info(
                    "double_blocks[-1].img_mod.lin.weight",
                    model.double_blocks[-1].img_mod.lin.weight,
                )
            if hasattr(model, "single_blocks") and len(model.single_blocks) > 0:
                print_weight_info(
                    "single_blocks[0].modulation.lin.weight",
                    model.single_blocks[0].modulation.lin.weight,
                )
                print_weight_info(
                    "single_blocks[-1].modulation.lin.weight",
                    model.single_blocks[-1].modulation.lin.weight,
                )
            print_weight_info(
                "final_layer.linear.weight", model.final_layer.linear.weight
            )
            print_weight_info("final_layer.linear.bias", model.final_layer.linear.bias)

            # VAE weights
            print("\n--- VAE ---")
            print_weight_info("post_kl.weight", vae.post_kl.weight)
            print_weight_info("post_kl.bias", vae.post_kl.bias)
            print_weight_info(
                "transformer.resblocks[0].attn.c_qkv.weight",
                vae.transformer.resblocks[0].attn.c_qkv.weight,
            )
            print_weight_info(
                "transformer.resblocks[-1].attn.c_qkv.weight",
                vae.transformer.resblocks[-1].attn.c_qkv.weight,
            )
            print_weight_info(
                "geo_decoder.output_proj.weight", vae.geo_decoder.output_proj.weight
            )
            print_weight_info(
                "geo_decoder.output_proj.bias", vae.geo_decoder.output_proj.bias
            )
            # Volume decoder weights
            if hasattr(vae, "volume_decoder"):
                vd = vae.volume_decoder
                if hasattr(vd, "dense_grid_proj"):
                    print_weight_info(
                        "volume_decoder.dense_grid_proj.weight",
                        vd.dense_grid_proj.weight,
                    )
                if hasattr(vd, "dense_points_proj"):
                    print_weight_info(
                        "volume_decoder.dense_points_proj.weight",
                        vd.dense_points_proj.weight,
                    )

            # Conditioner weights
            print("\n--- Conditioner ---")
            if hasattr(conditioner, "main_image_encoder"):
                enc = conditioner.main_image_encoder.model
                print_weight_info(
                    "encoder.patch_embed.projection.weight",
                    enc.embeddings.patch_embeddings.projection.weight,
                )
                print_weight_info(
                    "encoder.layer[0].attention.query.weight",
                    enc.encoder.layer[0].attention.attention.query.weight,
                )
                print_weight_info(
                    "encoder.layer[-1].attention.query.weight",
                    enc.encoder.layer[-1].attention.attention.query.weight,
                )
                print_weight_info("encoder.layernorm.weight", enc.layernorm.weight)

            print("=" * 70 + "\n")
            # ============================================================

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
