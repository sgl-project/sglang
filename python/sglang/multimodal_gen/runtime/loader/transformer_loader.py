import os
from copy import deepcopy
import inspect
from typing import Any

import torch
from safetensors.torch import load_file as safetensors_load_file
from torch import nn

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.component_loader import ComponentLoader
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    _normalize_component_type,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class TransformerLoader(ComponentLoader):
    """Shared loader for (video/audio) DiT transformers."""

    component_names = ["transformer", "audio_dit", "video_dit"]
    expected_library = "diffusers"

    def _is_svdquant_config(self, quant_config: Any) -> bool:
        return (
            quant_config is not None
            and hasattr(quant_config, "get_name")
            and quant_config.get_name() == "svdquant"
        )

    def _load_state_dict_safe(self, weights_path: str) -> dict | None:
        try:
            return safetensors_load_file(weights_path)
        except Exception as exc:
            logger.warning(
                "Failed to load weights from %s for Nunchaku scale patch: %s",
                weights_path,
                exc,
            )
            return None

    def _patch_native_svdq_linear(
        self, module: nn.Module, tensor: Any, svdq_linear_cls: type
    ) -> bool:
        if (
            isinstance(module, svdq_linear_cls)
            and getattr(module, "wtscale", None) is not None
        ):
            module.wtscale = tensor
            return True
        return False

    def _patch_sglang_svdq_linear(
        self, module: nn.Module, tensor: Any, svdq_method_cls: type
    ) -> bool:
        quant_method = getattr(module, "quant_method", None)
        if not isinstance(quant_method, svdq_method_cls):
            return False

        existing = getattr(module, "wtscale", None)
        if isinstance(existing, nn.Parameter):
            with torch.no_grad():
                existing.data.copy_(tensor.to(existing.data.dtype))
        else:
            module.wtscale = tensor

        # Keep alpha in sync (kernel reads `layer._nunchaku_alpha`)
        try:
            module._nunchaku_alpha = float(tensor.detach().cpu().item())
        except Exception:
            module._nunchaku_alpha = None
        return True

    def _patch_sglang_svdq_wcscales(
        self, module: nn.Module, tensor: Any, svdq_method_cls: type
    ) -> bool:
        quant_method = getattr(module, "quant_method", None)
        if not isinstance(quant_method, svdq_method_cls):
            return False

        existing = getattr(module, "wcscales", None)
        if isinstance(existing, nn.Parameter):
            with torch.no_grad():
                existing.data.copy_(tensor.to(existing.data.dtype))
        else:
            module.wcscales = tensor
        return True

    def _patch_nunchaku_scales(
        self,
        model: nn.Module,
        safetensors_list: list[str],
        quant_config: Any,
    ) -> None:
        """Patch Nunchaku scale tensors from safetensors weights.

        For NVFP4 checkpoints, correctness depends on `wtscale` and attention
        `wcscales`. The FSDP loader may skip some of these metadata tensors.
        """
        if not self._is_svdquant_config(quant_config):
            return

        if not safetensors_list:
            return

        if len(safetensors_list) != 1:
            logger.warning(
                "Nunchaku scale patch expects a single safetensors file, "
                "but got %d files. Skipping.",
                len(safetensors_list),
            )
            return

        try:
            from nunchaku.models.linear import SVDQW4A4Linear  # type: ignore[import]

            from sglang.multimodal_gen.runtime.layers.quantization.nunchaku_linear import (
                NunchakuSVDQLinearMethod,
            )
        except ImportError:
            logger.warning("Nunchaku is not available; skipping scale patch.")
            return

        state_dict = self._load_state_dict_safe(safetensors_list[0])
        if state_dict is None:
            return

        num_wtscale = 0
        num_wcscales = 0
        for name, module in model.named_modules():
            wt = state_dict.get(f"{name}.wtscale")
            if wt is not None:
                if self._patch_native_svdq_linear(module, wt, SVDQW4A4Linear):
                    num_wtscale += 1
                elif self._patch_sglang_svdq_linear(module, wt, NunchakuSVDQLinearMethod):
                    num_wtscale += 1

            wc = state_dict.get(f"{name}.wcscales")
            if wc is not None:
                # Some modules may have wcscales as a direct attribute/Parameter.
                existing = getattr(module, "wcscales", None)
                if isinstance(existing, nn.Parameter):
                    with torch.no_grad():
                        existing.data.copy_(wc.to(existing.data.dtype))
                    num_wcscales += 1
                elif existing is not None:
                    setattr(module, "wcscales", wc)
                    num_wcscales += 1
                elif self._patch_sglang_svdq_wcscales(
                    module, wc, NunchakuSVDQLinearMethod
                ):
                    num_wcscales += 1

        if num_wtscale > 0:
            logger.info("Patched wtscale for %d layers", num_wtscale)
        if num_wcscales > 0:
            logger.info("Patched wcscales for %d layers", num_wcscales)

    def _get_quant_config(self, server_args: ServerArgs) -> Any:
        ncfg = getattr(server_args, "nunchaku_config", None)
        if ncfg is None:
            return None
        if not getattr(ncfg, "enable_svdquant", False):
            return None
        if not getattr(ncfg, "quantized_model_path", None):
            return None

        from sglang.multimodal_gen.runtime.loader.nunchaku_loader import (
            create_nunchaku_config_from_server_args,
        )

        return create_nunchaku_config_from_server_args(server_args)

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        """Load the transformer based on the model path, and inference args."""
        config = get_diffusers_component_config(model_path=component_model_path)
        hf_config = deepcopy(config)
        cls_name = config.pop("_class_name")
        if cls_name is None:
            raise ValueError(
                "Model config does not contain a _class_name attribute. "
                "Only diffusers format is supported."
            )

        component_name = _normalize_component_type(component_name)
        server_args.model_paths[component_name] = component_model_path

        if component_name in ("transformer", "video_dit"):
            pipeline_dit_config_attr = "dit_config"
        elif component_name in ("audio_dit",):
            pipeline_dit_config_attr = "audio_dit_config"
        else:
            raise ValueError(f"Invalid module name: {component_name}")
        # Config from Diffusers supersedes sgl_diffusion's model config
        dit_config = getattr(server_args.pipeline_config, pipeline_dit_config_attr)
        dit_config.update_model_arch(config)

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        default_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]

        quant_config = self._get_quant_config(server_args)

        if quant_config is not None and getattr(quant_config, "quantized_model_path", None):
            weights_path = quant_config.quantized_model_path
            logger.info("Using quantized model weights from: %s", weights_path)
            if os.path.isfile(weights_path) and weights_path.endswith(".safetensors"):
                safetensors_list = [weights_path]
            else:
                safetensors_list = _list_safetensors_files(weights_path)
        else:
            weights_path = component_model_path
            safetensors_list = _list_safetensors_files(weights_path)

        if not safetensors_list:
            raise ValueError(f"No safetensors files found in {weights_path}")

        # Check if we should use custom initialization weights
        custom_weights_path = getattr(
            server_args, "init_weights_from_safetensors", None
        )
        if custom_weights_path is not None:
            logger.info(
                "Using custom initialization weights from: %s", custom_weights_path
            )
            if os.path.isdir(custom_weights_path):
                safetensors_list = _list_safetensors_files(custom_weights_path)
            else:
                if not custom_weights_path.endswith(".safetensors"):
                    raise ValueError(
                        f"Custom initialization weights must be a .safetensors file or directory, "
                        f"got: {custom_weights_path}"
                    )
                safetensors_list = [custom_weights_path]

        logger.info(
            "Loading %s from %s safetensors files, default_dtype: %s",
            cls_name,
            len(safetensors_list),
            default_dtype,
        )

        # Load the model using FSDP loader
        assert server_args.hsdp_shard_dim is not None
        init_params: dict[str, Any] = {"config": dit_config, "hf_config": hf_config}
        if (
            quant_config is not None
            and "quant_config" in inspect.signature(model_cls.__init__).parameters
        ):
            init_params["quant_config"] = quant_config

        model = maybe_load_fsdp_model(
            model_cls=model_cls,
            init_params=init_params,
            weight_dir_list=safetensors_list,
            device=get_local_torch_device(),
            hsdp_replicate_dim=server_args.hsdp_replicate_dim,
            hsdp_shard_dim=server_args.hsdp_shard_dim,
            cpu_offload=server_args.dit_cpu_offload,
            pin_cpu_memory=server_args.pin_cpu_memory,
            fsdp_inference=server_args.use_fsdp_inference,
            # TODO(will): make these configurable
            default_dtype=default_dtype,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=False,
        )

        if quant_config is not None:
            self._patch_nunchaku_scales(model, safetensors_list, quant_config)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded model with %.2fB parameters", total_params / 1e9)

        assert (
            next(model.parameters()).dtype == default_dtype
        ), "Model dtype does not match default dtype"

        return model
