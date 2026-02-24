import inspect
import json
import logging
import os
from typing import Any, Dict, List, Optional

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
    _patch_nunchaku_scales,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    _normalize_component_type,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
    get_metadata_from_safetensors_file,
    get_quant_config,
    get_quant_config_from_safetensors_metadata,
    maybe_download_model,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import get_log_level, init_logger
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE

logger = init_logger(__name__)


class TransformerLoader(ComponentLoader):
    """Shared loader for (video/audio) DiT transformers."""

    component_names = ["transformer", "audio_dit", "video_dit"]
    expected_library = "diffusers"

    def get_list_of_safetensors_to_load(
        self, server_args: ServerArgs, component_model_path: str
    ) -> list[str]:
        """
        get list of safetensors to load.

        If --transformer-weights-path is provided, load weights from that path
        instead of the base model's component directory.
        """
        quantized_path = server_args.transformer_weights_path

        if quantized_path:
            quantized_path = maybe_download_model(quantized_path)
            logger.info("using quantized transformer weights from: %s", quantized_path)
            if os.path.isfile(quantized_path) and quantized_path.endswith(
                ".safetensors"
            ):
                safetensors_list = [quantized_path]
            else:
                safetensors_list = _list_safetensors_files(quantized_path)
        else:
            safetensors_list = _list_safetensors_files(component_model_path)

        if not safetensors_list:
            raise ValueError(
                f"no safetensors files found in "
                f"{quantized_path or component_model_path}"
            )

        return safetensors_list

    def _resolve_quant_config(
        self,
        hf_config: Dict[str, List[str]],
        server_args: ServerArgs,
        safetensors_list: list[str],
    ) -> Optional[dict]:
        # priority: model config.json → safetensors metadata → nunchaku config
        quant_config = get_quant_config(hf_config)
        if quant_config is None and server_args.transformer_weights_path:
            # try to read quantization_config from the safetensors metadata header
            for safetensors_file in safetensors_list:
                quant_config = get_quant_config_from_safetensors_metadata(
                    safetensors_file
                )
                if quant_config:
                    break
        return quant_config

    def _resolve_target_param_dtype(
        self,
        quant_config: Optional[dict],
        nunchaku_config: Optional[NunchakuConfig],
        model_cls,
        server_args: ServerArgs,
    ) -> Optional[torch.dtype]:
        if quant_config is not None or nunchaku_config is not None:
            # TODO: improve the condition
            # respect dtype from checkpoint
            param_dtype = None
        else:
            param_dtype = PRECISION_TO_TYPE[server_args.pipeline_config.dit_precision]

        if nunchaku_config is not None:
            nunchaku_config.model_cls = model_cls
            # verify that the nunchaku checkpoint matches the selected model class
            original_dit_cls_name = json.loads(
                get_metadata_from_safetensors_file(
                    nunchaku_config.transformer_weights_path
                ).get("config")
            )["_class_name"]
            specified_dit_cls_name = str(model_cls.__name__)
            if original_dit_cls_name != specified_dit_cls_name:
                raise Exception(
                    f"Class name of DiT specified in nunchaku transformer_weights_path: {original_dit_cls_name} does not match that of specified DiT name: {specified_dit_cls_name}"
                )

        return param_dtype

    def load_customized(
        self, component_model_path: str, server_args: ServerArgs, component_name: str
    ):
        """Load the transformer based on the model path, and inference args."""
        # 1. hf config
        config = get_diffusers_component_config(component_path=component_model_path)

        # 2. quant config
        safetensors_list = self.get_list_of_safetensors_to_load(
            server_args, component_model_path
        )
        quant_config = self._resolve_quant_config(config, server_args, safetensors_list)

        # 3. dit config
        # Config from Diffusers supersedes sgl_diffusion's model config
        component_name = _normalize_component_type(component_name)
        server_args.model_paths[component_name] = component_model_path
        if component_name in ("transformer", "video_dit"):
            pipeline_dit_config_attr = "dit_config"
        elif component_name in ("audio_dit",):
            pipeline_dit_config_attr = "audio_dit_config"
        else:
            raise ValueError(f"Invalid module name: {component_name}")
        dit_config = getattr(server_args.pipeline_config, pipeline_dit_config_attr)
        dit_config.update_model_arch(config)

        cls_name = config.pop("_class_name")
        if cls_name is None:
            raise ValueError(
                "Model config does not contain a _class_name attribute. "
                "Only diffusers format is supported."
            )

        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        nunchaku_config = server_args.nunchaku_config
        param_dtype = self._resolve_target_param_dtype(
            quant_config, nunchaku_config, model_cls, server_args
        )

        logger.info(
            "Loading %s from %s safetensors file(s) %s, param_dtype: %s",
            cls_name,
            len(safetensors_list),
            f": {safetensors_list}" if get_log_level() == logging.DEBUG else "",
            param_dtype,
        )

        init_params: dict[str, Any] = {"config": dit_config, "hf_config": config}
        # prepare init_param
        if "quant_config" in inspect.signature(model_cls.__init__).parameters:
            init_params.update(
                {
                    "quant_config": (quant_config if quant_config else nunchaku_config),
                }
            )
            if init_params["quant_config"] is None:
                logger.warning(
                    f"transformer_weights_path provided, but quantization config not resolved, which is unexpected and likely to cause errors"
                )
            else:
                logger.debug("quantization config: %s", init_params["quant_config"])

        # Load the model using FSDP loader
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
            param_dtype=param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=False,
        )

        if nunchaku_config is not None:
            _patch_nunchaku_scales(model, safetensors_list)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded model with %.2fB parameters", total_params / 1e9)

        # considering the existent of mixed-precision models (e.g., nunchaku)
        if next(model.parameters()).dtype != param_dtype and param_dtype:
            logger.warning(
                f"Model dtype does not match expected param dtype, {next(model.parameters()).dtype} vs {param_dtype}"
            )

        return model
