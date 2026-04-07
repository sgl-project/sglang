import logging
from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.quantization import get_quantization_config
from sglang.multimodal_gen.runtime.layers.quantization.configs.nunchaku_config import (
    NunchakuConfig,
    _patch_nunchaku_scales,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    resolve_transformer_quant_load_spec,
    resolve_transformer_safetensors_to_load,
)
from sglang.multimodal_gen.runtime.loader.utils import _normalize_component_type
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import get_log_level, init_logger
from sglang.srt.utils import is_npu

_is_npu = is_npu()

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
        component_model_path: str,
    ) -> Optional[QuantizationConfig]:
        # priority: CLI flag → model config.json → safetensors metadata → quantization config (nvfp4, nunchaku, ...)

        if server_args.quantization:
            quant_cls = get_quantization_config(server_args.quantization)
            return quant_cls()

        quant_config = get_quant_config(hf_config, component_model_path)
        if quant_config is None and server_args.transformer_weights_path:
            # try to read quantization_config from the safetensors metadata header
            for safetensors_file in safetensors_list:
                quant_config = get_quant_config_from_safetensors_metadata(
                    safetensors_file
                )
                if quant_config:
                    return quant_config

            # fallback: handle nvfp4 per-layer format metadata
            # ({"format_version": ..., "layers": {"name": {"format": "nvfp4"}, ...}})
            param_names_mapping_dict = (
                server_args.pipeline_config.dit_config.arch_config.param_names_mapping
            )
            quant_config = build_nvfp4_config_from_safetensors_list(
                safetensors_list, param_names_mapping_dict
            )
            if quant_config:
                return quant_config
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

        safetensors_list = resolve_transformer_safetensors_to_load(
            server_args, component_model_path
        )

        # 2. dit config
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
        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)

        quant_spec = resolve_transformer_quant_load_spec(
            hf_config=config,
            server_args=server_args,
            safetensors_list=safetensors_list,
            component_model_path=component_model_path,
            model_cls=model_cls,
            cls_name=cls_name,
        )

        logger.info(
            "Loading %s from %s safetensors file(s) %s, param_dtype: %s",
            cls_name,
            len(safetensors_list),
            f": {safetensors_list}" if get_log_level() == logging.DEBUG else "",
            quant_spec.param_dtype,
        )
        # prepare init_param
        init_params: dict[str, Any] = {
            "config": dit_config,
            "hf_config": config,
            "quant_config": quant_spec.runtime_quant_config,
        }
        if (
            init_params["quant_config"] is None
            and server_args.transformer_weights_path is not None
        ):
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
            param_dtype=quant_spec.param_dtype,
            reduce_dtype=torch.float32,
            output_dtype=None,
            strict=False,
        )

        # post-hooks (e.g., patch scales (nunchaku))
        for post_load_hook in quant_spec.post_load_hooks:
            post_load_hook(model)

        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Loaded model with %.2fB parameters", total_params / 1e9)

        # considering the existent of mixed-precision models (e.g., nunchaku)
        if (
            next(model.parameters()).dtype != quant_spec.param_dtype
            and quant_spec.param_dtype
        ):
            logger.warning(
                "Model dtype does not match expected param dtype, %s vs %s",
                next(model.parameters()).dtype,
                quant_spec.param_dtype,
            )

        return model
