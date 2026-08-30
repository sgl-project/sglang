import copy
import logging
from collections.abc import Callable
from contextlib import nullcontext
from typing import Any

import torch

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.layers.attention.selector import (
    component_attn_backend_context_manager,
    get_component_forced_attn_backend,
    get_global_forced_attn_backend,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.fsdp_load import maybe_load_fsdp_model
from sglang.multimodal_gen.runtime.loader.gguf_weights import gguf_weights_iterator
from sglang.multimodal_gen.runtime.loader.minimax_h3_weights import (
    comfy_quant_key_filter,
    inspect_minimax_h3_safetensors,
    resolve_minimax_h3_checkpoint_quantization,
    validate_minimax_h3_checkpoint_variant,
)
from sglang.multimodal_gen.runtime.loader.transformer_load_utils import (
    TransformerQuantLoadSpec,
    resolve_transformer_checkpoint_files,
    resolve_transformer_gguf_to_load,
    resolve_transformer_quant_load_spec,
)
from sglang.multimodal_gen.runtime.loader.utils import _normalize_component_type
from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.platforms import (
    AttentionBackendEnum,
    current_platform,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import get_log_level, init_logger
from sglang.srt.utils import is_npu

_is_npu = is_npu()

logger = init_logger(__name__)


def _resolve_checkpoint_load_device(
    runtime_device: torch.device,
    *,
    component_starts_on_cpu: bool,
    runtime_quant_config: object | None,
    quantized_cpu_load_supported: bool = False,
) -> torch.device:
    if component_starts_on_cpu and (
        runtime_quant_config is None or quantized_cpu_load_supported
    ):
        return torch.device("cpu")
    return runtime_device


def _minimax_h3_adaln_cache_key_filter(name: str) -> bool:
    return ".adaln_proj.linear." not in name


def _default_quantized_attention_backend(
    quant_spec: TransformerQuantLoadSpec, server_args: ServerArgs
) -> AttentionBackendEnum | None:
    """Preserve stable NVFP4 numerics unless the user selected a backend."""
    if not current_platform.is_blackwell() or not quant_spec.is_modelopt_fp4:
        return None
    if (
        get_global_forced_attn_backend() is not None
        or get_component_forced_attn_backend() is not None
        or server_args.attention_backend is not None
    ):
        return None
    return AttentionBackendEnum.FA


def _warn_if_expected_param_dtype_missing(
    model: torch.nn.Module, expected_dtype: torch.dtype | None
) -> None:
    if expected_dtype is None:
        return
    param_dtypes = {param.dtype for param in model.parameters()}
    if expected_dtype not in param_dtypes:
        logger.warning(
            "Model parameter dtypes do not include expected param dtype, %s vs %s",
            param_dtypes,
            expected_dtype,
        )


def _server_args_for_transformer_component(
    server_args: ServerArgs, component_name: str
) -> ServerArgs:
    """Mask global quantized override flags for secondary transformer components."""
    component_weights_path = server_args.component_weights_paths.get(component_name)
    component_quantization = server_args.component_quantizations.get(component_name)
    component_ignored_layers = server_args.component_quantization_ignored_layers.get(
        component_name
    )
    if (
        component_weights_path is not None
        or component_quantization is not None
        or component_ignored_layers is not None
    ):
        component_server_args = copy.copy(server_args)
        if component_weights_path is not None:
            component_server_args.transformer_weights_path = component_weights_path
            component_server_args.nunchaku_config = None
            logger.info(
                "Using transformer_weights_path override for %s: %s",
                component_name,
                component_weights_path,
            )
        if component_quantization is not None:
            component_server_args.quantization = component_quantization
            logger.info(
                "Using quantization override %s for %s",
                component_quantization,
                component_name,
            )
        if component_ignored_layers is not None:
            component_server_args.quantization_ignored_layers = component_ignored_layers
        return component_server_args

    if component_name not in ("transformer_2", "unconditional_transformer"):
        return server_args

    if (
        server_args.transformer_weights_path is None
        and server_args.nunchaku_config is None
    ):
        return server_args

    component_server_args = copy.copy(server_args)
    component_server_args.transformer_weights_path = None
    component_server_args.nunchaku_config = None
    logger.info(
        "Ignoring global transformer_weights_path for %s; keep it on the base "
        "checkpoint unless a per-component override path is provided.",
        component_name,
    )
    return component_server_args


class TransformerLoader(ComponentLoader):
    """Shared loader for (video/audio) DiT transformers."""

    allow_global_attention_backend_fallback = False
    supports_online_quantization_override = True
    supports_fsdp_inference = True

    component_names = [
        "transformer",
        "unconditional_transformer",
        "audio_dit",
        "video_dit",
    ]
    expected_library = "diffusers"

    def customized_load_kwargs_for_component(
        self, server_args: ServerArgs, component_name: str
    ) -> dict[str, bool]:
        if (
            current_platform.is_mps()
            and server_args.should_configure_layerwise_offload_for_lazy_component(
                component_name
            )
        ):
            logger.info(
                "Loading %s on CPU first for MPS layerwise offload", component_name
            )
            return {"cpu_offload_flag": True}
        return {}

    def should_raise_customized_load_error(
        self, server_args: ServerArgs, component_name: str
    ) -> bool:
        component_server_args = _server_args_for_transformer_component(
            server_args, component_name
        )
        # Don't let a quantized load quietly fall back to the unquantized native
        # model. That would drop the requested precision and bury the real error.
        return (
            super().should_raise_customized_load_error(server_args, component_name)
            or component_server_args.transformer_weights_path is not None
            or component_server_args.quantization is not None
        )

    def validate_native_fallback(
        self, server_args: ServerArgs, component_name: str
    ) -> None:
        super().validate_native_fallback(server_args, component_name)
        requested_distributed_execution = []
        if server_args.tp_size is not None and server_args.tp_size > 1:
            requested_distributed_execution.append(f"tp_size={server_args.tp_size}")
        if server_args.sp_degree is not None and server_args.sp_degree > 1:
            requested_distributed_execution.append(f"sp_degree={server_args.sp_degree}")
        if server_args.ulysses_degree is not None and server_args.ulysses_degree > 1:
            requested_distributed_execution.append(
                f"ulysses_degree={server_args.ulysses_degree}"
            )
        if server_args.ring_degree is not None and server_args.ring_degree > 1:
            requested_distributed_execution.append(
                f"ring_degree={server_args.ring_degree}"
            )
        if (
            server_args.kv_gather_degree is not None
            and server_args.kv_gather_degree > 1
        ):
            requested_distributed_execution.append(
                f"kv_gather_degree={server_args.kv_gather_degree}"
            )
        if server_args.should_use_fsdp_for_component(component_name):
            requested_distributed_execution.append("FSDP")
        if requested_distributed_execution:
            raise RuntimeError(
                f"Native Diffusers fallback for transformer component "
                f"{component_name!r} cannot honor requested distributed execution: "
                f"{', '.join(requested_distributed_execution)}. Use an SGLang-native "
                "transformer implementation or set tp_size, sp_degree, "
                "ulysses_degree, ring_degree, and kv_gather_degree to 1 without "
                "FSDP."
            )

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        cpu_offload_flag: bool = False,
    ):
        """Load the transformer based on the model path, and inference args."""
        component_server_args = _server_args_for_transformer_component(
            server_args, component_name
        )

        # 1. hf config
        config = get_diffusers_component_config(component_path=component_model_path)

        gguf_file = resolve_transformer_gguf_to_load(
            component_server_args, component_name
        )
        if gguf_file is not None:
            # A GGUF file holds the whole transformer; the remaining components
            # still load from the base model path.
            safetensors_list = []
            transformer_override_config_path = None
        else:
            checkpoint_files = resolve_transformer_checkpoint_files(
                component_server_args, component_model_path
            )
            safetensors_list = list(checkpoint_files.safetensors)
            transformer_override_config_path = checkpoint_files.config_path

        # 2. dit config
        # Config from Diffusers supersedes sgl_diffusion's model config
        component_type = _normalize_component_type(component_name)
        server_args.model_paths[component_name] = component_model_path
        if component_type in (
            "transformer",
            "unconditional_transformer",
            "video_dit",
        ):
            pipeline_dit_config_attr = "dit_config"
        elif component_type == "audio_dit":
            pipeline_dit_config_attr = "audio_dit_config"
        else:
            raise ValueError(f"Invalid module name: {component_name}")
        dit_config = getattr(server_args.pipeline_config, pipeline_dit_config_attr)
        dit_config.update_model_arch(config)

        cls_name = config.pop("_class_name")
        model_cls, _ = ModelRegistry.resolve_model_cls(cls_name)
        is_minimax_h3 = model_cls.__name__ == "MiniMaxH3DiTModel"
        if is_minimax_h3:
            dit_config.arch_config.checkpoint_uses_diffusers_layout = (
                cls_name != model_cls.__name__
            )

        checkpoint_quant_config = None
        if is_minimax_h3:
            selected_variant = str(component_server_args.model_variant or "fl2va")
            if gguf_file is not None:
                validate_minimax_h3_checkpoint_variant([gguf_file], selected_variant)
            elif component_server_args.transformer_weights_path is not None:
                validate_minimax_h3_checkpoint_variant(
                    safetensors_list, selected_variant
                )
                adaln_curve_shape, layer_markers = inspect_minimax_h3_safetensors(
                    safetensors_list
                )
                checkpoint_quant_config = resolve_minimax_h3_checkpoint_quantization(
                    layer_markers,
                    safetensors_list,
                    dit_config.arch_config.param_names_mapping,
                    dit_config.arch_config.reverse_param_names_mapping,
                )
                if adaln_curve_shape is not None:
                    (
                        dit_config.arch_config.adaln_curve_grid,
                        dit_config.arch_config.time_embed_dim,
                    ) = adaln_curve_shape
                    if (
                        component_server_args.minimax_h3_adaln_cache_path is not None
                        or component_server_args.minimax_h3_adaln_online
                    ):
                        raise ValueError(
                            "MiniMax-H3 pruned curve checkpoints cannot use a "
                            "separate AdaLN cache or online AdaLN rebuild"
                        )

        quant_spec = resolve_transformer_quant_load_spec(
            hf_config=config,
            server_args=component_server_args,
            safetensors_list=safetensors_list,
            component_model_path=component_model_path,
            model_cls=model_cls,
            cls_name=cls_name,
            component_name=component_name,
            gguf_file=gguf_file,
            checkpoint_quant_config=checkpoint_quant_config,
            transformer_override_config_path=transformer_override_config_path,
        )
        if quant_spec.gguf_file is not None and is_minimax_h3:
            assert quant_spec.quant_config is not None
            curve = quant_spec.quant_config.tensor_meta.get("adaln_t_table")
            if curve is not None:
                if curve.is_quantized or len(curve.logical_shape) != 2:
                    raise ValueError(
                        "MiniMax-H3 adaln_t_table must be an unquantized 2D tensor"
                    )
                curve_grid, time_embed_dim = curve.logical_shape
                if curve_grid < 2:
                    raise ValueError("MiniMax-H3 adaln_t_table needs at least two rows")
                dit_config.arch_config.adaln_curve_grid = curve_grid
                dit_config.arch_config.time_embed_dim = time_embed_dim
        # Quantization adapters may require resident weights, so placement must
        # be resolved after they have validated the component configuration.
        component_starts_on_cpu = (
            server_args.should_start_component_on_cpu(component_name)
            or cpu_offload_flag
        )
        use_fsdp = server_args.should_use_fsdp_for_component(component_name)
        if quant_spec.uses_comfy_layer_markers and use_fsdp:
            raise ValueError(
                "Comfy quantized checkpoints do not support FSDP "
                "inference; use TP and/or sequence parallelism instead"
            )
        if (
            use_fsdp
            and quant_spec.quant_config is not None
            and quant_spec.quant_config.get_name() == "auto-round"
        ):
            raise ValueError(
                "AutoRound checkpoints do not support diffusion FSDP inference; "
                "use TP and/or sequence parallelism instead"
            )

        if quant_spec.gguf_file is not None:
            logger.info(
                "Loading %s from GGUF file %s, param_dtype: %s",
                cls_name,
                quant_spec.gguf_file,
                quant_spec.param_dtype,
            )
        else:
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
        checkpoint_key_filter: Callable[[str], bool] | None = (
            comfy_quant_key_filter if quant_spec.uses_comfy_layer_markers else None
        )
        adaln_cache_path = component_server_args.minimax_h3_adaln_cache_path
        if adaln_cache_path is not None:
            if not is_minimax_h3:
                raise ValueError(
                    "--minimax-h3-adaln-cache-path is only supported by MiniMax H3"
                )
            if component_server_args.model_variant not in ("fl2va", "ref2va"):
                raise ValueError(
                    "MiniMax H3 AdaLN cache requires --model-variant fl2va or ref2va"
                )
            init_params["adaln_cache_path"] = adaln_cache_path
            init_params["adaln_cache_model_variant"] = (
                component_server_args.model_variant
            )
            checkpoint_key_filter = _minimax_h3_adaln_cache_key_filter
        if component_server_args.minimax_h3_adaln_online:
            if not is_minimax_h3:
                raise ValueError(
                    "--minimax-h3-adaln-online is only supported by MiniMax H3"
                )
            if adaln_cache_path is not None:
                raise ValueError(
                    "--minimax-h3-adaln-online and --minimax-h3-adaln-cache-path "
                    "are mutually exclusive"
                )
            # Keep the weights off-device; the model rebuilds the AdaLN
            # outputs from the checkpoint for each request's timestep plan.
            init_params["adaln_weight_files"] = safetensors_list
            init_params["adaln_plan_width"] = (
                component_server_args.minimax_h3_adaln_plan_width
            )
            checkpoint_key_filter = _minimax_h3_adaln_cache_key_filter

        runtime_quant_config = init_params["quant_config"]
        if runtime_quant_config is not None:
            logger.debug(
                "Runtime quantization: %s", type(runtime_quant_config).__name__
            )
        elif component_server_args.transformer_weights_path is not None:
            logger.info(
                "Using an unquantized transformer weight override from %s",
                component_server_args.transformer_weights_path,
            )

        local_torch_device = get_local_torch_device()
        checkpoint_load_device = (
            torch.device("cpu")
            if cpu_offload_flag
            else _resolve_checkpoint_load_device(
                local_torch_device,
                component_starts_on_cpu=component_starts_on_cpu,
                runtime_quant_config=quant_spec.runtime_quant_config,
                quantized_cpu_load_supported=(
                    quant_spec.gguf_file is not None
                    or quant_spec.is_serialized_kitchen_int8
                    or quant_spec.is_serialized_kitchen_w4a8
                ),
            )
        )
        direct_gpu_weight_loading = bool(
            component_server_args.direct_gpu_weight_loading
        )
        if direct_gpu_weight_loading and quant_spec.runtime_quant_config is not None:
            raise ValueError(
                "--direct-gpu-weight-loading supports only unquantized DiT checkpoints"
            )
        weight_load_plan = WeightLoadPlan.for_component(
            checkpoint_load_device=checkpoint_load_device,
            needs_device_weight_postprocess=quant_spec.needs_device_weight_postprocess,
            component_starts_on_cpu=component_starts_on_cpu,
            load_full_state_dict_on_device=direct_gpu_weight_loading,
            mps_layerwise_cpu_staging=bool(
                cpu_offload_flag and current_platform.is_mps()
            ),
        )
        if direct_gpu_weight_loading:
            logger.warning(
                "Direct GPU weight loading is enabled for %s; compatible checkpoint "
                "tensors become model storage, while transformed tensors may still "
                "require temporary GPU allocations",
                component_name,
            )

        quantized_attn_backend = _default_quantized_attention_backend(
            quant_spec, component_server_args
        )
        if quantized_attn_backend is not None:
            logger.info(
                "Using %s attention for ModelOpt NVFP4 to preserve output precision",
                quantized_attn_backend.name.lower(),
            )
        attn_backend_context = (
            component_attn_backend_context_manager(
                quantized_attn_backend, component_name=component_name
            )
            if quantized_attn_backend is not None
            else nullcontext()
        )

        # Model construction resolves attention implementations, so apply the
        # quantization-specific default around FSDP initialization and loading.
        with attn_backend_context:
            model = maybe_load_fsdp_model(
                model_cls=model_cls,
                init_params=init_params,
                weight_dir_list=safetensors_list,
                device=local_torch_device,
                hsdp_replicate_dim=server_args.hsdp_replicate_dim,
                hsdp_shard_dim=server_args.hsdp_shard_dim,
                component_starts_on_cpu=component_starts_on_cpu,
                pin_cpu_memory=component_server_args.pin_cpu_memory,
                fsdp_inference=use_fsdp,
                param_dtype=quant_spec.param_dtype,
                reduce_dtype=torch.float32,
                output_dtype=None,
                strict=False,
                weight_load_plan=weight_load_plan,
                checkpoint_key_filter=checkpoint_key_filter,
                weights_iterator=(
                    gguf_weights_iterator(
                        quant_spec.gguf_file,
                        quant_spec.quant_config.tensor_meta,
                        key_filter=checkpoint_key_filter,
                    )
                    if quant_spec.gguf_file is not None
                    else None
                ),
            )

        # post-hooks (e.g., patch scales (nunchaku))
        for post_load_hook in quant_spec.post_load_hooks:
            post_load_hook(model)

        _warn_if_expected_param_dtype_missing(model, quant_spec.param_dtype)

        return model
