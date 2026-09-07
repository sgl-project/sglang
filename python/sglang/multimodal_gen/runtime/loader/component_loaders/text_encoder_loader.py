import os
import re
from collections.abc import Generator

import torch
import transformers
from torch import nn
from transformers import PretrainedConfig
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from sglang.multimodal_gen.configs.models import EncoderConfig
from sglang.multimodal_gen.configs.pipeline_configs.longcat_image import (
    LongCatImageEditPipelineConfig,
    LongCatImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImageEditPipelineConfig,
)
from sglang.multimodal_gen.runtime.distributed import (
    get_encoder_data_parallel_group,
    get_local_torch_device,
)
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    use_tensor_parallel_group,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    LinearBase,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.comfy_fp8 import ComfyFp8Config
from sglang.multimodal_gen.runtime.layers.quantization.comfy_nvfp4 import (
    ComfyNvfp4Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.base_config import (
    QuantizationConfig,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import (
    KitchenInt8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_w4a4_config import (
    KitchenW4A4Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_w4a8_config import (
    KitchenW4A8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.quanto_int8_config import (
    QuantoInt8Config,
    inspect_quanto_int8_checkpoint,
)
from sglang.multimodal_gen.runtime.layers.quantization.gguf import GGUFConfig
from sglang.multimodal_gen.runtime.layers.quantization.quanto_int8 import (
    normalize_quanto_int8_weights,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
    NativeComponentLoaderRequired,
    OnlineQuantizationComponentLoader,
    uses_native_transformers_quantization,
)
from sglang.multimodal_gen.runtime.loader.gguf_weights import (
    gguf_weights_iterator,
    names_gguf_checkpoint,
    read_gguf_tensor_meta,
    remap_gguf_tensor_meta,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    checkpoint_bytes,
    get_param_names_mapping,
    initialize_model,
    keep_checkpoint_mapped,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    checkpoint_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.encoders.base import (
    EncoderTensorParallelMixin,
    TextEncoder,
    finalize_encoder_folding,
    get_folding_tp_group,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_config,
    get_diffusers_component_config,
    load_dict,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.quantization_utils import (
    get_quant_config,
    get_quant_config_from_safetensors_metadata,
    inspect_comfy_quant_markers,
    process_model_weights_after_loading,
    resolve_comfy_checkpoint_quantization,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE
from sglang.srt.layers.linear import LinearBase as SrtLinearBase
from sglang.srt.layers.quantization.fp8 import Fp8Config as SrtFp8Config
from sglang.srt.layers.quantization.unquant import (
    UnquantizedLinearMethod as SrtUnquantizedLinearMethod,
)
from sglang.srt.model_loader.checkpoint_quantization import (
    resolve_checkpoint_quant_spec,
)

logger = init_logger(__name__)

_ONLINE_ENCODER_QUANTIZATIONS = frozenset({"fp8", "kitchen_int8", "mxfp4"})

_TRANSFORMERS_ENCODER_ONLY_CLASSES = {
    "T5EncoderModel": transformers.T5EncoderModel,
    "T5Model": transformers.T5EncoderModel,
    "T5ForConditionalGeneration": transformers.T5EncoderModel,
    "UMT5EncoderModel": transformers.UMT5EncoderModel,
    "UMT5Model": transformers.UMT5EncoderModel,
    "UMT5ForConditionalGeneration": transformers.UMT5EncoderModel,
    "MT5EncoderModel": transformers.MT5EncoderModel,
    "MT5Model": transformers.MT5EncoderModel,
    "MT5ForConditionalGeneration": transformers.MT5EncoderModel,
}


def _delegate_quantized_checkpoint_to_transformers(
    component_config: dict,
    component_name: str,
    *,
    methods: frozenset[str] | None = None,
) -> None:
    """Use Transformers when it owns the checkpoint's serialized format."""
    quant_spec = resolve_checkpoint_quant_spec(component_config)
    if quant_spec is None or (
        methods is not None and quant_spec.declared_method not in methods
    ):
        return
    if uses_native_transformers_quantization(component_config, component_name):
        method = quant_spec.declared_method or "unspecified"
        raise NativeComponentLoaderRequired(
            f"{component_name!r} delegates serialized quant_method={method!r} "
            "checkpoint loading to Transformers"
        )


def _get_srt_encoder_quant_config(
    component_config: dict,
    model_cls: type[EncoderTensorParallelMixin],
) -> SrtFp8Config | None:
    quant_spec = resolve_checkpoint_quant_spec(component_config)
    if quant_spec is None:
        return None
    if quant_spec.declared_method != "fp8":
        raise ComponentCheckpointUnsupportedError(
            "The SRT encoder checkpoint adapter supports only serialized 'fp8', "
            f"got {quant_spec.declared_method!r}"
        )

    config = dict(quant_spec.config)
    config["packed_modules_mapping"] = model_cls.packed_modules_mapping
    return SrtFp8Config.from_config(config)


def _get_encoder_quant_config(
    component_config: dict,
    component_model_path: str,
    component_weights_path: str,
    model_cls: type[nn.Module] | None = None,
):
    if (
        model_cls is not None
        and issubclass(model_cls, EncoderTensorParallelMixin)
        and model_cls.checkpoint_quantization_backend == "srt"
    ):
        srt_quant_config = _get_srt_encoder_quant_config(
            component_config,
            model_cls,
        )
        if srt_quant_config is not None:
            return srt_quant_config

    quant_config = get_quant_config(component_config, component_model_path)
    name_mapper = None
    parameter_name_mapper = None
    if model_cls is not None:
        mapping = vars(model_cls).get("param_names_mapping", {})
        if mapping:
            mapping_fn = get_param_names_mapping(mapping)

            def parameter_name_mapper(name: str) -> str:
                mapped_name, merge_index, _ = mapping_fn(name)
                if merge_index is not None:
                    raise ValueError(
                        "Serialized quantized component weights cannot use a "
                        "stacked parameter-name mapping"
                    )
                return mapped_name

            def name_mapper(name: str) -> str:
                # Layer-prefix metadata omits the suffix that many model
                # mappings use to delimit a parameter name.
                mapped_name = parameter_name_mapper(f"{name}.weight")
                return mapped_name.removesuffix(".weight")

    if names_gguf_checkpoint(component_weights_path):
        if quant_config is not None:
            raise ValueError(
                "A GGUF encoder checkpoint cannot be combined with a second "
                "quantization declaration"
            )
        tensor_meta = read_gguf_tensor_meta(component_weights_path)
        dequantize_prefixes = (
            vars(model_cls).get("gguf_dequantize_prefixes", ())
            if model_cls is not None
            else ()
        )
        tensor_meta = remap_gguf_tensor_meta(
            tensor_meta,
            parameter_name_mapper or (lambda name: name),
            dequantize_prefixes=dequantize_prefixes,
        )
        return GGUFConfig(component_weights_path, tensor_meta)

    if (
        quant_config is None
        and component_weights_path != component_model_path
        and component_weights_path.endswith(".safetensors")
    ):
        quant_config = get_quant_config_from_safetensors_metadata(
            component_weights_path
        )
    if quant_config is None and component_weights_path.endswith(".safetensors"):
        quant_config = inspect_quanto_int8_checkpoint(
            component_weights_path,
            param_name_mapper=name_mapper,
        )
        if quant_config is None:
            markers = inspect_comfy_quant_markers(
                [component_weights_path],
                param_name_mapper=name_mapper,
            )
            quant_config = resolve_comfy_checkpoint_quantization(markers)
    return quant_config


def _configure_encoder_quantization(
    model_config: EncoderConfig,
    model_cls: type[nn.Module],
    component_config: dict,
    component_model_path: str,
    component_weights_path: str,
    component_name: str,
    explicit_quantization: str | None = None,
    ignored_layers: list[str] | None = None,
) -> None:
    if (
        issubclass(model_cls, EncoderTensorParallelMixin)
        and model_cls.checkpoint_quantization_backend == "model"
    ):
        if explicit_quantization is not None:
            raise ComponentCheckpointUnsupportedError(
                f"{component_name!r} manages its own checkpoint quantization and "
                "does not support an online quantization override"
            )
        # Preserve model-owned formats such as Ideogram's bitsandbytes state.
        # Those models parse metadata, construct layers, and attach quant states
        # themselves; running the generic lifecycle as well would process twice.
        return

    _delegate_quantized_checkpoint_to_transformers(
        component_config,
        component_name,
        methods=frozenset({"bitsandbytes"}),
    )
    try:
        quant_config = _get_encoder_quant_config(
            component_config,
            component_model_path,
            component_weights_path,
            model_cls,
        )
    except (KeyError, NotImplementedError, TypeError, ValueError) as error:
        _delegate_quantized_checkpoint_to_transformers(
            component_config,
            component_name,
        )
        raise ComponentCheckpointUnsupportedError(
            f"Cannot configure checkpoint quantization for {component_name!r}: {error}"
        ) from error
    model_config.quant_config = quant_config
    if explicit_quantization is not None:
        if quant_config is not None:
            raise ComponentCheckpointUnsupportedError(
                f"{component_name!r} already declares checkpoint quantization; "
                "drop the explicit online quantization override"
            )
        if explicit_quantization not in _ONLINE_ENCODER_QUANTIZATIONS:
            raise ComponentCheckpointUnsupportedError(
                f"Online quantization {explicit_quantization!r} is not supported "
                f"for native encoders; choose one of "
                f"{sorted(_ONLINE_ENCODER_QUANTIZATIONS)}"
            )
        from sglang.multimodal_gen.runtime.layers.quantization import (
            get_quantization_config,
        )

        model_config.quant_config = get_quantization_config(explicit_quantization)(
            ignored_layers=ignored_layers
        )
        quant_config = model_config.quant_config
    if quant_config is None:
        _delegate_quantized_checkpoint_to_transformers(
            component_config,
            component_name,
        )
        return
    if not issubclass(model_cls, EncoderTensorParallelMixin):
        raise ComponentCheckpointUnsupportedError(
            f"A quantized {component_name!r} checkpoint requires an in-tree "
            "native encoder; "
            f"got {model_cls.__name__}"
        )


def _resolve_and_configure_encoder_quantization(
    model_config: EncoderConfig,
    component_config: dict,
    component_model_path: str,
    component_weights_path: str,
    component_name: str,
    explicit_quantization: str | None = None,
    ignored_layers: list[str] | None = None,
) -> type[nn.Module]:
    architectures = model_config.arch_config.architectures
    try:
        model_cls, _ = ModelRegistry.resolve_model_cls(architectures)
    except Exception as resolution_error:
        _delegate_quantized_checkpoint_to_transformers(
            component_config,
            component_name,
        )
        try:
            quant_config = _get_encoder_quant_config(
                component_config,
                component_model_path,
                component_weights_path,
            )
        except Exception as quantization_error:
            raise ComponentCheckpointUnsupportedError(
                f"Cannot parse checkpoint quantization for {component_name!r}: "
                f"{quantization_error}"
            ) from quantization_error
        if explicit_quantization is not None and quant_config is None:
            raise ComponentCheckpointUnsupportedError(
                f"Online quantization for {component_name!r} requires an in-tree "
                f"native encoder; unsupported architectures: {architectures}"
            ) from resolution_error
        if quant_config is None:
            raise
        raise ComponentCheckpointUnsupportedError(
            f"A quantized {component_name!r} checkpoint requires an in-tree "
            f"native encoder; unsupported architectures: {architectures}"
        ) from resolution_error

    _configure_encoder_quantization(
        model_config,
        model_cls,
        component_config,
        component_model_path,
        component_weights_path,
        component_name,
        explicit_quantization,
        ignored_layers,
    )
    return model_cls


def _require_quantized_encoder_layers(
    model: nn.Module,
    component_name: str,
    quant_config: QuantizationConfig | None = None,
) -> None:
    has_quantized_layers = any(
        isinstance(module, (LinearBase, SrtLinearBase))
        and module.quant_method is not None
        and not isinstance(
            module.quant_method,
            (UnquantizedLinearMethod, SrtUnquantizedLinearMethod),
        )
        for module in model.modules()
    )
    if not has_quantized_layers:
        raise ComponentCheckpointUnsupportedError(
            f"The native {type(model).__name__} implementation does not construct "
            f"quantized linear layers for {component_name!r}"
        )
    if isinstance(
        quant_config,
        (
            ComfyFp8Config,
            ComfyNvfp4Config,
            KitchenInt8Config,
            KitchenW4A4Config,
            KitchenW4A8Config,
        ),
    ):
        expected = set(quant_config.layer_markers)
        selected = set(quant_config.selected)
    elif isinstance(quant_config, QuantoInt8Config):
        expected = quant_config.layer_prefixes
        selected = quant_config.selected
    elif isinstance(quant_config, GGUFConfig):
        expected = quant_config.quantized_prefixes
        selected = quant_config.selected
    else:
        expected = set()
        selected = set()
    if expected:
        missing = expected - selected
        if missing:
            raise ComponentCheckpointUnsupportedError(
                f"The native {type(model).__name__} implementation did not consume "
                f"serialized quantization markers for {component_name!r}: "
                f"{sorted(missing)[:5]}"
            )


class TextEncoderLoader(OnlineQuantizationComponentLoader):
    """Loader for text encoders."""

    component_names = ["text_encoder"]
    expected_library = "transformers"

    def component_load_precision(
        self, server_args: ServerArgs, component_name: str
    ) -> str | None:
        override = server_args.component_precisions.get(component_name)
        if override is not None:
            return override
        return server_args.pipeline_config.text_encoder_precisions[
            self._extract_encoder_index(self.structural_component_name(component_name))
        ]

    def should_raise_customized_load_error(
        self, server_args: ServerArgs, component_name: str
    ) -> bool:
        return (
            super().should_raise_customized_load_error(server_args, component_name)
            or component_name in server_args.component_quantizations
        )

    def validate_component_weight_override(self, override: str) -> None:
        if names_gguf_checkpoint(override):
            if not current_platform.is_cuda():
                raise ValueError(
                    "GGUF encoder checkpoints require CUDA; the GGML kernels have "
                    f"no {current_platform.device_type} implementation"
                )

    def resolve_native_transformers_model_class(self, config: PretrainedConfig) -> type:
        """Resolve the concrete transformers class for a text encoder.

        AutoModel maps encoder-decoder model types (e.g. T5/UMT5) to full
        seq2seq classes, whose forward expects decoder inputs and raises when
        the module is used purely as a text encoder. For such checkpoints,
        prefer the encoder-only class from the config architectures or map the
        full seq2seq architecture to its encoder-only counterpart. Encoders that
        are not encoder-decoder keep using AutoModel unchanged.
        """
        if config.is_encoder_decoder:
            for arch in config.architectures or []:
                transformers_model_class = _TRANSFORMERS_ENCODER_ONLY_CLASSES.get(arch)
                if transformers_model_class is not None:
                    return transformers_model_class
        return transformers.AutoModel

    def _get_all_weights(
        self,
        model: EncoderTensorParallelMixin,
        model_path: str,
        to_cpu: bool,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        def include_checkpoint_weight(name: str) -> bool:
            return not name.endswith(
                ".comfy_quant"
            ) and model.should_materialize_checkpoint_weight(name)

        yield from checkpoint_weights_iterator(
            model_path,
            to_cpu=to_cpu,
            key_filter=include_checkpoint_weight,
            index_file=SAFE_WEIGHTS_INDEX_NAME,
        )

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        component_starts_on_cpu: bool | None = None,
    ):
        """Load the text encoders based on the model path, and inference args."""
        component_weights_path = self.resolve_component_weights_path(
            component_model_path,
            server_args,
            component_name,
        )
        model_config = get_diffusers_component_config(
            component_path=component_model_path
        )
        encoder_config = self.build_model_config(
            component_model_path, model_config, server_args, component_name
        )
        encoder_config.post_diffusers_config_update()
        model_cls = _resolve_and_configure_encoder_quantization(
            encoder_config,
            model_config,
            component_model_path,
            component_weights_path,
            component_name,
            server_args.component_quantizations.get(component_name),
            server_args.component_quantization_ignored_layers.get(component_name),
        )
        if issubclass(model_cls, EncoderTensorParallelMixin):
            model_cls.configure_component_paths(
                encoder_config,
                server_args.component_paths,
            )
        encoder_dp_group = get_encoder_data_parallel_group()
        prefer_dp = (
            server_args.batching_max_size > 1
            and encoder_dp_group is not None
            and encoder_dp_group.world_size > 1
            and issubclass(model_cls, TextEncoder)
            and model_cls.supports_dp_encode
        )
        # real dims are populated now; resolve fold vs replicate
        finalize_encoder_folding(
            encoder_config,
            server_args.encoder_parallel,
            prefer_dp=prefer_dp,
        )
        encoder_dtype = self.component_load_precision(server_args, component_name)
        assert encoder_dtype is not None
        # TODO(will): add support for other dtypes
        try:
            return self.load_model(
                component_weights_path,
                encoder_config,
                server_args,
                encoder_dtype,
                component_starts_on_cpu=component_starts_on_cpu,
                component_name=component_name,
            )
        except ComponentCheckpointUnsupportedError:
            raise
        except Exception as error:
            if encoder_config.quant_config is None:
                raise
            raise ComponentCheckpointUnsupportedError(
                f"Failed to load quantized native {component_name!r}: {error}"
            ) from error

    def build_model_config(
        self,
        component_model_path: str,
        model_config: dict,
        server_args: ServerArgs,
        component_name: str,
    ) -> EncoderConfig:
        diffusers_pretrained_config = get_config(
            component_model_path, trust_remote_code=True
        )
        encoder_index = self._extract_encoder_index(
            self.structural_component_name(component_name)
        )
        assert encoder_index < len(
            server_args.pipeline_config.text_encoder_configs
        ) and encoder_index < len(server_args.pipeline_config.text_encoder_precisions)

        encoder_config = server_args.pipeline_config.text_encoder_configs[encoder_index]
        encoder_config.update_model_arch(model_config)
        encoder_config.generation_config = load_dict(
            os.path.join(component_model_path, "generation_config.json")
        )

        if encoder_index == 0:
            for key, value in diffusers_pretrained_config.__dict__.items():
                setattr(encoder_config.arch_config, key, value)
        return encoder_config

    @staticmethod
    def _extract_encoder_index(component_name: str) -> int:
        """
        Map text encoder component names to zero-based indices.

        Examples:
        - text_encoder -> 0
        - text_encoder_2 -> 1
        - text_encoder_3 -> 2
        """
        match = re.search(r"_(\d+)$", component_name)
        if match is None:
            return 0

        suffix_num = int(match.group(1))
        if suffix_num <= 0:
            raise ValueError(
                f"Invalid text encoder component name '{component_name}': "
                "numeric suffix must be >= 1."
            )
        return suffix_num - 1

    def load_model(
        self,
        model_path: str,
        model_config: EncoderConfig,
        server_args: ServerArgs,
        dtype: str = "fp16",
        component_starts_on_cpu: bool | None = None,
        component_name: str = "text_encoder",
    ):
        local_torch_device = get_local_torch_device()
        quant_config = model_config.quant_config
        param_dtype = PRECISION_TO_TYPE[dtype]
        if quant_config is not None:
            if param_dtype not in quant_config.get_supported_act_dtypes():
                raise ValueError(
                    f"{component_name!r} quantization method "
                    f"{quant_config.get_name()!r} "
                    f"does not support activation dtype {param_dtype}"
                )
            if current_platform.is_mps():
                raise ValueError(
                    f"{component_name!r} quantization method "
                    f"{quant_config.get_name()!r} is not supported on MPS"
                )
            if current_platform.is_cuda():
                capability = current_platform.get_device_capability()
                if (
                    capability is not None
                    and capability.to_int() < quant_config.get_min_capability()
                ):
                    raise ValueError(
                        f"{component_name!r} quantization method "
                        f"{quant_config.get_name()!r} "
                        "requires CUDA compute capability "
                        f">= {quant_config.get_min_capability() / 10:.1f}; got "
                        f"{capability.to_int() / 10:.1f}"
                    )

        if not current_platform.is_cpu():
            component_starts_on_cpu = (
                component_starts_on_cpu
                if component_starts_on_cpu is not None
                else server_args.should_start_component_on_cpu(component_name)
            )
        else:
            component_starts_on_cpu = False

        if (
            getattr(
                model_config.arch_config, "requires_gpu_resident_text_encoder", False
            )
            and component_starts_on_cpu
        ):
            server_args.require_component_resident(
                component_name, feature_name="bitsandbytes 4-bit text encoder"
            )
            logger.warning(
                "Keeping bitsandbytes 4-bit text encoder GPU-resident; CUDA "
                "weights and quant states are required for this checkpoint."
            )
            component_starts_on_cpu = False

        if component_starts_on_cpu:
            model_device = torch.device("cpu")
        else:
            model_device = local_torch_device

        encoder_tp_group = get_folding_tp_group(model_config)
        with (
            use_tensor_parallel_group(encoder_tp_group),
            set_default_torch_dtype(PRECISION_TO_TYPE[dtype]),
        ):
            model_cls, _ = ModelRegistry.resolve_model_cls(
                model_config.arch_config.architectures
            )
            model_config.enable_image_understanding = isinstance(
                server_args.pipeline_config,
                (QwenImageEditPipelineConfig, LongCatImageEditPipelineConfig),
            )
            # longcat consumes the padded body without an attention cache
            model_config.honor_cache_free_padding_mask = isinstance(
                server_args.pipeline_config, LongCatImagePipelineConfig
            )
            model = initialize_model(
                model_cls, {"config": model_config}, param_dtype, model_device
            )

            if not isinstance(model, EncoderTensorParallelMixin):
                raise TypeError(
                    f"Native encoder {model_cls.__name__} must inherit "
                    "EncoderTensorParallelMixin"
                )
            model.bind_encoder_tp_group(encoder_tp_group)

            if isinstance(quant_config, GGUFConfig):
                quant_config.retain_tensor_meta(
                    model.should_materialize_checkpoint_weight
                )
            if quant_config is not None:
                _require_quantized_encoder_layers(
                    model, component_name, quant_config=quant_config
                )

            if component_starts_on_cpu and (
                current_platform.is_mps()
                or keep_checkpoint_mapped(
                    weight_bytes=checkpoint_bytes(model_path), component=component_name
                )
            ):
                model._keep_checkpoint_mapping = True

            weights_to_load = {name for name, _ in model.named_parameters()}
            if isinstance(quant_config, GGUFConfig):
                checkpoint_weights = gguf_weights_iterator(
                    model_path,
                    quant_config.tensor_meta,
                    key_filter=model.should_materialize_checkpoint_weight,
                )
            else:
                checkpoint_weights = self._get_all_weights(
                    model,
                    model_path,
                    to_cpu=component_starts_on_cpu,
                )
            if isinstance(quant_config, QuantoInt8Config):
                checkpoint_weights = normalize_quanto_int8_weights(checkpoint_weights)
            loaded_weights = model.load_weights(checkpoint_weights)
            self.validate_checkpoint_keys(
                weights_to_load - loaded_weights, [], component_name
            )

            if quant_config is not None and not isinstance(quant_config, GGUFConfig):
                postprocess_device: torch.device | None = local_torch_device
                if isinstance(quant_config, (ComfyNvfp4Config, QuantoInt8Config)) or (
                    isinstance(quant_config, KitchenInt8Config)
                    and quant_config.is_checkpoint_int8_serialized
                ):
                    postprocess_device = None
                processed_layers = process_model_weights_after_loading(
                    model,
                    postprocess_device,
                    quantized_only=True,
                )
                logger.info(
                    "Processed %d %s linear layers for %s",
                    processed_layers,
                    quant_config.get_name(),
                    component_name,
                )

            if component_starts_on_cpu:
                if current_platform.is_mps():
                    logger.info(
                        "Keeping %s on CPU for MPS layerwise offload",
                        model.__class__.__name__,
                    )
                else:
                    model = model.to("cpu")
            else:
                model = model.to(local_torch_device)

        return model
