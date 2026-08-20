import dataclasses
import glob
import os
import re
from collections.abc import Callable, Generator, Iterable
from itertools import chain
from typing import cast

import torch
from torch import nn
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from sglang.multimodal_gen.configs.models import EncoderConfig
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
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    set_default_torch_dtype,
    skip_init_modules,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    filter_duplicate_safetensors_files,
    filter_files_not_needed_for_inference,
    pt_weights_iterator,
    safetensors_weights_iterator,
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
from sglang.multimodal_gen.runtime.utils.precision import precision_to_dtype
from sglang.multimodal_gen.runtime.utils.quantization_utils import get_quant_config
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE
from sglang.srt.environ import envs

logger = init_logger(__name__)


def _configure_encoder_quantization(
    model_config: EncoderConfig,
    model_cls: type[nn.Module],
    component_config: dict,
    component_model_path: str,
    component_name: str,
) -> None:
    if getattr(model_cls, "manages_checkpoint_quantization", False):
        # Preserve model-owned formats such as Ideogram's bitsandbytes state.
        # Those models parse metadata, construct layers, and attach quant states
        # themselves; running the generic lifecycle as well would process twice.
        return

    try:
        quant_config = get_quant_config(
            component_config,
            component_model_path,
        )
    except (KeyError, ValueError) as error:
        raise ComponentCheckpointUnsupportedError(
            f"Cannot configure checkpoint quantization for {component_name!r}: {error}"
        ) from error
    model_config.quant_config = quant_config
    if quant_config is None:
        return
    if not issubclass(model_cls, EncoderTensorParallelMixin):
        raise ComponentCheckpointUnsupportedError(
            f"A quantized {component_name!r} checkpoint requires an in-tree "
            "native encoder; "
            f"got {model_cls.__name__}"
        )

    capability = model_cls.checkpoint_quantization_capability
    if capability is None:
        raise ComponentCheckpointUnsupportedError(
            f"{model_cls.__name__} does not support quantized checkpoints for "
            f"{component_name!r}: no checkpoint quantization capability is declared"
        )
    if capability.backend != "diffusion":
        raise ComponentCheckpointUnsupportedError(
            f"{model_cls.__name__} declares the {capability.backend!r} checkpoint "
            f"quantization backend for {component_name!r}, but the native encoder "
            "loader currently supports only the 'diffusion' backend"
        )

    quant_method = quant_config.get_name()
    if quant_method not in capability.methods:
        raise ComponentCheckpointUnsupportedError(
            f"{model_cls.__name__} does not support {component_name!r} checkpoints "
            f"quantized with {quant_method!r}; supported methods for the "
            f"{capability.backend!r} backend: {sorted(capability.methods)}"
        )


def _resolve_and_configure_encoder_quantization(
    model_config: EncoderConfig,
    component_config: dict,
    component_model_path: str,
    component_name: str,
) -> type[nn.Module]:
    architectures = getattr(model_config, "architectures", [])
    try:
        model_cls, _ = ModelRegistry.resolve_model_cls(architectures)
    except Exception as resolution_error:
        try:
            quant_config = get_quant_config(component_config, component_model_path)
        except Exception as quantization_error:
            raise ComponentCheckpointUnsupportedError(
                f"Cannot parse checkpoint quantization for {component_name!r}: "
                f"{quantization_error}"
            ) from quantization_error
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
        component_name,
    )
    return model_cls


def _module_tensor_device(module: nn.Module) -> torch.device | None:
    """Return the device of a module's own tensors.

    Quantized linear layers are expected to keep their parameters and buffers
    together.  Failing explicitly is safer than staging only part of a layer.
    """

    devices = {
        tensor.device
        for tensor in chain(
            module.parameters(recurse=False),
            module.buffers(recurse=False),
        )
    }
    if len(devices) > 1:
        raise ValueError(
            f"Cannot stage {type(module).__name__} with tensors on multiple "
            f"devices: {sorted(map(str, devices))}"
        )
    return next(iter(devices), None)


def _process_quantized_encoder_weights(
    model: nn.Module,
    process_device: torch.device,
    component_name: str,
) -> int:
    processed_layers = 0
    for module in model.modules():
        if not isinstance(module, LinearBase):
            continue
        quant_method = module.quant_method
        if quant_method is None or isinstance(quant_method, UnquantizedLinearMethod):
            continue

        origin_device = _module_tensor_device(module)
        should_stage = origin_device is not None and origin_device != process_device
        if should_stage:
            module.to(process_device)
        try:
            quant_method.process_weights_after_loading(module)
            processed_layers += 1
        finally:
            # Post-load methods may replace parameters or register buffers. Move
            # the complete layer back so component residency remains authoritative.
            if should_stage:
                module.to(origin_device)
    if processed_layers == 0:
        raise ValueError(
            f"The {component_name!r} checkpoint declares quantization, but the "
            "model did not construct any quantized linear layers"
        )
    return processed_layers


class TextEncoderLoader(ComponentLoader):
    """Loader for text encoders."""

    component_names = ["text_encoder"]
    expected_library = "transformers"

    @dataclasses.dataclass
    class Source:
        """A source for weights."""

        model_or_path: str
        """The model ID or path."""

        prefix: str = ""
        """A prefix to prepend to all weights."""

        fall_back_to_pt: bool = True
        """Whether .pt weights can be used."""

        allow_patterns_overrides: list[str] | None = None
        """If defined, weights will load exclusively using these patterns."""

    def load_native(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        transformers_or_diffusers: str,
        component_name: str | None = None,
    ):
        if transformers_or_diffusers != "transformers":
            return super().load_native(
                component_model_path,
                server_args,
                transformers_or_diffusers,
                component_name,
            )

        encoder_idx = (
            self._extract_encoder_index(component_name or "text_encoder_2")
            if component_name
            else 1 if component_model_path.rstrip("/").endswith("text_encoder_2") else 0
        )
        encoder_dtype = server_args.pipeline_config.text_encoder_precisions[encoder_idx]
        dtype = precision_to_dtype(
            encoder_dtype,
            f"text_encoder_precisions[{encoder_idx}]",
        )
        transformers_model_class = self._resolve_transformers_text_encoder_class(
            component_model_path, server_args
        )
        return transformers_model_class.from_pretrained(
            component_model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            torch_dtype=dtype,
        )

    @staticmethod
    def _resolve_transformers_text_encoder_class(component_model_path, server_args):
        """Resolve the concrete transformers class for a text encoder.

        AutoModel maps encoder-decoder model types (e.g. T5/UMT5) to full
        seq2seq classes, whose forward expects decoder inputs and raises when
        the module is used purely as a text encoder. For such checkpoints,
        prefer the encoder-only class from the config architectures or map the
        full seq2seq architecture to its encoder-only counterpart. Encoders that
        are not encoder-decoder keep using AutoModel unchanged.
        """
        import transformers
        from transformers import AutoConfig, AutoModel

        try:
            config = AutoConfig.from_pretrained(
                component_model_path,
                trust_remote_code=server_args.trust_remote_code,
                revision=server_args.revision,
            )
        except Exception:
            return AutoModel
        if getattr(config, "is_encoder_decoder", False):
            encoder_only_map = {
                "T5Model": "T5EncoderModel",
                "T5ForConditionalGeneration": "T5EncoderModel",
                "UMT5Model": "UMT5EncoderModel",
                "UMT5ForConditionalGeneration": "UMT5EncoderModel",
                "MT5Model": "MT5EncoderModel",
                "MT5ForConditionalGeneration": "MT5EncoderModel",
            }
            for arch in getattr(config, "architectures", None) or []:
                encoder_arch = encoder_only_map.get(arch, arch)
                transformers_model_class = getattr(transformers, encoder_arch, None)
                if isinstance(transformers_model_class, type):
                    return transformers_model_class
        return AutoModel

    def _prepare_weights(
        self,
        model_name_or_path: str,
        fall_back_to_pt: bool,
        allow_patterns_overrides: list[str] | None,
        key_filter: Callable[[str], bool] | None = None,
    ) -> tuple[str, list[str], bool]:
        """Prepare weights for the model.

        If the model is not local, it will be downloaded."""
        # model_name_or_path = (self._maybe_download_from_modelscope(
        #     model_name_or_path, revision) or model_name_or_path)

        is_local = os.path.isdir(model_name_or_path)
        assert is_local, "Model path must be a local directory"

        use_safetensors = False
        index_file = SAFE_WEIGHTS_INDEX_NAME
        allow_patterns = ["*.safetensors", "*.bin"]

        if fall_back_to_pt:
            allow_patterns += ["*.pt"]

        if allow_patterns_overrides is not None:
            allow_patterns = allow_patterns_overrides

        hf_folder = model_name_or_path

        hf_weights_files: list[str] = []
        for pattern in allow_patterns:
            hf_weights_files += glob.glob(os.path.join(hf_folder, pattern))
            if len(hf_weights_files) > 0:
                if pattern == "*.safetensors":
                    use_safetensors = True
                break

        if use_safetensors:
            hf_weights_files = filter_duplicate_safetensors_files(
                hf_weights_files,
                hf_folder,
                index_file,
                key_filter=key_filter,
            )
        else:
            hf_weights_files = filter_files_not_needed_for_inference(hf_weights_files)

        if len(hf_weights_files) == 0:
            raise RuntimeError(
                f"Cannot find any model weights with `{model_name_or_path}`"
            )

        # Sort weight files when SGLANG_SORT_WEIGHT_FILES >= 0 (default).
        # Staggering is not applicable to text-encoder loading (no TP split).
        if envs.SGLANG_SORT_WEIGHT_FILES.get() >= 0:
            hf_weights_files.sort()

        return hf_folder, hf_weights_files, use_safetensors

    def _get_weights_iterator(
        self,
        source: "Source",
        to_cpu: bool,
        key_filter: Callable[[str], bool] | None = None,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        """get an iterator for the model weights based on the load format."""
        source_key_filter: Callable[[str], bool] | None
        if key_filter is None:
            source_key_filter = None
        else:

            def include_source_weight(name: str) -> bool:
                return key_filter(source.prefix + name)

            source_key_filter = include_source_weight

        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path,
            source.fall_back_to_pt,
            source.allow_patterns_overrides,
            key_filter=source_key_filter,
        )
        if use_safetensors:
            weights_iterator = safetensors_weights_iterator(
                hf_weights_files,
                to_cpu=to_cpu,
                key_filter=source_key_filter,
            )
        else:
            weights_iterator = pt_weights_iterator(hf_weights_files, to_cpu=to_cpu)
            if source_key_filter is not None:
                weights_iterator = (
                    (name, tensor)
                    for name, tensor in weights_iterator
                    if source_key_filter(name)
                )

        # apply the prefix.
        return ((source.prefix + name, tensor) for (name, tensor) in weights_iterator)

    def _get_all_weights(
        self,
        model: nn.Module,
        model_path: str,
        to_cpu: bool,
    ) -> Generator[tuple[str, torch.Tensor], None, None]:
        key_filter = cast(
            Callable[[str], bool] | None,
            getattr(model, "should_materialize_checkpoint_weight", None),
        )
        primary_weights = TextEncoderLoader.Source(
            model_path,
            prefix="",
            fall_back_to_pt=getattr(model, "fall_back_to_pt_during_load", True),
            allow_patterns_overrides=getattr(model, "allow_patterns_overrides", None),
        )
        yield from self._get_weights_iterator(
            primary_weights,
            to_cpu,
            key_filter,
        )

        secondary_weights = cast(
            Iterable[TextEncoderLoader.Source],
            getattr(model, "secondary_weights", ()),
        )
        for source in secondary_weights:
            yield from self._get_weights_iterator(
                source,
                to_cpu,
                key_filter,
            )

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        component_starts_on_cpu: bool | None = None,
    ):
        """Load the text encoders based on the model path, and inference args."""
        diffusers_pretrained_config = get_config(
            component_model_path, trust_remote_code=True
        )
        model_config = get_diffusers_component_config(
            component_path=component_model_path
        )

        # TODO(mick): had to throw an exception for different text-encoder arch
        encoder_index = self._extract_encoder_index(component_name)
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
        post_diffusers_config_update = getattr(
            encoder_config, "post_diffusers_config_update", None
        )
        if post_diffusers_config_update is not None:
            post_diffusers_config_update()
        model_cls = _resolve_and_configure_encoder_quantization(
            encoder_config,
            model_config,
            component_model_path,
            component_name,
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
        encoder_dtype = server_args.pipeline_config.text_encoder_precisions[
            encoder_index
        ]
        # TODO(will): add support for other dtypes
        return self.load_model(
            component_model_path,
            encoder_config,
            server_args,
            encoder_dtype,
            component_starts_on_cpu=component_starts_on_cpu,
            component_name=component_name,
        )

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
        with use_tensor_parallel_group(encoder_tp_group), set_default_torch_dtype(
            PRECISION_TO_TYPE[dtype]
        ):
            with model_device, skip_init_modules():
                architectures = getattr(model_config, "architectures", [])
                model_cls, _ = ModelRegistry.resolve_model_cls(architectures)
                enable_image_understanding = (
                    True
                    if isinstance(
                        server_args.pipeline_config, QwenImageEditPipelineConfig
                    )
                    else False
                )
                model_config.enable_image_understanding = enable_image_understanding
                model = model_cls(model_config)

            if not isinstance(model, EncoderTensorParallelMixin):
                raise TypeError(
                    f"Native encoder {model_cls.__name__} must inherit "
                    "EncoderTensorParallelMixin"
                )
            model.bind_encoder_tp_group(encoder_tp_group)

            if current_platform.is_mps() and component_starts_on_cpu:
                # the h3 encoder is layered immediately after this loader returns
                # compatible CPU safetensors stay mapped instead of copying the
                # full Qwen checkpoint into unified memory
                model._mps_zero_copy_weight_loading = True

            weights_to_load = {name for name, _ in model.named_parameters()}
            loaded_weights = model.load_weights(
                self._get_all_weights(
                    model,
                    model_path,
                    to_cpu=component_starts_on_cpu,
                )
            )

            if quant_config is not None:
                processed_layers = _process_quantized_encoder_weights(
                    model,
                    local_torch_device,
                    component_name,
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
            # We only enable strict check for non-quantized models
            # that have loaded weights tracking currently.
            # if loaded_weights is not None:
            weights_not_loaded = weights_to_load - loaded_weights
            if weights_not_loaded:
                # NOTE:
                # If we silently continue with uninitialized weights, the text encoder can
                # produce NaNs/garbage embeddings that later fail stage verification in a
                # hard-to-debug way (e.g., `prompt_embeds` fails the NaN check).
                #
                # We allow a small set of known-optional parameters to be missing, but
                # default to strict behavior for the rest.
                allowed_missing_patterns = (
                    getattr(model, "_allowed_missing_weights_patterns", []) or []
                )
                unexpected_missing = {
                    n
                    for n in weights_not_loaded
                    if not any(pat in n for pat in allowed_missing_patterns)
                }
                if unexpected_missing:
                    raise ValueError(
                        "Following text encoder weights were not initialized from checkpoint: "
                        f"{sorted(unexpected_missing)}. "
                        "This usually indicates a checkpoint/model-arch mismatch or a broken "
                        "weight-name mapping. If these are truly optional, set "
                        "`model._allowed_missing_weights_patterns` to whitelist patterns."
                    )
                logger.warning(
                    "Following (allowed) text encoder weights were not initialized from "
                    "checkpoint: %s (allowed patterns: %s)",
                    sorted(weights_not_loaded),
                    allowed_missing_patterns,
                )

        return model
