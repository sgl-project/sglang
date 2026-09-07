import hashlib
import importlib.util
import os

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import safe_open
from safetensors.torch import save_file as safetensors_save_file

from sglang.multimodal_gen import envs
from sglang.multimodal_gen.configs.models.vaes.base import VAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.qwen_image import (
    QwenImagePipelineConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.wan import WanT2V480PConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
    NativeComponentLoaderRequired,
    WeightOverrideComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    _list_safetensors_files,
    _normalize_component_type,
    adopt_plain_weight_norm_state,
    checkpoint_bytes,
    initialize_model,
    keep_checkpoint_mapped,
    load_model_state_dict,
    set_default_torch_dtype,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    safetensors_weights_iterator,
)
from sglang.multimodal_gen.runtime.models.registry import ModelRegistry
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import (
    autocast_enabled,
    resolve_component_precision,
    resolve_decode_precision,
)
from sglang.multimodal_gen.runtime.weights.source import (
    filter_duplicate_precision_variant_safetensors,
)
from sglang.multimodal_gen.utils import PRECISION_TO_TYPE
from sglang.srt.model_loader.checkpoint_quantization import (
    resolve_checkpoint_quant_spec,
)

logger = init_logger(__name__)
VAE_CHANNELS_LAST_3D_ENV = "SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D"
_VAE_CHECKPOINT_ARCH_METADATA = ("latents_mean", "latents_std")


def _require_native_loader_for_quantized_vae(
    config: dict,
    component_name: str,
    *,
    native_only: bool = False,
    direct_gpu_weight_loading: bool = False,
) -> None:
    try:
        quant_spec = resolve_checkpoint_quant_spec(config)
    except (TypeError, ValueError) as error:
        raise ComponentCheckpointUnsupportedError(
            f"Cannot parse checkpoint quantization for {component_name!r}: {error}"
        ) from error
    if quant_spec is None:
        return

    method = quant_spec.declared_method or "unspecified"
    if direct_gpu_weight_loading:
        raise ComponentCheckpointUnsupportedError(
            f"Direct GPU loading for {component_name!r} cannot restore "
            f"quant_method={method!r}"
        )
    if native_only:
        raise ComponentCheckpointUnsupportedError(
            f"{component_name} uses a native-only SGLang implementation that "
            f"cannot restore quant_method={method!r}; Diffusers fallback is disabled."
        )
    if quant_spec.source != "quantization_config":
        raise ComponentCheckpointUnsupportedError(
            f"{component_name} checkpoint declares quantization metadata in "
            f"{quant_spec.source} (quant_method={method!r}), which the Diffusers "
            "component loader does not restore automatically."
        )

    raise NativeComponentLoaderRequired(
        f"{component_name} checkpoint declares quant_method={method!r}; routing "
        "through Diffusers from_pretrained because the SGLang VAE loader cannot "
        "restore serialized quantized state."
    )


def _backfill_ltx2_audio_vae_latent_stats(
    loaded: dict[str, torch.Tensor], component_type: str
) -> None:
    if component_type != "audio_vae":
        return
    mean_key = "per_channel_statistics.mean-of-means"
    std_key = "per_channel_statistics.std-of-means"
    if "latents_mean" not in loaded and mean_key in loaded:
        loaded["latents_mean"] = loaded[mean_key]
    if "latents_std" not in loaded and std_key in loaded:
        loaded["latents_std"] = loaded[std_key]


def _convert_conv3d_weights_to_channels_last_3d(module: nn.Module) -> int:
    """
    Convert Conv3d weights to channels_last_3d (NDHWC) memory format.
    Returns the number of Conv3d modules converted.
    """
    if not hasattr(torch, "channels_last_3d"):
        return 0
    num_converted = 0
    for m in module.modules():
        if isinstance(m, nn.Conv3d):
            try:
                m.weight.data = m.weight.data.to(memory_format=torch.channels_last_3d)
                num_converted += 1
            except Exception:
                # Best-effort; skip unsupported cases.
                continue
    return num_converted


def _should_use_channels_last_3d(
    server_args: ServerArgs | None, component_type: str
) -> bool:
    if component_type not in (
        "vae",
        "video_vae",
    ) or not (current_platform.is_cuda() or current_platform.is_rocm()):
        return False

    override = os.getenv(VAE_CHANNELS_LAST_3D_ENV)
    if override is not None and override.strip().lower() != "auto":
        return get_bool_env_var(VAE_CHANNELS_LAST_3D_ENV)

    if server_args is None:
        return False

    pipeline_config = server_args.pipeline_config
    if isinstance(pipeline_config, QwenImagePipelineConfig):
        return True
    if (
        isinstance(pipeline_config, (WanT2V480PConfig, LTX2PipelineConfig))
        and server_args.num_gpus == 1
    ):
        return True
    return False


def _decode_dtype_store_path(
    component_model_path: str, component_name: str, dtype: torch.dtype
) -> str:
    key = hashlib.sha1(
        f"{os.path.realpath(component_model_path)}|{component_name}|{dtype}".encode()
    ).hexdigest()[:16]
    return os.path.join(
        envs.SGLANG_DIFFUSION_CACHE_ROOT, "decode_dtype_store", f"{key}.safetensors"
    )


def _assign_matching_store(vae, mapped: dict, dtype: torch.dtype) -> bool:
    """Adopt a decode-dtype store if it matches the module, else refuse."""
    state = vae.state_dict()
    for name, tensor in mapped.items():
        param = state.get(name)
        if param is None or param.shape != tensor.shape or tensor.dtype != dtype:
            return False
    vae.load_state_dict(mapped, strict=False, assign=True)
    return True


def _rehome_cast_weights_to_file(
    vae, dtype: torch.dtype, component_model_path: str, component_name: str, prepare
) -> tuple[int, bool]:
    """Hold the decode-dtype weights in a file-backed mapping.

    The cast copies are anonymous host memory the kernel cannot reclaim, and
    on a budgeted host every one of those bytes comes out of the pin budget
    the stepped components live on. Written once to a cache file and mapped
    back, the same bytes become page cache: droppable under pressure, free to
    re-fault, and absent from the anonymous accounting. safetensors round-trips
    tensors byte-exactly, so the mapping holds the identical rounded values —
    and a later start adopts the store without paying the cast at all.

    Returns (weights held, file-backed?).
    """
    path = _decode_dtype_store_path(component_model_path, component_name, dtype)
    try:
        if os.path.exists(path):
            mapped = safetensors_load_file(path)
            if mapped and _assign_matching_store(vae, mapped, dtype):
                return len(mapped), True
            raise ValueError("existing decode-dtype store does not match the module")
        converted = prepare(dtype)
        if not converted:
            return 0, False
        cast_state = {
            name: tensor
            for name, tensor in vae.state_dict().items()
            if tensor.dtype == dtype and tensor.device.type == "cpu"
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp.{os.getpid()}"
        safetensors_save_file({k: v.contiguous() for k, v in cast_state.items()}, tmp)
        os.replace(tmp, path)
        mapped = safetensors_load_file(path)
        if set(mapped) != set(cast_state):
            raise ValueError("decode-dtype store does not match the cast weights")
        vae.load_state_dict(mapped, strict=False, assign=True)
        return converted, True
    except Exception as exc:
        logger.warning(
            "VAE: could not re-home %s decode-dtype weights to %s (%s); "
            "keeping in-memory copies",
            component_name,
            path,
            exc,
        )
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        return prepare(dtype), False


def _hold_decoder_weights_in_decode_dtype(
    vae,
    server_args: ServerArgs,
    component_name: str,
    component_model_path: str = "",
    *,
    component_type: str | None = None,
) -> None:
    """Round decoder weights to their decode compute dtype at load.

    The decode stage persists these frozen weights in the autocast dtype on
    first use (``prepare_autocast_linear_weights``), so the rounding itself is
    already part of the output. Doing it at load makes residency plans, host
    pins, and every host-to-device copy carry the halved size: MiniMax-H3's
    video decoder drops from 9.7 to ~4.9 GiB, which is the difference between
    restreaming a third of it per tile and holding all 36 blocks on a 12 GiB
    card for the decode.
    """
    component_type = component_type or _normalize_component_type(component_name)
    if component_type not in ("vae", "video_vae"):
        return
    if envs.SGLANG_DIFFUSION_DISABLE_EARLY_VAE_DECODER_CAST:
        return
    prepare = getattr(vae, "prepare_decoder_autocast_weights", None)
    if prepare is None:
        return
    dtype = resolve_decode_precision(server_args, component_name)
    if dtype == torch.float32:
        return
    if not autocast_enabled(dtype, server_args.disable_autocast):
        return
    if component_model_path and not envs.SGLANG_DIFFUSION_DISABLE_VAE_DECODER_STORE:
        held, file_backed = _rehome_cast_weights_to_file(
            vae, dtype, component_model_path, component_name, prepare
        )
    else:
        held, file_backed = prepare(dtype), False
    if held:
        logger.info(
            "VAE: %s holds %d decoder weights in %s from load (%s)",
            component_name,
            held,
            dtype,
            "file-backed" if file_backed else "anonymous host memory",
        )


def _vae_checkpoint_arch_metadata_names(
    vae_config: VAEConfig,
    target_state: dict[str, torch.Tensor],
) -> tuple[str, ...]:
    arch_values = vars(vae_config.arch_config)
    return tuple(
        name
        for name in _VAE_CHECKPOINT_ARCH_METADATA
        if name not in target_state and name in arch_values
    )


def _consume_vae_checkpoint_arch_metadata(
    loaded: dict[str, torch.Tensor],
    vae_config: VAEConfig,
    target_state: dict[str, torch.Tensor],
) -> tuple[str, ...]:
    """Move checkpoint-carried latent statistics into the VAE config."""
    arch_values = vars(vae_config.arch_config)
    consumed = []
    for name in _vae_checkpoint_arch_metadata_names(vae_config, target_state):
        tensor = loaded.get(name)
        if tensor is None:
            continue
        if tensor.ndim != 1:
            raise ValueError(
                f"VAE checkpoint metadata {name!r} must be one-dimensional, "
                f"got shape {tuple(tensor.shape)}"
            )
        arch_values[name] = tensor.tolist()
        del loaded[name]
        consumed.append(name)
    if consumed:
        vae_config.post_init()
    return tuple(consumed)


def _vae_checkpoint_tensor_names(weight_files: list[str]) -> set[str]:
    names: set[str] = set()
    for path in weight_files:
        with safe_open(path, framework="pt", device="cpu") as checkpoint:
            names.update(checkpoint.keys())
    return names


def _log_vae_checkpoint_adaptations(
    num_deparameterized: int, consumed_metadata: tuple[str, ...]
) -> None:
    if num_deparameterized:
        logger.info(
            "VAE: adopted %d deparameterized weight-normalized layers",
            num_deparameterized,
        )
    if consumed_metadata:
        logger.info(
            "VAE: loaded architecture metadata from checkpoint: %s",
            ", ".join(consumed_metadata),
        )


def _direct_gpu_vae_state_slots(
    vae: nn.Module, component_name: str
) -> tuple[dict[str, torch.Tensor], dict[str, tuple[nn.Module, str, bool]]]:
    """Return assignable parameter/buffer slots for a standard native VAE."""
    if type(vae).state_dict is not nn.Module.state_dict:
        raise ComponentCheckpointUnsupportedError(
            f"Direct GPU loading for {component_name!r} requires the standard "
            "torch.nn.Module state-dict ABI"
        )

    state = vae.state_dict(keep_vars=True)
    slots: dict[str, tuple[nn.Module, str, bool]] = {}
    object_names: dict[int, list[str]] = {}
    for prefix, module in vae.named_modules():
        for local_name, parameter in module._parameters.items():
            if parameter is None:
                continue
            name = f"{prefix}.{local_name}" if prefix else local_name
            slots[name] = (module, local_name, True)
            object_names.setdefault(id(parameter), []).append(name)
        for local_name, buffer in module._buffers.items():
            if buffer is None or local_name in module._non_persistent_buffers_set:
                continue
            name = f"{prefix}.{local_name}" if prefix else local_name
            slots[name] = (module, local_name, False)
            object_names.setdefault(id(buffer), []).append(name)

    if set(state) != set(slots):
        unsupported = sorted(set(state) ^ set(slots))
        raise ComponentCheckpointUnsupportedError(
            f"Direct GPU loading for {component_name!r} cannot assign custom "
            f"state entries: {unsupported}"
        )
    aliases = [names for names in object_names.values() if len(names) > 1]
    if aliases:
        raise ComponentCheckpointUnsupportedError(
            f"Direct GPU loading for {component_name!r} does not support tied "
            f"state entries: {aliases}"
        )
    return state, slots


def _assign_direct_gpu_vae_state(
    vae: nn.Module,
    weight_files: list[str],
    *,
    component_name: str,
    device: torch.device,
    vae_config: VAEConfig,
) -> tuple[int, tuple[str, ...]]:
    """Stream a complete standard VAE state directly onto its target device."""
    num_deparameterized = adopt_plain_weight_norm_state(
        vae, _vae_checkpoint_tensor_names(weight_files)
    )
    target_state, slots = _direct_gpu_vae_state_slots(vae, component_name)
    metadata_names = _vae_checkpoint_arch_metadata_names(vae_config, target_state)
    loaded_names: set[str] = set()
    metadata: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for raw_name, tensor in safetensors_weights_iterator(
            weight_files, to_cpu=device.type == "cpu"
        ):
            name = raw_name
            if name in metadata_names:
                metadata[name] = tensor
                continue
            if name in loaded_names:
                raise ComponentCheckpointUnsupportedError(
                    f"Direct GPU VAE checkpoint maps multiple tensors to {name!r}"
                )
            slot = slots.get(name)
            if slot is None:
                raise ComponentCheckpointUnsupportedError(
                    f"Direct GPU VAE checkpoint has unexpected tensor {raw_name!r}"
                )
            expected = target_state[name]
            if tensor.shape != expected.shape:
                raise ComponentCheckpointUnsupportedError(
                    f"Direct GPU VAE tensor {raw_name!r} has shape "
                    f"{tuple(tensor.shape)}, expected {tuple(expected.shape)}"
                )
            if tensor.dtype != expected.dtype:
                tensor = tensor.to(dtype=expected.dtype)

            module, local_name, is_parameter = slot
            if is_parameter:
                previous = module._parameters[local_name]
                module._parameters[local_name] = nn.Parameter(
                    tensor, requires_grad=previous.requires_grad
                )
            else:
                module._buffers[local_name] = tensor
            loaded_names.add(name)

    consumed_metadata = _consume_vae_checkpoint_arch_metadata(
        metadata, vae_config, target_state
    )
    missing = sorted(set(slots) - loaded_names)
    if missing:
        raise ComponentCheckpointUnsupportedError(
            f"Direct GPU VAE checkpoint is missing tensors: {missing}"
        )
    remaining_meta = sorted(
        name for name, tensor in vae.state_dict().items() if tensor.is_meta
    )
    if remaining_meta:
        raise RuntimeError(
            f"Direct GPU VAE loading left meta tensors: {remaining_meta}"
        )
    return num_deparameterized, consumed_metadata


class VAELoader(WeightOverrideComponentLoader):
    """Shared loader for (video/audio) VAE modules."""

    component_names = ["vae", "audio_vae", "video_vae"]
    expected_library = "diffusers"

    def resolve_component_direct_gpu_loading(
        self, server_args: ServerArgs, component_name: str
    ) -> bool:
        requested = server_args.should_direct_gpu_weight_load_component(component_name)
        if requested and component_name not in ("vae", "video_vae"):
            raise ComponentCheckpointUnsupportedError(
                f"Direct GPU loading is not implemented for {component_name!r}"
            )
        return requested

    def select_weight_files(
        self,
        safetensors_list: list[str],
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        vae_precision: str,
    ) -> list[str]:
        return server_args.pipeline_config.select_vae_weight_files(
            safetensors_list=safetensors_list,
            component_model_path=component_model_path,
            component_name=self.structural_component_type(component_name),
            vae_precision=vae_precision,
        )

    def component_load_precision(
        self, server_args: ServerArgs, component_name: str
    ) -> str | None:
        return server_args.component_precisions.get(component_name)

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

    def load_customized(
        self,
        component_model_path: str,
        server_args: ServerArgs,
        component_name: str,
        cpu_offload_flag: bool = False,
    ):
        """Load the VAE based on the model path, and inference args."""
        direct_gpu_weight_loading = self.resolve_component_direct_gpu_loading(
            server_args, component_name
        )
        component_weights_path = self.resolve_component_weights_path(
            component_model_path,
            server_args,
            component_name,
        )
        config = get_diffusers_component_config(component_path=component_model_path)
        server_args.model_paths[component_name] = component_model_path
        native_only = self.is_native_only_component(server_args, component_name)
        _require_native_loader_for_quantized_vae(
            config,
            component_name,
            native_only=native_only,
            direct_gpu_weight_loading=direct_gpu_weight_loading,
        )

        class_name = config.pop("_class_name", None)
        assert class_name is not None, (
            "Model config does not contain a _class_name attribute. Only diffusers format is supported."
        )

        component_type = self.structural_component_type(component_name)
        if component_type in ("vae", "video_vae"):
            pipeline_vae_config_attr = "vae_config"
            pipeline_vae_precision = "vae_precision"
        elif component_type == "audio_vae":
            pipeline_vae_config_attr = "audio_vae_config"
            pipeline_vae_precision = "audio_vae_precision"
        else:
            raise ValueError(
                f"Unsupported module name for VAE loader: {component_name}"
            )
        vae_config = getattr(server_args.pipeline_config, pipeline_vae_config_attr)
        vae_precision = getattr(server_args.pipeline_config, pipeline_vae_precision)
        resolved_vae_dtype = resolve_component_precision(server_args, component_name)
        vae_dtype = (
            resolved_vae_dtype
            if resolved_vae_dtype is not None
            else PRECISION_TO_TYPE[vae_precision]
        )
        vae_config.update_model_arch(config)
        if hasattr(vae_config, "post_init"):
            # NOTE: some post init logics are only available after updated with config
            vae_config.post_init()

        component_starts_on_cpu = (
            server_args.should_start_component_on_cpu(component_name)
            or cpu_offload_flag
        )
        target_device = self.target_device(component_starts_on_cpu)

        auto_map = config.get("auto_map", {})
        auto_model_map = auto_map.get("AutoModel")
        if direct_gpu_weight_loading and auto_model_map and not native_only:
            raise ComponentCheckpointUnsupportedError(
                f"Direct GPU loading for {component_name!r} requires a native "
                "ModelRegistry VAE; custom Diffusers auto_map code is unsupported"
            )
        if (
            auto_model_map
            and not native_only
            and component_weights_path != component_model_path
        ):
            raise ComponentCheckpointUnsupportedError(
                f"{component_name!r} uses a custom Diffusers class that cannot "
                "consume a weights-only override"
            )
        if auto_model_map and not native_only:
            module_path, cls_name = auto_model_map.rsplit(".", 1)
            custom_module_file = os.path.join(component_model_path, f"{module_path}.py")
            spec = importlib.util.spec_from_file_location("_custom", custom_module_file)
            custom_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_module)
            vae_cls = getattr(custom_module, cls_name)
            with set_default_torch_dtype(vae_dtype):
                vae = vae_cls.from_pretrained(
                    component_model_path,
                    revision=server_args.revision,
                    trust_remote_code=server_args.trust_remote_code,
                )
            vae = vae.to(device=target_device, dtype=vae_dtype)
            if _should_use_channels_last_3d(server_args, component_type):
                n = _convert_conv3d_weights_to_channels_last_3d(vae)
                if n > 0:
                    logger.info(
                        "VAE: converted %d Conv3d weights to channels_last_3d", n
                    )
            _hold_decoder_weights_in_decode_dtype(
                vae,
                server_args,
                component_name,
                component_model_path,
                component_type=component_type,
            )
            vae = current_platform.optimize_vae(vae)
            return vae

        # Load from ModelRegistry (standard VAE classes)
        vae_cls, _ = ModelRegistry.resolve_model_cls(class_name)
        vae = initialize_model(
            vae_cls,
            {"config": vae_config},
            vae_dtype,
            torch.device("meta") if direct_gpu_weight_loading else None,
        )
        if not direct_gpu_weight_loading:
            vae = vae.to(target_device)

        if os.path.isfile(component_weights_path):
            if not component_weights_path.endswith(".safetensors"):
                raise ValueError(
                    f"VAE weight overrides must be safetensors, got "
                    f"{component_weights_path!r}"
                )
            safetensors_list = [component_weights_path]
        else:
            # VAE configs may explicitly choose a precision variant, so their
            # selector must run before the canonical fallback.
            safetensors_list = _list_safetensors_files(
                component_weights_path, raw_candidates=True
            )
            safetensors_list = self.select_weight_files(
                safetensors_list,
                component_weights_path,
                server_args,
                component_name,
                vae_precision,
            )
            safetensors_list = filter_duplicate_precision_variant_safetensors(
                safetensors_list
            )

        assert len(safetensors_list) >= 1, (
            f"Found no safetensors files in {component_weights_path}"
        )
        if direct_gpu_weight_loading:
            adaptations = _assign_direct_gpu_vae_state(
                vae,
                safetensors_list,
                component_name=component_name,
                device=target_device,
                vae_config=vae_config,
            )
            _log_vae_checkpoint_adaptations(*adaptations)
            if _should_use_channels_last_3d(server_args, component_name):
                n = _convert_conv3d_weights_to_channels_last_3d(vae)
                if n > 0:
                    logger.info(
                        "VAE: converted %d Conv3d weights to channels_last_3d", n
                    )
            _hold_decoder_weights_in_decode_dtype(vae, server_args, component_name)
            return current_platform.optimize_vae(vae)

        loaded = {}
        for sf_path in safetensors_list:
            loaded.update(safetensors_load_file(sf_path))
        _backfill_ltx2_audio_vae_latent_stats(loaded, component_type)
        num_deparameterized = adopt_plain_weight_norm_state(vae, loaded)
        target_state = vae.state_dict()
        consumed_metadata = _consume_vae_checkpoint_arch_metadata(
            loaded, vae_config, target_state
        )
        _log_vae_checkpoint_adaptations(num_deparameterized, consumed_metadata)
        strict_load = native_only
        # `loaded` holds views into the safetensors mapping. When the component
        # starts on the CPU and the host cannot afford copies of the whole
        # deployment, assigning them keeps the weights file-backed instead of
        # copying them into anonymous host memory: the page cache can drop and
        # refetch file-backed bytes, and every anonymous byte here is a byte
        # the stepped components' pin budget loses -- MiniMax-H3's video VAE is
        # 9.70 GiB of a 32 GiB budget. On a host with room the copy stays the
        # default, because its pages are resident where a mapping's first use
        # pays a fault. MPS always assigns; the memory is unified. A tensor
        # whose dtype differs from its parameter's is converted, which copies
        # exactly the tensors that cannot stay.
        keep_mapping = component_starts_on_cpu and (
            current_platform.is_mps()
            or keep_checkpoint_mapped(
                # server_args.model_path can be a hub repo id, which is not a
                # directory anywhere; the component path is always local, and
                # its parent holds the rest of the variant being deployed.
                weight_bytes=checkpoint_bytes(
                    os.path.dirname(str(component_model_path))
                ),
                component=f"{component_name or 'vae'} (VAE)",
            )
        )
        load_model_state_dict(
            vae,
            loaded,
            strict=strict_load,
            assign=keep_mapping,
        )

        if not strict_load:
            state_keys = set(vae.state_dict().keys())
            loaded_keys = set(loaded.keys())
            missing_keys = sorted(state_keys - loaded_keys)
            unexpected_keys = sorted(loaded_keys - state_keys)
            if missing_keys:
                logger.warning("VAE missing keys: %s", missing_keys)
            if unexpected_keys:
                logger.warning("VAE unexpected keys: %s", unexpected_keys)

        if _should_use_channels_last_3d(server_args, component_type):
            n = _convert_conv3d_weights_to_channels_last_3d(vae)
            if n > 0:
                logger.info("VAE: converted %d Conv3d weights to channels_last_3d", n)

        _hold_decoder_weights_in_decode_dtype(
            vae,
            server_args,
            component_name,
            component_weights_path,
            component_type=component_type,
        )
        vae = current_platform.optimize_vae(vae)
        return vae
