from __future__ import annotations

import gc
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import diffusers
import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    T5EncoderModel,
    UMT5EncoderModel,
)

try:
    from transformers import AutoModelForImageTextToText as AutoVisionTextModel
except ImportError:
    try:
        from transformers import AutoModelForVision2Seq as AutoVisionTextModel
    except ImportError:
        AutoVisionTextModel = None

import sglang.multimodal_gen.runtime.managers.forward_context as fc_mod
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    destroy_model_parallel,
    get_local_torch_device,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
)
from sglang.multimodal_gen.runtime.managers.forward_context import ForwardContext
from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase
from sglang.multimodal_gen.test.single_test_file.component_accuracy.config import (
    DEFAULT_TIMESTEP,
    ComponentType,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.hooks import (
    resolve_component_native_profile,
)
from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
    build_accuracy_server_args,
    build_parameter_shard_contexts,
    build_state_lookup,
    copy_tensor,
    fuse_gate_up_proj,
    fuse_qkv,
    generate_name_candidates,
    initialize_parallel_runtime,
    load_checkpoint_weights,
    materialize_module,
    read_json_file,
    resolve_text_encoder_forward_module,
    select_component_source,
)

logger = init_logger(__name__)

MIN_MATCH_RATIO = float(os.getenv("SGLANG_DIFFUSION_WEIGHT_MATCH_RATIO", "0.98"))
VAE_CHANNELS_LAST_3D_ENV = "SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D"
VAE_CHANNELS_LAST_3D_PARITY_THRESHOLD = float(
    os.getenv("SGLANG_DIFFUSION_VAE_CHANNELS_LAST_3D_PARITY_THRESHOLD", "0.999")
)


@contextmanager
def _temporary_vae_channels_last_3d(enabled: bool):
    previous = os.environ.get(VAE_CHANNELS_LAST_3D_ENV)
    os.environ[VAE_CHANNELS_LAST_3D_ENV] = "true" if enabled else "false"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(VAE_CHANNELS_LAST_3D_ENV, None)
        else:
            os.environ[VAE_CHANNELS_LAST_3D_ENV] = previous


@dataclass
class Conv3dLayoutStats:
    calls: int = 0
    channels_last_input_calls: int = 0
    channels_last_weight_calls: int = 0
    mixed_layout_calls: int = 0


@contextmanager
def _record_conv3d_layouts():
    stats = Conv3dLayoutStats()
    original_conv3d = torch.nn.functional.conv3d

    def wrapped_conv3d(input, weight, *args, **kwargs):
        if (
            isinstance(input, torch.Tensor)
            and isinstance(weight, torch.Tensor)
            and input.dim() == 5
            and weight.dim() == 5
            and hasattr(torch, "channels_last_3d")
        ):
            input_channels_last = input.is_contiguous(
                memory_format=torch.channels_last_3d
            )
            weight_channels_last = weight.is_contiguous(
                memory_format=torch.channels_last_3d
            )
        else:
            input_channels_last = False
            weight_channels_last = False

        stats.calls += 1
        stats.channels_last_input_calls += int(input_channels_last)
        stats.channels_last_weight_calls += int(weight_channels_last)
        stats.mixed_layout_calls += int(
            weight_channels_last and not input_channels_last
        )
        return original_conv3d(input, weight, *args, **kwargs)

    torch.nn.functional.conv3d = wrapped_conv3d
    try:
        yield stats
    finally:
        torch.nn.functional.conv3d = original_conv3d


@dataclass(frozen=True)
class ComponentSpec:
    model_index_keys: Tuple[str, ...]
    reference_library: str


COMPONENT_SPECS: Dict[ComponentType, ComponentSpec] = {
    ComponentType.VAE: ComponentSpec(
        model_index_keys=(
            "vae",
            "vae_model",
            "autoencoder",
            "autoencoder_kl",
            "video_vae",
            "audio_vae",
        ),
        reference_library="diffusers",
    ),
    ComponentType.TRANSFORMER: ComponentSpec(
        model_index_keys=("transformer", "unet", "dit", "video_dit", "audio_dit"),
        reference_library="diffusers",
    ),
    ComponentType.TEXT_ENCODER: ComponentSpec(
        model_index_keys=(
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "image_encoder",
        ),
        reference_library="transformers",
    ),
}


# Component loading helpers
def _load_sglang_component(
    comp_path: str,
    sgl_args: ServerArgs,
    component: ComponentType,
    library: str,
    text_encoder_cpu_offload: bool | None = None,
) -> nn.Module:
    loader = ComponentLoader.for_component_type(component.value, library)
    if component == ComponentType.TEXT_ENCODER:
        component_model = loader.load_customized(
            comp_path,
            sgl_args,
            component.value,
            cpu_offload_flag=text_encoder_cpu_offload,
        )
    else:
        component_model = loader.load_customized(comp_path, sgl_args, component.value)
    if component_model is None:
        raise RuntimeError(f"Failed to load customized {component.value}")
    return component_model


def _load_wan_reference_vae(comp_path: str, pipeline_config) -> nn.Module:
    vae_config = pipeline_config.vae_config
    vae_config.update_model_arch(
        get_diffusers_component_config(component_path=comp_path)
    )
    if hasattr(vae_config, "post_init"):
        vae_config.post_init()

    vae = AutoencoderKLWan(vae_config)
    missing_keys, unexpected_keys = load_checkpoint_weights(vae, comp_path)
    if missing_keys:
        logger.warning("WAN VAE missing keys: %s", missing_keys)
    if unexpected_keys:
        logger.warning("WAN VAE unexpected keys: %s", unexpected_keys)
    return vae


def _load_reference_component_from_local_safetensors(
    component_cls: type[nn.Module],
    comp_path: str,
    component_name: str,
) -> nn.Module:
    config = component_cls.load_config(comp_path)
    component = component_cls.from_config(config)
    missing_keys, unexpected_keys = load_checkpoint_weights(component, comp_path)
    if missing_keys:
        logger.warning(
            "Reference %s missing keys from local safetensors: %s",
            component_name,
            missing_keys,
        )
    if unexpected_keys:
        logger.warning(
            "Reference %s unexpected keys from local safetensors: %s",
            component_name,
            unexpected_keys,
        )
    return component


def _load_reference_component(
    comp_path: str,
    component: ComponentType,
    hub_id: str,
    pipeline_config,
) -> nn.Module:
    # WAN VAE does not have a clean generic diffusers auto-load path here, and we
    # explicitly need checkpoint-loaded weights for reference-side transfer/parity.
    if component == ComponentType.VAE and "wan" in hub_id.lower():
        return _load_wan_reference_vae(comp_path, pipeline_config)

    if component == ComponentType.VAE:
        cfg = read_json_file(os.path.join(comp_path, "config.json"))
        class_name = cfg.get("_class_name") if cfg else None
        cls = getattr(diffusers, str(class_name), None) if class_name else None
        if cls is None:
            cls = diffusers.AutoencoderKL
        if cls is not diffusers.AutoencoderKL and os.path.exists(
            os.path.join(comp_path, "model.safetensors")
        ):
            return _load_reference_component_from_local_safetensors(
                cls, comp_path, component.value
            )
        return cls.from_pretrained(
            comp_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    if component == ComponentType.TRANSFORMER:
        cfg = read_json_file(os.path.join(comp_path, "config.json"))
        class_name = cfg.get("_class_name") if cfg else None
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if class_name:
            maybe_cls = getattr(diffusers, str(class_name), None)
            if maybe_cls is not None and os.path.exists(
                os.path.join(comp_path, "model.safetensors")
            ):
                return _load_reference_component_from_local_safetensors(
                    maybe_cls, comp_path, component.value
                )
        if cfg:
            for k, out_k in [
                ("in_dim", "in_channels"),
                ("dim", "hidden_size"),
                ("num_heads", "num_attention_heads"),
                ("out_dim", "out_channels"),
            ]:
                if k in cfg:
                    load_kwargs[out_k] = cfg[k]
        candidates = [diffusers.AutoModel]
        if class_name and maybe_cls is not None:
            candidates.insert(0, maybe_cls)
        last_error: Optional[Exception] = None
        for cls in candidates:
            try:
                return cls.from_pretrained(comp_path, **load_kwargs)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Failed to load transformer from {comp_path}: {last_error}")

    if component == ComponentType.TEXT_ENCODER:
        config = AutoConfig.from_pretrained(comp_path, trust_remote_code=True)
        kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
            "config": config,
        }
        architectures = tuple(getattr(config, "architectures", ()) or ())
        if (
            "UMT5EncoderModel" in architectures
            or getattr(config, "model_type", None) == "umt5"
        ):
            class_order = [UMT5EncoderModel, AutoModel, AutoModelForCausalLM]
        else:
            class_order = [
                AutoModel,
                AutoModelForCausalLM,
                UMT5EncoderModel,
                T5EncoderModel,
            ]
        if AutoVisionTextModel is not None:
            class_order.append(AutoVisionTextModel)
        last_error: Optional[Exception] = None
        for cls in class_order:
            try:
                return cls.from_pretrained(comp_path, **kwargs)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"Failed to load text encoder from {comp_path}: {last_error}"
        )

    raise RuntimeError(f"Unsupported component {component.value}")


# Public accuracy engine
class AccuracyEngine:
    @staticmethod
    def prepare_component_for_release(module: nn.Module) -> None:
        for submodule in module.modules():
            reset_teacache_state = getattr(submodule, "reset_teacache_state", None)
            if callable(reset_teacache_state):
                reset_teacache_state()

            seen_names: set[str] = set()
            for cls in type(submodule).__mro__:
                for name, attr in cls.__dict__.items():
                    if name in seen_names:
                        continue
                    seen_names.add(name)

                    cache_clear = getattr(attr, "cache_clear", None)
                    if callable(cache_clear):
                        cache_clear()

    @staticmethod
    def reset_parallel_runtime() -> None:
        if torch.distributed.is_initialized():
            if torch.distributed.get_world_size() == 1:
                cleanup_dist_env_and_memory()
        elif model_parallel_is_initialized():
            destroy_model_parallel()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @staticmethod
    def clear_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _execute_with_native_hook(call) -> Any:
        output: Any = None

        def _hook(_: nn.Module, __: tuple[Any, ...], captured: Any) -> None:
            nonlocal output
            output = captured

        handle = call.module.register_forward_hook(_hook)
        try:
            call.module(*call.args, **call.kwargs)
        finally:
            handle.remove()
        assert output is not None
        return output

    @staticmethod
    def _apply_output_transforms(tensor: torch.Tensor, call) -> torch.Tensor:
        if call.negate_output:
            return -tensor
        return tensor

    @staticmethod
    def check_accuracy(
        target: torch.Tensor, reference: torch.Tensor, name: str, threshold: float
    ) -> None:
        full_tensor = getattr(target, "full_tensor", None)
        if callable(full_tensor):
            target = full_tensor()
        t, r = target.detach().cpu().float(), reference.detach().cpu().float()

        logger.info(
            "[%s] Shape: SGL=%s, REF=%s | NaNs: SGL=%s, REF=%s",
            name,
            list(t.shape),
            list(r.shape),
            torch.isnan(t).sum(),
            torch.isnan(r).sum(),
        )

        if t.shape != r.shape:
            if t.ndim == 5 and t.shape[2] == 1:
                t = t.squeeze(2)
            if r.ndim == 5 and r.shape[2] == 1:
                r = r.squeeze(2)
            if t.shape != r.shape:
                raise RuntimeError(
                    f"Accuracy shape mismatch for {name}: {list(t.shape)} vs {list(r.shape)}"
                )

        cos_sim = torch.nn.functional.cosine_similarity(
            t.reshape(-1), r.reshape(-1), dim=0
        ).item()
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        logger.info("[%s] Rank %s CosSim=%.6f", name, rank, cos_sim)
        assert (
            cos_sim > threshold
        ), f"Accuracy failure in {name}: CosSim {cos_sim:.4f} < {threshold}"

    @staticmethod
    def transfer_weights(
        source: nn.Module,
        target: nn.Module,
        min_match_ratio: float = MIN_MATCH_RATIO,
        target_device: Optional[torch.device] = None,
    ) -> None:
        device = target_device or get_local_torch_device()
        dtype = torch.bfloat16
        materialize_module(target, device, dtype)

        source_state = source.state_dict()
        mapping = getattr(target, "param_names_mapping", None) or getattr(
            getattr(target, "module", None), "param_names_mapping", None
        )
        if mapping:
            source_state, _ = hf_to_custom_state_dict(
                source_state, get_param_names_mapping(mapping)
            )

        lookup = build_state_lookup(source_state)
        reverse_mapping = getattr(
            target, "reverse_param_names_mapping", None
        ) or getattr(
            getattr(target, "module", None), "reverse_param_names_mapping", None
        )
        tp_world = (
            get_tensor_model_parallel_world_size()
            if model_parallel_is_initialized()
            else 1
        )
        rank = (
            get_tensor_model_parallel_rank() if model_parallel_is_initialized() else 0
        )
        shard_contexts = build_parameter_shard_contexts(target)

        matched = 0
        total = 0
        unmatched_details: List[str] = []
        for name, tensor in target.named_parameters():
            total += 1
            src_tensor = None
            for cand in generate_name_candidates(name, reverse_mapping):
                if cand in lookup:
                    src_tensor = lookup[cand]
                    break
            if src_tensor is None:
                for cand in generate_name_candidates(name, reverse_mapping):
                    src_tensor = fuse_qkv(lookup, cand)
                    if src_tensor is not None:
                        break
            if src_tensor is None:
                for cand in generate_name_candidates(name, reverse_mapping):
                    src_tensor = fuse_gate_up_proj(lookup, cand)
                    if src_tensor is not None:
                        break
            if src_tensor is None:
                unmatched_details.append(f"{name}: no matching source tensor")
                continue
            shard_context = shard_contexts.get(name)
            shard_world_size = (
                shard_context.world_size if shard_context is not None else tp_world
            )
            shard_rank = shard_context.rank if shard_context is not None else rank
            if copy_tensor(tensor, src_tensor, shard_world_size, shard_rank):
                matched += 1
            else:
                unmatched_details.append(
                    f"{name}: source {list(src_tensor.shape)} -> target {list(tensor.shape)} unsupported for shard_world_size={shard_world_size}"
                )

        for name, tensor in target.named_buffers():
            src_tensor = None
            for cand in generate_name_candidates(name, reverse_mapping):
                if cand in lookup:
                    src_tensor = lookup[cand]
                    break
            if src_tensor is None:
                continue
            shard_context = shard_contexts.get(name)
            shard_world_size = (
                shard_context.world_size if shard_context is not None else tp_world
            )
            shard_rank = shard_context.rank if shard_context is not None else rank
            copy_tensor(tensor, src_tensor, shard_world_size, shard_rank)

        ratio = matched / max(total, 1)
        logger.info(
            "Weight transfer: %s/%s matched (%.2f%%).", matched, total, ratio * 100
        )
        if ratio < min_match_ratio:
            if rank == 0 and unmatched_details:
                logger.error(
                    "Unmatched parameter details:\n%s", "\n".join(unmatched_details)
                )
            raise RuntimeError(
                f"Weight transfer matched {matched}/{total} ({ratio:.2%}); below threshold {min_match_ratio:.2%}."
            )

    @staticmethod
    def run_component_pair_native(
        case: DiffusionTestCase,
        component: ComponentType,
        sgl_model: nn.Module,
        ref_model: nn.Module,
        device: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if component == ComponentType.TEXT_ENCODER:
            raise ValueError("Text encoder path is not migrated to native hooks yet")
        profile = resolve_component_native_profile(component)

        inputs = profile.build_inputs(case, sgl_model, device, ref_model)
        sgl_call = profile.prepare_sglang_call(sgl_model, inputs)
        ref_call = profile.prepare_reference_call(ref_model, inputs)

        with torch.no_grad():
            sgl_raw = AccuracyEngine._execute_with_native_hook(sgl_call)
            ref_raw = AccuracyEngine._execute_with_native_hook(ref_call)

        sgl_out = profile.normalize_sglang_output(sgl_raw)
        ref_out = profile.normalize_reference_output(ref_raw)
        sgl_out = AccuracyEngine._apply_output_transforms(sgl_out, sgl_call)
        ref_out = AccuracyEngine._apply_output_transforms(ref_out, ref_call)
        return sgl_out, ref_out

    @staticmethod
    def load_component_pair(
        case: DiffusionTestCase,
        component: ComponentType,
        library: str,
        num_gpus: int,
        materialize_sgl_on_device: bool = True,
        materialize_ref_on_device: bool = True,
    ) -> Tuple[nn.Module, nn.Module, str]:
        spec = COMPONENT_SPECS[component]
        if library != spec.reference_library:
            logger.warning(
                "Overriding library '%s' with '%s' for component '%s'.",
                library,
                spec.reference_library,
                component.value,
            )
            library = spec.reference_library
        hub_id = case.server_args.model_path
        component_selection = select_component_source(
            hub_id,
            case.server_args.extras,
            component,
            spec.model_index_keys,
        )
        sgl_args = build_accuracy_server_args(
            component_selection.base_model_id,
            component_selection.base_model_root,
            case,
            component,
            num_gpus,
            component_selection.component_paths,
        )
        if component == ComponentType.TRANSFORMER and not materialize_sgl_on_device:
            sgl_args.dit_cpu_offload = True
        initialize_parallel_runtime(sgl_args)
        set_global_server_args(sgl_args)

        device = get_local_torch_device()

        sgl_component = _load_sglang_component(
            component_selection.source_path,
            sgl_args,
            component,
            library,
            text_encoder_cpu_offload=(
                False
                if component != ComponentType.TEXT_ENCODER or materialize_sgl_on_device
                else True
            ),
        )
        if materialize_sgl_on_device:
            sgl_component = sgl_component.to(device=device, dtype=torch.bfloat16)

        ref_component = _load_reference_component(
            component_selection.source_path,
            component,
            hub_id,
            sgl_args.pipeline_config,
        )
        if materialize_ref_on_device:
            ref_component = ref_component.to(device=device, dtype=torch.bfloat16)

        if component == ComponentType.TRANSFORMER and "wan" in hub_id.lower():
            fc_mod._forward_context = ForwardContext(
                current_timestep=0, attn_metadata=None
            )

        ref_for_transfer = ref_component
        if (
            component == ComponentType.TEXT_ENCODER
            and getattr(ref_component, "shared", None) is None
        ):
            ref_for_transfer = resolve_text_encoder_forward_module(ref_component)
        AccuracyEngine.transfer_weights(
            ref_for_transfer,
            sgl_component,
            target_device=(
                device if materialize_sgl_on_device else torch.device("cpu")
            ),
        )

        if component != ComponentType.VAE:
            if not hasattr(fc_mod._forward_context, "attn_metadata"):
                fc_mod._forward_context = ForwardContext(
                    current_timestep=int(DEFAULT_TIMESTEP), attn_metadata=None
                )

        return sgl_component.eval(), ref_component.eval(), str(device)

    @staticmethod
    def run_vae_channels_last_3d_parity(
        case: DiffusionTestCase,
        num_gpus: int,
    ) -> None:
        component = ComponentType.VAE
        spec = COMPONENT_SPECS[component]
        hub_id = case.server_args.model_path
        component_selection = select_component_source(
            hub_id,
            case.server_args.extras,
            component,
            spec.model_index_keys,
        )
        sgl_args = build_accuracy_server_args(
            component_selection.base_model_id,
            component_selection.base_model_root,
            case,
            component,
            num_gpus,
            component_selection.component_paths,
        )

        baseline_vae = None
        channels_last_vae = None
        try:
            initialize_parallel_runtime(sgl_args)
            set_global_server_args(sgl_args)
            device = get_local_torch_device()

            with _temporary_vae_channels_last_3d(False):
                baseline_vae = _load_sglang_component(
                    component_selection.source_path,
                    sgl_args,
                    component,
                    spec.reference_library,
                ).to(device=device, dtype=torch.bfloat16)

            with _temporary_vae_channels_last_3d(True):
                channels_last_vae = _load_sglang_component(
                    component_selection.source_path,
                    sgl_args,
                    component,
                    spec.reference_library,
                ).to(device=device, dtype=torch.bfloat16)

            baseline_vae.eval()
            channels_last_vae.eval()

            profile = resolve_component_native_profile(component)
            inputs = profile.build_inputs(
                case, baseline_vae, str(device), channels_last_vae
            )
            baseline_call = profile.prepare_sglang_call(baseline_vae, inputs)
            channels_last_call = profile.prepare_sglang_call(channels_last_vae, inputs)

            with torch.no_grad():
                with _record_conv3d_layouts() as baseline_layout:
                    baseline_raw = AccuracyEngine._execute_with_native_hook(
                        baseline_call
                    )
                with _record_conv3d_layouts() as channels_last_layout:
                    channels_last_raw = AccuracyEngine._execute_with_native_hook(
                        channels_last_call
                    )

            baseline_out = profile.normalize_sglang_output(baseline_raw)
            channels_last_out = profile.normalize_sglang_output(channels_last_raw)

            AccuracyEngine.check_accuracy(
                channels_last_out,
                baseline_out,
                f"{case.id}_vae_channels_last_3d",
                VAE_CHANNELS_LAST_3D_PARITY_THRESHOLD,
            )

            logger.info(
                "[%s_vae_channels_last_3d] Conv3d layout baseline: calls=%d, "
                "input_cl3d=%d, weight_cl3d=%d, mixed=%d | channels_last: "
                "calls=%d, input_cl3d=%d, weight_cl3d=%d, mixed=%d",
                case.id,
                baseline_layout.calls,
                baseline_layout.channels_last_input_calls,
                baseline_layout.channels_last_weight_calls,
                baseline_layout.mixed_layout_calls,
                channels_last_layout.calls,
                channels_last_layout.channels_last_input_calls,
                channels_last_layout.channels_last_weight_calls,
                channels_last_layout.mixed_layout_calls,
            )
            if channels_last_layout.calls == 0:
                raise RuntimeError(
                    f"{case.id}: VAE channels_last_3d guard did not execute Conv3d"
                )
            if channels_last_layout.channels_last_weight_calls == 0:
                raise RuntimeError(
                    f"{case.id}: VAE channels_last_3d guard did not see channels_last_3d Conv3d weights"
                )
            if channels_last_layout.mixed_layout_calls:
                raise RuntimeError(
                    f"{case.id}: {channels_last_layout.mixed_layout_calls} Conv3d calls used "
                    "channels_last_3d weights with non-channels_last_3d inputs"
                )
        finally:
            if baseline_vae is not None:
                del baseline_vae
            if channels_last_vae is not None:
                del channels_last_vae
            AccuracyEngine.reset_parallel_runtime()
            AccuracyEngine.clear_memory()
