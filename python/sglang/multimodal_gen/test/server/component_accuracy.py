from __future__ import annotations

import gc
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import diffusers
import torch
import torch.nn as nn
from torch.distributed.tensor import distribute_tensor
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    T5EncoderModel,
    UMT5EncoderModel,
)

import sglang.multimodal_gen.runtime.managers.forward_context as fc_mod
from sglang.multimodal_gen.registry import _CONFIG_REGISTRY, get_model_info
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    get_local_torch_device,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.loader.component_loader import ComponentLoader
from sglang.multimodal_gen.runtime.loader.utils import (
    get_param_names_mapping,
    hf_to_custom_state_dict,
)
from sglang.multimodal_gen.runtime.managers.forward_context import ForwardContext
from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.accuracy_adapters import (
    ComponentAdapter,
    get_adapter,
)
from sglang.multimodal_gen.test.server.accuracy_config import ComponentType
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase

logger = init_logger(__name__)

DEFAULT_TIMESTEP = 500.0
MIN_MATCH_RATIO = float(os.getenv("SGLANG_DIFFUSION_WEIGHT_MATCH_RATIO", "0.98"))

MASTER_ADDR = os.getenv("MASTER_ADDR", "127.0.0.1")
MASTER_PORT = os.getenv("MASTER_PORT", "29505")

if "WORLD_SIZE" not in os.environ:
    os.environ.update(
        {
            "MASTER_ADDR": MASTER_ADDR,
            "MASTER_PORT": MASTER_PORT,
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }
    )


@dataclass(frozen=True)
class ComponentSpec:
    component: ComponentType
    model_index_keys: Tuple[str, ...]
    fallback_subfolders: Tuple[str, ...]
    reference_library: str


COMPONENT_SPECS: Dict[ComponentType, ComponentSpec] = {
    ComponentType.VAE: ComponentSpec(
        component=ComponentType.VAE,
        model_index_keys=(
            "vae",
            "vae_model",
            "autoencoder",
            "autoencoder_kl",
            "video_vae",
            "audio_vae",
        ),
        fallback_subfolders=(
            "vae",
            "vae_model",
            "autoencoder",
            "autoencoder_kl",
            "VAE",
        ),
        reference_library="diffusers",
    ),
    ComponentType.TRANSFORMER: ComponentSpec(
        component=ComponentType.TRANSFORMER,
        model_index_keys=("transformer", "unet", "dit", "video_dit", "audio_dit"),
        fallback_subfolders=("transformer", "unet", "dit"),
        reference_library="diffusers",
    ),
    ComponentType.TEXT_ENCODER: ComponentSpec(
        component=ComponentType.TEXT_ENCODER,
        model_index_keys=(
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "image_encoder",
        ),
        fallback_subfolders=(
            "text_encoder",
            "text_encoder_2",
            "text_encoder_3",
            "image_encoder",
            "umt5-xxl",
        ),
        reference_library="transformers",
    ),
}


def _read_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _has_component_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.exists(os.path.join(path, "config.json")):
        return True
    for ext in (".safetensors", ".bin", ".pth"):
        if any(name.endswith(ext) for name in os.listdir(path)):
            return True
    return False


def _is_text_encoder_config(path: str) -> bool:
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(cfg_path):
        return True
    cfg = _read_json(cfg_path)
    # 5120 is the known i2v text-encoder hidden size used by WAN video models.
    if cfg.get("model_type") == "i2v" or cfg.get("dim") == 5120:
        return False
    return True


def _resolve_component_subfolder(
    model_index: Dict[str, Any], key: str
) -> Optional[str]:
    entry = model_index.get(key)
    if isinstance(entry, dict):
        return entry.get("path") or entry.get("subfolder")
    if entry is not None:
        return key
    return None


def resolve_component_path(
    local_root: str, component: ComponentType
) -> Tuple[str, str]:
    """Resolve component subfolder from model_index.json or known fallbacks."""
    spec = COMPONENT_SPECS[component]
    model_index = _read_json(os.path.join(local_root, "model_index.json"))

    for key in spec.model_index_keys:
        subfolder = _resolve_component_subfolder(model_index, key)
        if not subfolder:
            continue
        candidate = os.path.join(local_root, subfolder)
        if not _has_component_files(candidate):
            continue
        if component == ComponentType.TEXT_ENCODER and not _is_text_encoder_config(
            candidate
        ):
            continue
        return candidate, subfolder

    for subfolder in spec.fallback_subfolders:
        candidate = os.path.join(local_root, subfolder)
        if not _has_component_files(candidate):
            continue
        if component == ComponentType.TEXT_ENCODER and not _is_text_encoder_config(
            candidate
        ):
            continue
        return candidate, subfolder

    raise FileNotFoundError(
        f"Could not locate {component.value} component under {local_root}"
    )


def _resolve_local_path(hub_id: str) -> str:
    if os.path.isdir(hub_id):
        return hub_id
    base_dir = os.getenv("SGLANG_DIFFUSION_MODEL_DIR")
    if base_dir and os.path.isdir(base_dir):
        short_name = hub_id.split("/")[-1]
        candidate = os.path.join(base_dir, short_name)
        if os.path.isdir(candidate):
            return candidate
    return maybe_download_model(hub_id)


def _build_pipeline_config(case: DiffusionTestCase, hub_id: str):
    info = get_model_info(hub_id) or get_model_info(hub_id.split("/")[-1])
    pipeline_config = (
        info.pipeline_config_cls()
        if info
        else _CONFIG_REGISTRY["0"].pipeline_config_cls()
    )
    if hasattr(pipeline_config, "dit_config") and (
        "i2v" in case.id.lower() or "i2v" in hub_id.lower() or "ti2v" in case.id.lower()
    ):
        pipeline_config.dit_config.image_dim = 1280
    return pipeline_config


def _init_parallel(
    case: DiffusionTestCase, component: ComponentType, num_gpus: int
) -> None:
    if model_parallel_is_initialized():
        return
    model_path = case.server_args.model_path.lower()
    if component == ComponentType.TEXT_ENCODER:
        tp_size, sp_size = 1, 1
        enable_cfg = False
        dp_size = max(1, num_gpus)
    elif component == ComponentType.TRANSFORMER and (
        "zimage" in model_path or "z-image" in model_path
    ):
        tp_size, sp_size = 1, 1
        enable_cfg = False
        dp_size = max(1, num_gpus)
    elif component == ComponentType.TRANSFORMER and "wan" in model_path:
        # Force SP=1 for accuracy comparisons against Diffusers to avoid SP-aware rotary mismatch.
        tp_size, sp_size = 1, 1
        enable_cfg = case.server_args.cfg_parallel or False
        dp_size = max(1, num_gpus)
    else:
        tp_size, sp_size = num_gpus, 1
        if case.server_args.cfg_parallel:
            tp_size = max(1, num_gpus // 2)
        if component == ComponentType.TRANSFORMER and (
            "video" in case.server_args.modality or "wan" in model_path
        ):
            tp_size, sp_size = 1, num_gpus
        enable_cfg = case.server_args.cfg_parallel or False
        dp_size = 1

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=tp_size,
        sp_size=sp_size,
        enable_cfg_parallel=enable_cfg,
        dp_size=dp_size,
    )


def _build_server_args(
    local_root: str,
    case: DiffusionTestCase,
    num_gpus: int,
    *,
    sp_degree: Optional[int] = None,
    tp_size: Optional[int] = None,
    ulysses_degree: Optional[int] = None,
    ring_degree: Optional[int] = None,
) -> ServerArgs:
    kwargs = {
        "model_path": local_root,
        "num_gpus": num_gpus,
        "trust_remote_code": True,
    }
    if sp_degree is not None:
        kwargs["sp_degree"] = sp_degree
    if tp_size is not None:
        kwargs["tp_size"] = tp_size
    if ulysses_degree is not None:
        kwargs["ulysses_degree"] = ulysses_degree
    if ring_degree is not None:
        kwargs["ring_degree"] = ring_degree
    sgl_args = ServerArgs(**kwargs)
    if tp_size is None and case.server_args.tp_size is not None:
        sgl_args.tp_size = case.server_args.tp_size
    if ulysses_degree is None and case.server_args.ulysses_degree is not None:
        sgl_args.ulysses_degree = case.server_args.ulysses_degree
    if ring_degree is None and case.server_args.ring_degree is not None:
        sgl_args.ring_degree = case.server_args.ring_degree
    sgl_args.text_encoder_cpu_offload = False
    sgl_args.dit_cpu_offload = False
    sgl_args.vae_cpu_offload = False
    sgl_args.image_encoder_cpu_offload = False
    sgl_args.enable_cfg_parallel = case.server_args.cfg_parallel or False
    sgl_args.enable_cache_dit = case.server_args.enable_cache_dit
    sgl_args.dit_layerwise_offload = case.server_args.dit_layerwise_offload
    sgl_args.dit_offload_prefetch_size = case.server_args.dit_offload_prefetch_size
    return sgl_args


def _load_sglang_component(
    comp_path: str,
    sgl_args: ServerArgs,
    component: ComponentType,
    library: str,
) -> nn.Module:
    loader = ComponentLoader.for_component_type(component.value, library)
    component_model = loader.load_customized(comp_path, sgl_args, component.value)
    if component_model is None:
        raise RuntimeError(f"Failed to load customized {component.value}")
    return component_model


def _load_reference_component(
    comp_path: str,
    local_root: str,
    component: ComponentType,
    hub_id: str,
    pipeline_config,
    subfolder: str,
) -> nn.Module:
    if component == ComponentType.VAE and "wan" in hub_id.lower():
        return AutoencoderKLWan(pipeline_config.vae_config)

    if component == ComponentType.VAE:
        cfg = _read_json(os.path.join(comp_path, "config.json"))
        class_name = cfg.get("_class_name") if cfg else None
        cls = getattr(diffusers, str(class_name), None) if class_name else None
        if cls is None:
            cls = diffusers.AutoencoderKL
        return cls.from_pretrained(
            local_root,
            subfolder=subfolder,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    if component == ComponentType.TRANSFORMER:
        cfg = _read_json(os.path.join(comp_path, "config.json"))
        class_name = cfg.get("_class_name") if cfg else None
        load_kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if cfg:
            for k, out_k in [
                ("in_dim", "in_channels"),
                ("dim", "hidden_size"),
                ("num_heads", "num_attention_heads"),
                ("out_dim", "out_channels"),
            ]:
                if k in cfg:
                    load_kwargs[out_k] = cfg[k]
        if (
            "i2v" in hub_id.lower()
            or "ti2v" in hub_id.lower()
            or "i2v" in comp_path.lower()
        ):
            load_kwargs["image_dim"] = 1280
        candidates = [diffusers.AutoModel]
        if class_name:
            maybe_cls = getattr(diffusers, str(class_name), None)
            if maybe_cls is not None:
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
            "ignore_mismatched_sizes": True,
            "config": config,
        }
        class_order = [
            AutoModel,
            AutoModelForCausalLM,
            UMT5EncoderModel,
            T5EncoderModel,
            AutoModelForVision2Seq,
        ]
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


def _set_module_attr(module: nn.Module, name: str, value: Any) -> None:
    attrs = name.split(".")
    parent = module
    for attr in attrs[:-1]:
        if hasattr(parent, attr):
            parent = getattr(parent, attr)
        elif isinstance(parent, (nn.ModuleList, nn.Sequential)):
            parent = parent[int(attr)]
        elif isinstance(parent, nn.ModuleDict):
            parent = parent[attr]
        else:
            raise AttributeError(
                f"Cannot resolve {name} on {module.__class__.__name__}"
            )
    setattr(parent, attrs[-1], value)


def _materialize_module(
    module: nn.Module, device: torch.device, dtype: torch.dtype
) -> None:
    for name, param in module.named_parameters():
        if param.device.type == "meta":
            new_data = torch.zeros(param.shape, device=device, dtype=dtype)
            if hasattr(param, "device_mesh") and param.device_mesh is not None:
                new_data = distribute_tensor(
                    new_data, param.device_mesh, param.placements
                )
            _set_module_attr(
                module, name, nn.Parameter(new_data, requires_grad=param.requires_grad)
            )
        elif torch.is_floating_point(param):
            param.data = param.data.to(device=device, dtype=dtype)

    for name, buf in module.named_buffers():
        if buf.device.type == "meta":
            new_buf = torch.zeros(buf.shape, device=device, dtype=buf.dtype)
            if hasattr(buf, "device_mesh") and buf.device_mesh is not None:
                new_buf = distribute_tensor(new_buf, buf.device_mesh, buf.placements)
            _set_module_attr(module, name, new_buf)
        elif torch.is_floating_point(buf):
            buf.data = buf.data.to(device=device, dtype=dtype)


SOURCE_PREFIXES = (
    "module.",
    "model.",
    "transformer.",
    "text_encoder.",
    "image_encoder.",
    "encoder.",
    "decoder.",
    "model.language_model.",
    "model.visual.",
)

TARGET_PREFIXES = (
    "module.",
    "model.",
    "transformer.",
    "text_encoder.",
    "image_encoder.",
    "encoder.",
    "decoder.",
)


def _build_state_lookup(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    lookup: Dict[str, torch.Tensor] = {}
    for key, val in state.items():
        lookup[key] = val
        for prefix in SOURCE_PREFIXES:
            if key.startswith(prefix):
                lookup[key[len(prefix) :]] = val
    return lookup


def _normalize_key(name: str) -> str:
    return (
        name.replace("_fsdp_wrapped_module.", "")
        .replace("_orig_mod.", "")
        .replace("gamma", "weight")
        .replace("beta", "bias")
        .replace("scale", "weight")
        .replace("shift", "bias")
    )


def _fuse_qkv(lookup: Dict[str, torch.Tensor], name: str) -> Optional[torch.Tensor]:
    if "qkv_proj" not in name:
        return None
    variants = ["q_proj", "q"]
    for repl in variants:
        q_name = name.replace("qkv_proj", repl)
        k_name = q_name.replace(".q_proj", ".k_proj").replace(".q", ".k")
        v_name = q_name.replace(".q_proj", ".v_proj").replace(".q", ".v")
        if q_name in lookup and k_name in lookup and v_name in lookup:
            return torch.cat([lookup[q_name], lookup[k_name], lookup[v_name]], dim=0)
    return None


def _generate_name_candidates(
    name: str, reverse_mapping: Optional[Dict[str, Tuple[str, Any, Any]]]
) -> List[str]:
    candidates: List[str] = []
    clean = _normalize_key(name)

    for cand in (name, clean):
        if cand not in candidates:
            candidates.append(cand)

    if reverse_mapping:
        for key in (name, clean):
            entry = reverse_mapping.get(key)
            if entry and entry[0] not in candidates:
                candidates.append(entry[0])

    for prefix in TARGET_PREFIXES:
        if clean.startswith(prefix):
            stripped = clean[len(prefix) :]
            if stripped and stripped not in candidates:
                candidates.append(stripped)

    parts = clean.split(".")
    for i in range(1, len(parts)):
        cand = ".".join(parts[i:])
        if cand not in candidates:
            candidates.append(cand)

    return candidates


def _copy_tensor(
    dest: torch.Tensor,
    src: torch.Tensor,
    tp_world: int,
    rank: int,
) -> bool:
    if src.numel() == 0:
        return False
    src = src.to(device=dest.device, dtype=dest.dtype)

    if hasattr(dest, "device_mesh") and dest.device_mesh is not None:
        if src.numel() == dest.numel():
            with torch.no_grad():
                dt = distribute_tensor(
                    src.view(dest.shape), dest.device_mesh, dest.placements
                )
                dest.copy_(dt)
            return True

    if src.numel() == dest.numel():
        with torch.no_grad():
            dest.copy_(src.view(dest.shape))
        return True

    if tp_world > 1 and src.numel() == dest.numel() * tp_world:
        if src.ndim >= 2 and src.shape[0] == dest.shape[0] * tp_world:
            with torch.no_grad():
                dest.copy_(src[rank * dest.shape[0] : (rank + 1) * dest.shape[0], ...])
            return True
        if src.ndim >= 2 and src.shape[1] == dest.shape[1] * tp_world:
            with torch.no_grad():
                dest.copy_(src[:, rank * dest.shape[1] : (rank + 1) * dest.shape[1]])
            return True
        flat = src.flatten()
        chunk = flat.numel() // tp_world
        with torch.no_grad():
            dest.copy_(flat[rank * chunk : (rank + 1) * chunk].view(dest.shape))
        return True

    if src.ndim == 4 and dest.ndim == 5 and dest.numel() == src.numel() * dest.shape[2]:
        with torch.no_grad():
            dest.copy_(src.unsqueeze(2).repeat(1, 1, dest.shape[2], 1, 1))
        return True

    return False


def _iter_named_params(module: nn.Module) -> Iterable[Tuple[str, torch.Tensor]]:
    for name, param in module.named_parameters():
        yield name, param


def _iter_named_buffers(module: nn.Module) -> Iterable[Tuple[str, torch.Tensor]]:
    for name, buf in module.named_buffers():
        yield name, buf


class AccuracyEngine:
    @staticmethod
    def clear_memory() -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    @staticmethod
    def check_accuracy(
        target: torch.Tensor, reference: torch.Tensor, name: str, threshold: float
    ) -> None:
        if hasattr(target, "full_tensor"):
            target = target.full_tensor()
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
                if t.numel() == r.numel():
                    logger.warning(
                        "[%s] Shape mismatch %s vs %s; reshaping by numel equality.",
                        name,
                        list(t.shape),
                        list(r.shape),
                    )
                    t = t.reshape(r.shape)
                else:
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
    ) -> None:
        """Copy matching parameters from reference to SGLang module with TP-aware slicing."""
        device = get_local_torch_device()
        dtype = torch.bfloat16
        _materialize_module(target, device, dtype)

        source_state = source.state_dict()
        mapping = getattr(target, "param_names_mapping", None) or getattr(
            getattr(target, "module", None), "param_names_mapping", None
        )
        if mapping:
            source_state, _ = hf_to_custom_state_dict(
                source_state, get_param_names_mapping(mapping)
            )

        lookup = _build_state_lookup(source_state)
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

        matched = 0
        total = 0
        for name, tensor in _iter_named_params(target):
            total += 1
            src_tensor = None
            for cand in _generate_name_candidates(name, reverse_mapping):
                if cand in lookup:
                    src_tensor = lookup[cand]
                    break
            if src_tensor is None:
                for cand in _generate_name_candidates(name, reverse_mapping):
                    src_tensor = _fuse_qkv(lookup, cand)
                    if src_tensor is not None:
                        break
            if src_tensor is None:
                continue
            if _copy_tensor(tensor, src_tensor, tp_world, rank):
                matched += 1

        for name, tensor in _iter_named_buffers(target):
            src_tensor = None
            for cand in _generate_name_candidates(name, reverse_mapping):
                if cand in lookup:
                    src_tensor = lookup[cand]
                    break
            if src_tensor is None:
                continue
            _copy_tensor(tensor, src_tensor, tp_world, rank)

        ratio = matched / max(total, 1)
        logger.info(
            "Weight transfer: %s/%s matched (%.2f%%).", matched, total, ratio * 100
        )
        if ratio < min_match_ratio:
            raise RuntimeError(
                f"Weight transfer matched {matched}/{total} ({ratio:.2%}); below threshold {min_match_ratio:.2%}."
            )

    @staticmethod
    def load_component_pair(
        case: DiffusionTestCase,
        component: ComponentType,
        library: str,
        num_gpus: int,
    ) -> Tuple[nn.Module, nn.Module, str, Optional[ComponentAdapter]]:
        """Load SGLang + reference components and align weights for accuracy checks."""
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
        local_root = _resolve_local_path(hub_id)
        comp_path, subfolder = resolve_component_path(local_root, component)

        _init_parallel(case, component, num_gpus)

        pipeline_config = _build_pipeline_config(case, hub_id)
        override_sp = None
        override_tp = None
        override_ulysses = None
        override_ring = None
        if component == ComponentType.TEXT_ENCODER or (
            component == ComponentType.TRANSFORMER
            and (
                "zimage" in hub_id.lower()
                or "z-image" in hub_id.lower()
                or "wan" in hub_id.lower()
            )
        ):
            override_sp = 1
            override_tp = 1
            override_ulysses = 1
            override_ring = 1
        sgl_args = _build_server_args(
            local_root,
            case,
            num_gpus,
            sp_degree=override_sp,
            tp_size=override_tp,
            ulysses_degree=override_ulysses,
            ring_degree=override_ring,
        )
        sgl_args.pipeline_config = pipeline_config
        set_global_server_args(sgl_args)

        adapter = get_adapter(case.id, component.value)
        device = get_local_torch_device()

        sgl_component = _load_sglang_component(
            comp_path, sgl_args, component, library
        ).to(device=device, dtype=torch.bfloat16)
        ref_component = _load_reference_component(
            comp_path, local_root, component, hub_id, pipeline_config, subfolder
        ).to(device=device, dtype=torch.bfloat16)

        if component == ComponentType.TRANSFORMER and "wan" in hub_id.lower():
            fc_mod._forward_context = ForwardContext(
                current_timestep=0, attn_metadata=None
            )

        if component == ComponentType.TEXT_ENCODER:
            # T5/UMT5 expose the token embedding as `shared`, which lives on the full model.
            # Using get_encoder() drops it and leads to a large accuracy drop.
            if hasattr(ref_component, "shared"):
                ref_for_transfer = ref_component
            elif hasattr(ref_component, "get_encoder"):
                ref_for_transfer = ref_component.get_encoder()
            else:
                ref_for_transfer = ref_component
        else:
            ref_for_transfer = ref_component
        AccuracyEngine.transfer_weights(ref_for_transfer, sgl_component)

        if component != ComponentType.VAE and not hasattr(
            fc_mod._forward_context, "attn_metadata"
        ):
            fc_mod._forward_context = ForwardContext(
                current_timestep=int(DEFAULT_TIMESTEP), attn_metadata=None
            )

        return sgl_component.eval(), ref_component.eval(), str(device), adapter
