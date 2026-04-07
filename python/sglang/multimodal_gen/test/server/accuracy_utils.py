from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors.torch import load_file as safetensors_load_file
from torch.distributed.tensor import distribute_tensor

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    destroy_model_parallel,
    get_data_parallel_world_size,
    get_sequence_parallel_world_size,
    get_tensor_model_parallel_world_size,
    maybe_init_distributed_environment_and_model_parallel,
    model_parallel_is_initialized,
)
from sglang.multimodal_gen.runtime.layers.utils import get_group_rank, get_group_size
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import maybe_download_model
from sglang.multimodal_gen.test.server.accuracy_config import (
    DEFAULT_TEXT_ENCODER_VOCAB_SIZE,
    I2V_TEXT_ENCODER_DIM,
    TEXT_ENCODER_INPUT_SEED,
    TEXT_ENCODER_TOKEN_LENGTH,
    TEXT_ENCODER_TOKEN_MAX,
    TEXT_ENCODER_TOKEN_MIN,
    ComponentType,
    get_threshold,
)

STAGED_1GPU_NATIVE_CASE_IDS = {
    "flux_2_image_t2i",
    "qwen_image_layered_i2i",
    "flux_2_image_t2i_upscaling_4x",
    "flux_2_ti2i",
    "flux_2_t2i_customized_vae_path",
    "flux_2_ti2i_multi_image_cache_dit",
}

# These case allowlists are accuracy-runner policy. They select the few 1-GPU
# cases that need sequential SGLang/reference execution to stay within memory
# limits during CI and local correctness runs.
STAGED_1GPU_TEXT_ENCODER_CASE_IDS = {
    "flux_2_image_t2i",
    "flux_2_image_t2i_upscaling_4x",
    "mova_360p_1gpu",
    "flux_2_ti2i",
    "flux_2_t2i_customized_vae_path",
    "flux_2_ti2i_multi_image_cache_dit",
}

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


@dataclass(frozen=True)
class ComponentSelection:
    base_model_id: str
    base_model_root: str
    component_paths: Dict[str, str]
    source_root: str
    source_path: str
    source_subfolder: str


@dataclass(frozen=True)
class ParameterShardContext:
    world_size: int
    rank: int


def seed_and_broadcast(seed: int, tensor: torch.Tensor) -> torch.Tensor:
    """Seed and broadcast tensor across ranks for determinism."""
    torch.manual_seed(seed)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        torch.distributed.broadcast(tensor, src=0)
    return tensor


def read_json_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def has_component_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    if os.path.exists(os.path.join(path, "config.json")):
        return True
    for ext in (".safetensors", ".bin", ".pth"):
        if any(name.endswith(ext) for name in os.listdir(path)):
            return True
    return False


def list_safetensor_files(path: str) -> List[str]:
    if not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(path, name)
        for name in os.listdir(path)
        if name.endswith(".safetensors")
    )


def is_text_encoder_config(path: str) -> bool:
    cfg_path = os.path.join(path, "config.json")
    if not os.path.exists(cfg_path):
        return False
    cfg = read_json_file(cfg_path)
    if cfg.get("model_type") == "i2v" or cfg.get("dim") == I2V_TEXT_ENCODER_DIM:
        return False
    return True


def _resolve_component_subfolder(
    model_index: Dict[str, Any], key: str
) -> Optional[str]:
    entry = model_index.get(key)
    if isinstance(entry, dict):
        return entry.get("path") or entry.get("subfolder")
    if isinstance(entry, str):
        return entry
    if entry is not None:
        return key
    return None


def resolve_component_path(
    local_root: str, component: ComponentType, model_index_keys: Tuple[str, ...]
) -> Tuple[str, str]:
    model_index_path = os.path.join(local_root, "model_index.json")
    model_index = read_json_file(model_index_path)

    if model_index:
        for key in model_index_keys:
            subfolder = _resolve_component_subfolder(model_index, key)
            if not subfolder:
                continue
            candidate = os.path.join(local_root, subfolder)
            if not has_component_files(candidate):
                continue
            if component == ComponentType.TEXT_ENCODER and not is_text_encoder_config(
                candidate
            ):
                continue
            return candidate, subfolder

    if has_component_files(local_root):
        if component != ComponentType.TEXT_ENCODER or is_text_encoder_config(
            local_root
        ):
            return local_root, ""

    raise FileNotFoundError(
        f"Could not resolve {component.value} from model_index.json under {local_root}"
    )


def extract_component_path_overrides(extra_args: List[str]) -> Dict[str, str]:
    component_paths: Dict[str, str] = {}
    index = 0
    while index < len(extra_args):
        arg = extra_args[index]
        key_part = arg.split("=", 1)[0] if "=" in arg else arg
        if key_part.startswith("--") and key_part.endswith("-path"):
            component = key_part[2:-5].replace("-", "_")
            if "=" in arg:
                component_paths[component] = arg.split("=", 1)[1]
            elif index + 1 < len(extra_args) and not extra_args[index + 1].startswith(
                "-"
            ):
                index += 1
                component_paths[component] = extra_args[index]
        index += 1

    for component, path in component_paths.items():
        component_paths[component] = os.path.expanduser(path)
    return component_paths


def load_checkpoint_weights(
    module: nn.Module, model_path: str
) -> tuple[list[str], list[str]]:
    safetensors_files = list_safetensor_files(model_path)
    assert safetensors_files, f"Found no safetensors files in {model_path}"

    loaded_state: Dict[str, torch.Tensor] = {}
    for safetensor_path in safetensors_files:
        loaded_state.update(safetensors_load_file(safetensor_path))

    module.load_state_dict(loaded_state, strict=False)

    state_keys = set(module.state_dict().keys())
    loaded_keys = set(loaded_state.keys())
    missing_keys = sorted(state_keys - loaded_keys)
    unexpected_keys = sorted(loaded_keys - state_keys)
    return missing_keys, unexpected_keys


def select_component_source(
    model_id: str,
    extra_args: List[str],
    component: ComponentType,
    model_index_keys: Tuple[str, ...],
) -> ComponentSelection:
    component_paths = extract_component_path_overrides(extra_args)
    base_model_root = maybe_download_model(model_id)
    search_keys = [component.value]
    for key in model_index_keys:
        if key not in search_keys:
            search_keys.append(key)

    source_root = base_model_root
    component_key = component.value
    for key in search_keys:
        override_path = component_paths.get(key)
        if override_path:
            source_root = maybe_download_model(override_path)
            component_key = key
            break

    ordered_keys = [component_key]
    for key in search_keys:
        if key not in ordered_keys:
            ordered_keys.append(key)
    source_path, source_subfolder = resolve_component_path(
        source_root,
        component,
        tuple(ordered_keys),
    )
    return ComponentSelection(
        base_model_id=model_id,
        base_model_root=base_model_root,
        component_paths=component_paths,
        source_root=source_root,
        source_path=source_path,
        source_subfolder=source_subfolder,
    )


def ensure_distributed_env_defaults() -> None:
    if "WORLD_SIZE" in os.environ:
        return
    os.environ.update(
        {
            "MASTER_ADDR": os.getenv("MASTER_ADDR", "127.0.0.1"),
            "MASTER_PORT": os.getenv("MASTER_PORT", "29505"),
            "RANK": "0",
            "LOCAL_RANK": "0",
            "WORLD_SIZE": "1",
        }
    )


def initialize_parallel_runtime(sgl_args: ServerArgs) -> None:
    tp_size = sgl_args.tp_size
    sp_degree = sgl_args.sp_degree
    ulysses_degree = sgl_args.ulysses_degree
    ring_degree = sgl_args.ring_degree
    dp_size = sgl_args.dp_size
    enable_cfg_parallel = bool(sgl_args.enable_cfg_parallel)

    if (
        tp_size is None
        or sp_degree is None
        or ulysses_degree is None
        or ring_degree is None
    ):
        raise RuntimeError(
            "ServerArgs must have tp_size, sp_degree, ulysses_degree, and ring_degree before init"
        )

    if not model_parallel_is_initialized() and torch.distributed.is_initialized():
        # A prior case may have failed while distributed groups were only partially
        # initialized. Clear any stale group objects before re-initializing.
        destroy_model_parallel()

    if model_parallel_is_initialized():
        current_tp = get_tensor_model_parallel_world_size()
        current_sp = get_sequence_parallel_world_size()
        current_dp = get_data_parallel_world_size()
        if current_tp == tp_size and current_sp == sp_degree and current_dp == dp_size:
            return
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        destroy_model_parallel()

    ensure_distributed_env_defaults()

    maybe_init_distributed_environment_and_model_parallel(
        tp_size=tp_size,
        sp_size=sp_degree,
        enable_cfg_parallel=enable_cfg_parallel,
        ulysses_degree=ulysses_degree,
        ring_degree=ring_degree,
        dp_size=dp_size,
    )
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def build_accuracy_server_args(
    base_model_id: str,
    base_model_root: str,
    case: Any,
    component: ComponentType,
    num_gpus: int,
    component_paths: Dict[str, str],
) -> ServerArgs:
    cfg_parallel = bool(case.server_args.cfg_parallel)
    kwargs = {
        "model_path": base_model_root,
        "model_id": base_model_id,
        "num_gpus": num_gpus,
        "trust_remote_code": True,
        "component_paths": component_paths,
        "enable_cfg_parallel": cfg_parallel,
    }

    if case.server_args.tp_size is not None:
        kwargs["tp_size"] = case.server_args.tp_size
    if case.server_args.ulysses_degree is not None:
        kwargs["ulysses_degree"] = case.server_args.ulysses_degree
    if case.server_args.ring_degree is not None:
        kwargs["ring_degree"] = case.server_args.ring_degree

    if component == ComponentType.TEXT_ENCODER:
        kwargs["enable_cfg_parallel"] = False

    sgl_args = ServerArgs.from_kwargs(**kwargs)
    sgl_args.text_encoder_cpu_offload = False
    sgl_args.dit_cpu_offload = False
    sgl_args.vae_cpu_offload = False
    sgl_args.image_encoder_cpu_offload = False
    sgl_args.enable_cache_dit = case.server_args.enable_cache_dit
    sgl_args.dit_layerwise_offload = case.server_args.dit_layerwise_offload
    sgl_args.dit_offload_prefetch_size = case.server_args.dit_offload_prefetch_size
    return sgl_args


def set_module_attr(module: nn.Module, name: str, value: Any) -> None:
    """Assign to a nested parameter/buffer path such as `blocks.0.attn.to_q.weight`."""
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


def materialize_module(
    module: nn.Module, device: torch.device, dtype: torch.dtype
) -> None:
    """Materialize meta tensors and cast floating tensors onto one target device/dtype."""
    for name, param in module.named_parameters():
        if param.device.type == "meta":
            new_data = torch.zeros(param.shape, device=device, dtype=dtype)
            if hasattr(param, "device_mesh") and param.device_mesh is not None:
                new_data = distribute_tensor(
                    new_data, param.device_mesh, param.placements
                )
            set_module_attr(
                module, name, nn.Parameter(new_data, requires_grad=param.requires_grad)
            )
        elif torch.is_floating_point(param):
            param.data = param.data.to(device=device, dtype=dtype)

    for name, buf in module.named_buffers():
        if buf.device.type == "meta":
            new_buf = torch.zeros(buf.shape, device=device, dtype=buf.dtype)
            if hasattr(buf, "device_mesh") and buf.device_mesh is not None:
                new_buf = distribute_tensor(new_buf, buf.device_mesh, buf.placements)
            set_module_attr(module, name, new_buf)
        elif torch.is_floating_point(buf):
            buf.data = buf.data.to(device=device, dtype=dtype)


def build_parameter_shard_contexts(
    module: nn.Module,
) -> Dict[str, ParameterShardContext]:
    """Record TP shard world/rank for each parameter owned by a TP-aware submodule."""
    shard_contexts: Dict[str, ParameterShardContext] = {}
    for module_name, submodule in module.named_modules():
        tp_group = getattr(submodule, "tp_group", None)
        if tp_group is None:
            continue

        context = ParameterShardContext(
            world_size=get_group_size(tp_group),
            rank=get_group_rank(tp_group),
        )
        if context.world_size <= 1:
            continue

        for name, _ in submodule.named_parameters(recurse=False):
            qualified_name = f"{module_name}.{name}" if module_name else name
            shard_contexts[qualified_name] = context
        for name, _ in submodule.named_buffers(recurse=False):
            qualified_name = f"{module_name}.{name}" if module_name else name
            shard_contexts[qualified_name] = context

    return shard_contexts


def build_state_lookup(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Index a source state dict under both original and prefix-stripped names."""
    lookup: Dict[str, torch.Tensor] = {}
    for key, val in state.items():
        lookup[key] = val
        for prefix in SOURCE_PREFIXES:
            if key.startswith(prefix):
                lookup[key[len(prefix) :]] = val
    return lookup


def normalize_state_key(name: str) -> str:
    """Normalize common naming differences between source and target state dicts."""
    return (
        name.replace("_fsdp_wrapped_module.", "")
        .replace("_orig_mod.", "")
        .replace("gamma", "weight")
        .replace("beta", "bias")
        .replace("scale", "weight")
        .replace("shift", "bias")
    )


def fuse_qkv(lookup: Dict[str, torch.Tensor], name: str) -> Optional[torch.Tensor]:
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


def fuse_gate_up_proj(
    lookup: Dict[str, torch.Tensor], name: str
) -> Optional[torch.Tensor]:
    if "gate_up_proj" not in name:
        return None

    for gate_token, up_token in (("gate_proj", "up_proj"), ("wi_0", "wi_1")):
        gate_name = name.replace("gate_up_proj", gate_token)
        up_name = name.replace("gate_up_proj", up_token)
        if gate_name in lookup and up_name in lookup:
            return torch.cat([lookup[gate_name], lookup[up_name]], dim=0)
    return None


def generate_name_candidates(
    name: str, reverse_mapping: Optional[Dict[str, Tuple[str, Any, Any]]]
) -> List[str]:
    candidates: List[str] = []
    clean = normalize_state_key(name)

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


def copy_tensor(
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
        if (
            src.ndim == dest.ndim
            and src.shape[0] == dest.shape[0] * tp_world
            and src.shape[1:] == dest.shape[1:]
        ):
            with torch.no_grad():
                dest.copy_(src[rank * dest.shape[0] : (rank + 1) * dest.shape[0], ...])
            return True
        if (
            src.ndim >= 2
            and dest.ndim >= 2
            and src.ndim == dest.ndim
            and src.shape[0] == dest.shape[0]
            and src.shape[1] == dest.shape[1] * tp_world
            and src.shape[2:] == dest.shape[2:]
        ):
            with torch.no_grad():
                dest.copy_(src[:, rank * dest.shape[1] : (rank + 1) * dest.shape[1]])
            return True

    if src.ndim == 4 and dest.ndim == 5 and dest.numel() == src.numel() * dest.shape[2]:
        with torch.no_grad():
            dest.copy_(src.unsqueeze(2).repeat(1, 1, dest.shape[2], 1, 1))
        return True

    return False


def _config_to_dict(config: Any) -> Dict[str, Any]:
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        return {}
    config_dict = to_dict()
    return config_dict if isinstance(config_dict, dict) else {}


def resolve_text_encoder_vocab_size(config: Any) -> int:
    config_dict = _config_to_dict(config)
    vocab_size = config_dict.get("vocab_size")
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size

    text_config = config_dict.get("text_config")
    if isinstance(text_config, dict):
        nested_vocab_size = text_config.get("vocab_size")
        if isinstance(nested_vocab_size, int) and nested_vocab_size > 0:
            return nested_vocab_size

    return DEFAULT_TEXT_ENCODER_VOCAB_SIZE


def build_deterministic_text_encoder_inputs(
    config: Any, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one stable token batch that works across text-encoder implementations."""
    vocab_size = resolve_text_encoder_vocab_size(config)
    max_token_id = max(
        TEXT_ENCODER_TOKEN_MIN + 1, min(vocab_size, TEXT_ENCODER_TOKEN_MAX)
    )

    torch.manual_seed(TEXT_ENCODER_INPUT_SEED)
    input_ids = torch.randint(
        TEXT_ENCODER_TOKEN_MIN,
        max_token_id,
        (1, TEXT_ENCODER_TOKEN_LENGTH),
        device="cpu",
        dtype=torch.long,
    ).to(device)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def resolve_text_encoder_forward_module(model: nn.Module) -> nn.Module:
    get_encoder = getattr(model, "get_encoder", None)
    return get_encoder() if callable(get_encoder) else model


def _module_device(module: nn.Module) -> torch.device:
    param = next(module.parameters(), None)
    if param is not None:
        return param.device

    buf = next(module.buffers(), None)
    if buf is not None:
        return buf.device

    return torch.device("cpu")


def extract_output_tensor(output: Any) -> torch.Tensor:
    """Best-effort extraction of a tensor from model outputs."""
    if isinstance(output, torch.Tensor):
        return output

    sample = getattr(output, "sample", None)
    if sample is not None:
        if isinstance(sample, (list, tuple)):
            sample = sample[0]
        if isinstance(sample, torch.Tensor):
            return sample

    last_hidden_state = getattr(output, "last_hidden_state", None)
    if last_hidden_state is not None:
        return last_hidden_state

    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]

    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output

    logits = getattr(output, "logits", None)
    if logits is not None:
        return logits

    if (
        isinstance(output, (list, tuple))
        and output
        and isinstance(output[0], torch.Tensor)
    ):
        return output[0]
    raise ValueError(f"Could not extract tensor from output of type {type(output)}")


def run_text_encoder_accuracy_pair(
    sgl: nn.Module, ref: nn.Module
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids, attention_mask = build_deterministic_text_encoder_inputs(
        ref.config, "cpu"
    )
    return (
        _run_single_text_encoder_forward(sgl, input_ids, attention_mask),
        _run_single_text_encoder_forward(ref, input_ids, attention_mask),
    )


def _should_stage_case(case: Any, component: ComponentType, num_gpus: int) -> bool:
    if num_gpus == 2:
        return True
    if num_gpus != 1:
        return False
    if component == ComponentType.TEXT_ENCODER:
        return case.id in STAGED_1GPU_TEXT_ENCODER_CASE_IDS
    return case.id in STAGED_1GPU_NATIVE_CASE_IDS


def _run_single_text_encoder_forward(
    model: nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Run one encoder forward and normalize its output into a tensor."""
    with torch.no_grad():
        forward_model = resolve_text_encoder_forward_module(model)
        model_device = _module_device(forward_model)
        output = forward_model(
            input_ids.to(device=model_device),
            attention_mask=attention_mask.to(device=model_device),
            output_hidden_states=True,
        )
    return extract_output_tensor(output)


def _run_staged_native_component_accuracy_case(
    engine_cls: Any,
    case: Any,
    component: ComponentType,
    library: str,
    num_gpus: int,
) -> None:
    from sglang.multimodal_gen.test.server.accuracy_hooks import (
        resolve_component_native_profile,
    )

    sgl = None
    ref = None
    try:
        sgl, ref, device = engine_cls.load_component_pair(
            case,
            component,
            library,
            num_gpus,
            materialize_ref_on_device=False,
        )
        profile = resolve_component_native_profile(component)
        inputs = profile.build_inputs(case, sgl, device, ref)

        sgl_call = profile.prepare_sglang_call(sgl, inputs)
        with torch.no_grad():
            sgl_raw = engine_cls._execute_with_native_hook(sgl_call)
        sgl_out = profile.normalize_sglang_output(sgl_raw)
        sgl_out = engine_cls._apply_output_transforms(sgl_out, sgl_call).detach().cpu()

        del sgl_call
        del sgl_raw
        del sgl
        sgl = None
        engine_cls.clear_memory()

        ref = ref.to(device=device, dtype=torch.bfloat16).eval()
        ref_call = profile.prepare_reference_call(ref, inputs)
        with torch.no_grad():
            ref_raw = engine_cls._execute_with_native_hook(ref_call)
        ref_out = profile.normalize_reference_output(ref_raw)
        ref_out = engine_cls._apply_output_transforms(ref_out, ref_call).detach().cpu()
        del ref_call
        del ref_raw

        engine_cls.check_accuracy(
            sgl_out,
            ref_out,
            f"{case.id}_{component.value}",
            get_threshold(case.id, component),
        )
    finally:
        if sgl is not None:
            del sgl
        if ref is not None:
            del ref
        engine_cls.reset_parallel_runtime()
        engine_cls.clear_memory()


def _run_staged_text_encoder_accuracy_case(
    engine_cls: Any, case: Any, num_gpus: int
) -> None:
    sgl = None
    ref = None
    try:
        sgl, ref, device = engine_cls.load_component_pair(
            case,
            ComponentType.TEXT_ENCODER,
            "transformers",
            num_gpus,
            materialize_sgl_on_device=False,
            materialize_ref_on_device=False,
        )
        input_ids, attention_mask = build_deterministic_text_encoder_inputs(
            ref.config, "cpu"
        )

        sgl = sgl.to(device=device, dtype=torch.bfloat16).eval()
        sgl_out = (
            _run_single_text_encoder_forward(sgl, input_ids, attention_mask)
            .detach()
            .cpu()
        )

        del sgl
        sgl = None
        engine_cls.clear_memory()

        ref = ref.to(device=device, dtype=torch.bfloat16).eval()
        ref_out = (
            _run_single_text_encoder_forward(ref, input_ids, attention_mask)
            .detach()
            .cpu()
        )

        engine_cls.check_accuracy(
            sgl_out,
            ref_out,
            f"{case.id}_encoder",
            get_threshold(case.id, ComponentType.TEXT_ENCODER),
        )
    finally:
        if sgl is not None:
            del sgl
        if ref is not None:
            del ref
        engine_cls.reset_parallel_runtime()
        engine_cls.clear_memory()


def run_native_component_accuracy_case(
    engine_cls: Any,
    case: Any,
    component: ComponentType,
    library: str,
    num_gpus: int,
) -> None:
    if _should_stage_case(case, component, num_gpus):
        _run_staged_native_component_accuracy_case(
            engine_cls, case, component, library, num_gpus
        )
        return
    engine_cls.clear_memory()
    sgl = None
    ref = None
    try:
        sgl, ref, device = engine_cls.load_component_pair(
            case, component, library, num_gpus
        )
        sgl_out, ref_out = engine_cls.run_component_pair_native(
            case, component, sgl, ref, device
        )
        engine_cls.check_accuracy(
            sgl_out,
            ref_out,
            f"{case.id}_{component.value}",
            get_threshold(case.id, component),
        )
    finally:
        if sgl is not None:
            del sgl
        if ref is not None:
            del ref
        engine_cls.reset_parallel_runtime()
        engine_cls.clear_memory()


def run_text_encoder_accuracy_case(engine_cls: Any, case: Any, num_gpus: int) -> None:
    if _should_stage_case(case, ComponentType.TEXT_ENCODER, num_gpus):
        _run_staged_text_encoder_accuracy_case(engine_cls, case, num_gpus)
        return
    engine_cls.clear_memory()
    sgl = None
    ref = None
    try:
        sgl, ref, _device = engine_cls.load_component_pair(
            case, ComponentType.TEXT_ENCODER, "transformers", num_gpus
        )
        sgl_out, ref_out = run_text_encoder_accuracy_pair(sgl, ref)
        engine_cls.check_accuracy(
            sgl_out,
            ref_out,
            f"{case.id}_encoder",
            get_threshold(case.id, ComponentType.TEXT_ENCODER),
        )
    finally:
        if sgl is not None:
            del sgl
        if ref is not None:
            del ref
        engine_cls.reset_parallel_runtime()
        engine_cls.clear_memory()
