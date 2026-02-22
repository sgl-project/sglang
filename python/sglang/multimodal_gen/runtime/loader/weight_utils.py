# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/model_loader/weight_utils.py
"""Utilities for downloading, loading, initializing and verifying model weights."""

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Generator, Iterable
from pathlib import Path

import filelock
import huggingface_hub.constants
import torch
from safetensors.torch import safe_open
from torch.distributed.tensor import DTensor
from tqdm.auto import tqdm

try:
    from runai_model_streamer import SafetensorsStreamer

    HAS_RUNAI_MODEL_STREAMER = True
except ImportError:
    HAS_RUNAI_MODEL_STREAMER = False

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# use system-level temp directory for file locks, so that multiple users
# can share the same lock without error.
# lock files in the temp directory will be automatically deleted when the
# system reboots, so users will not complain about annoying lock files
temp_dir = tempfile.gettempdir()


def enable_hf_transfer() -> None:
    """automatically activates hf_transfer"""
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        try:
            # enable hf hub transfer if available
            import hf_transfer  # type: ignore # noqa

            huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
        except ImportError:
            pass


enable_hf_transfer()


class DisabledTqdm(tqdm):

    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)


def get_lock(model_name_or_path: str | Path, cache_dir: str | None = None):
    lock_dir = cache_dir or temp_dir
    model_name_or_path = str(model_name_or_path)
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock


# For models like Mistral-7B-v0.3, there are both sharded
# safetensors files and a consolidated safetensors file.
# Passing both of these to the weight loader functionality breaks.
# So, we use the index_file to
# look up which safetensors files should be used.
def filter_duplicate_safetensors_files(
    hf_weights_files: list[str], hf_folder: str, index_file: str
) -> list[str]:
    # model.safetensors.index.json is a mapping from keys in the
    # torch state_dict to safetensors file holding that weight.
    index_file_name = os.path.join(hf_folder, index_file)
    if not os.path.isfile(index_file_name):
        return hf_weights_files

    # Iterate through the weight_map (weight_name: safetensors files)
    # to identify weights that we should use.
    with open(index_file_name) as f:
        weight_map = json.load(f)["weight_map"]
    weight_files_in_index = set()
    for weight_name in weight_map:
        weight_files_in_index.add(os.path.join(hf_folder, weight_map[weight_name]))
    # Filter out any fields that are not found in the index file.
    hf_weights_files = [f for f in hf_weights_files if f in weight_files_in_index]
    return hf_weights_files


def filter_files_not_needed_for_inference(hf_weights_files: list[str]) -> list[str]:
    """
    Exclude files that are not needed for inference.

    See https://github.com/huggingface/transformers/blob/v4.34.0/src/transformers/trainer.py#L227-L233
    """
    blacklist = [
        "training_args.bin",
        "optimizer.bin",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
    ]
    hf_weights_files = [
        f for f in hf_weights_files if not any(f.endswith(x) for x in blacklist)
    ]
    return hf_weights_files


# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501


def _validate_safetensors_file(file_path: str) -> bool:
    """
    Validate that a safetensors file is readable and not corrupted.

    Args:
        file_path: Path to the safetensors file

    Returns:
        True if file is valid, False if corrupted
    """
    try:
        with safe_open(file_path, framework="pt", device="cpu") as f:
            _ = list(f.keys())
        return True
    except Exception as e:
        logger.error(
            "Corrupted safetensors file detected: %s - %s: %s",
            file_path,
            type(e).__name__,
            str(e),
        )
        return False


def safetensors_weights_iterator(
    hf_weights_files: list[str],
    to_cpu: bool = True,
    use_runai_model_streamer: bool = HAS_RUNAI_MODEL_STREAMER,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    device = "cpu" if to_cpu else str(get_local_torch_device())

    # Validate files before loading
    corrupted_files = [
        st_file
        for st_file in hf_weights_files
        if not _validate_safetensors_file(st_file)
    ]

    if corrupted_files:
        # Delete corrupted files (both symlink and blob if applicable)
        for file_path in corrupted_files:
            try:
                if os.path.islink(file_path):
                    blob_path = os.path.realpath(file_path)
                    os.remove(file_path)
                    logger.info(
                        "Removed corrupted symlink: %s", os.path.basename(file_path)
                    )
                    if os.path.exists(blob_path):
                        os.remove(blob_path)
                        logger.info(
                            "Removed corrupted blob: %s", os.path.basename(blob_path)
                        )
                elif os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.info(
                        "Removed corrupted file: %s", os.path.basename(file_path)
                    )
            except Exception as e:
                logger.warning("Failed to remove corrupted file %s: %s", file_path, e)

        raise RuntimeError(
            f"Found {len(corrupted_files)} corrupted safetensors file(s). "
            f"Files have been removed: {[os.path.basename(f) for f in corrupted_files]}. "
            "Please retry - the files will be re-downloaded automatically."
        )

    if use_runai_model_streamer:
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(hf_weights_files)
            for name, tensor in streamer.get_tensors():
                if to_cpu:
                    yield name, tensor.clone().detach()
                else:
                    yield name, tensor.to(device)
    else:
        for st_file in tqdm(
            hf_weights_files,
            desc="Loading safetensors checkpoint shards",
            disable=not enable_tqdm,
            bar_format=_BAR_FORMAT,
        ):
            with safe_open(st_file, framework="pt", device=device) as f:
                for name in f.keys():  # noqa: SIM118
                    param = f.get_tensor(name)
                    yield name, param


def _load_pt_file(bin_file: str, device: str) -> dict:
    """Load a PyTorch checkpoint file, handling legacy tar format.

    PyTorch 2.6 changed the default of weights_only from False to True.
    Legacy tar format files cannot be loaded with weights_only=True.
    This function tries weights_only=True first, then falls back to False
    for legacy tar format files from trusted sources (HuggingFace Hub).
    """
    try:
        return torch.load(bin_file, map_location=device, weights_only=True)
    except RuntimeError as e:
        if "legacy .tar format" in str(e):
            logger.warning(
                "Loading %s with weights_only=False (legacy tar format)",
                os.path.basename(bin_file),
            )
            return torch.load(bin_file, map_location=device, weights_only=False)
        raise


def pt_weights_iterator(
    hf_weights_files: list[str],
    to_cpu: bool = True,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model bin/pt files."""
    device = "cpu" if to_cpu else str(get_local_torch_device())
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    for bin_file in tqdm(
        hf_weights_files,
        desc="Loading pt checkpoint shards",
        disable=not enable_tqdm,
        bar_format=_BAR_FORMAT,
    ):
        state = _load_pt_file(bin_file, device)
        yield from state.items()
        del state


def default_weight_loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
    """Default weight loader."""
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise


def maybe_remap_kv_scale_name(name: str, params_dict: dict) -> str | None:
    """Remap the name of FP8 k/v_scale parameters.

    This function handles the remapping of FP8 k/v_scale parameter names.
    It detects if the given name ends with a suffix and attempts to remap
    it to the expected name format in the model. If the remapped name is not
    found in the params_dict, a warning is printed and None is returned.

    Args:
        name (str): The original loaded checkpoint parameter name.
        params_dict (dict): Dictionary containing the model's named parameters.

    Returns:
        str: The remapped parameter name if successful, or the original name
             if no remapping is needed.
        None: If the remapped name is not found in params_dict.
    """
    if name.endswith(".kv_scale"):
        logger.warning_once(
            "DEPRECATED. Found kv_scale in the checkpoint. "
            "This format is deprecated in favor of separate k_scale and "
            "v_scale tensors and will be removed in a future release. "
            "Functionally, we will remap kv_scale to k_scale and duplicate "
            "k_scale to v_scale"
        )
        # NOTE: we remap the deprecated kv_scale to k_scale
        remapped_name = name.replace(".kv_scale", ".attn.k_scale")
        if remapped_name not in params_dict:
            logger.warning_once(
                f"Found kv_scale in the checkpoint (e.g. {name}), "
                "but not found the expected name in the model "
                f"(e.g. {remapped_name}). kv_scale is "
                "not loaded."
            )
            return None
        return remapped_name

    possible_scale_names = [".k_scale", ".v_scale"]
    modelopt_scale_names = [".self_attn.k_proj.k_scale", ".self_attn.v_proj.v_scale"]
    for scale_name in possible_scale_names:
        if name.endswith(scale_name):
            if any(mo_scale_name in name for mo_scale_name in modelopt_scale_names):
                remapped_name = name.replace(
                    f".self_attn.{scale_name[1]}_proj{scale_name}",
                    f".self_attn.attn{scale_name}",
                )
            else:
                remapped_name = name.replace(scale_name, f".attn{scale_name}")
            if remapped_name not in params_dict:
                logger.warning_once(
                    f"Found {scale_name} in the checkpoint (e.g. {name}), "
                    "but not found the expected name in the model "
                    f"(e.g. {remapped_name}). {scale_name} is "
                    "not loaded."
                )
                return None
            return remapped_name

    # If there were no matches, return the untouched param name
    return name


def _tensor_content_signature(tensor: torch.Tensor) -> tuple[tuple[int, ...], bytes]:
    """Return (shape, raw_bytes) for hashing. Shape as tuple for canonical order."""
    t = tensor.detach()
    if isinstance(t, DTensor):
        t = t._local_tensor
    arr = t.cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return (tuple(tensor.shape), arr.data.tobytes())


def compute_weights_checksum(
    named_params: Iterable[tuple[str, torch.Tensor]],
) -> str:
    """Compute a SHA-256 checksum for a set of (name, tensor) pairs.

    Used to verify the correctness of weight refitting. After a refit,
    compare the checksum of the in-GPU model weights against the checksum
    of the on-disk tensors or the tensors in the training engine.
    """
    hasher = hashlib.sha256()
    for name, tensor in sorted(named_params, key=lambda x: x[0]):
        hasher.update(name.encode())
        t = tensor.detach()
        # DTensor doesn't support .numpy(); extract the local tensor.
        if isinstance(t, DTensor):
            t = t._local_tensor
        hasher.update(t.cpu().contiguous().reshape(-1).view(torch.uint8).numpy().data)
    return hasher.hexdigest()


def compute_weights_checksum_content_only(
    named_params: Iterable[tuple[str, torch.Tensor]],
) -> str:
    """Compute a SHA-256 checksum from tensor contents only (ignore parameter names).

    Use when comparing weights that may have different names on disk vs in the
    model (e.g. after weight-name remapping or prefix in loaders). Two sets of
    (name, tensor) with the same multiset of (shape, bytes) produce the same
    checksum. Tensors are sorted by (shape, bytes) for a canonical order.
    """
    sigs = [_tensor_content_signature(t) for _, t in named_params]
    sigs.sort(key=lambda s: (s[0], s[1]))
    hasher = hashlib.sha256()
    for shape, raw in sigs:
        hasher.update(str(shape).encode())
        hasher.update(raw)
    return hasher.hexdigest()

VAE_CHECKSUM_EXCLUDE_NAMES: frozenset[str] = frozenset({
    "bn.num_batches_tracked",
    "bn.running_mean",
    "bn.running_var",
})


def filter_weights_for_checksum(
    named_weights: Iterable[tuple[str, torch.Tensor]],
    exclude_names: frozenset[str],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield (name, tensor) pairs excluding given names (e.g. VAE BatchNorm buffers)."""
    for name, tensor in named_weights:
        if name not in exclude_names:
            yield name, tensor


_QWEN3_STACKED_PARAMS_MAPPING: list[tuple[str, str, str | int]] = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]


def _qwen3_remap_scale_name_for_checksum(name: str) -> str | None:
    """Remap scale param names for checksum (no params_dict). Drops unknown scales."""
    if name.endswith(".kv_scale"):
        return name.replace(".kv_scale", ".attn.k_scale")
    for suffix in (".k_scale", ".v_scale"):
        if name.endswith(suffix):
            return name.replace(suffix, f".attn{suffix}")
    return None


def _disk_to_model_weights_qwen3(
    disk_weights: Iterable[tuple[str, torch.Tensor]],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield (model_param_name, tensor) from disk (name, tensor) using Qwen3 load_weights rules."""
    disk_list = list(disk_weights)
    simple: list[tuple[str, torch.Tensor]] = []
    stacked: dict[str, list[tuple[str | int, torch.Tensor]]] = {}

    for name, tensor in disk_list:
        if name.startswith("model."):
            name = name[6:]
        if "rotary_emb.inv_freq" in name or "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
            continue
        if "scale" in name:
            remapped = _qwen3_remap_scale_name_for_checksum(name)
            if remapped is not None:
                simple.append((remapped, tensor))
            continue
        handled = False
        for param_name, weight_name, shard_id in _QWEN3_STACKED_PARAMS_MAPPING:
            if weight_name not in name:
                continue
            model_name = name.replace(weight_name, param_name)
            stacked.setdefault(model_name, []).append((shard_id, tensor))
            handled = True
            break
        if not handled:
            simple.append((name, tensor))

    for n, t in simple:
        yield n, t

    for model_name in sorted(stacked.keys()):
        shards = stacked[model_name]
        order = [
            s[2]
            for s in _QWEN3_STACKED_PARAMS_MAPPING
            if s[0] in model_name
        ]
        shards_by_id = {s[0]: s[1] for s in shards}
        ordered_tensors = [shards_by_id[sid] for sid in order if sid in shards_by_id]
        if not ordered_tensors:
            continue
        merged = torch.cat(ordered_tensors, dim=0)
        yield model_name, merged


def _disk_to_model_weights_identity(
    disk_weights: Iterable[tuple[str, torch.Tensor]],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Identity: disk keys are model keys (e.g. VAE, transformer)."""
    yield from disk_weights


_DISK_TO_MODEL_REGISTRY: dict[str, Callable[..., Generator[tuple[str, torch.Tensor], None, None]]] = {
    "Qwen3ForCausalLM": _disk_to_model_weights_qwen3,
    "FSDPQwen3ForCausalLM": _disk_to_model_weights_qwen3,
}


def get_disk_to_model_weights(
    arch: str,
    disk_weights: Iterable[tuple[str, torch.Tensor]],
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield (model_param_name, tensor) from disk weights using loader-style mapping.

    Use this to compute a name-based checksum from disk that is comparable to
    the server's name-based checksum after load_weights.
    """
    fn = _DISK_TO_MODEL_REGISTRY.get(arch, _disk_to_model_weights_identity)
    yield from fn(disk_weights)


def get_module_arch_from_disk(model_path: str, module_name: str) -> str | None:
    """Read architectures[0] from model_path/module_name/config.json if present."""
    config_path = Path(model_path) / module_name / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path) as f:
            data = json.load(f)
        archs = data.get("architectures")
        if archs and len(archs) > 0:
            return archs[0]
        return None
    except Exception:
        return None


def compute_weights_checksum_from_disk_mapped(
    model_path: str,
    module_name: str,
    safetensors_files: list[str],
    arch: str | None = None,
) -> str:
    """Compute name-based checksum from disk after applying disk->model name mapping.

    Use when the module uses remapping (e.g. text_encoder Qwen3). If arch is None,
    it is read from model_path/module_name/config.json (architectures[0]).
    """
    if arch is None:
        arch = get_module_arch_from_disk(model_path, module_name) or ""
    it = safetensors_weights_iterator(safetensors_files)
    mapped = get_disk_to_model_weights(arch, it)
    return compute_weights_checksum(mapped)
