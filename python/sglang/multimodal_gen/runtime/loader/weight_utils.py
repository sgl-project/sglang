# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/model_loader/weight_utils.py
"""Utilities for downloading, loading, initializing and verifying model weights."""

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Generator, Iterable
from pathlib import Path

import filelock
import torch
from safetensors.torch import safe_open
from torch.distributed.tensor import DTensor
from tqdm.auto import tqdm

from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.loader.utils import (
    _DEFAULT_SAFETENSORS_INDEX,
    _list_safetensors_files,
)
from sglang.multimodal_gen.runtime.loader.weight_load_plan import WeightLoadPlan
from sglang.multimodal_gen.runtime.loader.weight_readers import (
    FALLBACK_READER,
    RunaiStreamerReader,
    select_weight_reader,
)
from sglang.multimodal_gen.runtime.loader.weight_readers.runai_streamer import (
    HAS_RUNAI_MODEL_STREAMER,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def checkpoint_weights_iterator(
    model_path: str,
    *,
    to_cpu: bool = True,
    key_filter: Callable[[str], bool] | None = None,
    index_file: str = _DEFAULT_SAFETENSORS_INDEX,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Read a materialized component checkpoint, preferring indexed safetensors."""
    files = _list_safetensors_files(
        model_path, index_file=index_file, key_filter=key_filter
    )
    if files:
        yield from safetensors_weights_iterator(
            files, to_cpu=to_cpu, key_filter=key_filter
        )
        return
    if os.path.isfile(model_path):
        files = [model_path] if model_path.endswith((".bin", ".pt")) else []
    else:
        for suffix in ("*.bin", "*.pt"):
            files = filter_files_not_needed_for_inference(
                sorted(str(path) for path in Path(model_path).glob(suffix))
            )
            if files:
                break
    if not files:
        raise ValueError(
            f"No safetensors, bin, or pt checkpoint found at {model_path!r}"
        )
    for name, tensor in pt_weights_iterator(files, to_cpu=to_cpu):
        if key_filter is None or key_filter(name):
            yield name, tensor


def _disable_runai_streamer_rank_discovery_collective() -> None:
    """RunAI Model Streamer's ``find_local_ranks()`` fires a full-world
    collective on the first ``stream_files()`` of every streamer instance even
    when the caller passes ``is_distributed=False`` — it only populates an env
    var for the library's distributed-streaming path, which this loader never
    uses (each rank loads its own full copy). Ranks reach it with divergent
    timing, so it can fire out of lockstep and hang
    (https://github.com/run-ai/runai-model-streamer/issues/84).

    Patch it to the single-process early return it already has; the only
    behavior lost is the collective this loader never wanted.
    """
    try:
        from runai_model_streamer.distributed_streamer.distributed_streamer import (
            _distributedStreamerParams,
        )
    except ImportError:
        return
    if not hasattr(_distributedStreamerParams, "find_local_ranks"):
        logger.warning(
            "runai_model_streamer find_local_ranks not found; skipping the "
            "rank-discovery-collective workaround (multi-rank loads may hang, "
            "see run-ai/runai-model-streamer#84)."
        )
        return

    def _find_local_ranks_no_collective(self):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return 1, rank, [[rank]]

    _distributedStreamerParams.find_local_ranks = _find_local_ranks_no_collective


if HAS_RUNAI_MODEL_STREAMER:
    _disable_runai_streamer_rank_discovery_collective()

# use system-level temp directory for file locks, so that multiple users
# can share the same lock without error.
# lock files in the temp directory will be automatically deleted when the
# system reboots, so users will not complain about annoying lock files
temp_dir = tempfile.gettempdir()


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
    # Linux filesystems commonly cap one filename at 255 bytes. Absolute
    # snapshot paths can exceed that even though the full path is valid.
    # The digest is already collision-resistant, so fall back to it alone
    # while preserving the historical name for ordinary paths.
    if len(os.fsencode(lock_file_name)) > 255:
        lock_file_name = hash_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock


# For models like Mistral-7B-v0.3, there are both sharded
# safetensors files and a consolidated safetensors file.
# Passing both of these to the weight loader functionality breaks.
# So, we use the index_file to
# look up which safetensors files should be used.
def filter_duplicate_safetensors_files(
    hf_weights_files: list[str],
    hf_folder: str,
    index_file: str,
    key_filter: Callable[[str], bool] | None = None,
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
        # remove only shards whose indexed tensors are all filtered
        if key_filter is not None and not key_filter(weight_name):
            continue
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


def _scan_safetensors_files(
    hf_weights_files: list[str],
) -> tuple[list[str], dict[str, set[str]]]:
    """Validate headers and detect cross-file duplicate keys in one pass."""
    corrupted_files: list[str] = []
    key_to_file: dict[str, str] = {}
    duplicate_files_by_key: dict[str, set[str]] = defaultdict(set)

    for st_file in hf_weights_files:
        try:
            with safe_open(st_file, framework="pt", device="cpu") as f:
                for name in f.keys():  # noqa: SIM118
                    previous_file = key_to_file.get(name)
                    if previous_file is None:
                        key_to_file[name] = st_file
                    elif previous_file != st_file:
                        duplicate_files_by_key[name].update((previous_file, st_file))
        except Exception as e:
            logger.error(
                "Corrupted safetensors file detected: %s - %s: %s",
                st_file,
                type(e).__name__,
                str(e),
            )
            corrupted_files.append(st_file)

    return corrupted_files, duplicate_files_by_key


def _raise_if_duplicate_safetensors_keys(
    duplicate_files_by_key: dict[str, set[str]],
) -> None:
    if not duplicate_files_by_key:
        return

    examples = []
    for key in sorted(duplicate_files_by_key)[:8]:
        files = ", ".join(
            sorted(os.path.basename(p) for p in duplicate_files_by_key[key])
        )
        examples.append(f"{key} [{files}]")

    raise ValueError(
        "Duplicate tensor names detected across safetensors files. Refusing to load "
        "because final weights would depend on file or streamer ordering. "
        f"Found {len(duplicate_files_by_key)} duplicate tensor name(s). "
        f"Examples: {examples}. "
        "This usually means multiple precision variants or consolidated+sharded "
        "checkpoints were passed together."
    )


def safetensors_weights_iterator(
    hf_weights_files: list[str],
    to_cpu: bool = True,
    use_runai_model_streamer: bool | None = None,
    key_filter: Callable[[str], bool] | None = None,
    clone_streamed_tensors: bool = True,
    weight_load_plan: WeightLoadPlan | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files."""
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )
    if weight_load_plan is not None:
        checkpoint_device = torch.device(weight_load_plan.checkpoint_load_device)
        to_cpu = checkpoint_device.type == "cpu"
        device = str(checkpoint_device)
    else:
        device = "cpu" if to_cpu else str(get_local_torch_device())
    # The caller may still pass the old boolean; map it onto a backend name so
    # there is one place that decides, and it is the place that knows which
    # backends can skip keys.
    requested = None
    if use_runai_model_streamer is not None:
        requested = (
            RunaiStreamerReader.name
            if use_runai_model_streamer
            else FALLBACK_READER.name
        )
    elif to_cpu:
        # A host-bound load keeps the checkpoint mapping: mapped pages are the
        # zero-copy optimum there, and everything downstream that budgets host
        # memory (layerwise offload, pinning, the mapped-weight gate) assumes
        # them. The streamer materializes anonymous copies instead -- measured
        # as the whole 61.7 GB DiT landing in host anon on the 5090 CI runner
        # -- and its strengths (direct-to-GPU, remote streaming) do not apply
        # to a local file headed for the CPU.
        requested = FALLBACK_READER.name
    backend = select_weight_reader(
        requested=requested, needs_key_filter=key_filter is not None
    )

    # Validate files before loading
    corrupted_files, duplicate_files_by_key = _scan_safetensors_files(hf_weights_files)

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

    _raise_if_duplicate_safetensors_keys(duplicate_files_by_key)

    yield from backend.iter_weights(
        hf_weights_files,
        device=device,
        to_cpu=to_cpu,
        key_filter=key_filter,
        clone_tensors=clone_streamed_tensors,
        show_progress=enable_tqdm,
    )


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
