# SPDX-License-Identifier: Apache-2.0
"""Tensor-level parity between a disk-loaded model and a daemon-loaded one.

The weight cache's shipped tests assert that a client-mode server answers and
that its greedy text matches a disk-loading server's. Neither looks at a tensor,
so a transport that rebuilds a view at the wrong storage offset, loses a tied
name, or silently skips a tensor can still produce a green run -- that is how
the ``as_strided`` offset bug survived one. This module fingerprints every
parameter and buffer of a loaded model and diffs two fingerprints, and runs as
``python -m sglang.test.weight_cache_parity_utils`` to produce one.

What a fingerprint records, and why each field is load-bearing:

- ``shape``, ``stride``, ``storage_offset``: the view metadata the vmm_fd wire
  layout carries per tensor. ``storage_offset`` is what distinguishes a
  correctly rebuilt sub-allocated tensor from one rebased to the start of its
  slab, and ``stride`` is what distinguishes a padded XPU MoE expert weight
  from a contiguous one.
- ``dtype``: a dtype view over the same bytes is a different tensor.
- ``storage_nbytes``: the size of the storage behind the view, which catches a
  slot sized to a padded weight's logical extent rather than its allocation.
- ``storage_group``: an index per distinct storage, in first-seen order. Two
  tensors that shared a storage in the daemon must share one in the engine --
  that is what keeps tied weights, transposes and dtype views aliasing.
- ``sha256``: a digest over the tensor's LOGICAL elements, gathered through the
  view.

The digest is deliberately not taken over the storage. An XPU MoE expert weight
is a narrowed view of an over-allocated buffer (see
``xpu_moe_ld_padding_elems``), and neither side ever writes the pad bytes, so
hashing the storage would compare uninitialized memory and report a mismatch
where the weights agree. The pad is still covered, as ``stride`` and
``storage_nbytes`` metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _logical_digest(tensor: torch.Tensor) -> str:
    """sha256 over the elements ``tensor`` addresses, in row-major order.

    The gather happens in the device-to-host copy, so a strided view costs one
    host buffer of its logical size and no device allocation -- ``reshape(-1)``
    on the device would materialize a second copy of a padded expert weight on
    a card that already holds the model.
    """
    if tensor.numel() == 0:
        return hashlib.sha256().hexdigest()
    host = tensor.detach().to("cpu").contiguous().reshape(-1)
    return hashlib.sha256(memoryview(host.view(torch.uint8).numpy())).hexdigest()


def tensor_fingerprint(
    tensor: torch.Tensor, *, storage_group: int, is_param: bool
) -> Dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "storage_offset": tensor.storage_offset(),
        "storage_nbytes": tensor.untyped_storage().nbytes(),
        "storage_group": storage_group,
        "device_type": tensor.device.type,
        "is_param": is_param,
        "sha256": _logical_digest(tensor),
    }


def collect_weight_manifest(model: torch.nn.Module) -> Dict[str, Any]:
    """Fingerprint every parameter and buffer of ``model``.

    ``remove_duplicate=False`` on both walks, matching the daemon's export, so a
    tied weight appears under every name it is reachable by and a name the
    transport dropped shows up as missing rather than as a deduplicated alias.
    Returns ``{"tensors": {name: fingerprint}, "duplicate_names": [...]}``;
    duplicate names cannot survive the dict, so they are reported separately
    instead of silently collapsing.
    """
    tensors: Dict[str, Dict[str, Any]] = {}
    duplicate_names: List[str] = []
    storage_groups: Dict[int, int] = {}

    def record(name: str, tensor: torch.Tensor, *, is_param: bool) -> None:
        if name in tensors:
            duplicate_names.append(name)
            return
        key = tensor.untyped_storage().data_ptr()
        group = storage_groups.setdefault(key, len(storage_groups))
        tensors[name] = tensor_fingerprint(
            tensor, storage_group=group, is_param=is_param
        )

    for name, param in model.named_parameters(remove_duplicate=False):
        record(name, param.data, is_param=True)
    for name, buf in model.named_buffers(remove_duplicate=False):
        record(name, buf, is_param=False)

    return {"tensors": tensors, "duplicate_names": sorted(set(duplicate_names))}


# Fields compared verbatim. Device type is excluded: both sides land on the same
# device by construction, and comparing it would mask nothing.
_COMPARED_FIELDS = (
    "shape",
    "stride",
    "dtype",
    "storage_offset",
    "storage_nbytes",
    "storage_group",
    "is_param",
    "sha256",
)


def compare_manifests(
    *,
    reference: Mapping[str, Any],
    actual: Mapping[str, Any],
    reference_label: str = "disk",
    actual_label: str = "daemon",
    max_reported: int = 20,
) -> List[str]:
    """Every way ``actual`` differs from ``reference``, as readable lines.

    An empty list is parity. The caller decides what to do with a non-empty one;
    nothing here is tolerated as close enough, because every compared field is
    either equal or evidence of a transport bug.
    """
    problems: List[str] = []
    ref_tensors: Mapping[str, Any] = reference["tensors"]
    act_tensors: Mapping[str, Any] = actual["tensors"]

    for label, manifest in ((reference_label, reference), (actual_label, actual)):
        duplicates = manifest["duplicate_names"]
        if duplicates:
            problems.append(
                f"{label}: {len(duplicates)} tensor name(s) appeared twice while "
                f"walking the model: {duplicates[:max_reported]}"
            )

    missing = sorted(set(ref_tensors) - set(act_tensors))
    extra = sorted(set(act_tensors) - set(ref_tensors))
    if missing:
        problems.append(
            f"{len(missing)} tensor(s) in {reference_label} are absent from "
            f"{actual_label}: {missing[:max_reported]}"
        )
    if extra:
        problems.append(
            f"{len(extra)} tensor(s) in {actual_label} are absent from "
            f"{reference_label}: {extra[:max_reported]}"
        )

    field_mismatches: Dict[str, List[str]] = {}
    for name in sorted(set(ref_tensors) & set(act_tensors)):
        ref, act = ref_tensors[name], act_tensors[name]
        for field in _COMPARED_FIELDS:
            if ref[field] != act[field]:
                field_mismatches.setdefault(field, []).append(
                    f"{name}: {reference_label}={ref[field]!r} "
                    f"{actual_label}={act[field]!r}"
                )
    for field, lines in sorted(field_mismatches.items()):
        problems.append(
            f"{len(lines)} tensor(s) differ in {field}:\n    "
            + "\n    ".join(lines[:max_reported])
        )

    return problems


def manifest_summary(manifest: Mapping[str, Any]) -> str:
    tensors = manifest["tensors"]
    params = sum(1 for entry in tensors.values() if entry["is_param"])
    storages = len({entry["storage_group"] for entry in tensors.values()})
    logical_bytes = sum(
        _dtype_itemsize(entry["dtype"]) * _numel(entry["shape"])
        for entry in tensors.values()
    )
    return (
        f"{len(tensors)} tensors ({params} parameters, {len(tensors) - params} "
        f"buffers) over {storages} storages, {logical_bytes / (1 << 30):.3f} GiB "
        f"logical"
    )


def _numel(shape: List[int]) -> int:
    total = 1
    for size in shape:
        total *= size
    return total


def _dtype_itemsize(name: str) -> int:
    dtype = getattr(torch, name.removeprefix("torch."), None)
    return dtype.itemsize if isinstance(dtype, torch.dtype) else 0


# ---------------------------------------------------------------------------
# Manifest worker
# ---------------------------------------------------------------------------
#
# Both modes go through the same public entry point -- get_model_loader() then
# load_model() -- with the same ServerArgs, the same ModelConfig and the same
# distributed setup, so a difference in the manifests can only come from the
# loading source. That setup is the daemon's own _init_distributed, called on a
# daemon instance used purely as a setup helper: a second copy of the sequence
# here could drift from the one the daemon shards its weights with, and then a
# parity failure would say nothing about the transport.


def load_model_for_manifest(
    *,
    model_path: str,
    mode: str,
    tp_size: int,
    tp_rank: int,
    base_gpu_id: int,
    device: str,
    dist_init_method: str,
    socket_path: Optional[str] = None,
    weight_cache_transport: str = "auto",
    load_format: str = "auto",
    dtype: str = "auto",
    quantization: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Tuple[torch.nn.Module, Any]:
    """Load one TP rank's weights from disk or from a daemon.

    Both modes build their ``LoadConfig`` the same way and go through
    ``get_model_loader``; ``mode="disk"`` names ``load_format`` so the loader is
    the one a ``--weight-cache-mode off`` server would use, and
    ``mode="daemon"`` names ``ipc_cache`` with the daemon's socket.

    ``load_format="dummy"`` substitutes ``initialize_dummy_weights``, which
    reseeds a generator per parameter from a fixed seed, so two processes
    produce byte-identical weights. That makes a real architecture's layout
    comparable without reading its checkpoint -- it says nothing about whether
    checkpoint values load correctly, which a real-weight case must cover.
    """
    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig, LoadFormat
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.model_loader.loader import get_model_loader
    from sglang.srt.runtime_context import SpawnRanks, publish, spawn_world_rank
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.weight_cache.daemon import WeightCacheDaemon
    from sglang.srt.weight_cache.protocol import compute_local_gpu_id

    if mode not in ("disk", "daemon"):
        raise ValueError(f"mode must be 'disk' or 'daemon', got {mode!r}")

    gpu_id = compute_local_gpu_id(
        pp_rank=0,
        tp_rank=tp_rank,
        pp_size_per_node=1,
        tp_size_per_node=tp_size,
        base_gpu_id=base_gpu_id,
    )
    server_args = ServerArgs(
        model_path=model_path,
        device=device,
        tp_size=tp_size,
        pp_size=1,
        dp_size=1,
        base_gpu_id=base_gpu_id,
        dtype=dtype,
        quantization=quantization,
        trust_remote_code=trust_remote_code,
        load_format=load_format,
        weight_cache_transport=weight_cache_transport,
    )
    setup = WeightCacheDaemon(
        server_args=server_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        pp_rank=0,
        dist_init_method=dist_init_method,
    )
    publish(
        server_args,
        role="weight_cache_daemon",
        ranks=SpawnRanks(
            world_rank=spawn_world_rank(server_args, tp_rank=tp_rank, pp_rank=0),
            gpu_id=gpu_id,
        ),
    )
    initialize_moe_config()

    from sglang.srt.platforms import current_platform

    current_platform.set_device(current_platform.get_device(gpu_id))
    torch.set_num_threads(1)

    model_config = ModelConfig(
        model_path=model_path,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        quantization=quantization,
    )
    setup._init_distributed(server_args, model_config)

    if mode == "daemon":
        load_config = LoadConfig(
            load_format=LoadFormat.IPC_CACHE,
            tp_rank=tp_rank,
            weight_cache_mode="daemon",
            weight_cache_socket=socket_path,
            weight_cache_transport=weight_cache_transport,
            fallback_load_format=LoadFormat(load_format),
        )
    else:
        # No weight-cache fields at all, so the reference cannot reach a daemon
        # even if one is running. "daemon" mode above, for the same reason,
        # refuses to disk-load instead of falling back.
        load_config = LoadConfig(load_format=LoadFormat(load_format), tp_rank=tp_rank)
    loader = get_model_loader(load_config=load_config, model_config=model_config)
    model = loader.load_model(
        model_config=model_config,
        device_config=DeviceConfig(current_platform.device_type, gpu_id),
    )
    return model, loader


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Dump a weight manifest for one TP rank, from disk or a daemon."
    )
    parser.add_argument("--mode", choices=("disk", "daemon"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True, help="path to write the JSON manifest")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--tp-rank", type=int, default=0)
    parser.add_argument("--base-gpu-id", type=int, default=0)
    parser.add_argument("--device", default="xpu")
    parser.add_argument("--dist-init-method", required=True)
    parser.add_argument("--socket-path", default=None)
    parser.add_argument("--weight-cache-transport", default="auto")
    parser.add_argument("--load-format", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    model, loader = load_model_for_manifest(
        model_path=args.model_path,
        mode=args.mode,
        tp_size=args.tp_size,
        tp_rank=args.tp_rank,
        base_gpu_id=args.base_gpu_id,
        device=args.device,
        dist_init_method=args.dist_init_method,
        socket_path=args.socket_path,
        weight_cache_transport=args.weight_cache_transport,
        load_format=args.load_format,
        dtype=args.dtype,
        quantization=args.quantization,
        trust_remote_code=args.trust_remote_code,
    )
    manifest = collect_weight_manifest(model)
    # Which path actually ran. The reference must not have come off a daemon and
    # the daemon side must not have fallen back to disk; either would compare one
    # path against itself, which always passes. The loader attaches this
    # attribute only after a zero-copy map, so its presence is the witness.
    manifest["zero_copy_mapped"] = hasattr(model, "_weight_cache_transport_backend")
    manifest["loader_preloaded_weights_bytes"] = loader.preloaded_weights_bytes
    manifest["mode"] = args.mode
    manifest["tp_rank"] = args.tp_rank
    manifest["load_format"] = args.load_format
    logger.info(
        "[parity] %s rank %d: %s (zero_copy_mapped=%s)",
        args.mode,
        args.tp_rank,
        manifest_summary(manifest),
        manifest["zero_copy_mapped"],
    )

    tmp = f"{args.out}.partial"
    with open(tmp, "w") as handle:
        json.dump(manifest, handle)
    os.replace(tmp, args.out)
    logger.info("[parity] wrote %s", args.out)


if __name__ == "__main__":
    _main()
