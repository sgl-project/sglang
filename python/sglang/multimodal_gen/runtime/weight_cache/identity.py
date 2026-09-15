# SPDX-License-Identifier: Apache-2.0
"""Strict published checkpoint/source identity, using shared cache utilities."""

import dataclasses
import hashlib
import importlib.metadata
import os
import re
from pathlib import Path

import torch

import sglang
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.weight_cache.adapters.dit_wan import (
    fingerprint_fields,
)
from sglang.multimodal_gen.runtime.weight_cache.placement import local_device_index
from sglang.multimodal_gen.runtime.weight_cache.plan import (
    CacheCompatibilityPlan,
    PlannedRankContext,
)
from sglang.srt.weight_cache.protocol import compute_env_stamp
from sglang.weight_cache_common.checkpoint import (
    MANIFEST_FILENAME,
    CheckpointManifest,
    verify_manifest,
)
from sglang.weight_cache_common.identity import FileStamp, socket_path, source_digest


def consumed_files(prepared):
    root = Path(prepared.model_path)
    recipe = prepared.transformer.thaw()
    files = [
        root / "model_index.json",
        Path(recipe.server_args.model_paths["transformer"]) / "config.json",
    ]
    files.extend(Path(path) for path in recipe.weight_files)
    index = root / "transformer/diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        files.append(index)
    # Exact consumed paths, including any index which selected the shards.
    if any(not path.absolute().is_relative_to(root.absolute()) for path in files):
        raise ValueError(
            "Weight cache initial adapter requires component weights/configs inside the model directory"
        )
    return root, tuple(sorted(path.relative_to(root).as_posix() for path in files))


def checkpoint_identity(prepared, args, *, verify=False):
    root, names = consumed_files(prepared)
    stamps = {name: dataclasses.asdict(FileStamp.read(root / name)) for name in names}
    manifest_path = root / MANIFEST_FILENAME
    if manifest_path.exists():
        manifest = CheckpointManifest.read(manifest_path)
        if tuple(entry.path for entry in manifest.files) != names:
            raise ValueError(
                "Checkpoint content manifest must cover exactly the prepared config/index/weight files"
            )
        if verify:
            verify_manifest(root, manifest, list(names))
        identity = {"kind": "content_manifest", "digest": manifest.digest}
    elif (
        root.parent.name == "snapshots"
        and re.fullmatch(r"[0-9a-f]{40}", root.name)
        and root.parent.parent.name.startswith("models--")
    ):
        repo_root = root.parent.parent.resolve()
        if any(
            not (root / name).resolve(strict=True).is_relative_to(repo_root)
            for name in names
        ):
            raise ValueError("HF checkpoint symlink escapes its published repository")
        identity = {
            "kind": "hf_snapshot",
            "repo": repo_root.name,
            "revision": root.name,
        }
    elif args.weight_cache_allow_weak_checkpoint_identity:
        import logging

        logging.getLogger(__name__).warning(
            "Weight cache uses WEAK stat-only checkpoint identity; republishing changed files is required"
        )
        identity = {"kind": "weak_local", "path": str(root.resolve())}
    else:
        raise ValueError(
            f"Local weight cache requires {manifest_path}; publish a content manifest with the weight-cache digest command"
        )
    return {**identity, "files": stamps}


def environment_identity(args):
    try:
        source = source_digest(Path(sglang.__file__).parent)
    except (OSError, ValueError):
        if not args.weight_cache_allow_unverified_build:
            raise
        source = "UNVERIFIED-DEVELOPMENT-BUILD"
    distributions = {}
    # RECORD is the installer's content manifest, including native binaries.
    # As with checkpoints, this assumes trusted published installations; manual
    # binary edits without republishing their RECORD are outside that contract.
    for name in (
        "torch",
        "transformers",
        "diffusers",
        "sgl-kernel",
        "flash-attn",
        "flash-attn-3",
        "flashinfer-python",
    ):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            distributions[name] = None
            continue
        record = dist.read_text("RECORD")
        distributions[name] = {
            "version": dist.version,
            "record": hashlib.sha256(record.encode()).hexdigest() if record else None,
        }
    cap = current_platform.get_device_capability(local_device_index(args))
    if cap is None:
        raise ValueError("Cannot identify planned CUDA device capability")
    return {
        **compute_env_stamp(device_capability=f"{cap.major}.{cap.minor}"),
        "source": source,
        "distributions": distributions,
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
    }


def compatibility_plan(prepared, args, *, verify_checkpoint=False):
    index = local_device_index(args)
    rank = PlannedRankContext(0, index, current_platform.get_device_uuid(index))
    rank_fields = dataclasses.asdict(rank)
    # A process-local CUDA ordinal is a locator, not an allocation identity.
    # CUDA_VISIBLE_DEVICES may renumber the same physical GPU in the consumer.
    rank_fields.pop("local_device")
    return CacheCompatibilityPlan.from_fields(
        pipeline=prepared.pipeline_cls.__name__,
        rank=rank_fields,
        requested=["transformer"],
        component=fingerprint_fields(prepared.transformer),
        checkpoint=checkpoint_identity(prepared, args, verify=verify_checkpoint),
        environment=environment_identity(args),
    )


def locate(plan, args):
    device_uuid = plan.to_dict()["rank"]["device_uuid"]
    if args.weight_cache_socket is not None:
        path = Path(args.weight_cache_socket.format(device_uuid=device_uuid))
        if not path.is_absolute() or len(os.fsencode(path)) > 107:
            raise ValueError(
                "weight_cache_socket must be an absolute Unix socket path of at most 107 bytes"
            )
        return path
    return socket_path(device_uuid, plan.digest)
