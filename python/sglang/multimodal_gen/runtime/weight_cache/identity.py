# SPDX-License-Identifier: Apache-2.0
"""Strict published checkpoint/source identity, using shared cache utilities."""

import hashlib
import importlib.metadata
import logging
import os
import re
from pathlib import Path

import msgspec
import torch

import sglang
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.weight_cache.placement import local_device_index
from sglang.multimodal_gen.runtime.weight_cache.plan import (
    CacheCompatibilityPlan,
    PlannedRankContext,
)
from sglang.srt.environ import envs
from sglang.srt.weight_cache.common.checkpoint import (
    MANIFEST_FILENAME,
    CheckpointManifest,
    verify_manifest,
)
from sglang.srt.weight_cache.common.identity import (
    FileStamp,
    socket_path,
    source_digest,
)
from sglang.srt.weight_cache.protocol import compute_env_stamp


def consumed_files(prepared):
    root = Path(prepared.model_path)
    files = {root / "model_index.json"}
    for component in prepared.cached_components:
        files.update(component.consumed_files())
    # Exact consumed paths, including any index which selected the shards.
    if any(not path.absolute().is_relative_to(root.absolute()) for path in files):
        raise ValueError(
            "Weight cache requires component weights/configs inside the published model directory"
        )
    return root, tuple(sorted(path.relative_to(root).as_posix() for path in files))


def checkpoint_identity(prepared, args, *, verify=False):
    root, names = consumed_files(prepared)
    # HF native releases may keep independently admitted pipelines below the
    # immutable snapshot (e.g. FL2VA). Keep the exact subfolder in the identity;
    # every consumed symlink must still resolve within this published repo.
    snapshot = next(
        (
            candidate
            for candidate in (root, *root.parents)
            if candidate.parent.name == "snapshots"
            and re.fullmatch(r"[0-9a-f]{40}", candidate.name)
            and candidate.parent.parent.name.startswith("models--")
        ),
        None,
    )
    stamps = {name: msgspec.to_builtins(FileStamp.read(root / name)) for name in names}
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
    elif snapshot is not None:
        repo_root = snapshot.parent.parent.resolve()
        if any(
            not (root / name).resolve(strict=True).is_relative_to(repo_root)
            for name in names
        ):
            raise ValueError("HF checkpoint symlink escapes its published repository")
        identity = {
            "kind": "hf_snapshot",
            "repo": repo_root.name,
            "revision": snapshot.name,
        }
        if root != snapshot:
            identity["subfolder"] = root.relative_to(snapshot).as_posix()
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


def dependency_identity(args):
    # The dynamic Hugging Face FA3 loader can select artifacts outside the
    # installed distributions. It has no audited publication receipt here.
    if not envs.SGLANG_USE_SGL_FA3_KERNEL.get():
        raise ValueError("Weight cache requires SGLANG_USE_SGL_FA3_KERNEL=1")
    pip_fa4 = envs.SGLANG_INKLING_FA4_USE_PIP.get() == "1"
    providers = {"fa3": "sglang-kernel", "fa4": "pip" if pip_fa4 else "vendored"}
    required = {
        "torch",
        "transformers",
        "diffusers",
        "sglang-kernel",
        "triton",
        "nvidia-cutlass-dsl",
    }
    if pip_fa4:
        required.add("flash-attn-4")
    distributions = {}
    # Python import names are not distribution names: sgl_kernel is provided
    # by sglang-kernel. Include native/JIT providers even when vendored FA4
    # Python is already covered by source_digest.
    names = required | {
        "flash-attn-4",
        "flashinfer-python",
        "flashinfer-cubin",
        "flashinfer-jit-cache",
        "nvidia-cutlass-dsl-libs-base",
        "nvidia-cutlass-dsl-libs-core",
        "nvidia-cutlass-dsl-libs-cu12",
        "nvidia-cutlass-dsl-libs-cu13",
    }
    for name in sorted(names):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError as error:
            if name in required:
                raise ValueError(
                    f"Weight-cache required provider is missing: {name}"
                ) from error
            distributions[name] = None
            continue
        record = dist.read_text("RECORD")
        if not record:
            if not args.weight_cache_allow_unverified_build:
                raise ValueError(
                    f"Weight-cache provider lacks published RECORD: {name}"
                )
            logging.getLogger(__name__).warning(
                "Weight cache uses UNVERIFIED provider without RECORD: %s", name
            )
        distributions[name] = {
            "version": dist.version,
            "record": hashlib.sha256(record.encode()).hexdigest() if record else None,
        }
    return {"providers": providers, "distributions": distributions}


def environment_identity(args):
    # Even development providers without RECORD require a complete, stable
    # Python source identity. A shared placeholder could admit different code.
    source = source_digest(Path(sglang.__file__).parent)
    # RECORD is the installer's content manifest, including native binaries.
    # As with checkpoints, this assumes trusted published installations; manual
    # binary edits without republishing their RECORD are outside that contract.
    cap = current_platform.get_device_capability(local_device_index(args))
    if cap is None:
        raise ValueError("Cannot identify planned CUDA device capability")
    return {
        **compute_env_stamp(device_capability=f"{cap.major}.{cap.minor}"),
        "source": source,
        **dependency_identity(args),
        "torch_git": torch.version.git_version,
        "cuda": torch.version.cuda,
    }


def compatibility_plan(prepared, args, *, verify_checkpoint=False):
    index = local_device_index(args)
    rank = PlannedRankContext(0, index, current_platform.get_device_uuid(index))
    rank_fields = msgspec.to_builtins(rank)
    # A process-local CUDA ordinal is a locator, not an allocation identity.
    # CUDA_VISIBLE_DEVICES may renumber the same physical GPU in the consumer.
    rank_fields.pop("local_device")
    return CacheCompatibilityPlan.from_fields(
        pipeline={
            "class": f"{prepared.pipeline_cls.__module__}.{prepared.pipeline_cls.__qualname__}",
            "binding": prepared.binding_id,
        },
        rank=rank_fields,
        requested=list(prepared.cached_component_names),
        components={
            component.name: {
                **component.fingerprint_fields(),
                "checkpoint_files": sorted(
                    path.relative_to(prepared.model_path).as_posix()
                    for path in component.consumed_files()
                ),
            }
            for component in prepared.cached_components
        },
        checkpoint=checkpoint_identity(prepared, args, verify=verify_checkpoint),
        environment=environment_identity(args),
    )


def locate(plan, args):
    device_uuid = plan.to_dict()["rank"]["device_uuid"]
    if args.weight_cache_socket is not None:
        path = Path(args.weight_cache_socket.replace("{device_uuid}", device_uuid))
        if not path.is_absolute() or len(os.fsencode(path)) > 107:
            raise ValueError(
                "weight_cache_socket must be an absolute Unix socket path of at most 107 bytes"
            )
        # Resolve directory aliases, but never follow a socket-node symlink.
        path = path.parent.resolve() / path.name
        if len(os.fsencode(path)) > 107:
            raise ValueError("Resolved weight-cache socket exceeds 107 bytes")
        return path
    return socket_path(device_uuid, plan.digest)
