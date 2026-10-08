# SPDX-License-Identifier: Apache-2.0
"""Portable paths, content identities and atomic result storage."""
from __future__ import annotations
import hashlib
import json
import os
import re
import shutil
from pathlib import Path


# NOTE (yiakwy) : same to https://github.com/multimodal-art-projection/YuE/blob/main/src/yue2/storage.py


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


# Explicit public model files. Extra reports, caches and arbitrary code are not exported.
MODEL_FILES = frozenset({
    "config.json", "generation_config.json", "yue2_generation_config.json",
    "weights_manifest.json", "model.safetensors", "model.safetensors.index.json",
    "qwen.tiktoken", "modeling_yue2.py", "modeling_vae.py",
    "LICENSE", "THIRD_PARTY_NOTICES.md",
})
MODEL_LICENSES = frozenset({"stable-audio-tools-MIT.txt", "SnakeBeta-NVIDIA-MIT.txt"})
SHARD_NAME = re.compile(r"model-[0-9]{5}-of-[0-9]{5}\.safetensors")


def resolve_model(model, revision=None, local_files_only=False, token=None, cache_dir=None, **kwargs):
    path = Path(model).expanduser()
    if path.is_dir():
        return path.resolve()
    if path.is_absolute() or str(model).startswith("."):
        raise FileNotFoundError(path)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(model), revision=revision, local_files_only=local_files_only,
                                  token=token, cache_dir=cache_dir,
                                  allow_patterns=sorted(MODEL_FILES) +
                                      ["model-?????-of-?????.safetensors"] +
                                      ["licenses/" + name for name in sorted(MODEL_LICENSES)]))


def copy_model_files(source, destination):
    """Export a model to an empty directory with a documented file boundary."""
    source, destination = Path(source).resolve(), Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("save_pretrained needs an empty model destination")
    config = json.loads((source / "config.json").read_text())
    kind = config.get("model_type")

    if kind == "yue2":
        from .modeling_yue2 import YuE2Config
        clean_config = YuE2Config(**config).to_dict()
        code_name = "modeling_yue2.py"
    elif kind == "yue2_vae":
        from .modeling_vae import YuE2VAEConfig
        clean_config = YuE2VAEConfig(**config).to_dict()
        code_name = "modeling_vae.py"
    else:
        raise ValueError("Unsupported model configuration")
    
    weights = model_identity(source)
    names = set(MODEL_FILES) - {"config.json", "modeling_yue2.py", "modeling_vae.py", "weights_manifest.json"}
    names.update(name for name in weights["files"] if SHARD_NAME.fullmatch(name))
    if set(weights["files"]) - names:
        raise ValueError("Unexpected model weight filename")
    destination.mkdir(parents=True, exist_ok=True)
    for name in sorted(names):
        file = source / name
        if file.is_file():
            if name == "yue2_generation_config.json":
                from .protocol import GenerationConfig
                write_json(destination / name, GenerationConfig.from_dict(json.loads(file.read_text())).to_dict())
            elif name == "generation_config.json":
                from transformers import GenerationConfig as HFGenerationConfig
                known = HFGenerationConfig().to_dict()
                value = {key: value for key, value in json.loads(file.read_text()).items()
                         if key in known and not key.startswith("_")}
                write_json(destination / name, value)
            else:
                shutil.copyfile(file, destination / name)

    # Use the installed, reviewed implementation instead of arbitrary source-folder code.
    shutil.copyfile(Path(__file__).with_name(code_name), destination / code_name)
    for name in sorted(MODEL_LICENSES):
        file = source / "licenses" / name
        if file.is_file():
            (destination / "licenses").mkdir(exist_ok=True)
            shutil.copyfile(file, destination / "licenses" / name)
    write_json(destination / "config.json", clean_config)
    write_json(destination / "weights_manifest.json", {"files": weights["files"]})


def model_identity(path, verify=True):
    path = Path(path)
    manifest = path / "weights_manifest.json"
    expected = json.loads(manifest.read_text()) if manifest.exists() else None
    files = sorted(path.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors weights in {path}")
    entries = {}
    for file in files:
        digest = sha256_file(file)
        if verify and expected is not None:
            wanted = expected.get("files", {}).get(file.name, {}).get("sha256")
            if wanted != digest:
                raise ValueError(f"Weight integrity failed: {file.name}")
        entries[file.name] = {"sha256": digest, "bytes": file.stat().st_size}
    if expected is not None and set(entries) != set(expected.get("files", {})):
        raise ValueError("Weight manifest has missing or unexpected shards")
    return {"files": entries, "config_sha256": sha256_file(path / "config.json")}


def collect_hashes(directory, exclude=("result.json",)):
    directory = Path(directory)
    return {str(p.relative_to(directory)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in exclude}


def export_song_artifacts(directory, *, request, latents, semantic_ids, prefix_ids,
                          abc_ids, timing, truncated):
    """Write one request's WSB-style artifact bundle (latent/token arrays + plan).

    ``latents`` is the NAR output (``[1, 64, T]`` or ``[64, T]``); other token
    lists are raw ids. The manifest hashes the frozen plan inputs so the scorer
    can verify the bundle later with :func:`verify_result`.
    """
    import numpy as np

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    latent = latents.squeeze(0)
    if latent.shape[0] == 64 and latent.shape[-1] != 64:
        latent = latent.T  # [64, T] -> [T, 64]
    np.save(directory / "latent.npy", latent.detach().float().cpu().numpy())
    np.save(directory / "semantic.npy", np.asarray(semantic_ids, dtype=np.int32))
    np.save(directory / "prefix.npy", np.asarray(prefix_ids, dtype=np.int32))
    np.save(directory / "abc_tokens.npy", np.asarray(abc_ids, dtype=np.int32))
    request_dict = request.to_dict()
    write_json(directory / "plan.json", {
        "request": request_dict, "timing": timing, "truncated": truncated,
        "prefix": prefix_ids, "abc_ids": abc_ids, "abc": request.abc,
    })
    write_json(directory / "plan_manifest.json", {
        name: sha256_file(directory / name)
        for name in ("plan.json", "abc_tokens.npy", "prefix.npy")
    })
    write_json(directory / "request.json", request_dict)


def verify_result(directory, expected_identity=None):
    directory = Path(directory)
    result = json.loads((directory / "result.json").read_text())
    if result.get("status") != "complete":
        raise ValueError("Saved request did not complete")
    if expected_identity is not None and result.get("identity") != expected_identity:
        raise ValueError("Request/config/weight identity changed; use a new output directory")
    required = {"audio.flac", "prefix.npy", "semantic.npy", "latent.npy", "request.json", "config.json"}
    artifacts = result.get("artifacts", {})
    if not required <= set(artifacts):
        raise ValueError("Incomplete result artifact manifest")
    for name, expected in artifacts.items():
        p = directory / name
        if Path(name).is_absolute() or not p.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Invalid artifact path")
        if not p.is_file() or p.stat().st_size != expected["bytes"] or sha256_file(p) != expected["sha256"]:
            raise ValueError(f"Missing or corrupt result: {name}")
    return result
