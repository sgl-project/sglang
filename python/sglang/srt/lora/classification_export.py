"""Export serving metadata for a PEFT sequence-classification adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from safetensors import safe_open

from sglang.srt.lora.classification_head import load_classification_head


def write_classification_manifest(
    adapter_dir,
    *,
    id2label,
    hidden_size: int,
    max_length: int,
    add_special_tokens: bool,
) -> Path:
    """Bind a saved PEFT linear head, labels and preprocessing to its weights.

    Call this after ``PeftModel.save_pretrained(..., safe_serialization=True)``.
    The adapter tensors and config remain in their original PEFT format.
    Existing manifests are rejected; export a new directory for a new version.
    """
    path = Path(adapter_dir)
    destination = path / "classification_config.json"
    if destination.exists():
        raise FileExistsError(f"Classification manifest already exists: {destination}")
    if not isinstance(id2label, dict):
        raise ValueError("id2label must be an index-to-label mapping")
    labels = {}
    for index, label in id2label.items():
        if type(index) not in (str, int) or str(index) in labels:
            raise ValueError("id2label contains invalid or duplicate class indices")
        labels[str(index)] = label
    unsupported = (
        "classification_head.pt",
        "label_mapping.json",
        "added_tokens.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
    )
    if any((path / name).exists() for name in unsupported):
        raise ValueError(
            "Export a PEFT adapter without legacy heads or tokenizer files"
        )
    # Validate a complete staged bundle before publishing the manifest. Runtime
    # loading verifies the same hashes again and takes its own immutable copy.
    with tempfile.TemporaryDirectory(prefix=".classifier-export-", dir=path) as tmp:
        stage = Path(tmp)
        artifacts = {}
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            target = stage / name
            shutil.copyfile(path / name, target)
            digest = hashlib.sha256()
            with target.open("rb") as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            artifacts[name] = digest.hexdigest()
        with safe_open(stage / "adapter_model.safetensors", framework="pt") as tensors:
            keys = set(tensors.keys())
            heads = [
                name
                for name in keys
                if name.endswith((".score.weight", ".classifier.weight"))
            ]
            if len(heads) != 1:
                raise ValueError(
                    "Expected exactly one saved PEFT score/classifier head"
                )
            weight_key = heads[0]
            shape = tensors.get_slice(weight_key).get_shape()
            if len(shape) != 2:
                raise ValueError("Classification head must be a linear weight matrix")
        head = {"weight_key": weight_key}
        bias_key = weight_key.removesuffix("weight") + "bias"
        if bias_key in keys:
            head["bias_key"] = bias_key
        manifest = {
            "schema_version": 1,
            "problem_type": "single_label_classification",
            "num_labels": shape[0],
            "hidden_size": hidden_size,
            "id2label": labels,
            "pooling": "last",
            "max_length": max_length,
            "add_special_tokens": add_special_tokens,
            "head": head,
            "artifacts": artifacts,
        }
        staged_manifest = stage / destination.name
        staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        load_classification_head(str(stage), hidden_size)
        # Atomic no-clobber publication, even if another exporter wins the race.
        os.link(staged_manifest, destination)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("adapter_dir", type=Path)
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Training model's HF config JSON, including hidden_size and id2label",
    )
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument(
        "--add-special-tokens", action=argparse.BooleanOptionalAction, required=True
    )
    args = parser.parse_args()
    config = json.loads(args.model_config.read_text())
    path = write_classification_manifest(
        args.adapter_dir,
        id2label=config["id2label"],
        hidden_size=config.get("text_config", config)["hidden_size"],
        max_length=args.max_length,
        add_special_tokens=args.add_special_tokens,
    )
    print(path)


if __name__ == "__main__":
    main()
