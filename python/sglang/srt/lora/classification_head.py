"""Version-bound CPU heads for multiclass LoRA adapters on generation engines.

A classifier is an immutable local bundle. Its manifest binds the head and
label mapping to the exact adapter weights; incomplete bundles fail before a
GPU load. Ordinary generation adapters need no manifest.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors import SafetensorError, safe_open


async def finish_on_cancel(awaitable):
    """Finish a transaction/CPU job despite repeated caller cancellations."""
    task = asyncio.ensure_future(awaitable)
    cancelled = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as error:
            cancelled = error
        except Exception:
            break
    if cancelled is not None:
        if not task.cancelled():
            task.exception()  # Retrieve a failure before propagating cancellation.
        raise cancelled
    return task.result()


@dataclass(frozen=True)
class ClassificationHead:
    weight: torch.Tensor
    bias: torch.Tensor | None
    labels: tuple[str, ...]
    max_length: int
    add_special_tokens: bool

    def prepare_input(self, obj, tokenizer, pad_token_id=None):
        # Raw training text, no conversation template. Slice explicitly on the
        # right without mutating the tokenizer shared with chat requests.
        if obj.text is not None:
            if tokenizer is None:
                raise ValueError("Classification text requires a tokenizer")
            texts = [obj.text] if obj.is_single else obj.text
            ids = tokenizer(
                texts,
                add_special_tokens=self.add_special_tokens,
                truncation=False,
                padding=False,
            )["input_ids"]
        else:
            ids = [obj.input_ids] if obj.is_single else obj.input_ids
        ids = [row[: self.max_length] for row in ids]
        if any(not row for row in ids):
            raise ValueError("Classification input must contain at least one token")
        # LAST captures the final submitted token. HF sequence classifiers pool
        # the final non-pad token instead, so explicitly padded input would
        # silently select a different representation (including literal PAD text).
        if pad_token_id is not None and any(pad_token_id in row for row in ids):
            raise ValueError("Classification input must not contain padding tokens")
        obj.text = None
        obj.input_ids = ids[0] if obj.is_single else ids
        # Request states may already reference normalized batch sub-objects.
        for i, sub_obj in obj.__dict__.get("_sub_obj_cache", {}).items():
            sub_obj.text = None
            sub_obj.input_ids = ids[i]

    def classify(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        data = []
        with torch.inference_mode():
            for i, result in enumerate(results):
                meta = result.get("meta_info", {})
                if meta.get("finish_reason", {}).get("type") == "abort":
                    raise ValueError("Classification generation was aborted")
                hidden = meta.get("hidden_states")
                if hidden is None:
                    raise ValueError("Classification requires last hidden states")
                hidden = torch.as_tensor(hidden, dtype=self.weight.dtype)
                if hidden.ndim == 2 and hidden.shape[0] == 1:
                    hidden = hidden[0]
                if hidden.shape != (self.weight.shape[1],):
                    raise ValueError("Classification hidden-state shape mismatch")
                if not torch.isfinite(hidden).all():
                    raise ValueError("Classification hidden states are not finite")
                logits = F.linear(hidden, self.weight, self.bias).float()
                if not torch.isfinite(logits).all():
                    raise ValueError("Classification logits are not finite")
                probs = logits.softmax(dim=-1)
                data.append(
                    {
                        "index": i,
                        "label": self.labels[probs.argmax().item()],
                        "probs": probs.tolist(),
                        "num_classes": len(self.labels),
                    }
                )
        return data


def _read_json(path: Path):
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path.name}: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(), object_pairs_hook=unique_keys)


def _labels(mapping, size: int) -> tuple[str, ...]:
    if not isinstance(mapping, dict) or set(mapping) != {str(i) for i in range(size)}:
        raise ValueError("id2label must contain every class index as a string")
    labels = [mapping[str(i)] for i in range(size)]
    if (
        len(labels) != size
        or any(not isinstance(v, str) or not v.strip() for v in labels)
        or len(set(labels)) != size
    ):
        raise ValueError("id2label must contain unique nonempty class names")
    return tuple(labels)


_ARTIFACTS = {"adapter_config.json", "adapter_model.safetensors"}
_MANIFEST = "classification_config.json"
_LEGACY_MARKERS = ("classification_head.pt", "label_mapping.json")


def _has_manifest(path: Path) -> bool:
    if any((path / name).exists() for name in _LEGACY_MARKERS):
        raise ValueError("Legacy classification artifacts require offline conversion")
    if (path / _MANIFEST).exists():
        return True
    adapter_config = path / "adapter_config.json"
    if adapter_config.exists():
        config = _read_json(adapter_config)
        if not isinstance(config, dict):
            raise ValueError("LoRA adapter config must be a JSON object")
        if config.get("task_type") == "SEQ_CLS":
            raise ValueError("SEQ_CLS adapter requires classification_config.json")
    # Remote HF identifiers and ordinary generation adapters keep their loader.
    return False


def _artifacts(config: dict) -> dict:
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != _ARTIFACTS:
        raise ValueError("Classification manifest must bind both adapter artifacts")
    for name, digest in artifacts.items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdefABCDEF" for c in digest)
        ):
            raise ValueError(f"Invalid SHA256 for {name}")
    return artifacts


def reject_static_classification_adapter(lora_path: str):
    if _has_manifest(Path(lora_path)):
        raise ValueError(
            "Classification adapters must use /load_lora_adapter "
            "for immutable snapshot loading"
        )


def _head_keys(config: dict, adapter: dict) -> tuple[str, str | None]:
    if adapter.get("peft_type") != "LORA" or adapter.get("task_type") != "SEQ_CLS":
        raise ValueError("Classification requires a PEFT LORA SEQ_CLS adapter")
    # These PEFT variants change the decoder computation. Until their serving
    # parity is covered, accepting them could return plausible, incorrect labels.
    unsupported = (
        "use_rslora",
        "use_dora",
        "use_qalora",
        "fan_in_fan_out",
        "lora_bias",
        "rank_pattern",
        "alpha_pattern",
        "layer_replication",
        "trainable_token_indices",
        "alora_invocation_tokens",
        "target_parameters",
    )
    if (
        any(adapter.get(name) for name in unsupported)
        or adapter.get("bias", "none") != "none"
    ):
        raise ValueError(
            "Classification currently supports standard, uniform-rank LoRA"
        )
    rank, alpha = adapter.get("r"), adapter.get("lora_alpha")
    if type(rank) is not int or rank < 1:
        raise ValueError("Classification adapter LoRA rank must be positive")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha < 0:
        raise ValueError(
            "Classification adapter lora_alpha must be finite and nonnegative"
        )
    targets = adapter.get("target_modules")
    if not (
        isinstance(targets, str)
        and targets
        or isinstance(targets, list)
        and targets
        and all(isinstance(target, str) and target for target in targets)
    ):
        raise ValueError("Classification adapter must declare LoRA target_modules")
    modules = adapter.get("modules_to_save")
    if not isinstance(modules, list) or not all(
        isinstance(module, str) and module for module in modules
    ):
        raise ValueError("Classification adapter must declare modules_to_save")
    if any(module.split(".")[-1] not in {"score", "classifier"} for module in modules):
        raise ValueError("Only score/classifier modules_to_save are supported")
    head = config.get("head")
    if not isinstance(head, dict) or not {"weight_key"} <= head.keys() <= {
        "weight_key",
        "bias_key",
    }:
        raise ValueError("Classification head must declare its exact tensor keys")
    weight_key, bias_key = head["weight_key"], head.get("bias_key")
    if not isinstance(weight_key, str) or not weight_key.endswith(".weight"):
        raise ValueError("Invalid classification head weight_key")
    module_path = weight_key.removesuffix(".weight")
    if module_path.split(".")[-1] not in {"score", "classifier"} or not any(
        module_path == module or module_path.endswith("." + module)
        for module in modules
    ):
        raise ValueError(
            "Classification head must select a linear saved score/classifier"
        )
    if "bias_key" in head and bias_key != module_path + ".bias":
        raise ValueError("Classification bias_key must match the selected head")
    return weight_key, bias_key


def load_classification_head(
    lora_path: str, hidden_size: int
) -> ClassificationHead | None:
    path = Path(lora_path)
    try:
        if not _has_manifest(path):
            return None
        config = _read_json(path / _MANIFEST)
        if not isinstance(config, dict):
            raise ValueError("Classification manifest must be a JSON object")
        if (
            type(config.get("schema_version")) is not int
            or config["schema_version"] != 1
        ):
            raise ValueError("Unsupported classification schema_version")
        if config.get("problem_type") != "single_label_classification":
            raise ValueError("Only single_label_classification heads are supported")
        if config.get("pooling") != "last":
            raise ValueError("Classification pooling must be last")
        size = config.get("num_labels")
        if type(size) is not int or size < 2:
            raise ValueError("Classification num_labels must be at least 2")
        if (
            type(hidden_size) is not int
            or hidden_size < 1
            or type(config.get("hidden_size")) is not int
            or config["hidden_size"] != hidden_size
        ):
            raise ValueError("Classification hidden_size does not match the base model")
        max_length = config.get("max_length")
        if type(max_length) is not int or max_length < 1:
            raise ValueError("Classification max_length must be a positive integer")
        add_special_tokens = config.get("add_special_tokens")
        if type(add_special_tokens) is not bool:
            raise ValueError("Classification add_special_tokens must be explicit")
        for name, expected in _artifacts(config).items():
            with (path / name).open("rb") as artifact:
                digest = hashlib.sha256()
                for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                    digest.update(chunk)
                actual = digest.hexdigest()
            if actual != expected.lower():
                raise ValueError(f"Classification artifact checksum mismatch: {name}")
        labels = _labels(config.get("id2label"), size)
        adapter = _read_json(path / "adapter_config.json")
        if not isinstance(adapter, dict):
            raise ValueError("LoRA adapter config must be a JSON object")
        weight_key, bias_key = _head_keys(config, adapter)
        with safe_open(
            path / "adapter_model.safetensors", framework="pt", device="cpu"
        ) as state:
            keys = set(state.keys())
            expected = {weight_key} | ({bias_key} if bias_key is not None else set())
            saved_head_keys = {
                key for key in keys if {"score", "classifier"} & set(key.split("."))
            }
            if saved_head_keys != expected or not expected <= keys:
                raise ValueError(
                    "Saved classification head keys do not match the manifest"
                )
            weight = state.get_tensor(weight_key)
            bias = state.get_tensor(bias_key) if bias_key is not None else None
        if (
            not isinstance(weight, torch.Tensor)
            or weight.shape != (size, hidden_size)
            or weight.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or not torch.isfinite(weight).all()
        ):
            raise ValueError("Invalid classification head weight")
        if bias is not None and (
            not isinstance(bias, torch.Tensor)
            or bias.shape != (size,)
            or bias.dtype != weight.dtype
            or not torch.isfinite(bias).all()
        ):
            raise ValueError("Invalid classification head bias")
        return ClassificationHead(
            weight.clone().contiguous(),
            None if bias is None else bias.clone().contiguous(),
            labels,
            max_length,
            add_special_tokens,
        )
    except (
        OSError,
        TypeError,
        OverflowError,
        RuntimeError,
        json.JSONDecodeError,
        SafetensorError,
        EOFError,
    ) as error:
        raise ValueError(f"Invalid classification bundle: {error}") from error


@dataclass
class ClassificationBundle:
    head: ClassificationHead
    directory: tempfile.TemporaryDirectory

    @property
    def path(self):
        return self.directory.name


def prepare_classification_bundle(
    lora_path: str, hidden_size: int
) -> ClassificationBundle | None:
    """Copy a checksum-bound bundle into a private immutable runtime directory.

    Only manifest-listed files reach the backend, so unbound extra weight files
    cannot override the verified weights selected by the model loader.
    """
    source = Path(lora_path)
    try:
        if not _has_manifest(source):
            return None
    except (OSError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid classification bundle: {error}") from error
    tokenizer_files = (
        "added_tokens.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "tokenizer.model",
        "spiece.model",
    )
    if any((source / name).exists() for name in tokenizer_files):
        raise ValueError(
            "Classification adapter tokenizer/vocabulary extensions are unsupported"
        )
    directory = tempfile.TemporaryDirectory(prefix="sglang-classifier-")
    try:
        snapshot = Path(directory.name)
        shutil.copyfile(source / _MANIFEST, snapshot / _MANIFEST)
        config = _read_json(snapshot / _MANIFEST)
        if not isinstance(config, dict):
            raise ValueError("Classification manifest must be a JSON object")
        for name in _artifacts(config):
            shutil.copyfile(source / name, snapshot / name)
        head = load_classification_head(str(snapshot), hidden_size)
        for name in _ARTIFACTS | {_MANIFEST}:
            (snapshot / name).chmod(0o400)
        return ClassificationBundle(head, directory)
    except BaseException as error:
        directory.cleanup()
        if isinstance(error, (OSError, TypeError, json.JSONDecodeError)):
            raise ValueError(f"Invalid classification bundle: {error}") from error
        raise


class ClassificationLease:
    """Hold the exact resolved LoRA version through CPU postprocessing.

    This object is passed internally, never deserialized from an HTTP request.
    The generation request owns its normal scheduler lease independently.
    """

    def __init__(self, manager):
        self.manager = manager
        self.lora_id = None
        self.head = None

    async def acquire(self, obj):
        ids = obj.lora_id if isinstance(obj.lora_id, list) else [obj.lora_id]
        if not ids or not ids[0] or any(value != ids[0] for value in ids):
            raise ValueError("Classification requires one LoRA adapter per request")
        head = self.manager.classification_heads.get(ids[0])
        if head is None:
            raise ValueError(
                "LoRA adapter has no validated classification head and label mapping"
            )
        await self.manager.lora_registry.retain(ids[0])
        self.lora_id, self.head = ids[0], head
        await self.run_cpu(self._prepare_input, obj)

    def _prepare_input(self, obj):
        # HF fast tokenizers mutate padding/truncation state on every call.
        # Use an independent tokenizer and serialize only classifier tokenization.
        with self.manager.classification_tokenizer_lock:
            if obj.text is not None and self.manager.classification_tokenizer is None:
                self.manager.classification_tokenizer = copy.deepcopy(
                    self.manager.tokenizer
                )
            self.head.prepare_input(
                obj,
                self.manager.classification_tokenizer,
                getattr(self.manager.model_config.hf_config, "pad_token_id", None),
            )

    async def run_cpu(self, fn, *args):
        return await finish_on_cancel(asyncio.to_thread(fn, *args))

    async def close(self):
        if self.lora_id is not None:
            lora_id, self.lora_id = self.lora_id, None
            await finish_on_cancel(self.manager.lora_registry.release(lora_id))
