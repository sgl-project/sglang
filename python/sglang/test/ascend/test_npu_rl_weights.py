"""Generic deterministic weight source for NPU RL weight-transfer tests.

The published checkpoints used by these tests are far too large to read (GLM-5.2
is ~1.4 TB). Instead a case describes a layer-reduced, expert-reduced variant of
a HuggingFace model, and this module builds it from ``config.json`` alone:

* a meta-device Transformers model enumerates the exact checkpoint-facing
  parameter names, shapes and dtypes of the reduced model;
* every tensor is regenerated from a per-name seed, so the payload is identical
  across processes and runs;
* ``write_checkpoint`` materializes the same payload as a normally loadable
  checkpoint, which is the startup-load oracle for the tests.

The case/adapter layout mirrors vLLM-Ascend's weight-transfer tests so another
model can be added by declaring a ``WeightUpdateModelCase`` plus, where needed,
a checkpoint name map / expert-shape adapter.

Tensors are serialized as NPU IPC handles by the callers, so their bytes move
through NPU shared memory rather than the HTTP request body.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch

from sglang.test.ascend.test_ascend_utils import GLM_5_2_BF16_MODEL_PATH

WEIGHT_SEED = 20260915
CACHE_ROOT = os.path.join(tempfile.gettempdir(), "sglang-weight-update-models")

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
)

# Per-layer config lists whose length must match num_hidden_layers once the
# model is reduced (transformers validates this for GlmMoeDsaConfig).
PER_LAYER_CONFIG_LISTS = ("mlp_layer_types", "indexer_types")

SIGNATURE_PROMPTS = (
    "The capital of France is",
    "1 + 1 =",
    "def hello():\n    return",
)


@dataclass(frozen=True)
class WeightUpdateModelCase:
    """A layer-reduced model whose complete parameter set is reloaded."""

    id: str
    model: str
    hf_overrides: Dict[str, Any]
    # Local checkpoint directory. ``local_path_env`` overrides it so the same
    # case runs on hosts that keep the model elsewhere.
    local_path: Optional[str] = None
    local_path_env: Optional[str] = None
    # HF attribute holding the text config (VLMs expose a nested one).
    meta_config_attribute: Optional[str] = None
    # Namespace prefix applied to raw meta-model names (e.g. "model.language_model.").
    checkpoint_model_prefix: Optional[str] = None
    # Rename map applied to the expanded (per-expert) checkpoint-facing names.
    checkpoint_name_map: Optional[Callable[[str], str]] = None
    # Checkpoint expert MLP width when it differs from HF's from_config default.
    expert_intermediate_size: Optional[int] = None
    # Floating buffers that are checkpoint tensors (routers keep them as buffers
    # in HF). They are promoted to parameters before the payload is enumerated.
    buffer_to_parameter: Tuple[str, ...] = ()
    # Override for architectures without an AutoModelForCausalLM entry; the
    # factory receives the (text) config and must build on the meta device.
    meta_model_factory: Optional[Callable[[Any], torch.nn.Module]] = None
    extra_server_args: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()


def model_path(case: WeightUpdateModelCase) -> str:
    """Return the local checkpoint directory that provides config + tokenizer."""
    candidates = []
    if case.local_path_env:
        candidates.append(os.environ.get(case.local_path_env))
    candidates.append(case.local_path)
    candidates.append(case.model)
    for path in candidates:
        if path and os.path.isfile(os.path.join(path, "config.json")):
            return path
    raise RuntimeError(
        f"{case.id}: config.json not found; set {case.local_path_env or 'the local path'} "
        f"or provision one of {candidates}"
    )


@lru_cache(maxsize=None)
def _raw_config(case_id: str, path: str) -> str:
    with open(os.path.join(path, "config.json"), encoding="utf-8") as handle:
        return json.dumps(json.load(handle), sort_keys=True)


def _config_dict(case: WeightUpdateModelCase) -> Dict[str, Any]:
    return json.loads(_raw_config(case.id, model_path(case)))


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _slice_per_layer_lists(config: Dict[str, Any], layers: Optional[int]) -> None:
    """Trim per-layer lists to ``num_hidden_layers`` at every nesting level."""
    for key, value in list(config.items()):
        if isinstance(value, dict):
            _slice_per_layer_lists(value, value.get("num_hidden_layers", layers))
        elif key in PER_LAYER_CONFIG_LISTS and isinstance(value, list) and layers:
            config[key] = value[:layers]


def _fill_and_slice_overrides(
    raw_config: Dict[str, Any], overrides: Dict[str, Any]
) -> None:
    """Add the source's per-layer lists (sliced) at each reduced level."""
    layers = overrides.get("num_hidden_layers")
    if isinstance(layers, int):
        for key in PER_LAYER_CONFIG_LISTS:
            values = raw_config.get(key)
            if not isinstance(values, list):
                continue
            if key in overrides and isinstance(overrides[key], list):
                overrides[key] = overrides[key][:layers]
            elif len(values) != layers:
                overrides[key] = values[:layers]
        return
    for key, value in overrides.items():
        if isinstance(value, dict):
            _fill_and_slice_overrides(raw_config.get(key) or {}, value)


def resolved_overrides(case: WeightUpdateModelCase) -> Dict[str, Any]:
    """Case overrides plus the per-layer lists sliced to the reduced depth."""
    overrides = copy.deepcopy(case.hf_overrides)
    _fill_and_slice_overrides(_config_dict(case), overrides)
    return overrides


def server_override_args(case: WeightUpdateModelCase) -> Tuple[str, ...]:
    return ("--json-model-override-args", json.dumps(resolved_overrides(case)))


def reduced_config(case: WeightUpdateModelCase):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path(case))
    _apply_config_overrides(config, resolved_overrides(case))
    return config


def _apply_config_overrides(config, overrides: Dict[str, Any]) -> None:
    from transformers import PretrainedConfig

    for key, value in overrides.items():
        current = getattr(config, key, None)
        if isinstance(value, dict) and isinstance(current, PretrainedConfig):
            _apply_config_overrides(current, value)
        else:
            setattr(config, key, value)


def expand_fused_expert_params(
    name: str, shape: Tuple[int, ...]
) -> List[Tuple[str, Tuple[int, ...]]]:
    """Split HuggingFace's fused routed-expert tensors into checkpoint names.

    The meta model exposes ``mlp.experts.gate_up_proj`` ``[E, 2I, H]`` and
    ``mlp.experts.down_proj`` ``[E, H, I]``, while the published checkpoint -
    and therefore SGLang's ``load_weights`` - stores one tensor per expert.
    """
    for suffix, kind in (
        (".mlp.experts.gate_up_proj", "gate_up"),
        (".mlp.experts.down_proj", "down"),
    ):
        if not name.endswith(suffix):
            continue
        prefix = name[: -len(suffix.rsplit(".", 1)[-1])]
        if kind == "gate_up":
            num_experts, fused_inter, hidden = shape
            inter = fused_inter // 2
            return [
                (f"{prefix}{expert}.{proj}_proj.weight", (inter, hidden))
                for expert in range(num_experts)
                for proj in ("gate", "up")
            ]
        num_experts, hidden, inter = shape
        return [
            (f"{prefix}{expert}.down_proj.weight", (hidden, inter))
            for expert in range(num_experts)
        ]
    return [(name, shape)]


def resize_expert_shape(
    name: str, shape: Tuple[int, ...], intermediate_size: Optional[int]
) -> Tuple[int, ...]:
    """Force expert MLP tensors to the checkpoint's intermediate dimension."""
    if intermediate_size is None:
        return shape
    for suffix in (
        ".mlp.shared_experts.gate_proj.weight",
        ".mlp.shared_experts.up_proj.weight",
        ".mlp.experts.gate_proj.weight",
        ".mlp.experts.up_proj.weight",
    ):
        if name.endswith(suffix):
            return (intermediate_size, shape[-1])
    if name.endswith(
        (".mlp.shared_experts.down_proj.weight", ".mlp.experts.down_proj.weight")
    ):
        return (shape[0], intermediate_size)
    return shape


def _checkpoint_name(name: str, case: WeightUpdateModelCase) -> str:
    if case.checkpoint_model_prefix is not None and name.startswith("model."):
        return case.checkpoint_model_prefix + name[len("model.") :]
    return name


def _apply_name_map(name: str, case: WeightUpdateModelCase) -> str:
    if case.checkpoint_name_map is not None:
        return case.checkpoint_name_map(name)
    return name


def _promote_buffers(model: torch.nn.Module, suffixes: Tuple[str, ...]) -> None:
    """Make checkpoint-facing floating buffers visible as parameters."""
    if not suffixes:
        return
    for name, buffer in list(model.named_buffers()):
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        if attr not in parent._buffers:
            continue
        del parent._buffers[attr]
        parent.register_parameter(attr, torch.nn.Parameter(buffer))


def payload_metadata(
    case: WeightUpdateModelCase,
) -> List[Tuple[str, Tuple[int, ...], torch.dtype]]:
    """Enumerate the reduced model's checkpoint-facing tensors."""
    from transformers import AutoModelForCausalLM

    config = reduced_config(case)
    meta_config = (
        getattr(config, case.meta_config_attribute)
        if case.meta_config_attribute
        else config
    )
    with torch.device("meta"):
        if case.meta_model_factory is not None:
            model = case.meta_model_factory(meta_config)
        else:
            model = AutoModelForCausalLM.from_config(meta_config)
        _promote_buffers(model, case.buffer_to_parameter)

    tensors = [
        (name, tuple(param.shape), param.dtype)
        for name, param in model.named_parameters()
    ]
    # Learned integer tables are buffers; float buffers that are checkpoint
    # tensors must be listed in ``buffer_to_parameter`` by the case.
    tensors += [
        (name, tuple(buffer.shape), buffer.dtype)
        for name, buffer in model.named_buffers()
        if not buffer.dtype.is_floating_point
    ]

    expanded: List[Tuple[str, Tuple[int, ...], torch.dtype]] = []
    for name, shape, dtype in tensors:
        for expanded_name, expanded_shape in expand_fused_expert_params(name, shape):
            checkpoint_name = _apply_name_map(
                _checkpoint_name(expanded_name, case), case
            )
            expanded.append(
                (
                    checkpoint_name,
                    resize_expert_shape(
                        expanded_name, expanded_shape, case.expert_intermediate_size
                    ),
                    dtype,
                )
            )
    expanded.sort(key=lambda item: item[0])

    names = [name for name, _, _ in expanded]
    assert names, f"{case.id}: reduced meta model contains no parameters"
    assert len(set(names)) == len(names), f"{case.id}: generated names are not unique"
    return expanded


def _tensor_seed(name: str) -> int:
    digest = hashlib.sha256(f"{WEIGHT_SEED}:{name}".encode()).hexdigest()
    return int(digest[:16], 16) % (2**63 - 1)


def make_tensor(name: str, shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Regenerate one deterministic tensor (bit-identical across processes)."""
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_tensor_seed(name))
    if name.endswith("norm.weight"):
        tensor = torch.empty(shape, dtype=torch.float32).uniform_(
            0.9, 1.1, generator=generator
        )
    else:
        tensor = torch.empty(shape, dtype=torch.float32).uniform_(
            -0.02, 0.02, generator=generator
        )
    return tensor.to(dtype)


def iter_payload(
    case: WeightUpdateModelCase,
) -> Iterator[Tuple[str, torch.Tensor]]:
    for name, shape, dtype in payload_metadata(case):
        yield name, make_tensor(name, shape, dtype)


class FixedRandomWeightSource:
    """Deterministic payload for one case, in the shape of vLLM's weight source."""

    def __init__(self, case: WeightUpdateModelCase) -> None:
        self.case = case
        self._metadata = payload_metadata(case)

    @property
    def metadata(self) -> List[Tuple[str, Tuple[int, ...], torch.dtype]]:
        return list(self._metadata)

    def __iter__(self) -> Iterator[Tuple[str, torch.Tensor]]:
        for name, shape, dtype in self._metadata:
            yield name, make_tensor(name, shape, dtype)

    @staticmethod
    def tensor_bytes(name: str, shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        return math.prod(shape) * dtype.itemsize

    def total_bytes(self) -> int:
        return sum(self.tensor_bytes(*item) for item in self._metadata)

    def largest_tensor(self) -> Tuple[int, str]:
        name, shape, dtype = max(
            self._metadata, key=lambda item: self.tensor_bytes(*item)
        )
        return self.tensor_bytes(name, shape, dtype), name

    def write_checkpoint(self, destination: str) -> None:
        from safetensors.torch import save_file

        path = model_path(self.case)
        os.makedirs(destination, exist_ok=True)
        config = _config_dict(self.case)
        config.update(resolved_overrides(self.case))
        with open(
            os.path.join(destination, "config.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")
        for name in TOKENIZER_FILES:
            source = os.path.join(path, name)
            if os.path.isfile(source):
                shutil.copyfile(source, os.path.join(destination, name))
        state = {name: tensor.contiguous() for name, tensor in self}
        save_file(
            state,
            os.path.join(destination, "model.safetensors"),
            metadata={"format": "pt"},
        )


def _cache_key(case: WeightUpdateModelCase) -> str:
    path = model_path(case)
    config_path = os.path.join(path, "config.json")
    stat = os.stat(config_path)
    raw = "|".join(
        [
            os.path.realpath(path),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            json.dumps(resolved_overrides(case), sort_keys=True),
            str(WEIGHT_SEED),
        ]
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def ensure_checkpoint(case: WeightUpdateModelCase) -> str:
    """Return a cached reference checkpoint, building it on first use."""
    target = os.path.join(CACHE_ROOT, f"{case.id}-{_cache_key(case)}")
    checkpoint = os.path.join(target, "model.safetensors")
    config_path = os.path.join(target, "config.json")
    if os.path.isfile(checkpoint) and os.path.isfile(config_path):
        return target

    os.makedirs(CACHE_ROOT, exist_ok=True)
    staging = f"{target}.tmp-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        FixedRandomWeightSource(case).write_checkpoint(staging)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


GLM52_CASE = WeightUpdateModelCase(
    id="glm52-6l8e",
    model="zai-org/GLM-5.2",
    local_path=GLM_5_2_BF16_MODEL_PATH,
    local_path_env="SGLANG_TEST_GLM52_PATH",
    hf_overrides={
        "num_hidden_layers": 6,
        "n_routed_experts": 8,
        "num_nextn_predict_layers": 0,
    },
    buffer_to_parameter=("e_score_correction_bias",),
)

MODEL_CASES = (GLM52_CASE,)
