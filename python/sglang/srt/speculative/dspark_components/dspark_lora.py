# SPDX-License-Identifier: Apache-2.0
"""Startup-only PEFT LoRA merge for dense DSpark draft checkpoints.

This is deliberately separate from the target's LoRAManager: adapter IDs and
per-request target routing never enter this loader. Merging before TP sharding
also covers the draft's direct weight accesses and fused context-KV projection.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

import torch

_MODULE = re.compile(
    r"(?:layers\.\d+\.(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|mlp\.(?:gate_proj|up_proj|down_proj))|fc)"
)
_KEY = re.compile(r"(.+)\.lora_([AB])\.weight")
_SUPPORTED_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "fc",
}


def validate_dspark_lora_args(*, algorithm: str | None, draft_load_format: str) -> None:
    if (algorithm or "").upper() != "DSPARK":
        raise ValueError("--speculative-dspark-lora-path requires DSPARK.")
    if draft_load_format not in {"auto", "safetensors", "pt"}:
        raise ValueError(
            "Static DSpark draft LoRA requires the auto, safetensors, or pt "
            "draft checkpoint loader (not dummy, sharded, or quantized loaders)."
        )


def _canonical_module(name: str) -> str:
    # PEFT adds base_model.model.; the underlying HF model may also use model.
    return name.removeprefix("base_model.model.").removeprefix("model.")


def _validate_config(config: Mapping) -> tuple[int, float]:
    if config.get("peft_type") != "LORA":
        raise ValueError("DSpark draft adapter must have peft_type='LORA'.")
    if config.get("bias", "none") != "none":
        raise ValueError("DSpark draft LoRA does not support bias updates.")
    if config.get("init_lora_weights", True) not in (True, False, "gaussian"):
        raise ValueError(
            "DSpark draft LoRA does not support initialization methods that "
            "may require a transformed base checkpoint."
        )
    # These change merge semantics or introduce extra trainable parameters.
    # Reject explicitly instead of silently treating their tensors as plain LoRA.
    for option in (
        "fan_in_fan_out",
        "use_dora",
        "use_rslora",
        "use_qalora",
        "lora_bias",
        "rank_pattern",
        "alpha_pattern",
        "modules_to_save",
        "target_parameters",
        "layer_replication",
        "megatron_config",
        "loftq_config",
        "eva_config",
        "corda_config",
        "trainable_token_indices",
        "alora_invocation_tokens",
        "arrow_config",
        "ensure_weight_tying",
    ):
        if config.get(option):
            raise ValueError(f"DSpark draft LoRA does not support {option}.")
    modules = config.get("target_modules")
    if (
        not isinstance(modules, list)
        or not modules
        or not all(isinstance(module, str) for module in modules)
        or not set(modules) <= _SUPPORTED_MODULES
    ):
        raise ValueError(
            "DSpark draft LoRA target_modules must be an explicit nonempty list "
            f"from {sorted(_SUPPORTED_MODULES)}; shared embeddings/LM head, "
            "Markov heads, fused names, and regex targets are not supported."
        )
    rank = config.get("r")
    alpha = config.get("lora_alpha")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
        raise ValueError("DSpark draft LoRA rank r must be a positive integer.")
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("DSpark draft LoRA lora_alpha must be finite and positive.")
    return rank, float(alpha) / rank


def _read_adapter(adapter_path: str) -> tuple[dict, dict[str, torch.Tensor]]:
    from safetensors.torch import load_file

    path = Path(adapter_path)
    if not path.is_dir():
        raise ValueError("DSpark draft LoRA requires a local adapter directory.")
    with (path / "adapter_config.json").open() as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must contain an object.")
    _validate_config(config)
    # No pickle fallback or network resolution: use an explicit local artifact.
    return config, load_file(str(path / "adapter_model.safetensors"), device="cpu")


def merge_dspark_lora_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    adapter_path: str,
    *,
    model_parameter_names: Iterable[str],
) -> Iterator[tuple[str, torch.Tensor]]:
    """Merge W + (alpha/r) B@A without mutating any checkpoint tensor.

    Consume the iterator completely before loading model parameters. This
    ensures every adapter pair matched exactly one unsharded checkpoint weight.
    Packed/quantized input and unknown adapter tensors fail closed.
    """
    config, adapter = _read_adapter(adapter_path)
    rank, scaling = _validate_config(config)
    model_parameters = {_canonical_module(name) for name in model_parameter_names}
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in adapter.items():
        match = _KEY.fullmatch(name)
        if match is None:
            raise ValueError(f"Unsupported DSpark draft adapter tensor: {name}")
        module = _canonical_module(match[1])
        if _MODULE.fullmatch(module) is None:
            raise ValueError(f"Unsupported DSpark draft LoRA module: {module}")
        if module.rsplit(".", 1)[-1] not in config["target_modules"]:
            raise ValueError(f"Adapter tensor not declared in target_modules: {name}")
        packed_module = module
        for source, target in (
            ("q_proj", "qkv_proj"),
            ("k_proj", "qkv_proj"),
            ("v_proj", "qkv_proj"),
            ("gate_proj", "gate_up_proj"),
            ("up_proj", "gate_up_proj"),
        ):
            if module.endswith("." + source):
                packed_module = module.removesuffix(source) + target
                break
        if packed_module + ".weight" not in model_parameters:
            raise ValueError(
                f"LoRA module absent from the DSpark draft model: {module}"
            )
        pair = pairs.setdefault(module, {})
        if match[2] in pair:
            raise ValueError(f"Duplicate DSpark draft adapter tensor for {module}")
        if tensor.ndim != 2 or tensor.dtype not in {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }:
            raise ValueError(f"LoRA tensor must be a floating point matrix: {name}")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Non-finite DSpark draft adapter tensor: {name}")
        pair[match[2]] = tensor
    if not pairs:
        raise ValueError("DSpark draft adapter contains no LoRA tensor pairs.")
    for module, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"Missing LoRA A/B partner for {module}")
        if pair["A"].shape[0] != rank or pair["B"].shape[1] != rank:
            raise ValueError(f"LoRA rank disagrees with adapter config for {module}")

    seen = set()
    for name, weight in weights:
        module = _canonical_module(name.removesuffix(".weight"))
        if not name.endswith(".weight") or module not in pairs:
            yield name, weight
            continue
        if module in seen:
            raise ValueError(f"Duplicate draft checkpoint weight for {module}")
        seen.add(module)
        a, b = pairs[module]["A"], pairs[module]["B"]
        if weight.ndim != 2 or tuple(weight.shape) != (b.shape[0], a.shape[1]):
            raise ValueError(
                f"LoRA shape does not match unsharded draft weight {name}: "
                f"weight={tuple(weight.shape)}, A={tuple(a.shape)}, B={tuple(b.shape)}"
            )
        if weight.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise ValueError(f"Quantized draft weight is unsupported: {name}")
        # The checkpoint may already be on the loading device. FP32 accumulation
        # avoids performing a low-rank update in BF16 before the final cast.
        merged = weight.float() + scaling * (
            b.to(device=weight.device, dtype=torch.float32)
            @ a.to(device=weight.device, dtype=torch.float32)
        )
        merged = merged.to(dtype=weight.dtype)
        if not torch.isfinite(merged).all().item():
            raise ValueError(f"Non-finite merged DSpark draft weight: {name}")
        yield name, merged
    missing = pairs.keys() - seen
    if missing:
        raise ValueError(
            "DSpark draft adapter weights not found in checkpoint: "
            f"{sorted(missing)}. Packed QKV/gate-up checkpoint weights are not supported."
        )


class DSparkDraftAdapterBank:
    """Immutable merged variants for TP=1, eager, homogeneous draft batches.

    This deliberately trades GPU memory for reuse of the model's existing
    linear and fused KV kernels. Only modified packed weights are duplicated;
    shared embeddings, LM head, norms, and untouched draft parameters are not.
    No request-time disk IO, additive merge/unmerge drift, or target mutation.
    """

    @torch.no_grad()
    def __init__(self, model, adapter_paths: Mapping[str, str]):
        self.model = model
        parameters = dict(model.named_parameters())
        self.variants = {}
        self.base = {}
        self.active = None
        self.switch_count = 0

        # Present the same logical unsharded weights accepted by the startup
        # merger. TP=1 is enforced before loading; no shard offsets are guessed.
        logical = {}
        destinations = {}
        for name, param in parameters.items():
            if name.endswith(".self_attn.qkv_proj.weight"):
                parent = model.get_submodule(name.removesuffix(".qkv_proj.weight"))
                sizes = (parent.q_size, parent.kv_size, parent.kv_size)
                if sum(sizes) != param.shape[0]:
                    raise ValueError(f"Unsupported packed QKV layout: {name}")
                offset = 0
                for suffix, size in zip(("q_proj", "k_proj", "v_proj"), sizes):
                    key = name.replace("qkv_proj.weight", suffix + ".weight")
                    rows = slice(offset, offset + size)
                    logical[key] = param.detach()[rows]
                    destinations[key] = (name, rows)
                    offset += size
            elif name.endswith(".mlp.gate_up_proj.weight"):
                if param.shape[0] % 2:
                    raise ValueError(f"Unsupported packed gate/up layout: {name}")
                width = param.shape[0] // 2
                for i, suffix in enumerate(("gate_proj", "up_proj")):
                    key = name.replace("gate_up_proj.weight", suffix + ".weight")
                    rows = slice(i * width, (i + 1) * width)
                    logical[key] = param.detach()[rows]
                    destinations[key] = (name, rows)
            else:
                logical[name] = param.detach()
                destinations[name] = (name, slice(None))

        # Build all variants without modifying the live model. A bad adapter
        # aborts startup before any request can observe partial weights.
        for adapter, path in adapter_paths.items():
            replacement = {}
            for key, merged in merge_dspark_lora_weights(
                logical.items(), path, model_parameter_names=parameters
            ):
                if merged is logical[key]:
                    continue
                name, rows = destinations[key]
                if name not in replacement:
                    replacement[name] = parameters[name].detach().clone()
                replacement[name][rows].copy_(merged)
            self.variants[adapter] = replacement
        names = {name for variant in self.variants.values() for name in variant}
        self.parameters = {name: parameters[name] for name in names}
        self.base = {
            name: param.detach().clone() for name, param in self.parameters.items()
        }
        self.resident_bytes = sum(
            tensor.numel() * tensor.element_size()
            for weights in [self.base, *self.variants.values()]
            for tensor in weights.values()
        )

    @torch.no_grad()
    def activate(self, name: str | None) -> bool:
        if name is not None and name not in self.variants:
            raise ValueError(f"Unknown draft adapter: {name!r}")
        if name == self.active:
            return False
        # Non-overlap execution is required, but kernels may still be queued.
        # Drain all device streams before replacing weights or cached KV views.
        device = next(iter(self.parameters.values())).device
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        variant = self.variants.get(name, {})
        for key, param in self.parameters.items():
            param.copy_(variant.get(key, self.base[key]))
        # The fused writer stacks weights: clearing just its pointers is not
        # enough. Discard the entire bundle so it is rebuilt for this adapter.
        self.model._fused_kv_write_cache = None
        # FP16 / unsupported fused-write layouts use a second stacked-weight
        # cache. False means "rebuild"; None means "do not use this fast path".
        self.model._stacked_ctx_kv_cache = False
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        self.active = name
        self.switch_count += 1
        return True
