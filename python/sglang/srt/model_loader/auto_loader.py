# Adapted from vLLM's AutoWeightsLoader (vllm/model_executor/models/utils.py).
"""Walker-style weight loader for sglang models.

The :class:`AutoWeightsLoader` walks a module tree once and dispatches each
incoming ``(name, tensor)`` pair to the right place:

* If a child module defines its own ``load_weights``, defer to it.
* Otherwise recurse into the child.
* At the leaf, look up ``param.weight_loader`` (already the convention for
  ``QKVParallelLinear`` / ``MergedColumnParallelLinear`` / ``FusedMoE``) and
  call it. If no ``weight_loader`` attribute exists, fall back to
  :func:`default_weight_loader`.

A :class:`WeightsMapper` lets a model declaratively rename or drop checkpoint
keys before the walk (e.g. ``weight_scale_inv -> weight_scale``).
"""

from __future__ import annotations

import itertools
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Quirks registry (RFC §3)
# ---------------------------------------------------------------------------

QuirkFn = Callable[[nn.Module, "WeightsMapper"], "WeightsMapper"]


class WeightQuirkRegistry:
    """Architecture-keyed registry for ``WeightsMapper`` producers.

    Use to externalize per-model name remaps that today live as
    ``if isinstance / if quant_config.get_name() == "..."`` branches inside
    ``load_weights``. A quirk takes the model instance (so it can introspect
    quant config, fused-projection flags, etc.) plus the caller-supplied base
    mapper, and returns a (possibly extended) mapper.

    Quirks are looked up by ``model.__class__.__name__``. Multiple quirks may
    be registered for the same architecture; they compose left-to-right via
    :meth:`WeightsMapper.__or__` in registration order.

    Example::

        @WeightQuirkRegistry.register("DeepseekV2ForCausalLM")
        def deepseek_fused_qkv_a_proj(model, mapper):
            if not getattr(model, "fuse_qkv_a_proj", False):
                return mapper
            return mapper | WeightsMapper(
                orig_to_new_substr={
                    "q_a_proj": "fused_qkv_a_proj_with_mqa",
                    "kv_a_proj_with_mqa": "fused_qkv_a_proj_with_mqa",
                },
            )
    """

    _quirks: dict[str, list[QuirkFn]] = {}

    @classmethod
    def register(
        cls, architectures: str | Iterable[str]
    ) -> Callable[[QuirkFn], QuirkFn]:
        if isinstance(architectures, str):
            architectures = [architectures]
        archs = list(architectures)

        def decorator(fn: QuirkFn) -> QuirkFn:
            for arch in archs:
                cls._quirks.setdefault(arch, []).append(fn)
            return fn

        return decorator

    @classmethod
    def compose(
        cls, model: nn.Module, base: "WeightsMapper" | None = None
    ) -> "WeightsMapper":
        """Return ``base`` extended by every quirk registered for ``model``'s class."""
        mapper = base if base is not None else WeightsMapper()
        for arch in _arch_lookup_keys(model):
            for fn in cls._quirks.get(arch, ()):
                mapper = fn(model, mapper)
        return mapper

    @classmethod
    def _reset_for_tests(cls) -> None:
        cls._quirks.clear()


def _arch_lookup_keys(model: nn.Module) -> list[str]:
    """Lookup keys for ``WeightQuirkRegistry``. Includes the concrete class
    name plus every base class name in MRO (so ``Phi3ForCausalLM`` inherits
    quirks registered on ``LlamaForCausalLM``)."""
    return [c.__name__ for c in type(model).__mro__ if c is not object]


# ---------------------------------------------------------------------------
# Multi-input fusion contract (RFC §4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MultiInputFusion:
    """Declare that one runtime parameter is the concat (or other reduction) of
    several checkpoint tensors that live under sibling child names.

    Attached to the parent module via the class attribute
    ``checkpoint_fusions: list[MultiInputFusion]``. The walker holds the
    partial-input buffer for the duration of one ``load_weights`` call; once
    every input has been seen, it calls :attr:`combine` and feeds the result
    to the target child's ``weight_loader`` (or :func:`default_weight_loader`).
    If a load completes with required inputs missing, the walker raises rather
    than silently leaving the fused parameter unloaded.

    Buffering is keyed by the *trailing* part of the checkpoint name (e.g.
    ``weight``, ``weight_scale``, ``bias``), so e.g. ``q_a_proj.weight`` and
    ``kv_a_proj_with_mqa.weight`` fuse into ``target.weight``, while
    ``q_a_proj.weight_scale`` and ``kv_a_proj_with_mqa.weight_scale`` fuse
    independently into ``target.weight_scale``.

    Example::

        class DeepseekV2Attention(nn.Module):
            checkpoint_fusions = [
                MultiInputFusion(
                    inputs=("q_a_proj", "kv_a_proj_with_mqa"),
                    target="fused_qkv_a_proj_with_mqa",
                    combine=lambda module, parts: torch.cat(
                        [parts["q_a_proj"], parts["kv_a_proj_with_mqa"]],
                        dim=_qkv_a_cat_dim(module),
                    ),
                ),
            ]
    """

    inputs: tuple[str, ...]
    """Sibling child-module names in the checkpoint that feed this fusion.
    Order is significant — passed to ``combine`` as a dict keyed by name."""

    target: str
    """Child-module name on the *runtime* module tree that owns the fused
    parameter (e.g. ``"fused_qkv_a_proj_with_mqa"``)."""

    combine: Callable[[nn.Module, dict[str, torch.Tensor]], torch.Tensor]
    """``(parent_module, {input_name: tensor}) → fused_tensor``. The parent
    module is passed so the combiner can introspect quant config or other
    attributes (e.g. AWQ requires ``dim=1`` instead of ``dim=0``)."""


# Standard fused-linear stacking shared by Llama-style decoder models. The
# tuples are ``(packed_param_name, source_shard_name, shard_id)`` and apply
# anywhere in the parameter tree (matched by substring), so they intentionally
# include the leading ``.``.
STACKED_PARAMS_MAPPING_LLAMA: list[tuple[str, str, str | int]] = [
    (".qkv_proj", ".q_proj", "q"),
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),
    (".gate_up_proj", ".up_proj", 1),
]


def default_moe_load(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    *,
    stacked_params_mapping: list[tuple[str, str, Any]],
    expert_params_mapping: list[tuple[str, str, int, Any]],
    skip_pp_out_of_range: bool = True,
) -> set[str]:
    """Per-tensor weight dispatch for an MoE-bearing module.

    Same shape as :func:`default_stacked_params_load` but adds the second
    dispatch table for fused experts. The expert path matches by checkpoint
    weight-name fragment, renames into the runtime parameter name, and calls
    ``param.weight_loader(param, w, name, shard_id=..., expert_id=...)`` —
    the signature that ``FusedMoE`` exposes.

    Stacked-params attempts come first; ``mlp.experts`` matches are skipped
    in that pass so the expert dispatch table can claim them. (Without this
    skip, ``mlp.experts.0.gate_proj`` would get renamed to
    ``mlp.experts.0.gate_up_proj`` and then mis-routed to the expert table
    as ``mlp.experts.0.gate_gate_up_proj``.)
    """
    params_dict = dict(module.named_parameters())
    loaded: set[str] = set()

    for name, loaded_weight in weights:
        if skip_pp_out_of_range:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(module, "start_layer")
                and (layer_id < module.start_layer or layer_id >= module.end_layer)
            ):
                continue

        for packed_name, shard_name, shard_id in stacked_params_mapping:
            if shard_name not in name:
                continue
            # Experts are handled by the next table; skip-before-rename so
            # the rename doesn't poison their names.
            if "mlp.experts" in name:
                continue
            mapped = name.replace(shard_name, packed_name)
            # If the fused target doesn't exist on this module (e.g. a
            # cross-attn with split q/k/v), don't claim the weight — let the
            # for-else fall-through try to load it under its original name.
            if mapped.endswith(".bias") and mapped not in params_dict:
                continue
            if mapped not in params_dict:
                continue
            param = params_dict[mapped]
            param.weight_loader(param, loaded_weight, shard_id)
            loaded.add(mapped)
            break
        else:
            is_expert_weight = False
            for packed_name, shard_name, expert_id, shard_id in expert_params_mapping:
                if shard_name not in name:
                    continue
                is_expert_weight = True
                mapped = name.replace(shard_name, packed_name)
                if mapped not in params_dict:
                    # Expert lives on a different EP rank; nothing to load
                    # locally but mark so we don't fall through to "missing".
                    continue
                param = params_dict[mapped]
                param.weight_loader(
                    param, loaded_weight, mapped, shard_id=shard_id, expert_id=expert_id
                )
                loaded.add(mapped)
                break
            else:
                if is_expert_weight:
                    # Expert weight that doesn't live on this EP rank — drop.
                    continue
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name.endswith(".kv_scale") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning("Parameter %s not found in params_dict", name)
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded.add(name)

    return loaded


def default_stacked_params_load(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    stacked_params_mapping: list[tuple[str, str, Any]],
    *,
    skip_pp_out_of_range: bool = True,
) -> set[str]:
    """Per-tensor weight dispatch for a module that owns fused linear layers.

    This is the body of the loop every Llama-derived ``load_weights`` writes by
    hand today: walk the incoming stream, route each tensor either through a
    ``(packed_param, shard_id)`` weight loader (for fused QKV / gate_up) or
    fall through to ``param.weight_loader`` / :func:`default_weight_loader`.

    PP-out-of-range layers, rotary buffer junk, and other top-level filtering
    are expected to be done upstream (e.g. by the AutoWeightsLoader walker
    that wraps this call). Only the FP8 kv-scale dynamic remap stays here
    because it depends on the live ``params_dict``.
    """
    params_dict = dict(module.named_parameters())
    loaded: set[str] = set()

    for name, loaded_weight in weights:
        if skip_pp_out_of_range:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(module, "start_layer")
                and (layer_id < module.start_layer or layer_id >= module.end_layer)
            ):
                continue

        if "scale" in name:
            remapped = maybe_remap_kv_scale_name(name, params_dict)
            if remapped is None:
                continue
            name = remapped

        for packed_name, shard_name, shard_id in stacked_params_mapping:
            if shard_name not in name:
                continue
            mapped = name.replace(shard_name, packed_name)
            # If the fused target doesn't exist on this module (split q/k/v
            # in a cross-attn, missing GPTQ bias, missing PP shard), don't
            # claim the weight here — the for-else falls through and tries
            # to load it under its original name.
            if mapped.endswith(".bias") and mapped not in params_dict:
                continue
            if mapped not in params_dict:
                continue
            param = params_dict[mapped]
            param.weight_loader(param, loaded_weight, shard_id)
            loaded.add(mapped)
            break
        else:
            if name.endswith(".bias") and name not in params_dict:
                continue
            if name.endswith(".kv_scale") and name not in params_dict:
                continue
            if name not in params_dict:
                logger.warning("Parameter %s not found in params_dict", name)
                continue
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded.add(name)

    return loaded


@dataclass
class WeightsMapper:
    """Declaratively rename (or drop) checkpoint tensor names.

    Each mapping entry maps an *old* fragment of the key to a *new* fragment;
    a value of ``None`` drops the matching tensor entirely.

    Mappers compose with ``|``; the right-hand mapper's entries override the
    left-hand mapper's on conflicts.
    """

    orig_to_new_regex: Mapping[re.Pattern[str], str | None] = field(
        default_factory=dict
    )
    orig_to_new_substr: Mapping[str, str | None] = field(default_factory=dict)
    orig_to_new_prefix: Mapping[str, str | None] = field(default_factory=dict)
    orig_to_new_suffix: Mapping[str, str | None] = field(default_factory=dict)

    def __or__(self, other: "WeightsMapper") -> "WeightsMapper":
        return WeightsMapper(
            orig_to_new_regex={**self.orig_to_new_regex, **other.orig_to_new_regex},
            orig_to_new_substr={**self.orig_to_new_substr, **other.orig_to_new_substr},
            orig_to_new_prefix={**self.orig_to_new_prefix, **other.orig_to_new_prefix},
            orig_to_new_suffix={**self.orig_to_new_suffix, **other.orig_to_new_suffix},
        )

    def _map_name(self, key: str) -> str | None:
        for pattern, new_key in self.orig_to_new_regex.items():
            if pattern.search(key):
                if new_key is None:
                    return None
                key = pattern.sub(new_key, key)

        for substr, new_key in self.orig_to_new_substr.items():
            if substr in key:
                if new_key is None:
                    return None
                key = key.replace(substr, new_key, 1)

        for prefix, new_key in self.orig_to_new_prefix.items():
            if key.startswith(prefix):
                if new_key is None:
                    return None
                key = key.replace(prefix, new_key, 1)

        for suffix, new_key in self.orig_to_new_suffix.items():
            if key.endswith(suffix):
                if new_key is None:
                    return None
                key = new_key.join(key.rsplit(suffix, 1))

        return key

    def apply(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, torch.Tensor]]:
        return (
            (out_name, data)
            for name, data in weights
            if (out_name := self._map_name(name)) is not None
        )

    def apply_list(self, values: list[str]) -> list[str]:
        return [
            out_name
            for name in values
            if (out_name := self._map_name(name)) is not None
        ]

    def apply_dict(self, values: dict[str, Any]) -> dict[str, Any]:
        return {
            out_name: value
            for name, value in values.items()
            if (out_name := self._map_name(name)) is not None
        }


class AutoWeightsLoader:
    """Walk a module tree and load weights into it.

    Typical use from a top-level model::

        def load_weights(self, weights):
            loader = AutoWeightsLoader(
                self,
                skip_prefixes=["lm_head."] if self.config.tie_word_embeddings else None,
            )
            return loader.load_weights(weights)

    Submodules with non-trivial loading (fused linear, MoE) just expose their
    own ``weight_loader`` on each parameter — the walker will find it.
    A submodule may override the whole loader by defining ``load_weights``;
    the walker defers to it for everything under that subtree.
    """

    # Models trained on early ColossalAI versions (or quantized via GPTQModel)
    # ship these tensors in the checkpoint. They are not parameters of the
    # current sglang implementations and should be dropped silently.
    ROTARY_EMBEDS_UNUSED_WEIGHTS = (
        "rotary_pos_emb.inv_freq",
        "rotary_emb.inv_freq",
        "rotary_emb.cos_cached",
        "rotary_emb.sin_cached",
    )

    def __init__(
        self,
        module: nn.Module,
        *,
        skip_prefixes: list[str] | None = None,
        skip_substrs: list[str] | None = None,
        ignore_unexpected_prefixes: list[str] | None = None,
        ignore_unexpected_suffixes: list[str] | None = None,
    ) -> None:
        self.module = module
        self.skip_prefixes = list(skip_prefixes or [])
        self.skip_substrs = list(skip_substrs or [])
        self.ignore_unexpected_prefixes = list(ignore_unexpected_prefixes or [])
        self.ignore_unexpected_suffixes = list(ignore_unexpected_suffixes or [])
        self.skip_substrs.extend(self.ROTARY_EMBEDS_UNUSED_WEIGHTS)

    @staticmethod
    def _get_qualname(prefix: str, rest: str) -> str:
        if prefix == "":
            return rest
        if rest == "":
            return prefix
        return f"{prefix}.{rest}"

    def _can_skip(self, qualname: str) -> bool:
        return any(qualname.startswith(p) for p in self.skip_prefixes) or any(
            substr in qualname for substr in self.skip_substrs
        )

    def _can_ignore_unexpected(self, qualname: str) -> bool:
        return any(
            qualname.startswith(p) for p in self.ignore_unexpected_prefixes
        ) or any(qualname.endswith(s) for s in self.ignore_unexpected_suffixes)

    def _groupby_prefix(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[tuple[str, Iterable[tuple[str, torch.Tensor]]]]:
        weights_by_parts = ((name.split(".", 1), data) for name, data in weights)
        for prefix, group in itertools.groupby(weights_by_parts, key=lambda x: x[0][0]):
            yield (
                prefix,
                (("" if len(parts) == 1 else parts[1], data) for parts, data in group),
            )

    def _load_param(
        self,
        base_prefix: str,
        param: nn.Parameter,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        for weight_name, weight_data in weights:
            weight_qualname = self._get_qualname(base_prefix, weight_name)

            if self._can_skip(weight_qualname):
                logger.debug("Skipping weight %s", weight_qualname)
                continue

            if weight_name != "":
                if self._can_ignore_unexpected(weight_qualname):
                    logger.debug("Ignoring weight %s", weight_qualname)
                    continue
                raise ValueError(
                    f"Attempted to load nested weight {weight_qualname!r} "
                    f"into a single parameter {base_prefix!r}"
                )

            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, weight_data)
            logger.debug("Loaded weight %s with shape %s", weight_qualname, param.shape)
            yield weight_qualname

    def _load_module(
        self,
        base_prefix: str,
        module: nn.Module,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> Iterable[str]:
        # Pipeline-parallel: every weight that lands on a missing rank should be
        # silently dropped. The walker is the single place that knows this.
        if isinstance(module, PPMissingLayer):
            for _ in weights:
                pass
            return

        if module is not self.module:
            module_load_weights = getattr(module, "load_weights", None)
            if callable(module_load_weights):
                loaded = module_load_weights(weights)
                if loaded is None:
                    logger.debug(
                        "Module %s.load_weights returned None; loaded-name "
                        "tracking disabled for this subtree",
                        base_prefix,
                    )
                    return
                for name in loaded:
                    yield self._get_qualname(base_prefix, name)
                return

        # Multi-input fusion: collect the inputs that feed each declared
        # fusion under this module before they're routed to a child module.
        # ``input_to_fusion[child_name]`` is the fusion that consumes that
        # checkpoint child; ``buffer[(target, sub_name)][input_name]`` holds
        # tensors waiting on their siblings.
        fusions: list[MultiInputFusion] = list(
            getattr(module, "checkpoint_fusions", ())
        )
        input_to_fusion: dict[str, MultiInputFusion] = {}
        for fusion in fusions:
            for inp in fusion.inputs:
                input_to_fusion[inp] = fusion
        fusion_buffers: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

        child_modules = dict(module.named_children())
        child_params = dict(module.named_parameters(recurse=False))
        child_buffers = dict(module.named_buffers(recurse=False))

        for child_prefix, child_weights in self._groupby_prefix(weights):
            prefix = self._get_qualname(base_prefix, child_prefix)

            if child_prefix in input_to_fusion:
                fusion = input_to_fusion[child_prefix]
                # Drain this child's weights into the buffer keyed by the
                # trailing checkpoint name (``weight``, ``weight_scale``, …).
                for sub_name, tensor in child_weights:
                    if self._can_skip(self._get_qualname(prefix, sub_name)):
                        continue
                    bucket = fusion_buffers.setdefault((fusion.target, sub_name), {})
                    bucket[child_prefix] = tensor
                continue

            if child_prefix in child_modules:
                if self._can_skip(prefix + "."):
                    logger.debug("Skipping module %s", prefix)
                    for _ in child_weights:
                        pass
                    continue
                yield from self._load_module(
                    prefix, child_modules[child_prefix], child_weights
                )
            elif child_prefix in child_params:
                if self._can_skip(prefix):
                    logger.debug("Skipping param %s", prefix)
                    for _ in child_weights:
                        pass
                    continue
                yield from self._load_param(
                    prefix, child_params[child_prefix], child_weights
                )
            elif child_prefix in child_buffers:
                if self._can_skip(prefix):
                    logger.debug("Skipping buffer %s", prefix)
                    for _ in child_weights:
                        pass
                    continue
                yield from self._load_param(
                    prefix, child_buffers[child_prefix], child_weights
                )
            else:
                if self._can_skip(prefix + ".") or self._can_skip(prefix):
                    logger.debug("Skipping missing %s", prefix)
                    for _ in child_weights:
                        pass
                    continue
                if self._can_ignore_unexpected(
                    prefix + "."
                ) or self._can_ignore_unexpected(prefix):
                    logger.debug("Ignoring missing %s", prefix)
                    for _ in child_weights:
                        pass
                    continue

                # Drain child_weights so the outer generator doesn't deadlock,
                # then surface a useful error with what *is* available.
                for _ in child_weights:
                    pass
                desc_param_keys = sorted(
                    self._get_qualname(base_prefix, k)
                    for k, _ in module.named_parameters(recurse=True)
                )
                raise ValueError(
                    f"There is no module or parameter named {prefix!r} "
                    f"in {self.module.__class__.__name__}. The available "
                    f"parameters belonging to {base_prefix!r} "
                    f"({module.__class__.__name__}) are: {desc_param_keys}"
                )

        # Drain any complete fusion buffers into the target's parameter.
        # Incomplete fusions are an error: we'd otherwise leave the runtime
        # parameter half-loaded with no warning.
        for fusion in fusions:
            target_module = child_modules.get(fusion.target)
            for (target, sub_name), parts in list(fusion_buffers.items()):
                if target != fusion.target:
                    continue
                missing = [n for n in fusion.inputs if n not in parts]
                if missing:
                    raise ValueError(
                        f"MultiInputFusion {fusion.inputs!r} → "
                        f"{self._get_qualname(base_prefix, fusion.target)}.{sub_name} "
                        f"is missing checkpoint inputs {missing!r}; "
                        f"only got {sorted(parts)}"
                    )
                fused = fusion.combine(module, parts)
                target_qualname = self._get_qualname(base_prefix, fusion.target)
                if target_module is None:
                    # The fused-target module has been swapped for a
                    # PPMissingLayer (or doesn't exist on this rank); drop.
                    logger.debug(
                        "Skipping fusion %s (no target module on this rank)",
                        target_qualname,
                    )
                    continue
                if isinstance(target_module, PPMissingLayer):
                    logger.debug("Skipping fusion %s (PPMissingLayer)", target_qualname)
                    continue
                # Resolve ``sub_name`` (e.g. ``weight``) on the target module
                # and call its weight_loader.
                target_param = getattr(target_module, sub_name, None)
                if target_param is None:
                    raise ValueError(
                        f"MultiInputFusion target {target_qualname}.{sub_name} "
                        f"does not exist on {target_module.__class__.__name__}"
                    )
                weight_loader = getattr(
                    target_param, "weight_loader", default_weight_loader
                )
                weight_loader(target_param, fused)
                yield self._get_qualname(target_qualname, sub_name)
                fusion_buffers.pop((target, sub_name), None)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
        *,
        mapper: WeightsMapper | None = None,
        run_post_load: bool = True,
    ) -> set[str]:
        # Quirks registered via :class:`WeightQuirkRegistry` for this
        # architecture extend the caller-supplied mapper. This is where things
        # like deepseek's ``q_a_proj → fused_qkv_a_proj_with_mqa`` rename get
        # composed in without the model's own ``load_weights`` having to know.
        mapper = WeightQuirkRegistry.compose(self.module, mapper)
        weights = mapper.apply(weights)

        # Filter top-level skips first so we don't even group them.
        weights = (
            (name, weight) for name, weight in weights if not self._can_skip(name)
        )
        loaded = set(self._load_module("", self.module, weights))

        if run_post_load:
            run_post_load_weights(self.module, loaded)
        return loaded


# ---------------------------------------------------------------------------
# Post-load hook (RFC §5)
# ---------------------------------------------------------------------------


def run_post_load_weights(
    module: nn.Module, loaded_shards: set[str], *, base_prefix: str = ""
) -> None:
    """Walk ``module`` children-first and call any ``post_load_weights`` hook.

    Hooks are invoked with ``(self, loaded_shards_for_this_subtree)`` where
    the second argument is the subset of ``loaded_shards`` whose qualified
    name starts with the module's prefix (so a layer hook only sees the
    shards that landed in its layer). Children are visited before the parent
    — this matches ``torch.nn.Module._load_from_state_dict`` order and lets
    higher-level hooks observe child-derived state.

    A module that takes no kwargs and only wants the parameter-changed
    notification can declare ``def post_load_weights(self):`` — the helper
    falls back to the no-arg form if the kwarg form raises ``TypeError`` on a
    keyword binding.
    """
    # Children-first. Only descend into modules that actually exist on this
    # rank (PPMissingLayer has no parameters and nothing meaningful to do).
    for child_name, child in module.named_children():
        if isinstance(child, PPMissingLayer):
            continue
        run_post_load_weights(
            child,
            loaded_shards,
            base_prefix=f"{base_prefix}.{child_name}" if base_prefix else child_name,
        )

    hook = getattr(module, "post_load_weights", None)
    if not callable(hook):
        return
    # Pre-filter: drop shards that don't belong to this subtree so a layer-
    # local hook doesn't have to re-filter on every call.
    if base_prefix:
        prefix_dot = base_prefix + "."
        local_shards = {s for s in loaded_shards if s.startswith(prefix_dot)}
    else:
        local_shards = loaded_shards
    try:
        hook(loaded_shards=local_shards)
    except TypeError:
        # Hook doesn't accept the kwarg — fall back to the no-arg form.
        hook()
