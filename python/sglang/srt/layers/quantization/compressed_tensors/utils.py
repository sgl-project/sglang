# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from types import MappingProxyType
from typing import Any, Callable, Iterable, List, Mapping, Optional

from compressed_tensors import CompressionFormat
from torch.nn import Module

from sglang.srt.layers.quantization.utils import _FALLBACK_FUSED_SHARDS


def is_activation_quantization_format(format: str) -> bool:
    _ACTIVATION_QUANTIZATION_FORMATS = [
        CompressionFormat.naive_quantized.value,
        CompressionFormat.int_quantized.value,
        CompressionFormat.float_quantized.value,
        CompressionFormat.nvfp4_pack_quantized.value,
        CompressionFormat.mxfp4_pack_quantized.value,
        CompressionFormat.pack_quantized.value,
        CompressionFormat.mxfp8_quantized.value,
    ]
    return format in _ACTIVATION_QUANTIZATION_FORMATS


def should_ignore_layer(
    layer_name: Optional[str],
    ignore: Iterable[str] = tuple(),
    fused_mapping: Mapping[str, List[str]] = MappingProxyType({}),
) -> bool:
    if layer_name is None:
        return False

    # layer_name = model.layers.0.self_attn.qkv_proj
    # proj_name = qkv_proj
    proj_name = layer_name.split(".")[-1]

    # Some checkpoints store a real qkv_proj / gate_up_proj, so an entry that
    # names the fused layer itself wins.
    if check_equal_or_regex_match(
        layer_name=layer_name, targets=[t for t in ignore if not t.startswith("re:")]
    ):
        return True
    fused_name_ignored = check_equal_or_regex_match(
        layer_name=layer_name, targets=ignore
    )

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    shard_proj_names = fused_mapping.get(proj_name)
    if shard_proj_names is None:
        # The model declares no fusion, so its checkpoint may store the fused
        # tensor (MiMoV2 Pro's qkv_proj): a regex on the fused name decides.
        if fused_name_ignored:
            return True
        shard_proj_names = _FALLBACK_FUSED_SHARDS.get(proj_name)

    # Unfused layers like down_proj and o_proj will match
    # the safetensors checkpoint already.
    if shard_proj_names is None:
        return False

    # Convert fused_name --> [shard_names]
    shard_names = [
        layer_name.replace(proj_name, shard_proj_name)
        for shard_proj_name in shard_proj_names
    ]
    shards_ignored = {
        check_equal_or_regex_match(layer_name=shard_name, targets=ignore)
        or check_equal_or_regex_match(layer_name=f"{shard_name}.linear", targets=ignore)
        for shard_name in shard_names
    }

    # Reject a list that covers only some shards. With a declared fusion this
    # holds even if a regex also matches the fused name: `re:.*v_proj$` matches
    # qkv_proj too, but the checkpoint's q/k shards are packed.
    if len(shards_ignored) > 1:
        raise _mixed_shard_schemes_error(shard_proj_names, layer_name)
    return fused_name_ignored or shards_ignored.pop()


def _mixed_shard_schemes_error(shards: List[str], layer_name: str) -> ValueError:
    return ValueError(
        f"Found different quantization schemes for {shards} in {layer_name}. "
        "SGLang requires all to use the same scheme."
    )


def check_equal_or_regex_match(layer_name: str, targets: Iterable[str]) -> bool:
    """
    Checks whether a layer_name is exactly equal or a regex match for
    if target starts with 're:' to any target in list.

    A plain (non-regex) target additionally matches a dotted-path *suffix* of
    layer_name, so a target may be written as a module suffix, e.g.
    "self_attn.kv_b_proj" matches "model.layers.0.self_attn.kv_b_proj".

    It must never match a *prefix*: llm-compressor writes parent modules into
    the `ignore` list (e.g. "model.layers.0.mlp.experts.0"), and a plain
    substring match would let such an entry silently swallow its quantized
    children ("model.layers.0.mlp.experts.0.gate_proj"), dropping the layer
    back to an unquantized method.
    """
    for target in targets:
        if target.startswith("re:"):
            if _is_equal_or_regex_match(layer_name, target):
                return True
        elif target == layer_name or layer_name.endswith("." + target):
            return True
    return False


def find_matched_target(
    layer_name: Optional[str],
    module: Module,
    targets: Iterable[str],
    fused_mapping: Mapping[str, List[str]] = MappingProxyType({}),
    scheme_key: Optional[Callable[[str], Any]] = None,
) -> str:
    """
    Helper function to look up which "target" in the compressed-tensors
    config that a layer corresponds to.

    Recall that a compressed-tensors configs has a concept of
    config_groups, where each layer can be quantized with with a different
    scheme.

    targets in each config_group will be a list of either layer names
    (or regexes corresponding to layer names) or names of torch Modules.

    First, we try to match the layer_name with a target
    Second, we try to match the module's name with a target
    Third, we try to map the layer_name to a list of fused module names.
        *All* component module names must match in order for a match to be
        successful. A successful match returns the first component target

    :param layer_name: layer name
    :param module: torch.nn.Module
    :param targets: list of targets to match the layer against
    :param fused_mapping: map from fused layer names to its components
    :param scheme_key: target -> what the loader builds from it; when given,
        the components of a fused layer must resolve to the same key
    :param fused_strategy: either "all" or "any". If using "all", fused
        layers match if "all" of its components match
    """

    if layer_name is None:
        layer_name = ""

    matched_target = (
        _find_first_match(layer_name, targets)
        or _find_first_match(module.__class__.__name__, targets, True)
        or _match_fused_layer(
            layer_name=layer_name,
            target_layers=targets,
            fused_mapping=fused_mapping,
            scheme_key=scheme_key,
        )
    )

    if matched_target is None:
        raise ValueError(
            f"Unable to find matching target for {layer_name} in the "
            "compressed-tensors config."
        )

    return matched_target


def _find_first_match(
    value: str, targets: Iterable[str], check_contains: bool = False
) -> Optional[str]:
    """
    Returns first element of target that matches value either
    exactly or as a regex after 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.

    :param value: string to compare the list of targets against
    :param targets: list of targets to match the layer against
    :param check_contains: whether or not to do a substring match
    """

    for target in targets:
        if _is_equal_or_regex_match(value, target, check_contains=check_contains):
            return target
    return None


def _is_equal_or_regex_match(
    value: str, target: str, check_contains: bool = False
) -> bool:
    """
    Checks whether a value is exactly equal or a regex match for target
    if target starts with 're:'. If check_contains is set to True,
    additionally checks if the target string is contained within the value.
    """

    if target.startswith("re:"):
        pattern = target[3:]
        if re.match(pattern, value):
            return True
    elif check_contains:
        if target.lower() in value.lower():
            return True
    elif target == value:
        return True
    return False


def _match_fused_layer(
    layer_name: str,
    target_layers: Iterable[str],
    fused_mapping: Mapping[str, List[str]],
    scheme_key: Optional[Callable[[str], Any]] = None,
) -> Optional[str]:
    """
    Match a fused layer name to its corresponding individual layer in
    target_layers. Returns first value in fused_mapping which matches targets

    Implements an "all" matching strategy where a fused layer matches iff
    "all" of its components match

    :param layer_name: layer name
    :param target_layers: list of targets to match the layer against
    :param fused_mapping: map from fused layer names to its components

    Examples:
        layer_name = "model.layers.0.self_attn.qkv_proj"
        target_layers = ["model.layers.0.self_attn.q_proj",
                        "model.layers.0.self_attn.k_proj",
                        "model.layers.0.self_attn.v_proj"]
    """
    # find layer_name in mapping
    fused = next((key for key in fused_mapping if layer_name.endswith(key)), None)
    if fused is None:
        return None

    # expand path of unfused components
    unfused_paths = [
        layer_name.replace(fused, unfused) for unfused in fused_mapping[fused]
    ]

    # for each unfused component, find a match in targets
    unfused_matches: List[Optional[str]] = []
    for unfused in unfused_paths:
        for target in target_layers:
            if _is_equal_or_regex_match(unfused, target):
                unfused_matches.append(target)
                break
        else:
            unfused_matches.append(None)

    if not all(unfused_matches):
        return None

    # Exact per-shard targets can give the shards different schemes (FP8 q/k,
    # INT4 v); taking the first one would build v with q's scheme.
    if scheme_key is not None and any(
        scheme_key(match) != scheme_key(unfused_matches[0])
        for match in unfused_matches[1:]
    ):
        raise _mixed_shard_schemes_error(fused_mapping[fused], layer_name)
    return unfused_matches[0]
