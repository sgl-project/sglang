# Adapted from https://github.com/vllm-project/vllm/tree/main/vllm/model_executor/layers/quantization/compressed_tensors
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from types import MappingProxyType
from typing import Iterable, List, Mapping, Optional

from compressed_tensors import CompressionFormat
from torch.nn import Module


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

    # Fused layers like gate_up_proj or qkv_proj will not be fused
    # in the safetensors checkpoint. So, we convert the name
    # from the fused version to unfused + check to make sure that
    # each shard of the fused layer has the same scheme.
    if proj_name in fused_mapping and layer_name not in ignore:
        shard_proj_names = fused_mapping[proj_name]

        # Convert fused_name --> [shard_names]
        shard_names = [
            layer_name.replace(proj_name, shard_proj_name)
            for shard_proj_name in shard_proj_names
        ]

        # Layer should be ignored if shards are ignored.
        should_ignore_layer = None
        for shard_name in shard_names:
            should_ignore_shard = check_equal_or_regex_match(
                layer_name=shard_name, targets=ignore
            )
            if not should_ignore_shard:
                should_ignore_shard = check_equal_or_regex_match(
                    layer_name=f"{shard_name}.linear", targets=ignore
                )

            # If shard_idx=0, set layer ignore to match shard.
            if should_ignore_layer is None:
                should_ignore_layer = should_ignore_shard

            # If shard_idx=1+ confirm scheme matches prior shards.
            elif should_ignore_shard != should_ignore_layer:
                raise ValueError(
                    f"Found different quantization schemes for "
                    f"{shard_proj_names} in {layer_name}. SGLang "
                    "requires all to use the same scheme."
                )

        if not should_ignore_layer:
            should_ignore_layer = check_equal_or_regex_match(
                layer_name=layer_name, targets=ignore
            )

    # Unfused layers like down_proj and o_proj will match
    # the safetensors checkpoint already.
    else:
        should_ignore_layer = check_equal_or_regex_match(
            layer_name=layer_name, targets=ignore
        )

    assert should_ignore_layer is not None
    return should_ignore_layer


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


def _match_fused_shard_targets(
    layer_name: str,
    module: Module,
    target_list: List[str],
    fused_mapping: Mapping[str, List[str]],
) -> Optional[List[str]]:
    fused = next(
        (
            key
            for key in fused_mapping
            if layer_name == key or layer_name.endswith("." + key)
        ),
        None,
    )
    if fused is None:
        return None
    unfused_paths = [
        layer_name.replace(fused, unfused) for unfused in fused_mapping[fused]
    ]
    shard_name_matches = [
        _find_first_match(unfused, target_list) for unfused in unfused_paths
    ]
    if not any(match is not None for match in shard_name_matches):
        return None
    class_match = _find_first_match(module.__class__.__name__, target_list, True)
    resolved = [
        match if match is not None else class_match for match in shard_name_matches
    ]
    if all(match is not None for match in resolved):
        return resolved
    raise ValueError(
        f"Unable to find matching targets for all fused components of "
        f"{layer_name} ({fused_mapping[fused]}) in the "
        "compressed-tensors config."
    )


def find_matched_targets(
    layer_name: Optional[str],
    module: Module,
    targets: Iterable[str],
    fused_mapping: Mapping[str, List[str]] = MappingProxyType({}),
) -> List[str]:
    if layer_name is None:
        layer_name = ""

    target_list = list(targets)
    fused_matches = _match_fused_shard_targets(
        layer_name=layer_name,
        module=module,
        target_list=target_list,
        fused_mapping=fused_mapping,
    )
    if fused_matches is not None:
        return fused_matches

    direct_match = _find_first_match(layer_name, target_list) or _find_first_match(
        module.__class__.__name__, target_list, True
    )
    if direct_match is not None:
        return [direct_match]

    raise ValueError(
        f"Unable to find matching target for {layer_name} in the "
        "compressed-tensors config."
    )


def find_matched_target(
    layer_name: Optional[str],
    module: Module,
    targets: Iterable[str],
    fused_mapping: Mapping[str, List[str]] = MappingProxyType({}),
) -> str:
    return find_matched_targets(
        layer_name=layer_name,
        module=module,
        targets=targets,
        fused_mapping=fused_mapping,
    )[0]


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
