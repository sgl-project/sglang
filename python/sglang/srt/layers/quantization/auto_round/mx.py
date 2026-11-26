# SPDX-License-Identifier: Apache-2.0
"""AutoRound MX config translation.

Both AutoRound MX exporters normalize onto CompressedTensorsConfig, so the
checkpoint metadata is rewritten into a compressed-tensors config here and
every MX kernel decision stays in the compressed-tensors path.
"""

from __future__ import annotations

from typing import Any, Optional

_MX_FORMATS = {"mxfp4-pack-quantized", "mxfp8-quantized"}


def _ct_target(target: str) -> str:
    if target.startswith("re:") or not any(char in target for char in "*+?^$()[]{}|\\"):
        return target
    return f"re:{target}"


def _from_compressed_tensors_config(cls, config: dict[str, Any]):
    """Accept AutoRound-produced CT metadata behind the AutoRound CLI."""
    provider = str(config.get("provider", "")).lower()
    quant_format = str(config.get("format", "")).lower()
    is_mxfp_format = quant_format in _MX_FORMATS or (
        quant_format == "mixed-precision"
        and any(
            group.get("format") in _MX_FORMATS
            for group in config["config_groups"].values()
        )
    )
    if provider and provider not in {"auto-round", "autoround", "auto_round"}:
        raise ValueError(
            "--quantization auto-round accepts compressed-tensors metadata "
            f"only from AutoRound, got provider={provider!r}"
        )
    if not is_mxfp_format:
        raise ValueError(
            "--quantization auto-round accepts compressed-tensors metadata "
            "only for MXFP4/MXFP8 formats"
        )
    from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    delegate = CompressedTensorsConfig.from_config(config)
    obj = cls(
        weight_bits=4,
        group_size=32,
        data_type="mx_fp",
        packing_format="auto_round:llm_compressor",
        mx_delegate=delegate,
    )
    return obj


def _from_native_mxfp_config(cls, config: dict[str, Any]):
    """Translate native AutoRound MX metadata to the existing CT schemes."""
    packing_format = cls.get_from_keys_or(
        config, ["packing_format"], "auto_round:llm_compressor"
    )
    # Only llm_compressor packing is accepted for MX; auto_round:sglang is
    # deliberately not enabled, so it must not be silently relabelled here.
    if packing_format != "auto_round:llm_compressor":
        raise ValueError(
            f"Unsupported packing_format for AutoRound MX: {packing_format!r}, "
            "only 'auto_round:llm_compressor' is supported"
        )
    bits = int(cls.get_from_keys(config, ["bits"]))
    act_bits = int(cls.get_from_keys_or(config, ["act_bits"], bits))
    group_size = int(cls.get_from_keys(config, ["group_size"]))
    act_group_size = int(cls.get_from_keys_or(config, ["act_group_size"], group_size))
    if bits not in (4, 8):
        raise NotImplementedError("AutoRound MX loading supports MXFP4 and MXFP8 only")
    # MXFP4 executes as W4A16 on XPU (moe_grouped_mm_nt_xe20_w4a16 consumes
    # bf16 activations), so an unquantized-activation export is the common case.
    if act_bits not in (bits, 16):
        raise NotImplementedError(
            f"AutoRound MX loading supports act_bits in {{{bits}, 16}}, got {act_bits}"
        )
    if group_size != 32:
        raise NotImplementedError("AutoRound MX loading requires weight group_size=32")
    if act_bits < 16 and act_group_size != 32:
        raise NotImplementedError(
            "AutoRound MX loading requires activation group_size=32"
        )

    def scheme(bits_: int, overrides: Optional[dict[str, Any]] = None):
        overrides = overrides or {}
        local_bits = int(overrides.get("bits", bits_))
        local_act_bits = int(
            overrides.get("act_bits", act_bits if local_bits == bits else local_bits)
        )
        if local_act_bits not in (local_bits, 16) or (
            local_bits == 8 and local_act_bits == 16
        ):
            raise NotImplementedError(
                f"Unsupported AutoRound MXFP{local_bits} activation bits: {local_act_bits}"
            )
        if int(overrides.get("group_size", group_size)) != 32 or (
            local_act_bits < 16
            and int(overrides.get("act_group_size", act_group_size)) != 32
        ):
            raise NotImplementedError("AutoRound MX overrides require group size 32")
        group = {
            "weights": {
                "num_bits": local_bits,
                "type": "float",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
                "dynamic": False,
            },
            "format": (
                "mxfp4-pack-quantized" if local_bits == 4 else "mxfp8-quantized"
            ),
        }
        # Omitted for W4A16: an input_activations entry would declare activation
        # quantization the XPU MXFP4 path does not perform.
        if local_act_bits < 16:
            group["input_activations"] = {
                "num_bits": local_act_bits,
                "type": "float",
                "symmetric": True,
                "strategy": "group",
                "group_size": 32,
                "dynamic": True,
            }
        return group

    ignore = []
    groups = {}
    default = scheme(bits)
    default["targets"] = ["Linear"]
    groups["group_0"] = default
    for index, (target, override) in enumerate(
        (config.get("extra_config") or {}).items(), start=1
    ):
        override_bits = int(override.get("bits", bits))
        override_type = str(override.get("data_type", "mx_fp")).lower()
        if override_bits >= 16 or override_type in {"float", "fp", "bf16"}:
            ignore.append(_ct_target(target))
            continue
        if override_type != "mx_fp" or override_bits not in (4, 8):
            raise NotImplementedError(
                f"Unsupported AutoRound MX override for {target!r}: {override}"
            )
        group = scheme(bits, override)
        group["targets"] = [_ct_target(target)]
        groups[f"group_{index}"] = group

    ct_config = {
        "quant_method": "compressed-tensors",
        "provider": "auto-round",
        "expert_layout": "split",
        "format": default["format"]
        if all(group["format"] == default["format"] for group in groups.values())
        else "mixed-precision",
        "config_groups": groups,
        "ignore": ignore,
        "packed_modules_mapping": config.get("packed_modules_mapping", {}),
    }
    obj = cls._from_compressed_tensors_config(ct_config)
    obj.weight_bits = bits
    obj.group_size = group_size
    return obj
