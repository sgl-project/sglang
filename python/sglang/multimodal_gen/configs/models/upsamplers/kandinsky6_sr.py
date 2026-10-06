# SPDX-License-Identifier: Apache-2.0
"""Config of one Kandinsky 6 SR latent-upscaler bank entry (``latent_upscaler/config.json``).

The component config is ``{"models": [{"target_scale": "2x"|"4x", "model": {...}}, ...],
"scaling_factor": f}``.  Each ``model`` mapping describes one cascaded 2x+2x upsampler.  Only
the architecture of the released checkpoints is implemented (mirrors FastVideo's
``fastvideo/configs/models/upsamplers/kandinsky6_sr.py``), so every key that would select a
different architecture must carry its released value; a value that is not supported raises a
``ValueError`` naming the key instead of silently building something else.

Unlike FastVideo's ``Kandinsky6SRLatentUpscalerConfig``, there is no bank-level config class
here: ``Kandinsky6SRLatentUpscalerBank`` (``runtime/models/upsampler/kandinsky6_sr_latent_upscaler.py``)
takes the bundle's raw ``models`` list and ``scaling_factor`` directly, matching
``LatentUpscalerLoader``'s existing call contract -- only the per-entry validation lives here,
as a ``msgspec.Struct`` (this repo's house rule bans new ``@dataclass``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import msgspec

# key -> (required value, value a config that omits the key stands for).  The omitted-key values
# are the training code's defaults, so e.g. a config without ``input_skip`` means
# ``input_skip=True`` and is rejected.
_FIXED_KEYS: dict[str, tuple[Any, Any]] = {
    "upscale_factor": (4, 4),
    "dims": (3, 2),
    "temporal_padding": ("replicate", "zeros"),
    "upsample_mode": ("pxs_v2", "pixel_shuffle"),
    "upsample_padding_mode": ("zeros", "reflect"),
    "modulated_norm": (True, False),
    "modulated_output_proj": (True, False),
    "bare_stem": (True, False),
    "input_skip": (False, True),
    "global_skip": (False, False),
    "grn": (False, False),
    "layer_scale_init": (None, None),
    "depthwise": (False, False),
    "bottleneck_channels": (None, None),
    "motion_attention": (None, None),
    # Only read by depthwise blocks; the residual convs are always 3x3x3.
    "kernel_size": (3, 3),
}
# Fixed for entries built with the x2 entry (``enable_x2_entry=true``).
_X2_FIXED_KEYS: dict[str, tuple[Any, Any]] = {
    "x2_tail_mode": ("private_full", "shared"),
    "x2_finisher": ("pxs_residual", "none"),
}
# Accepted without effect at inference: training-time settings, and switches that only apply to
# the pixel-shuffle upsample modes (the pxs_v2 upsample has no temporal kernel and no ICNR init).
_IGNORED_KEYS = frozenset(
    {
        "gradient_checkpointing",
        "stochastic_depth_rate",
        "loss_weight_2x",
        "loss_weight_4x",
        "temporal_mix",
        "icnr",
        "upsample_position",
    }
)
_REQUIRED_INT_KEYS = (
    "in_channels",
    "hidden_channels",
    "num_pre_blocks",
    "num_mid_blocks",
    "num_post_blocks",
    "expand_ratio",
)
_SUPPORTED_SCALES = (2, 4)


def _positive_int(where: str, key: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{where}: `{key}` must be a positive integer, got {value!r}")
    return value


def _parse_target_scale(where: str, value: Any) -> int:
    text = str(value).removesuffix("x")
    if (
        isinstance(value, bool)
        or not text.isdigit()
        or int(text) not in _SUPPORTED_SCALES
    ):
        raise ValueError(
            f"{where}: `target_scale` must be one of '2x', '4x', got {value!r}"
        )
    return int(text)


class Kandinsky6SRLatentUpscalerEntryConfig(msgspec.Struct, frozen=True, kw_only=True):
    """One bank entry: a cascaded 2x+2x upsampler serving ``target_scale``.

    ``stage_channels`` are the widths at the 1x, 2x and 4x latent grids.  With
    ``enable_x2_entry`` the entry also has a private x2 path (its own input stem,
    ``x2_adapter_blocks`` residual blocks at 1x, and private copies of the mid stage and second
    stage) that upsamples by 2 instead of 4.
    """

    target_scale: int
    in_channels: int
    hidden_channels: int
    stage_channels: tuple[int, int, int]
    num_pre_blocks: int
    num_mid_blocks: int
    num_post_blocks: int
    expand_ratio: int
    enable_x2_entry: bool = False
    x2_adapter_blocks: int = 0

    @classmethod
    def from_dict(
        cls, spec: Any, index: int = 0
    ) -> Kandinsky6SRLatentUpscalerEntryConfig:
        """Validate one ``models[index]`` entry of ``latent_upscaler/config.json``."""
        where = f"latent upscaler models[{index}]"
        if (
            not isinstance(spec, Mapping)
            or set(spec) != {"target_scale", "model"}
            or not isinstance(spec["model"], Mapping)
        ):
            raise ValueError(
                f"{where} must be a mapping with exactly `target_scale` and a `model` "
                f"mapping, got {spec!r}"
            )
        target_scale = _parse_target_scale(where, spec["target_scale"])
        model = dict(spec["model"])

        if model.get("architecture") != "multi_scale":
            raise ValueError(
                f"{where}: `architecture` must be 'multi_scale', got "
                f"{model.get('architecture')!r}"
            )
        enable_x2_entry = model.get("enable_x2_entry", False)
        if not isinstance(enable_x2_entry, bool):
            raise ValueError(
                f"{where}: `enable_x2_entry` must be a boolean, got {enable_x2_entry!r}"
            )

        fixed = dict(_FIXED_KEYS)
        if enable_x2_entry:
            fixed.update(_X2_FIXED_KEYS)
        for key, (required, omitted) in fixed.items():
            value = model.get(key, omitted)
            if value != required or type(value) is not type(required):
                raise ValueError(
                    f"{where}: unsupported `{key}`={value!r}"
                    f"{' (omitted)' if key not in model else ''}; only {required!r} is "
                    "implemented"
                )

        x2_defaults = {
            "x2_adapter_blocks": 0,
            "x2_adapter_sources": None,
            "x2_tail_mode": "shared",
            "x2_finisher": "none",
        }
        known = {
            "architecture",
            "enable_x2_entry",
            "stage_channels",
            *_REQUIRED_INT_KEYS,
            *fixed,
            *x2_defaults,
            *_IGNORED_KEYS,
        }
        unknown = sorted(set(model) - known)
        if unknown:
            raise ValueError(f"{where}: unknown keys {unknown}")
        if not enable_x2_entry:
            present = sorted(
                key
                for key, default in x2_defaults.items()
                if model.get(key, default) != default
            )
            if present:
                raise ValueError(f"{where}: {present} require `enable_x2_entry`=true")
        if target_scale == 2 and not enable_x2_entry:
            raise ValueError(
                f"{where}: `target_scale`='2x' requires `enable_x2_entry`=true"
            )

        missing = [key for key in _REQUIRED_INT_KEYS if key not in model]
        if missing:
            raise ValueError(f"{where}: missing keys {missing}")
        ints = {
            key: _positive_int(where, key, model[key]) for key in _REQUIRED_INT_KEYS
        }
        hidden = ints["hidden_channels"]

        raw_stages = model.get("stage_channels")
        if raw_stages is None:
            stage_channels = (hidden, hidden, hidden)
        else:
            if not isinstance(raw_stages, (list, tuple)) or len(raw_stages) != 3:
                raise ValueError(
                    f"{where}: `stage_channels` must list 3 widths, got {raw_stages!r}"
                )
            first, second, third = (
                _positive_int(where, "stage_channels", width) for width in raw_stages
            )
            stage_channels = (first, second, third)
            if stage_channels[0] != hidden:
                raise ValueError(
                    f"{where}: `hidden_channels` ({hidden}) must equal `stage_channels[0]` "
                    f"({stage_channels[0]})"
                )

        x2_adapter_blocks = 0
        if enable_x2_entry:
            x2_adapter_blocks = _positive_int(
                where, "x2_adapter_blocks", model.get("x2_adapter_blocks", 0)
            )
            # Training warm-starts the adapter from these pre_blocks; only their consistency
            # matters here.
            sources = model.get("x2_adapter_sources")
            if sources is not None and (
                not isinstance(sources, (list, tuple))
                or len(sources) != x2_adapter_blocks
                or any(
                    not isinstance(i, int) or not 0 <= i < ints["num_pre_blocks"]
                    for i in sources
                )
            ):
                raise ValueError(
                    f"{where}: `x2_adapter_sources` must list {x2_adapter_blocks} indices in "
                    f"[0, num_pre_blocks={ints['num_pre_blocks']}), got {sources!r}"
                )

        return cls(
            target_scale=target_scale,
            stage_channels=stage_channels,
            enable_x2_entry=enable_x2_entry,
            x2_adapter_blocks=x2_adapter_blocks,
            **ints,
        )


__all__ = ["Kandinsky6SRLatentUpscalerEntryConfig"]
