# SPDX-License-Identifier: Apache-2.0
"""VDN-H3 hybrid attention architecture config (window softmax + linear branch)."""

from typing import Any

import msgspec

VDN_H3_DELTA_RULES = ("vdn_solve", "sana_scaled", "vdn_scaled")
VDN_H3_BRIDGE_MODES = ("alpha", "none")
VDN_H3_ANCHOR_FRAME_MODES = ("none", "columns", "rows", "both")
VDN_H3_SHORT_CONV_TARGETS = ("q", "k", "v")


class VDNHybridAttentionArchConfig(msgspec.Struct):
    """VDN-H3 hybrid attention (window softmax + frame-wise linear branch); the
    resolved ``hybrid_attention`` transform config the overlay copies into
    ``transformer/config.json``. A dense checkpoint has none."""

    # frame t is in chunk t // chunk and attends chunks [c - radius, c + radius];
    # chunk 0 means a centered frame window
    chunk: int = 5
    radius: int = 1
    # "both": frames 0 and F-1 dense as rows and columns, so the branch skips them
    anchor_frames: str = "both"
    enable_softmax_gate: bool = True
    delta_rule: str = "vdn_solve"
    linear_head_dim: int = 128
    bridge: str = "alpha"
    a_fp32: bool = True
    enable_text_state: bool = True
    short_conv: tuple[str, ...] = ("k", "v")

    def __post_init__(self) -> None:
        if self.delta_rule not in VDN_H3_DELTA_RULES:
            raise ValueError(
                f"hybrid_attention.delta_rule={self.delta_rule!r}; expected one of "
                f"{VDN_H3_DELTA_RULES}"
            )
        if self.bridge not in VDN_H3_BRIDGE_MODES:
            raise ValueError(
                f"hybrid_attention.bridge={self.bridge!r}; expected one of "
                f"{VDN_H3_BRIDGE_MODES}"
            )
        if self.anchor_frames not in VDN_H3_ANCHOR_FRAME_MODES:
            raise ValueError(
                f"hybrid_attention.anchor_frames={self.anchor_frames!r}; expected "
                f"one of {VDN_H3_ANCHOR_FRAME_MODES}"
            )
        if any(t not in VDN_H3_SHORT_CONV_TARGETS for t in self.short_conv) or len(
            set(self.short_conv)
        ) != len(self.short_conv):
            raise ValueError(
                f"hybrid_attention.short_conv={self.short_conv!r}; expected a "
                f"distinct subset of {VDN_H3_SHORT_CONV_TARGETS}"
            )
        if self.chunk < 0 or self.radius < 0:
            raise ValueError("hybrid_attention.chunk and radius must be >= 0")
        if self.linear_head_dim <= 0:
            raise ValueError("hybrid_attention.linear_head_dim must be positive")

    @classmethod
    def from_transform_config(
        cls, config: dict[str, Any]
    ) -> "VDNHybridAttentionArchConfig":
        """Build from VDN's nested v2 transform config."""
        soft = dict(config.get("softmax_attention", {}))
        lin = dict(config.get("linear_attention", {}))
        short_conv = lin.get("short_conv", {"targets": []})
        targets = (
            short_conv.get("targets", [])
            if isinstance(short_conv, dict)
            else list(short_conv or [])
        )
        return cls(
            chunk=int(soft.get("chunk", 0)),
            radius=int(soft["radius"]),
            anchor_frames=str(config.get("anchor_frames", "none")),
            enable_softmax_gate=bool(config.get("enable_softmax_gate", True)),
            delta_rule=str(lin.get("delta_rule", "vdn_solve")),
            linear_head_dim=int(lin["linear_head_dim"]),
            bridge=str(lin.get("bridge", "alpha")),
            a_fp32=bool(lin.get("a_fp32", True)),
            enable_text_state=bool(lin.get("enable_text_state", False)),
            short_conv=tuple(targets),
        )

    def window_bounds(self, num_frames: int) -> list[tuple[int, int]]:
        """Per-frame inclusive softmax-window bounds [lo, hi], unclamped."""
        if self.chunk <= 0:
            return [(t - self.radius, t + self.radius) for t in range(num_frames)]
        return [
            (
                ((t // self.chunk) - self.radius) * self.chunk,
                ((t // self.chunk) + self.radius + 1) * self.chunk - 1,
            )
            for t in range(num_frames)
        ]

    def full_cover(self, num_frames: int) -> bool:
        """True when every frame's window already spans the whole clip, i.e.
        the softmax branch IS dense attention and the linear branch is off."""
        return all(
            lo <= 0 and hi >= num_frames - 1
            for lo, hi in self.window_bounds(num_frames)
        )


__all__ = ["VDNHybridAttentionArchConfig"]
