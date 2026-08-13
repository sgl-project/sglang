"""AoH sidecar configuration and validation helpers.

The offline effective-rank analysis produces a small JSON sidecar instead of
changing model weights.  The runtime only needs the per-layer decision for
each global KV group.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_VALID_MODES = frozenset(("retrieval", "streaming"))


def get_aoh_cacheable_prefix_len(sequence_len: int, sink_size: int) -> int:
    """Return the AoH prefix that can safely be shared by a radix tree.

    Streaming-layer KV after the anchor is request-private: the middle is
    reclaimed and the tail moves as decoding progresses.  Keeping the radix
    tree to the permanent anchor avoids retaining a stale SWA mapping.
    """
    return min(sequence_len, sink_size)


def get_aoh_kv_group(
    *, total_kv_heads: int, kv_tp_size: int, kv_tp_rank: int, local_kv_heads: int
) -> int:
    """Return the first global KV group owned by an attention-TP rank."""
    if total_kv_heads <= kv_tp_size:
        return kv_tp_rank // (kv_tp_size // total_kv_heads)
    return kv_tp_rank * local_kv_heads


@dataclass(frozen=True)
class AoHConfig:
    """Validated AoH v1 sidecar.

    Expected JSON shape::

        {
          "version": 1,
          "layers": {
            "3": ["streaming", "retrieval"],
            "7": ["retrieval", "streaming"]
          }
        }

    Each list is indexed by the global KV-head group.  For Qwen3.6-35B-A3B
    there are two entries per full-attention layer.
    """

    layer_modes: dict[int, tuple[str, ...]]

    @classmethod
    def load(cls, path: str) -> AoHConfig:
        config_path = Path(path)
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ValueError(f"AoH config does not exist: {config_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid AoH JSON in {config_path}: {exc}") from exc

        if not isinstance(raw, dict) or raw.get("version") != 1:
            raise ValueError("AoH config must be a JSON object with version=1.")
        raw_layers = raw.get("layers")
        if not isinstance(raw_layers, dict) or not raw_layers:
            raise ValueError("AoH config must contain a non-empty 'layers' object.")

        layer_modes: dict[int, tuple[str, ...]] = {}
        for raw_layer_id, raw_modes in raw_layers.items():
            try:
                layer_id = int(raw_layer_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"AoH layer id {raw_layer_id!r} is not an integer."
                ) from exc
            if layer_id < 0 or not isinstance(raw_modes, list) or not raw_modes:
                raise ValueError(
                    f"AoH layer {raw_layer_id!r} must map to a non-empty mode list."
                )
            modes = tuple(raw_modes)
            invalid = set(modes).difference(_VALID_MODES)
            if invalid:
                raise ValueError(
                    f"AoH layer {layer_id} has invalid modes {sorted(invalid)}; "
                    "expected 'retrieval' or 'streaming'."
                )
            layer_modes[layer_id] = modes

        return cls(layer_modes=layer_modes)

    def mode_for(self, layer_id: int, kv_group: int) -> str:
        try:
            modes = self.layer_modes[layer_id]
        except KeyError as exc:
            raise ValueError(
                f"AoH config is missing full-attention layer {layer_id}."
            ) from exc
        if kv_group < 0 or kv_group >= len(modes):
            raise ValueError(
                f"AoH layer {layer_id} has {len(modes)} KV-group modes, "
                f"but rank owns KV group {kv_group}."
            )
        return modes[kv_group]
