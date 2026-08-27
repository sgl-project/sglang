"""Residue NVFP4 metadata contract.

A residue checkpoint is a stock ModelOpt NVFP4 export plus a
``residue_kernel_metadata.json`` file in the model directory. This module is
the single validated reader for that file. The runtime never inspects the
JSON directly and never infers a layer's residue representation from tensor
shapes or from ``num_salient == 0`` (that value historically meant both
"plain NVFP4" and "mext_r1", which produced silent-wrong-numerics bugs in the
reference implementation; here the representation is always an explicit enum).

Supported representations:

- ``k_ext`` (on-disk ``runtime_mode: "extended_k"``): selective residue for
  ratios 1/8, 2/8, 4/8. The weight is stored K-extended
  (``K_ext = K_base + num_salient``) and the activation quantizer appends the
  selected residue channels.
- ``mext_r1`` (on-disk ``runtime_mode: "mext_r1"``): full-rank ratio 1.0. The
  weight stays at ``K_base``; the activation quantizer doubles the token rows
  and a fold GEMM sums both contributions.

Anything else in the file that would change runtime behavior we do not
support (MoE residue, unknown modes, unsupported ratios) fails at load time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

METADATA_FILENAME = "residue_kernel_metadata.json"

# Salient channels are selected top-k within blocks of this many channels.
# The k_ext quant kernels and the TP sharding rules both assume it.
SALIENT_BLOCK = 8

# ratio -> salient channels per SALIENT_BLOCK. Ratio 1.0 is deliberately not
# here: full-rank residue is only served as mext_r1 (K-extension at ratio 1.0
# would store the weight twice and was removed from the reference kernels).
SUPPORTED_K_EXT_RATIOS: Mapping[float, int] = {
    0.125: 1,
    0.25: 2,
    0.5: 4,
}


class ResidueMetadataError(ValueError):
    """The residue metadata file is missing required data, self-contradictory,
    or requests something this integration does not support."""


class ResidueMode(Enum):
    """How a dense linear layer represents its residue. Values match the
    on-disk ``runtime_mode`` strings."""

    K_EXT = "extended_k"
    MEXT_R1 = "mext_r1"


@dataclass(frozen=True)
class ResidueLayerSpec:
    """Validated residue description of one dense linear layer.

    ``salient_indices`` is populated for K_EXT only. A MEXT_R1 layer's salient
    set is by definition every input channel; storing it would be redundant.
    """

    name: str
    mode: ResidueMode
    k_base: int
    num_salient: int
    salient_indices: tuple[int, ...] = field(default=())

    @property
    def k_ext(self) -> int:
        """Extended K consumed by the k_ext GEMM. Equals k_base for mext_r1."""
        if self.mode is ResidueMode.MEXT_R1:
            return self.k_base
        return self.k_base + self.num_salient

    @property
    def ratio(self) -> float:
        return self.num_salient / self.k_base

    @property
    def residue_per_block(self) -> int:
        """Salient channels per SALIENT_BLOCK ("residue_per_8" in the kernels)."""
        if self.mode is ResidueMode.MEXT_R1:
            return SALIENT_BLOCK
        return SUPPORTED_K_EXT_RATIOS[self.ratio]


# vLLM/SGLang fuse these HF projections into one linear; the metadata is
# written against HF names, so a fused layer resolves through its parts.
_FUSED_SUFFIXES: Mapping[str, tuple[str, ...]] = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
    # Qwen3.5/Qwen3-Next Gated DeltaNet checkpoints serialize these as two
    # HF tensors, while the SGLang runtime executes one fused projection.
    "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
}


def layer_name_candidates(layer_name: str) -> list[str]:
    """Metadata lookup candidates for a possibly-fused layer name."""
    candidates = [layer_name]
    for fused_suffix, part_suffixes in _FUSED_SUFFIXES.items():
        if layer_name.endswith(fused_suffix):
            base = layer_name[: -len(fused_suffix)]
            candidates.extend(base + part for part in part_suffixes)
    return candidates


@dataclass(frozen=True)
class ResidueModelSpec:
    """All residue layers of one checkpoint, keyed by exported layer name."""

    layers: Mapping[str, ResidueLayerSpec]
    block_size: int
    source: str

    def spec_for(self, layer_name: str) -> Optional[ResidueLayerSpec]:
        """Resolve a runtime layer name, including fused-layer fallbacks.

        Fused layers share one activation quantization, so every constituent
        that carries residue metadata must agree on the residue contract; a
        disagreement is an export this integration cannot serve.
        """
        direct = self.layers.get(layer_name)
        if direct is not None:
            return direct

        found: list[ResidueLayerSpec] = [
            self.layers[name]
            for name in layer_name_candidates(layer_name)[1:]
            if name in self.layers
        ]
        if not found:
            return None

        head = found[0]
        for other in found[1:]:
            if (
                other.mode is not head.mode
                or other.k_base != head.k_base
                or other.num_salient != head.num_salient
                or other.salient_indices != head.salient_indices
            ):
                raise ResidueMetadataError(
                    f"fused layer {layer_name!r} resolves to constituents with "
                    f"conflicting residue metadata ({head.name!r} vs "
                    f"{other.name!r}); a fused linear shares one activation "
                    "quantization and cannot serve two contracts"
                )
        return ResidueLayerSpec(
            name=layer_name,
            mode=head.mode,
            k_base=head.k_base,
            num_salient=head.num_salient,
            salient_indices=head.salient_indices,
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResidueMetadataError(message)


def _validate_salient_indices(
    name: str, indices: Sequence[Any], k_base: int
) -> tuple[int, ...]:
    _require(
        all(isinstance(i, int) and not isinstance(i, bool) for i in indices),
        f"layer {name!r}: salient_indices must be integers",
    )
    out = tuple(int(i) for i in indices)
    _require(
        all(0 <= i < k_base for i in out),
        f"layer {name!r}: salient_indices out of range [0, {k_base})",
    )
    _require(
        all(a < b for a, b in zip(out, out[1:])),
        f"layer {name!r}: salient_indices must be strictly increasing "
        "(sorted, no duplicates); the TP sharding contract depends on it",
    )
    return out


def _parse_k_ext_layer(
    name: str, entry: Mapping[str, Any], extended_dim: Any
) -> ResidueLayerSpec:
    _require(
        extended_dim is not None,
        f"layer {name!r}: runtime_mode is 'extended_k' but the layer has no "
        "entry in extended_linear_dims; the stored weight width is unknown",
    )
    _require(
        isinstance(extended_dim, int) and not isinstance(extended_dim, bool),
        f"layer {name!r}: extended_linear_dims entry must be an integer",
    )

    raw_indices = entry.get("salient_indices")
    _require(
        isinstance(raw_indices, list) and len(raw_indices) > 0,
        f"layer {name!r}: extended_k requires a non-empty salient_indices list",
    )
    num_salient = entry.get("num_salient", len(raw_indices))
    _require(
        num_salient == len(raw_indices),
        f"layer {name!r}: num_salient={num_salient} does not match "
        f"len(salient_indices)={len(raw_indices)}",
    )

    k_base = int(extended_dim) - int(num_salient)
    _require(
        k_base > 0 and k_base % 16 == 0,
        f"layer {name!r}: derived K_base={k_base} must be a positive multiple "
        "of 16 (NVFP4 group size)",
    )

    ratio = num_salient / k_base
    supported = ", ".join(f"{r:g}" for r in sorted(SUPPORTED_K_EXT_RATIOS))
    _require(
        any(abs(ratio - r) < 1e-3 for r in SUPPORTED_K_EXT_RATIOS),
        f"layer {name!r}: residue ratio {ratio:.4f} ({num_salient}/{k_base}) "
        f"is not supported; supported extended_k ratios: {supported}. "
        "Ratio 1.0 must be exported as mext_r1.",
    )

    indices = _validate_salient_indices(name, raw_indices, k_base)
    return ResidueLayerSpec(
        name=name,
        mode=ResidueMode.K_EXT,
        k_base=k_base,
        num_salient=int(num_salient),
        salient_indices=indices,
    )


def _parse_mext_r1_layer(
    name: str, entry: Mapping[str, Any], extended_dim: Any
) -> ResidueLayerSpec:
    _require(
        extended_dim is None,
        f"layer {name!r}: runtime_mode is 'mext_r1' but the layer appears in "
        "extended_linear_dims; mext_r1 weights must stay at the original K",
    )
    raw_indices = entry.get("salient_indices")
    _require(
        isinstance(raw_indices, list) and len(raw_indices) > 0,
        f"layer {name!r}: mext_r1 requires salient_indices covering every "
        "input channel (the export writes them; their length defines K)",
    )
    num_salient = entry.get("num_salient", len(raw_indices))
    _require(
        num_salient == len(raw_indices),
        f"layer {name!r}: num_salient={num_salient} does not match "
        f"len(salient_indices)={len(raw_indices)}",
    )
    k_base = int(num_salient)
    _require(
        k_base > 0 and k_base % 16 == 0,
        f"layer {name!r}: K={k_base} must be a positive multiple of 16",
    )
    indices = _validate_salient_indices(name, raw_indices, k_base)
    _require(
        indices == tuple(range(k_base)),
        f"layer {name!r}: mext_r1 means ratio 1.0, so salient_indices must be "
        "exactly every channel 0..K-1",
    )
    return ResidueLayerSpec(
        name=name,
        mode=ResidueMode.MEXT_R1,
        k_base=k_base,
        num_salient=k_base,
    )


def _validate_moe_section(moe: Any) -> None:
    """Residue MoE is out of scope. A checkpoint whose MoE policy is entirely
    'off' is fine (the experts are plain NVFP4); anything else is a format we
    must refuse rather than silently serve without its residue."""
    _require(isinstance(moe, dict), "'moe' section must be an object")
    layers = moe.get("layers")
    _require(isinstance(layers, dict), "'moe' section must contain 'layers'")
    active = sorted(
        name
        for name, policy in layers.items()
        if not (isinstance(policy, dict) and policy.get("impl") == "off")
    )
    _require(
        not active,
        "checkpoint requests residue MoE, which this integration does not "
        f"support (non-'off' impl for: {active[:5]}"
        f"{'...' if len(active) > 5 else ''})",
    )


def parse_residue_metadata(data: Any, source: str = "<memory>") -> ResidueModelSpec:
    """Validate a decoded residue_kernel_metadata.json into a ResidueModelSpec.

    Raises ResidueMetadataError on anything missing, ambiguous, or
    unsupported. A spec that parses is fully servable by this integration.
    """
    _require(isinstance(data, dict), f"{source}: metadata root must be an object")

    raw_layers = data.get("layers")
    _require(
        isinstance(raw_layers, dict),
        f"{source}: metadata must contain a 'layers' object",
    )

    global_section = data.get("global", {})
    _require(
        isinstance(global_section, dict),
        f"{source}: 'global' section must be an object",
    )
    block_size = global_section.get("block_size", SALIENT_BLOCK)
    _require(
        block_size == SALIENT_BLOCK,
        f"{source}: block_size={block_size} is not supported; the residue "
        f"kernels assume salient selection in blocks of {SALIENT_BLOCK}",
    )

    if "moe" in data:
        _validate_moe_section(data["moe"])

    extended_dims = data.get("extended_linear_dims", {})
    _require(
        isinstance(extended_dims, dict),
        f"{source}: 'extended_linear_dims' must be an object",
    )

    layers: dict[str, ResidueLayerSpec] = {}
    for name, entry in raw_layers.items():
        _require(
            isinstance(entry, dict),
            f"{source}: layer {name!r} entry must be an object",
        )
        if "salient_indices" not in entry and "runtime_mode" not in entry:
            # Entry carries no residue contract (e.g. only debug fields).
            continue

        mode_str = entry.get("runtime_mode")
        _require(
            mode_str is not None,
            f"{source}: layer {name!r} has residue data but no explicit "
            "runtime_mode; refusing to guess the representation",
        )
        extended_dim = extended_dims.get(name)
        if mode_str == ResidueMode.K_EXT.value:
            spec = _parse_k_ext_layer(name, entry, extended_dim)
        elif mode_str == ResidueMode.MEXT_R1.value:
            spec = _parse_mext_r1_layer(name, entry, extended_dim)
        elif mode_str == "standard":
            _require(
                not entry.get("salient_indices"),
                f"{source}: layer {name!r} is marked 'standard' but carries "
                "salient_indices; the entry contradicts itself",
            )
            continue
        else:
            raise ResidueMetadataError(
                f"{source}: layer {name!r} has unknown runtime_mode "
                f"{mode_str!r}; supported: 'extended_k', 'mext_r1', 'standard'"
            )
        layers[name] = spec

    orphaned = sorted(set(extended_dims) - set(layers))
    _require(
        not orphaned,
        f"{source}: extended_linear_dims names layers without a valid residue "
        f"entry: {orphaned[:5]}{'...' if len(orphaned) > 5 else ''}",
    )

    return ResidueModelSpec(layers=layers, block_size=SALIENT_BLOCK, source=source)


def find_residue_metadata(model_dir: str | Path) -> Optional[Path]:
    """Path of the residue metadata file in a local model dir, if present."""
    path = Path(model_dir) / METADATA_FILENAME
    return path if path.is_file() else None


def load_residue_model_spec(model_dir: str | Path) -> Optional[ResidueModelSpec]:
    """Load and validate the residue metadata of a local checkpoint.

    Returns None when the checkpoint has no residue metadata file (the stock
    modelopt_fp4 path applies). Raises ResidueMetadataError when the file
    exists but cannot be served: a present-but-broken file must never fall
    back to plain NVFP4 silently.
    """
    path = find_residue_metadata(model_dir)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ResidueMetadataError(f"cannot read {path}: {e}") from e
    return parse_residue_metadata(data, source=str(path))
