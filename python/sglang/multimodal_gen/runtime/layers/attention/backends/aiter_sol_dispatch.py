# SPDX-License-Identifier: Apache-2.0
#
# Sol-Attn on AITER's mha_v4 mode-2 ASM rows (ROCm, gfx950).
#
# Block-sparse attention routes each query tile to a subset of KV blocks. Plain
# block-sparse attention drops the rest; Sol-Attn (arXiv 2607.24027) recovers
# them from pooled (mean) K/V under the same softmax, so a skipped block costs
# its higher-order terms rather than all of its mass.
#
# Unrelated to `sol_attn.py`, which wraps the NVIDIA `sol_attn` package.

from __future__ import annotations

import inspect
from typing import Any, Callable

import msgspec
import torch

from sglang.multimodal_gen.runtime.platforms.aiter import USE_AITER
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# Every Sol-Attn manifest row is gfx950. Which recipes a device actually serves
# is read from aiter's manifest, per recipe.
_SUPPORTED_ARCHS = ("gfx950",)

# Below this many KV blocks the routing threshold's mean and standard deviation
# describe too few samples to separate anything. A heuristic.
_MIN_USEFUL_KV_BLOCKS = 16


class SolAttnUnsupported(RuntimeError):
    """The inputs or the environment cannot be served correctly by Sol-Attn."""


def _keep_bf16(tensor: torch.Tensor, *args: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Pass a BF16 operand through, in the (tensor, descale) shape the quantizers return.

    The NONE scale mode makes the kernel ignore the descale; the tensor is the
    placeholder, as in aiter's own mha_v4.
    """
    return tensor, tensor


# Defined unconditionally (as None) so the names stay patchable and so that
# referencing one yields the construction-time message rather than a NameError.
#
# Held as module globals rather than fields of a namespace object: the geometry
# queries are functools.cache wrappers, and Dynamo binds one reached through an
# attribute as a method, which fails with "Too many positional arguments".
_AiterAttentionFormat = None
_AiterAttentionPack = None
_AiterAttentionScaleMode = None
_aiter_native_fp8_format = None
_aiter_sol_mode = None
_aiter_mha_v4_sol = None
_aiter_mha_v4_packed = None
_aiter_sol_prepare = None
_aiter_pool_q = None
_aiter_pool_reuse_descale = None
_aiter_q_multiplier = None
_aiter_operands = None
_aiter_block_tile = None
_aiter_kv_tile_for_q_tile = None
_aiter_block_tiles = None
_aiter_ragged_kv = None
_aiter_block_scale_storage = None
_aiter_scale_tile_rows = None
_aiter_scale_lookahead_rows = None
_aiter_mxfp4_k_view = None
_aiter_mxfp4_v_view = None
_aiter_mxfp6_k_view = None
_QUANTIZERS: dict[str, Callable | None] = {}
AITER_SOL_AVAILABLE = False
FUSED_ROUTING_AVAILABLE = False

# The head dimension every mode-2 Sol-Attn row takes.
SOL_ATTN_HEAD_DIM = 128

# Imported at module load so torch.compile sees stable symbols, and gated on
# USE_AITER because importing aiter without it is not guaranteed to fail with an
# ImportError on a machine that lacks the package or the ROCm runtime.
if USE_AITER:
    try:
        from aiter.ops.mha_v4 import MHA_V4_SOL_MODE as _aiter_sol_mode
        from aiter.ops.mha_v4 import AttentionFormat as _AiterAttentionFormat
        from aiter.ops.mha_v4 import AttentionPack as _AiterAttentionPack
        from aiter.ops.mha_v4 import AttentionScaleMode as _AiterAttentionScaleMode
        from aiter.ops.mha_v4 import mha_v4_block_tile as _aiter_block_tile
        from aiter.ops.mha_v4 import mha_v4_block_tiles as _aiter_block_tiles
        from aiter.ops.mha_v4 import (
            mha_v4_kv_tile_for_q_tile as _aiter_kv_tile_for_q_tile,
        )
        from aiter.ops.mha_v4 import mha_v4_operands as _aiter_operands
        from aiter.ops.mha_v4 import mha_v4_packed as _aiter_mha_v4_packed
        from aiter.ops.mha_v4 import mha_v4_q_multiplier as _aiter_q_multiplier
        from aiter.ops.mha_v4 import mha_v4_sol as _aiter_mha_v4_sol
        from aiter.ops.mha_v4 import mxfp4_k_view as _aiter_mxfp4_k_view
        from aiter.ops.mha_v4 import mxfp4_v_view as _aiter_mxfp4_v_view
        from aiter.ops.mha_v4 import mxfp6_k_view as _aiter_mxfp6_k_view
        from aiter.ops.mha_v4 import native_fp8_format as _aiter_native_fp8_format
        from aiter.ops.mha_v4 import (
            quantize_fp8,
            quantize_fp8_rotated,
            quantize_int8,
            quantize_mxfp4_k,
            quantize_mxfp4_q,
            quantize_mxfp6_k,
            quantize_mxfp6_q,
            quantize_mxfp8_k,
            quantize_mxfp8_q,
            quantize_v_mxfp4_fp6_p,
            quantize_v_mxfp6_fp6_p,
        )
        from aiter.ops.mha_v4_quant import (
            MHA_V4_KV_SCALE_LOOKAHEAD_ROWS as _aiter_scale_lookahead_rows,
        )
        from aiter.ops.mha_v4_quant import MHA_V4_KV_TILE_ROWS as _aiter_scale_tile_rows
        from aiter.ops.mha_v4_quant import (
            block_scale_storage as _aiter_block_scale_storage,
        )
        from aiter.ops.triton.attention.utils import sol_prepare as _aiter_sol_prepare

        try:
            from aiter.ops.mha_v4 import quantize_v_fp8
        except ImportError:
            # Predates the MXFP6 Q/K, FP8 V row, which is then reported as unserved.
            quantize_v_fp8 = None

        try:
            from aiter.ops.mha_v4 import mha_v4_ragged_kv as _aiter_ragged_kv
        except ImportError:
            # Predates per-row ragged KV; every row then wants K/V padded to the tile.
            _aiter_ragged_kv = None

        _QUANTIZERS = {
            "quantize_bf16": _keep_bf16,
            "quantize_fp8": quantize_fp8,
            "quantize_fp8_rotated": quantize_fp8_rotated,
            "quantize_int8": quantize_int8,
            "quantize_mxfp4_k": quantize_mxfp4_k,
            "quantize_mxfp4_q": quantize_mxfp4_q,
            "quantize_mxfp6_k": quantize_mxfp6_k,
            "quantize_mxfp6_q": quantize_mxfp6_q,
            "quantize_mxfp8_k": quantize_mxfp8_k,
            "quantize_mxfp8_q": quantize_mxfp8_q,
            "quantize_v_fp8": quantize_v_fp8,
            "quantize_v_mxfp4_fp6_p": quantize_v_mxfp4_fp6_p,
            "quantize_v_mxfp6_fp6_p": quantize_v_mxfp6_fp6_p,
        }
        AITER_SOL_AVAILABLE = True
    except ImportError:
        pass

    # The fused routing path reads aiter's pooling helpers directly, which are
    # internal names. Their absence costs the fused path only, so it is gated on
    # its own and the backend keeps aiter's routing.
    try:
        from aiter.ops.triton.attention.utils import _sol_attn_pool_q as _aiter_pool_q
        from aiter.ops.triton.attention.utils import (
            _sol_attn_pool_reuse_descale as _aiter_pool_reuse_descale,
        )

        FUSED_ROUTING_AVAILABLE = True
    except ImportError:
        pass

# Whether aiter's sol_prepare can add forced blocks to its selection. Builds that
# predate force_block_mask still serve every unforced call.
SOL_PREPARE_FORCES_BLOCKS = (
    AITER_SOL_AVAILABLE
    and "force_block_mask" in inspect.signature(_aiter_sol_prepare).parameters
)


class _Recipe(msgspec.Struct, frozen=True):
    """One mode-2 manifest row: its formats, its quantizers, and how its scales
    survive pooling.

    A per-tensor or per-channel scale does not vary along the pooled sequence
    axis, so the pooled operand reuses it. An E8M0 1x32 scale does, so that
    operand pools in dequantized space and hands the kernel a pooled scale of
    its own.

    `quantize_q` takes a multiplier because the MX quantizers fold
    softmax_scale * log2(e) into Q; the per-tensor ones ignore it and the kernel
    applies the scale itself.
    """

    name: str
    qk_format: str  # AttentionFormat member name, or "native"
    qk_scale_mode: str  # "none", "per_tensor" or "block"
    quantize_q: str  # key into _QUANTIZERS
    quantize_k: str
    quantize_v: str = "quantize_fp8"
    v_format: str = "native"
    v_scale_mode: str = "per_tensor"  # or "none", "per_channel" or "block"
    # Set, with sol_prepare's name for the packing, when that operand's codes
    # are sub-byte and in tile order. They cannot be pooled or routed on, so
    # both use the BF16 source.
    k_packed: str | None = None
    v_packed: str | None = None

    @property
    def v_pack(self) -> str:
        """The AttentionPack V is stored in: the FP6-P order for every MX V, else
        the default."""
        return "V_FOR_FP6_P" if (self.v_packed or "").endswith("_fp6_p") else "DEFAULT"

    @property
    def routes_through_raw(self) -> bool:
        """Whether mha_v4_sol serves this row.

        MXFP8's addressable block-scaled K needs its scale handed to the
        pooling, which the raw entry point cannot take, so it goes packed.
        """
        return not (self.qk_scale_mode == "block" and self.k_packed is None)


# fmt: off
# Hand-aligned: a row reads across and a column reads down, against the header.
# Q/K format then V format, as the dense mha_v4 table names them: f8f6 is FP8
# Q/K with MXFP6 V, mxfp6 is MXFP6 Q/K with FP8 V, f6f6 MXFP6 throughout. The
# MX-V rows store V in the order the kernel's FP6 P operand reads it.
_RECIPE_ROWS = (
    #       name       Q/K       Q/K scale    quantize Q              quantize K
    _Recipe("bf16",    "BF16",   "none",      "quantize_bf16",        "quantize_bf16",
            quantize_v="quantize_bf16", v_format="BF16", v_scale_mode="none"),
    _Recipe("bf16fp8", "BF16",   "none",      "quantize_bf16",        "quantize_bf16"),
    _Recipe("fp8",     "native", "per_tensor", "quantize_fp8_rotated", "quantize_fp8_rotated"),
    _Recipe("i8fp8",   "INT8",   "per_tensor", "quantize_int8",        "quantize_int8"),
    _Recipe("mxfp8",   "native", "block",     "quantize_mxfp8_q",     "quantize_mxfp8_k"),
    _Recipe("f8f6",    "native", "per_tensor", "quantize_fp8_rotated", "quantize_fp8_rotated",
            quantize_v="quantize_v_mxfp6_fp6_p", v_format="MXFP6", v_scale_mode="block",
            v_packed="mxfp6_fp6_p"),
    _Recipe("mxfp6",   "MXFP6",  "block",     "quantize_mxfp6_q",     "quantize_mxfp6_k",
            quantize_v="quantize_v_fp8", v_scale_mode="per_channel", k_packed="mxfp6"),
    _Recipe("f6f6",    "MXFP6",  "block",     "quantize_mxfp6_q",     "quantize_mxfp6_k",
            quantize_v="quantize_v_mxfp6_fp6_p", v_format="MXFP6", v_scale_mode="block",
            k_packed="mxfp6", v_packed="mxfp6_fp6_p"),
    _Recipe("f6f4",    "MXFP6",  "block",     "quantize_mxfp6_q",     "quantize_mxfp6_k",
            quantize_v="quantize_v_mxfp4_fp6_p", v_format="MXFP4", v_scale_mode="block",
            k_packed="mxfp6", v_packed="mxfp4_fp6_p"),
    _Recipe("mxfp4",   "MXFP4",  "block",     "quantize_mxfp4_q",     "quantize_mxfp4_k",
            quantize_v="quantize_v_mxfp4_fp6_p", v_format="MXFP4", v_scale_mode="block",
            k_packed="mxfp4", v_packed="mxfp4_fp6_p"),
)
# fmt: on

_RECIPES = {row.name: row for row in _RECIPE_ROWS}
SOL_ATTN_RECIPES = tuple(_RECIPES)


def _resolve_recipe(recipe: str | _Recipe) -> _Recipe:
    """The named recipe, or an error listing what this build actually ships."""
    if isinstance(recipe, _Recipe):
        return recipe
    try:
        return _RECIPES[recipe]
    except KeyError:
        raise SolAttnUnsupported(
            f"Unknown Sol-Attn recipe {recipe!r}; this build has "
            f"{', '.join(SOL_ATTN_RECIPES)}."
        ) from None


def _format(name: str):
    """An AttentionFormat by name, with "native" resolved against the active GPU."""
    if name == "native":
        return _aiter_native_fp8_format()
    return getattr(_AiterAttentionFormat, name)


def _scale_mode(name: str):
    return {
        "none": _AiterAttentionScaleMode.NONE,
        "per_tensor": _AiterAttentionScaleMode.F32_PER_TENSOR,
        "per_channel": _AiterAttentionScaleMode.F32_PER_CHANNEL,
        "block": _AiterAttentionScaleMode.E8M0_PER_1X32,
    }[name]


def _recipe_operands(recipe: _Recipe) -> tuple[int, ...]:
    """The seven manifest values identifying this row, for aiter's geometry queries.

    Resolved per device, since a recipe's format may be "native".
    """
    qk_format, v_format = _format(recipe.qk_format), _format(recipe.v_format)
    qk_scale = _scale_mode(recipe.qk_scale_mode)
    v_scale = _scale_mode(recipe.v_scale_mode)
    return _aiter_operands(
        qk_format,
        qk_format,
        v_format,
        qk_scale,
        qk_scale,
        v_scale,
        getattr(_AiterAttentionPack, recipe.v_pack),
    )


def _device_arch(device: torch.device | None = None) -> str | None:
    """The supported arch this device is, or None. gcnArchName carries a
    target-feature suffix."""
    if device is None:
        if not torch.cuda.is_available():
            return None
        device = torch.device("cuda", torch.cuda.current_device())
    name = torch.cuda.get_device_properties(device).gcnArchName or ""
    return next((arch for arch in _SUPPORTED_ARCHS if name.startswith(arch)), None)


def _require_aiter() -> None:
    """Raise unless this build imported the aiter surface Sol-Attn needs.

    The one place that states the import requirement, so the build check and
    the call check cannot drift apart.
    """
    if not AITER_SOL_AVAILABLE:
        raise SolAttnUnsupported(
            "Sol-Attn requires SGLANG_USE_AITER=1 and an aiter that ships "
            "aiter.ops.mha_v4.mha_v4_sol and aiter.ops.triton.attention.utils."
            "sol_prepare; please update AITER."
        )


def _serves_recipe(recipe: _Recipe) -> bool:
    """Whether the active device's manifest has a mode-2 row, at any geometry."""
    if _QUANTIZERS.get(recipe.quantize_v) is None:
        return False
    return bool(_aiter_block_tiles(_recipe_operands(recipe), _aiter_sol_mode))


def sol_attn_recipes_served() -> list[str]:
    """The recipe names this device has a Sol-Attn row for, in declaration order."""
    return [name for name in SOL_ATTN_RECIPES if _serves_recipe(_RECIPES[name])]


def sol_attn_block_tile(
    recipe: str | _Recipe, block_tile: tuple[int, int] | None = None
) -> tuple[int, int]:
    """The (q_tile, kv_tile) a call routes, pools and dispatches this recipe at.

    Routing, pooling and dispatch must agree on it, and a mismatch is silently
    wrong, so all three read it from here. None takes the recipe's default,
    which differs between rows.
    """
    if block_tile is not None:
        return tuple(block_tile)
    return _aiter_block_tile(_recipe_operands(_resolve_recipe(recipe)), _aiter_sol_mode)


def check_sol_attn_build(
    recipe_name: str,
    device: torch.device | None = None,
    block_tile: tuple[int, int] | None = None,
) -> None:
    """Raise unless this build, this device and this recipe can run Sol-Attn.

    Everything that a run can settle before it starts: the aiter import, the
    arch, the manifest row and the geometry. Checked once at backend
    construction so a bad configuration fails at startup, not mid-generation.
    """
    _require_aiter()
    if device is None:
        if not torch.cuda.is_available():
            raise SolAttnUnsupported(
                "Sol-Attn is a GPU kernel and no GPU is available."
            )
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type != "cuda":
        raise SolAttnUnsupported(f"Sol-Attn is a GPU kernel, got device {device}.")
    if _device_arch(device) is None:
        reported = torch.cuda.get_device_properties(device).gcnArchName or ""
        raise SolAttnUnsupported(
            f"Sol-Attn ships kernels for {', '.join(_SUPPORTED_ARCHS)}, this "
            f"device reports {reported!r}."
        )

    recipe = _resolve_recipe(recipe_name)
    if not _serves_recipe(recipe):
        served = ", ".join(sol_attn_recipes_served()) or "none"
        raise SolAttnUnsupported(
            f"Sol-Attn recipe {recipe_name!r} has no {_device_arch(device)} "
            f"manifest row; this device serves {served}."
        )
    if block_tile is None:
        return
    q_tile, kv_tile = block_tile
    operands = _recipe_operands(recipe)
    if _aiter_kv_tile_for_q_tile(q_tile, operands, _aiter_sol_mode) == kv_tile:
        return
    served = _aiter_block_tiles(operands, _aiter_sol_mode)
    default = "x".join(str(part) for part in sol_attn_block_tile(recipe))
    raise SolAttnUnsupported(
        f"block_size {q_tile}x{kv_tile} has no Sol-Attn kernel with the "
        f"{recipe.name!r} recipe on this GPU; it serves "
        f"{', '.join(f'{q}x{kv}' for q, kv in served) or 'none'}. Omit it for "
        f"the default geometry ({default})."
    )


def check_sol_attn_values(
    *,
    head_dim: int | None = None,
    dtypes: tuple[torch.dtype, ...] = (),
    is_causal: bool = False,
) -> None:
    """Raise unless these operand values can be served.

    Everything that depends on what the call carries rather than on how the run
    was configured. The backend checks what it knows at construction and the
    dispatch checks the tensors, so each constraint is stated once.
    """
    _require_aiter()
    if head_dim is not None and head_dim != SOL_ATTN_HEAD_DIM:
        raise SolAttnUnsupported(
            f"Sol-Attn's mha_v4 rows take head_dim == {SOL_ATTN_HEAD_DIM}, got "
            f"{head_dim}."
        )
    if dtypes and any(dtype != torch.bfloat16 for dtype in dtypes):
        got = " ".join(str(dtype) for dtype in dtypes)
        raise SolAttnUnsupported(
            f"Sol-Attn's mha_v4 row takes bf16 Q/K/V and returns bf16, got "
            f"{got}. Select another attention backend for other dtypes."
        )
    if is_causal:
        raise SolAttnUnsupported(
            "Sol-Attn has no causal variant: its pooled correction assumes every "
            "skipped block is fully attendable, which a causal mask breaks."
        )


def _takes_ragged_kv(recipe: _Recipe, block_tile: tuple[int, int]) -> bool:
    """Whether this row masks a short last KV block itself. A row that does not
    needs K/V padded to a whole number of blocks."""
    if _aiter_ragged_kv is None:
        return False
    return _aiter_ragged_kv(_recipe_operands(recipe), _aiter_sol_mode, block_tile)


def _pad_kv_to_tile(
    key: torch.Tensor,
    value: torch.Tensor,
    recipe: _Recipe,
    block_tile: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad BSHD K/V with zero tokens to a whole number of KV blocks, unless
    the row is ragged.

    A zero key scores 0 rather than -inf, so the pad draws a little softmax
    weight: at most one block's worth, exponentially small in the row max.
    """
    pad = -key.shape[1] % block_tile[1]
    if pad == 0 or _takes_ragged_kv(recipe, block_tile):
        return key, value
    widths = (0, 0, 0, 0, 0, pad)
    return (
        torch.nn.functional.pad(key, widths),
        torch.nn.functional.pad(value, widths),
    )


_short_kv_warned: set[int] = set()


def _warn_if_kv_too_short(seqlen_k: int, kv_tile: int) -> None:
    """Warn, in eager runs only, when a call is too short to route usefully."""
    if torch.compiler.is_compiling():
        return
    blocks = -(-seqlen_k // kv_tile)
    if blocks >= _MIN_USEFUL_KV_BLOCKS or blocks in _short_kv_warned:
        return
    _short_kv_warned.add(blocks)
    logger.warning(
        "Sol-Attn was handed a %d-token KV, i.e. %d blocks of %d. Its routing "
        "threshold is a mean plus a multiple of a standard deviation over the "
        "blocks, which says little at that count, and the pooled correction is "
        "then standing in for much of the sequence. Expect a worse result than "
        "a dense backend, not a faster one. If this is a short text-only "
        "attention inside a model whose main attention is long, that site "
        "should be on a dense backend.",
        seqlen_k,
        blocks,
        kv_tile,
    )


def _quantize(
    recipe: _Recipe,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
):
    """Quantize BSHD bf16 Q/K/V for one recipe, returning (tensor, descale, source)
    per operand.

    As mha_v4_sol quantizes the same row, except that K is not smoothed: once
    K's shared component passes aiter's smoothing gate, this path's output
    differs from the raw one. The MX quantizers fold softmax_scale * log2(e)
    into Q; the per-tensor ones leave the scale to the kernel.

    source is the BF16 input, kept only for a packed operand, whose codes
    pooling and routing cannot read.
    """
    multiplier = _aiter_q_multiplier(softmax_scale)

    if recipe.qk_scale_mode == "block":
        q, q_descale = _QUANTIZERS[recipe.quantize_q](query, multiplier)
    else:
        q, q_descale = _QUANTIZERS[recipe.quantize_q](query)

    k, k_descale = _QUANTIZERS[recipe.quantize_k](key)
    # The packers return a flat buffer; the kernel takes the strided view aiter
    # builds over it.
    if recipe.k_packed == "mxfp4":
        k = _aiter_mxfp4_k_view(k, k_descale)
    elif recipe.k_packed == "mxfp6":
        k, k_descale = _aiter_mxfp6_k_view(k, k_descale, *key.shape[:3])

    v, v_descale = _QUANTIZERS[recipe.quantize_v](value)
    if recipe.v_packed == "mxfp4_fp6_p":
        v = _aiter_mxfp4_v_view(v, v_descale, value.shape[1])

    qk_packed = recipe.k_packed is not None
    return (
        (q, q_descale, query if qk_packed else None),
        (k, k_descale, key if qk_packed else None),
        (v, v_descale, value if recipe.v_packed is not None else None),
    )


def sol_attn_routing_for(
    q,
    k,
    v,
    beta,
    recipe: _Recipe,
    block_attn_mask: torch.Tensor | None = None,
    block_tile: tuple[int, int] | None = None,
    force_blocks: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Pooled K/V, ragged LUT and selection bitmap for one call, as the kernel
    consumes them.

    q/k/v are the (tensor, descale, source) triples `_quantize` returns.
    block_attn_mask supplies an already chosen selection instead of routing from
    beta; pass exactly one of the two. force_blocks, a (num_kv_blocks,) bool
    mask, is added to the selection either way.
    """
    if force_blocks is not None and not SOL_PREPARE_FORCES_BLOCKS:
        raise SolAttnUnsupported(
            "Exact KV tokens on this recipe need an aiter whose sol_prepare takes "
            "force_block_mask; update aiter."
        )
    qk_packed = recipe.k_packed is not None
    # An addressable E8M0-scaled operand's codes are not proportional to its
    # values, so routing and pooling are handed the scale.
    qk_block_scaled = recipe.qk_scale_mode == "block" and not qk_packed
    v_block_scaled = recipe.v_scale_mode == "block" and recipe.v_packed is None
    # A packed recipe routes on the BF16 Q. Routing is invariant to the scale and
    # to the packers' orthogonal rotation.
    q_routing = q[2] if qk_packed else q[0]
    return _aiter_sol_prepare(
        q_routing,
        k[0],
        v[0],
        beta,
        # Not sol_prepare's default tile, which is only one of the rows' tiles.
        *sol_attn_block_tile(recipe, block_tile),
        num_heads=q_routing.shape[2],
        k_scale=k[1] if qk_block_scaled else None,
        v_scale=v[1] if v_block_scaled else None,
        k_source=k[2],
        v_source=v[2],
        k_packed_format=recipe.k_packed,
        v_packed_format=recipe.v_packed,
        block_attn_mask=block_attn_mask,
        q_scale=q[1] if qk_block_scaled else None,
        # Only when set, so a build without the argument serves unforced calls.
        **({} if force_blocks is None else {"force_block_mask": force_blocks}),
    )


def _takes_fused_routing(recipe: _Recipe, routing: dict[str, Any] | None) -> bool:
    """Whether this call builds its routing with the fused kernel.

    Nothing to build when the caller supplied a selection, and the kernel pools
    the stored codes, so a recipe it cannot pool keeps aiter's routing.
    """
    return routing is None and _fused_routing_serves(recipe)


def _takes_raw_entry(
    recipe: _Recipe,
    routing: dict[str, Any] | None,
    return_head_cost: bool,
    use_fused: bool,
    forced: bool,
) -> bool:
    """Whether the call goes to mha_v4_sol rather than mha_v4_packed.

    The raw entry point routes internally, so it serves only a call that wants
    neither its own routing in, nor the mask out, nor the fused kernel's, nor
    forced blocks.
    """
    return (
        routing is None
        and not return_head_cost
        and not use_fused
        and not forced
        and recipe.routes_through_raw
    )


def sol_attn_routing_mode(recipe_name: str) -> str:
    """The routing path this recipe takes, from the recipe alone.

    The fused kernel covers a subset of the recipes and falls back to aiter's
    own routing for the others. A row too wide for one program falls back as
    well, which only the call itself can know.
    """
    if _fused_routing_serves(_resolve_recipe(recipe_name)):
        return "fused"
    return "eager"


def _fused_routing_serves(recipe: _Recipe) -> bool:
    """Whether the fused routing kernel covers this recipe.

    It pools the stored codes and routes on them, which holds for a scale that
    does not vary along the pooled sequence axis. A block-granular scale and a
    packed operand both need the dequantized or pre-packing tensor, which aiter's
    own routing reads.
    """
    return (
        FUSED_ROUTING_AVAILABLE
        and recipe.k_packed is None
        and recipe.v_packed is None
        and recipe.qk_scale_mode != "block"
        and recipe.v_scale_mode != "block"
    )


def sol_attn_routing_fused(
    q,
    k,
    v,
    beta,
    recipe: _Recipe,
    block_tile: tuple[int, int] | None = None,
    force_blocks: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Routing built by the fused kernel, in the shape mha_v4_packed consumes.

    Does the work of aiter's sol_prepare for the recipes _fused_routing_serves
    accepts, with the threshold, the selection, the per-row count, the bitmap
    and the index list in one kernel rather than in many tensor operations.
    Each row owns a segment of the index list, so the starts follow from the
    shape and need no running sum over the counts.
    """
    from sglang.multimodal_gen.runtime.layers.attention.backends.aiter_sol_kernels import (
        route_and_pack,
        row_lut_start,
    )

    block_m, block_n = sol_attn_block_tile(recipe, block_tile)
    q_routing, k_quant, v_quant = q[0], k[0], v[0]
    batch, seqlen_q, nheads_q, _ = q_routing.shape
    seqlen_k, nheads_kv = k_quant.shape[1], k_quant.shape[2]
    num_q_tiles = (seqlen_q + block_m - 1) // block_m
    num_kv_blocks = (seqlen_k + block_n - 1) // block_n
    bitmap_ds = (block_n // 32) * ((num_kv_blocks + block_n - 1) // block_n)

    mean_k = _aiter_pool_reuse_descale(k_quant, block_n)
    mean_v = _aiter_pool_reuse_descale(v_quant, block_n)
    q_mean = _aiter_pool_q(q_routing, block_m)

    group = nheads_q // nheads_kv
    k_rep = mean_k.float()
    if group != 1:
        k_rep = k_rep.repeat_interleave(group, dim=2)
    proxy = torch.einsum("bihd,bjhd->bhij", q_mean.float(), k_rep)

    mask, lut_count, block_bitmap, kv_block_indices = route_and_pack(
        proxy,
        beta,
        num_kv_blocks,
        bitmap_ds,
        seqlen_k % block_n != 0,
        force_blocks=force_blocks,
    )
    lut_start = row_lut_start(
        batch * nheads_q * num_q_tiles, num_kv_blocks, proxy.device
    )

    return {
        "mean_k": mean_k,
        "mean_v": mean_v,
        "mean_k_scale": None,
        "mean_v_scale": None,
        "block_bitmap": block_bitmap,
        "kv_block_indices": kv_block_indices,
        "lut_start": lut_start,
        "lut_count": lut_count,
        "num_kv_blocks": num_kv_blocks,
        "bitmap_Ds": bitmap_ds,
        "num_q_tiles": num_q_tiles,
        "block_attn_mask": mask,
    }


def _with_scale_lookahead(scale: torch.Tensor) -> torch.Tensor:
    # The MXFP8 row reads past the last pooled scale tile; drop this copy once
    # sol_prepare allocates the pooled scale with the same lookahead rows.
    backed = _aiter_block_scale_storage(
        scale,
        *scale.shape,
        _aiter_scale_tile_rows,
        lookahead_rows=_aiter_scale_lookahead_rows,
    )
    backed.copy_(scale)
    return backed


def _force_blocks_from_tokens(
    exact_kv_tokens: torch.Tensor, seqlen_k: int, kv_tile: int
) -> torch.Tensor:
    """(num_kv_blocks,) bool: the KV blocks holding any flagged token.

    Flags past seqlen_k belong to a pad the caller has already dropped.
    """
    if exact_kv_tokens.dim() != 1 or exact_kv_tokens.dtype != torch.bool:
        raise SolAttnUnsupported(
            "exact_kv_tokens must be a 1-D bool tensor, got "
            f"{tuple(exact_kv_tokens.shape)} {exact_kv_tokens.dtype}."
        )
    if exact_kv_tokens.shape[0] < seqlen_k:
        raise SolAttnUnsupported(
            f"exact_kv_tokens covers {exact_kv_tokens.shape[0]} keys, the call "
            f"has {seqlen_k}."
        )
    flags = exact_kv_tokens[:seqlen_k]
    pad = -seqlen_k % kv_tile
    if pad:
        flags = torch.nn.functional.pad(flags, (0, pad))
    return flags.view(-1, kv_tile).any(dim=1)


def _head_cost_from_routing(routing: dict[str, Any]) -> torch.Tensor:
    """Per-head count of exactly-computed KV blocks, float32 (nheads_q,)."""
    # (batch, nheads_q, num_q_tiles, num_kv_blocks) bool
    mask = routing["block_attn_mask"]
    return mask.sum(dim=(0, 2, 3), dtype=torch.float32)


def sol_attn_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool = False,
    beta: float | torch.Tensor = 0.5,
    softmax_scale: float | None = None,
    recipe: str | _Recipe = "fp8",
    key_seqlen: int | None = None,
    block_tile: tuple[int, int] | None = None,
    routing: dict[str, Any] | None = None,
    return_head_cost: bool = False,
    exact_kv_tokens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sol-Attn over BSHD tensors, returning (BSHD bf16 output, per-head cost or None).

    query/key/value are (batch, seqlen, nheads, head_dim) bf16 tensors, which is
    the layout both this runtime's attention layers and aiter's mha_v4 use, so
    nothing is permuted here. See SOL_ATTN_RECIPES for the rows this build ships.

    return_head_cost asks for the per-head count of exactly-computed KV blocks,
    float32 (nheads_q,). The raw entry point keeps its routing internal, so
    asking for it moves the call onto aiter's packed API with quantization and
    routing done here. MXFP8 and a caller-supplied `routing` always take that path.

    key_seqlen is the number of real KV tokens when the caller has padded the
    sequence. Sol-Attn takes no mask, so the tail is dropped rather than attended.

    block_tile is the (q_tile, kv_tile) to route and dispatch at. None takes the
    recipe's default.

    exact_kv_tokens is a (seqlen_k,) bool mask, or longer by the caller's pad.
    Every KV block holding a flagged key is computed exactly for every query,
    on top of what routing selects: routing judges each block against
    statistics over all blocks, so a small part of a mixed sequence would
    otherwise be routed by the majority. It does not apply to a caller-supplied
    `routing`.
    """
    check_sol_attn_values(
        head_dim=query.shape[-1],
        dtypes=(query.dtype, key.dtype, value.dtype),
        is_causal=is_causal,
    )
    recipe = _resolve_recipe(recipe)
    block_tile = sol_attn_block_tile(recipe, block_tile)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()

    # Before the tile pad, so the caller's pad and the tile pad do not stack.
    if key_seqlen is not None and key_seqlen < key.shape[1]:
        key, value = key[:, :key_seqlen], value[:, :key_seqlen]
    force_blocks = (
        None
        if exact_kv_tokens is None or routing is not None
        else _force_blocks_from_tokens(exact_kv_tokens, key.shape[1], block_tile[1])
    )
    key, value = _pad_kv_to_tile(key, value, recipe, block_tile)
    _warn_if_kv_too_short(key.shape[1], block_tile[1])

    if softmax_scale is None:
        softmax_scale = query.shape[-1] ** -0.5

    use_fused = _takes_fused_routing(recipe, routing)
    if _takes_raw_entry(
        recipe, routing, return_head_cost, use_fused, force_blocks is not None
    ):
        qk_fmt, v_fmt = _format(recipe.qk_format), _format(recipe.v_format)
        out = _aiter_mha_v4_sol(
            query,
            key,
            value,
            qk_fmt,
            qk_fmt,
            v_fmt,
            beta=beta,
            softmax_scale=softmax_scale,
            block_tile=block_tile,
        )
        return out, None

    q, k, v = _quantize(recipe, query, key, value, softmax_scale)
    if routing is None:
        builder = sol_attn_routing_fused if use_fused else sol_attn_routing_for
        routing = builder(
            q, k, v, beta, recipe, block_tile=block_tile, force_blocks=force_blocks
        )
    qk_fmt, v_fmt = _format(recipe.qk_format), _format(recipe.v_format)
    qk_scale = _scale_mode(recipe.qk_scale_mode)
    v_scale = _scale_mode(recipe.v_scale_mode)
    out = _aiter_mha_v4_packed(
        q[0],
        k[0],
        v[0],
        q[1],
        k[1],
        v[1],
        qk_fmt,
        qk_fmt,
        v_fmt,
        qk_scale,
        qk_scale,
        v_scale,
        softmax_scale=softmax_scale,
        kv_block_indices=routing["kv_block_indices"],
        lut_start=routing["lut_start"],
        lut_count=routing["lut_count"],
        mean_k=routing["mean_k"],
        mean_v=routing["mean_v"],
        block_bitmap=routing["block_bitmap"],
        # None where the pooled operand reuses the source descale.
        mean_k_scale=(
            _with_scale_lookahead(routing["mean_k_scale"])
            if recipe.name == "mxfp8"
            else routing["mean_k_scale"]
        ),
        mean_v_scale=routing["mean_v_scale"],
        block_tile=block_tile,
        v_pack=getattr(_AiterAttentionPack, recipe.v_pack),
    )
    return out, _head_cost_from_routing(routing) if return_head_cost else None
