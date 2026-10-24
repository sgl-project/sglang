"""Load-time decoding of CSF-compressed NVFP4 block scales.

ModelOpt checkpoints mark a layer with ``"weight_scale_encoding": "csf"`` when
its E4M3 block-scale matrix of shape ``[rows, columns]`` is stored losslessly as
two tensors instead of ``<prefix>.weight_scale``:

- ``<prefix>.weight_scale.nvfp4_csf_fixed`` (uint8,
  ``[rows // 16, 16 * (1 + columns // 2)]``): one slab per 16 rows, holding the
  16 rows' base bytes, then each row's 4-bit offsets, low nibble first.
  A scale byte is ``base + offset``.
- ``<prefix>.weight_scale.nvfp4_csf_exceptions`` (uint32): one record per scale
  outside its row's window, sorted by position and unique. Bits 0-23 hold the
  position ``row * columns + column``, bits 24-31 the scale byte itself.

The layout matches FlashInfer's b12x ``make_nvfp4_csf_batch`` (codec 0). The
decoded bytes equal the uncompressed scales exactly, so every NVFP4 kernel
consumes them unchanged.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, Iterable, Iterator, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# The only `weight_scale_encoding` value this module decodes
CSF_ENCODING = "csf"

# Name suffixes of the two streams that replace `<prefix>.weight_scale`
CSF_FIXED_SUFFIX = ".weight_scale.nvfp4_csf_fixed"
CSF_EXCEPTIONS_SUFFIX = ".weight_scale.nvfp4_csf_exceptions"

# Rows per slab of the fixed stream
SLAB_ROWS = 16

# The largest row base; base + 15 still fits in a byte
MAX_BASE = 240

# Width of the position field in an exception record, in bits
POSITION_BITS = 24


def csf_stream_of(name: str) -> Optional[Tuple[str, str]]:
    """The layer prefix and stream kind ("fixed" or "exceptions") of a CSF stream."""
    if name.endswith(CSF_FIXED_SUFFIX):
        return name.removesuffix(CSF_FIXED_SUFFIX), "fixed"
    if name.endswith(CSF_EXCEPTIONS_SUFFIX):
        return name.removesuffix(CSF_EXCEPTIONS_SUFFIX), "exceptions"
    return None


def decode_csf_scales(
    prefix: str, fixed: torch.Tensor, exceptions: torch.Tensor
) -> torch.Tensor:
    """The `[rows, columns]` float8_e4m3fn scales of a stream pair, decoded on CPU."""
    fixed = fixed.cpu()
    rows, columns = _scale_shape_of(prefix, fixed)
    positions, values = _exception_records_of(prefix, exceptions.cpu())
    _check_exception_positions(prefix, positions, scale_count=rows * columns)

    bases = fixed[:, :SLAB_ROWS].reshape(rows, 1)
    packed = fixed[:, SLAB_ROWS:].reshape(rows, columns // 2)
    offsets = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(rows, columns)
    # MAX_BASE + 15 fits in a byte, so the uint8 sum never wraps.
    scales = bases + offsets
    scales.view(-1)[positions] = values
    return scales.view(torch.float8_e4m3fn)


def _scale_shape_of(prefix: str, fixed: torch.Tensor) -> Tuple[int, int]:
    """The `(rows, columns)` a fixed stream encodes; raises if it is malformed."""
    name = prefix + CSF_FIXED_SUFFIX
    if fixed.dtype != torch.uint8 or fixed.dim() != 2 or fixed.shape[0] == 0:
        raise ValueError(
            f"{name}: expected a non-empty 2-D uint8 tensor, "
            f"got {fixed.dtype} of shape {tuple(fixed.shape)}."
        )
    slabs, slab_bytes = fixed.shape
    if slab_bytes % SLAB_ROWS or slab_bytes // SLAB_ROWS < 2:
        raise ValueError(
            f"{name}: a slab must be {SLAB_ROWS} * (1 + columns / 2) bytes "
            f"with columns >= 2, got {slab_bytes} bytes."
        )
    if int(fixed[:, :SLAB_ROWS].max()) > MAX_BASE:
        raise ValueError(f"{name}: row bases must be at most {MAX_BASE}.")
    rows = slabs * SLAB_ROWS
    columns = (slab_bytes // SLAB_ROWS - 1) * 2
    if rows * columns > 1 << POSITION_BITS:
        raise ValueError(
            f"{name}: {rows} x {columns} scales do not fit the "
            f"{POSITION_BITS}-bit exception positions."
        )
    return rows, columns


def _exception_records_of(
    prefix: str, exceptions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """The int64 positions and uint8 scale bytes of an exception stream."""
    if exceptions.dtype != torch.uint32 or exceptions.dim() != 1:
        raise ValueError(
            f"{prefix}{CSF_EXCEPTIONS_SUFFIX}: expected a 1-D uint32 tensor, "
            f"got {exceptions.dtype} of shape {tuple(exceptions.shape)}."
        )
    # torch.uint32 has few CPU ops, so widen through a same-size int32 view.
    words = exceptions.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    positions = words & ((1 << POSITION_BITS) - 1)
    values = (words >> POSITION_BITS).to(torch.uint8)
    return positions, values


def _check_exception_positions(
    prefix: str, positions: torch.Tensor, scale_count: int
) -> None:
    """Raise unless POSITIONS are sorted, unique and below SCALE_COUNT."""
    if positions.numel() == 0:
        return
    is_increasing = bool((positions[1:] > positions[:-1]).all())
    if not is_increasing or int(positions[-1]) >= scale_count:
        raise ValueError(
            f"{prefix}{CSF_EXCEPTIONS_SUFFIX}: positions must be sorted, unique "
            f"and below {scale_count}."
        )


def decode_csf_weight_scales(
    weights: Iterable[Tuple[str, torch.Tensor]],
    scale_encoding_of: Callable[[str], Optional[str]],
) -> Iterator[Tuple[str, torch.Tensor]]:
    """WEIGHTS with each CSF stream pair replaced by its `<prefix>.weight_scale`.

    A pair whose layer SCALE_ENCODING_OF does not mark "csf" raises.
    """
    waiting: Dict[str, Dict[str, torch.Tensor]] = {}
    decoded_count = 0
    decode_seconds = 0.0
    for name, tensor in weights:
        stream = csf_stream_of(name)
        if stream is None:
            yield name, tensor
            continue

        prefix, kind = stream
        pair = waiting.setdefault(prefix, {})
        pair[kind] = tensor
        if len(pair) < 2:
            continue
        del waiting[prefix]

        encoding = scale_encoding_of(prefix)
        if encoding != CSF_ENCODING:
            raise ValueError(
                f"{prefix}: the checkpoint stores CSF scale streams, but "
                f"quantized_layers gives weight_scale_encoding={encoding!r}."
            )
        start = time.perf_counter()
        scales = decode_csf_scales(prefix, pair["fixed"], pair["exceptions"])
        decode_seconds += time.perf_counter() - start
        decoded_count += 1
        yield f"{prefix}.weight_scale", scales

    if waiting:
        unpaired = [f"{prefix} ({', '.join(pair)})" for prefix, pair in waiting.items()]
        raise ValueError(
            f"{len(waiting)} CSF scale stream(s) have no partner, e.g. "
            f"{unpaired[:4]}; the checkpoint is incomplete."
        )
    if decoded_count:
        logger.info(
            "Decoded %d CSF weight scales in %.1f s", decoded_count, decode_seconds
        )
