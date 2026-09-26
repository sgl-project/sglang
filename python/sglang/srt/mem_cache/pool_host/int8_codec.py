"""INT8 record codec for the HiCache L2 host pool.

Self-contained by design: it imports nothing from the pool that uses it, so the
record layout and the quantiser can be tested on any device, with no GPU, CUDA
driver or SGLang runtime.

Packed record (one per ``(layer, K|V, token)`` row)::

    offset 0                        1023
    |        1024 INT8 payload       |    head_dim bytes per head, one byte each
    1024                            1039
    |      8 BF16 scales (16 B)      |    one scale per KV head
    1040                            1151
    |        112 bytes padding       |

``ROW_BYTES = 1152 = 9 x 128`` is what keeps the row admissible on SGLang's CUDA
JIT HiCache mover, which requires ``element_size % 128 == 0``.

Torch-only, no host synchronisation, no optional dependencies: every operation
below is a normal tensor op so the codec runs inside the existing HiCache
transfer streams without introducing a second synchronisation system.
"""

from __future__ import annotations

import torch

#: Bytes of INT8 payload: one byte per (head, head_dim) element.
PAYLOAD_BYTES = 1024

#: One BF16 scale per KV head.
SCALE_BYTES = 16

#: Padding, chosen so that a row is a whole number of 128-byte groups.
PADDING_BYTES = 112

#: Total encoded bytes per K-or-V row.
ROW_BYTES = PAYLOAD_BYTES + SCALE_BYTES + PADDING_BYTES  # 1152

#: Byte offset of the BF16 scale block inside a record.
SCALE_OFFSET = PAYLOAD_BYTES  # 1024

#: KV heads the fixed V1 record is sized for: Qwen3-8B at TP=1.
#: ``EXPECTED_KV_HEADS * head_dim == PAYLOAD_BYTES``.
EXPECTED_KV_HEADS = 8

#: Quantiser full scale. ``-128`` is deliberately unused so the range is
#: symmetric and sign handling cannot drift.
QUANT_MAX = 127

#: CUDA JIT mover alignment requirement.
ALIGNMENT_BYTES = 128

#: Lower bound on the per-head absmax: ``2**-112``, a power of two so the clamp
#: is exact on every backend. ``AMAX_FLOOR / QUANT_MAX`` is still normal in BF16
#: (the normal range starts at ``2**-126``), so a stored scale is never
#: subnormal. A head below this floor quantises to all zeros and decodes to exact
#: zero, which is the correct answer for negligible input.
AMAX_FLOOR = 2.0**-112

#: Convenience alias used in tests.
SCALE_FLOOR = AMAX_FLOOR / QUANT_MAX


def check_layout(head_num: int, head_dim: int, itemsize: int) -> None:
    """Fail fast if the record cannot represent this K/V geometry.

    The V1 record is a *fixed* 1024-byte payload plus 16 scale bytes, sized for
    Qwen3-8B at TP=1 (8 local KV heads x 128 dims). ``head_num`` here is the
    per-rank local KV head count, so a different TP size changes it and the
    record no longer fits. That case is reported as a TP restriction rather than
    a byte mismatch, because "payload is 1024 but geometry is 512" tells the
    reader nothing about what to do.
    """
    expected_payload = head_num * head_dim
    if PAYLOAD_BYTES != expected_payload:
        if head_dim == 128 and head_num != EXPECTED_KV_HEADS:
            tp = EXPECTED_KV_HEADS / head_num
            tp_note = (
                f" This looks like TP={tp:g}: {head_num} local KV heads."
                if tp == int(tp) and tp > 1
                else ""
            )
            raise ValueError(
                f"INT8 HiCache L2 records support {EXPECTED_KV_HEADS} local KV "
                f"heads only "
                f"({EXPECTED_KV_HEADS} local KV heads, {PAYLOAD_BYTES} payload "
                f"bytes). This pool has {head_num} local KV heads "
                f"({expected_payload} bytes).{tp_note} TP-sharded record formats "
                f"are future work."
            )
        raise ValueError(
            f"INT8 record payload is {PAYLOAD_BYTES} bytes but this pool has "
            f"head_num*head_dim={expected_payload} elements per row."
        )
    expected_scales = head_num * 2
    if SCALE_BYTES != expected_scales:
        raise ValueError(
            f"INT8 record reserves {SCALE_BYTES} scale bytes but one BF16 scale "
            f"per head_num={head_num} needs {expected_scales}."
        )
    if ROW_BYTES % ALIGNMENT_BYTES != 0:
        raise ValueError(
            f"INT8 record is {ROW_BYTES} bytes, not a multiple of "
            f"{ALIGNMENT_BYTES}; the CUDA JIT HiCache mover requires "
            f"element_size % {ALIGNMENT_BYTES} == 0."
        )
    if itemsize != 2:
        raise ValueError(
            f"INT8 record assumes 2-byte source elements (BF16/FP16), got {itemsize}."
        )


def bytes_per_token(layer_num: int) -> int:
    """Encoded host bytes for one token across all layers, K and V combined."""
    return 2 * layer_num * ROW_BYTES


def compute_scales(x: torch.Tensor) -> torch.Tensor:
    """Per-head scales for ``[..., head_num, head_dim]``, same dtype as ``x``."""
    amax = x.abs().amax(dim=-1).clamp_min(AMAX_FLOOR)
    return amax / QUANT_MAX


def quantize_rows(x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Quantise to INT8 with per-head ``scales``.

    The quotient is formed in **float32**. A BF16 division near full scale has
    ~0.25 of a scale step of granularity (8 mantissa bits at magnitude 127),
    which would push the normalised reconstruction error from ``<= 0.5`` to
    ``~0.75``.

    The clamp is load-bearing, not defensive: with the scale rounded to BF16,
    ``|x / s|`` can reach ``127 * (1 + 2**-8) < 128``, so ``-128`` is never
    produced and the range stays symmetric.
    """
    quotient = (x.float() / scales.float().unsqueeze(-1)).round()
    return quotient.clamp_(-QUANT_MAX, QUANT_MAX).to(torch.int8)


def dequantize_rows(
    q: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    """Restore ``q * s`` in ``dtype`` (BF16 for L1 attention)."""
    return q.to(dtype) * scales.unsqueeze(-1)


def encode_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[..., H, D]`` -> ``(int8 payload, bf16 scales)``."""
    scales = compute_scales(x)
    return quantize_rows(x, scales), scales


def decode_records(
    records: torch.Tensor, *, head_num: int, head_dim: int, dtype: torch.dtype
) -> torch.Tensor:
    """``[..., ROW_BYTES]`` uint8 -> ``[..., head_num, head_dim]`` ``dtype``."""
    batch = records.shape[:-1]
    payload = records[..., :PAYLOAD_BYTES].contiguous()
    payload = payload.view(torch.int8).reshape(*batch, head_num, head_dim)
    scale_bytes = records[..., SCALE_OFFSET : SCALE_OFFSET + head_num * 2].contiguous()
    scales = scale_bytes.view(torch.bfloat16).reshape(*batch, head_num)
    return dequantize_rows(payload, scales, dtype)


def write_record(
    x: torch.Tensor, dst: torch.Tensor, *, scales: torch.Tensor | None = None
) -> torch.Tensor:
    """Encode ``[N, H, D]`` directly into ``dst`` ``[N, ROW_BYTES]`` uint8.

    Writes in place so the caller can reuse a persistent staging buffer instead
    of allocating a fresh record tensor per layer on the transfer stream.
    Only the payload and the scale block are written; padding is left as the
    zero it was allocated with, which keeps the record deterministic.
    """
    if scales is None:
        scales = compute_scales(x)
    n = x.shape[0]
    payload = quantize_rows(x, scales)
    dst[:n, :PAYLOAD_BYTES] = payload.reshape(n, -1).view(torch.uint8)
    dst[:n, SCALE_OFFSET : SCALE_OFFSET + scales.shape[-1] * 2] = scales.reshape(
        n, -1
    ).view(torch.uint8)
    return dst[:n]
