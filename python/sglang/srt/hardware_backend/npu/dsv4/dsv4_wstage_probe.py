"""Read-only, env-gated write-side staged probe for the DSV4/A5 divergence.

The observable symptom is ``cache-HIT != MISS`` with the first divergence at
model **layer 0 OUTPUT, pos 17598**, on the NPU/A5 bf16 ``store_cache`` KV-write
path. This probe emits, per ABSOLUTE position, the md5 of the KV row at each
stage of the L0 KV write, so a MISS-vs-HIT diff at the SAME position isolates:

    S0  pre-GEMM layer input    ``x_linear``                          (optional)
    S1  post-GEMM bf16 kv       before ``self.kv_norm``
    S2  post-norm bf16 kv       after ``self.kv_norm``, before RoPE
    S3  post-rope bf16 kv       after ``Dsv4NpuRoPE.apply_rotary_mul_inplace``
    S4  pre-epilog bf16 row     input row to ``kv_compress_epilog``
    S5  stored packed fp8 row   cache row after ``kv_compress_epilog`` (same slot)

Enabled **only** when ``DSV4_DUMP_WSTAGE`` is truthy; emits **only** positions
``p`` in ``[DSV4_DUMP_MIN_POS, DSV4_DUMP_MAX_POS]``. Read-only: no tensor is
mutated, and the disabled path is a single env lookup that returns an empty row
list. Every hashing guard is wrapped in try/except so a probe fault can never
break the forward.

Emitted format (one line per position per stage)::

    [WSTAGE] layer=<L> pos=<p> stage=<S> md5=<h>

Run (NPU host)::

    DSV4_DUMP_WSTAGE=1 DSV4_DUMP_MIN_POS=17598 DSV4_DUMP_MAX_POS=17598 \
        python -m sglang.launch_server ... 2>&1 | grep WSTAGE

S1-S4 are bf16 (hashed via a loss-free fp32 expansion); S5 is the raw stored
packed row (fp8/int8 bytes), hashed byte-for-byte.
"""

from __future__ import annotations

import hashlib
import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_MD5_LEN = 16


def enabled() -> bool:
    """True iff ``DSV4_DUMP_WSTAGE`` is set to a truthy value."""
    try:
        return os.environ.get("DSV4_DUMP_WSTAGE", "").strip().lower() in _TRUTHY
    except Exception:  # noqa: BLE001 - probe must never raise
        return False


def _window() -> tuple[int | None, int | None]:
    """(min_pos, max_pos) from env, or (None, None) when unset/unparsable."""
    lo = os.environ.get("DSV4_DUMP_MIN_POS")
    hi = os.environ.get("DSV4_DUMP_MAX_POS")
    try:
        lo_i = int(lo) if lo is not None else None
    except (TypeError, ValueError):
        lo_i = None
    try:
        hi_i = int(hi) if hi is not None else None
    except (TypeError, ValueError):
        hi_i = None
    return lo_i, hi_i


def select_rows(positions) -> list[tuple[int, int]]:
    """Return ``[(row_idx, abs_pos)]`` for ``abs_pos`` in the dump window.

    Empty list (a cheap no-op) when the probe is disabled, the window is not
    fully specified, or anything goes wrong. ``positions`` is the forward
    batch's ``[T]`` absolute-position tensor; row ``r`` of the stage tensor is
    token ``r``, whose absolute position is ``positions[r]``.
    """
    if not enabled():
        return []
    try:
        lo, hi = _window()
        if lo is None or hi is None:
            return []
        import torch

        pos = positions.detach().reshape(-1).to(torch.int64).cpu().numpy()
        return [
            (r, int(pos[r])) for r in range(pos.shape[0]) if lo <= int(pos[r]) <= hi
        ]
    except Exception as exc:  # noqa: BLE001 - probe must never break the forward
        print(f"[WSTAGE] select_rows skipped: {exc}", flush=True)
        return []


def _emit(stage: str, layer: int, pos: int, digest: str) -> None:
    print(
        f"[WSTAGE] layer={layer} pos={pos} stage={stage} md5={digest}",
        flush=True,
    )


def emit_bf16(stage: str, layer, tensor, rows: list[tuple[int, int]]) -> None:
    """Hash rows of a bf16 stage tensor (S0-S4) as fp32-expanded bytes.

    ``tensor`` is ``[T, ...]``; only rows named in ``rows`` are read/cast, so the
    cost is O(window), not O(T).
    """
    if not rows or tensor is None:
        return
    try:
        import torch

        n = int(tensor.shape[0])
        flat = tensor.detach().reshape(n, -1)
        for r, p in rows:
            if r < 0 or r >= n:
                continue
            row = flat[r].to(torch.float32).cpu().numpy()
            _emit(stage, layer, p, hashlib.md5(row.tobytes()).hexdigest()[:_MD5_LEN])
    except Exception as exc:  # noqa: BLE001
        print(f"[WSTAGE] {stage} skipped: {exc}", flush=True)


def emit_raw_rows(
    stage: str,
    layer,
    buf_flat,
    slots,
    rows: list[tuple[int, int]],
) -> None:
    """Hash the raw stored bytes of the packed cache row at each selected slot (S5).

    ``buf_flat`` is ``[N, W]`` (fp8/int8 packed rows); ``slots`` is ``[T]``; row
    ``r`` is stored at slot ``slots[r]``. Hashes byte-for-byte so the fp8/int8
    encoding, not a dequantized float, is compared.
    """
    if not rows or buf_flat is None or slots is None:
        return
    try:
        import torch

        n = int(buf_flat.shape[0])
        flat = buf_flat.detach().reshape(n, -1)
        slot_list = slots.detach().reshape(-1)
        for r, p in rows:
            if r < 0 or r >= slot_list.numel():
                continue
            s = int(slot_list[r].item())
            if s < 0 or s >= n:
                continue
            raw = (
                flat[s]
                .contiguous()
                .view(torch.uint8)
                .cpu()
                .numpy()
                .tobytes()
            )
            _emit(stage, layer, p, hashlib.md5(raw).hexdigest()[:_MD5_LEN])
    except Exception as exc:  # noqa: BLE001
        print(f"[WSTAGE] {stage} skipped: {exc}", flush=True)
