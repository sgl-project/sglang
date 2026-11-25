import logging
from typing import Optional

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_MAX_GROUP = 5
_GROUP_SHIFT = 26
_OPS = None
_WS = {}  # (device index, name) -> grow-only workspace (packed indices, lengths)
_CALLS = {"packed": 0, "unpacked": 0, "declined": 0}
_DECLINE_REASONS = {}
_DISABLED = None  # the error that turned the reader off for this process
_WARNED_UNAVAILABLE = False


class _OverlapGuard:
    def __init__(self, max_fraction, probe_every=64, hold=1024):
        self.max_fraction = max_fraction
        self.probe_every = probe_every
        self.hold = hold
        self.until_probe = 0
        self.pending = None
        self.unpacked_left = 0
        self.last_fraction = None
        self.switches = 0

    def want_packed(self, capturing: bool) -> bool:
        if self.max_fraction >= 1.0:
            return True
        if capturing:
            # Neither read nor count inside a capture; the graph keeps the current choice.
            return self.unpacked_left == 0
        if self.pending is not None:
            self.last_fraction = float(self.pending)
            self.pending = None
            if self.last_fraction > self.max_fraction:
                self.unpacked_left = self.hold
                self.switches += 1
                if self.switches in (1, 10, 100, 1000):
                    logger.info(
                        "Kimi-K3.1 sparse prefill reader: union fraction %.2f above %.2f, next %d calls unpacked "
                        "(switch %d)",
                        self.last_fraction,
                        self.max_fraction,
                        self.hold,
                        self.switches,
                    )
        if self.unpacked_left > 0:
            self.unpacked_left -= 1
            if self.unpacked_left == 0:
                self.until_probe = 0
            return False
        return True

    def observe(self, union_lengths, valid_lengths, capturing: bool) -> None:
        if self.max_fraction >= 1.0 or capturing:
            return
        if self.until_probe > 0:
            self.until_probe -= 1
            return
        self.until_probe = self.probe_every - 1
        self.pending = union_lengths.sum() / valid_lengths.sum().clamp_min(1)


_GUARD = None


def _guard() -> _OverlapGuard:
    global _GUARD
    if _GUARD is None:
        _GUARD = _OverlapGuard(envs.SGLANG_K31_PREFILL_READER_MAX_UNION_FRACTION.get())
    return _GUARD


def reader_stats() -> dict:
    guard = _guard()
    return {
        **_CALLS,
        "declined_reasons": dict(_DECLINE_REASONS),
        "disabled": repr(_DISABLED) if _DISABLED is not None else None,
        "last_union_fraction": guard.last_fraction,
        "overlap_switches": guard.switches,
    }


def _note(kind, q, indices, reason=None):
    _CALLS[kind] += 1
    if reason is not None:
        first = reason not in _DECLINE_REASONS
        _DECLINE_REASONS[reason] = _DECLINE_REASONS.get(reason, 0) + 1
        if first:
            logger.info(
                "Kimi-K3.1 sparse prefill reader declines a call (stock path runs): %s",
                reason,
            )
    n = sum(_CALLS.values())
    if n in (1, 100, 10000, 1000000):
        logger.info(
            "Kimi-K3.1 sparse prefill reader call %d: %s, q %s, top-k %s, totals %s",
            n,
            kind,
            tuple(q.shape),
            tuple(indices.shape),
            _CALLS,
        )


def _flash_mla():
    from sglang.kernels.ops.attention import flash_mla_interface

    return flash_mla_interface


def _load():
    global _OPS
    if _OPS is None:
        fm = _flash_mla()
        if not fm.supports_sparse_prefill_token_packing():
            raise RuntimeError(
                "the installed sgl-flash-mla has no sparse prefill with token packing "
                "(flash_mla_sparse_fwd_packed)"
            )
        _OPS = (
            fm.flash_mla_sparse_fwd_packed,
            fm.flash_mla_sparse_pack_indices,
            fm.flash_mla_dequant_v32_kv,
        )
    return _OPS


def k31_prefill_reader_available() -> bool:
    global _WARNED_UNAVAILABLE
    try:
        _load()
        return True
    except Exception as e:
        if not _WARNED_UNAVAILABLE:
            _WARNED_UNAVAILABLE = True
            logger.warning(
                "Kimi-K3.1 sparse prefill reader unavailable, the stock path runs: %r",
                e,
            )
        return False


def _ws(name, like, shape, dtype, capturing=False):
    if capturing:
        # A captured graph keeps the pointers it saw; the shared workspace can move.
        return torch.empty(tuple(shape), dtype=dtype, device=like.device)
    key = (like.device.index, name)
    w = _WS.get(key)
    if w is None or any(a < b for a, b in zip(w.shape, shape)):
        grow = (
            tuple(max(b, a) for a, b in zip(w.shape, shape))
            if w is not None
            else tuple(shape)
        )
        w = torch.empty(grow, dtype=dtype, device=like.device)
        _WS[key] = w
    return w


def _forward_packed(q_all, kv, indices, sm_scale, topk_length, capturing=False):
    op, pack, _ = _load()
    T, h = q_all.shape[0], q_all.shape[1]
    topk = indices.shape[1]
    G = min(64 // h, _MAX_GROUP)
    n = T // G
    out = torch.empty((T, h, 512), dtype=q_all.dtype, device=q_all.device)
    if n:
        W = (G * topk + 63) // 64 * 64
        pi = _ws("pi", q_all, (n, W), torch.int32, capturing)
        ln = _ws("ln", q_all, (n,), torch.int32, capturing)
        pack(indices[: n * G], G, pi, ln)
        _guard().observe(ln[:n], topk_length[: n * G], capturing)
        op(
            q_all[: n * G].view(n, G * h, 576),
            kv,
            pi[:n, :W].unsqueeze(1),
            float(sm_scale),
            h,
            ln[:n],
            out[: n * G].view(n, G * h, 512),
        )
    if n * G < T:
        op(
            q_all[n * G :],
            kv,
            indices[n * G :].unsqueeze(1),
            float(sm_scale),
            0,
            topk_length[n * G :],
            out[n * G :],
        )
    return out


def _decline_reason(q, q_rope, kv_cache, indices, max_bf16_bytes) -> Optional[str]:
    if q.dtype != torch.bfloat16 or q.dim() != 3:
        return "q is not a 3-d bf16 tensor"
    if q.shape[0] == 0:
        return "empty batch"
    if not 1 <= q.shape[1] <= 32:
        return "head count outside 1..32"
    if q.shape[2] != (512 if q_rope is not None else 576):
        return "query width is not 512 + 64"
    if q_rope is not None and (
        q_rope.dtype != q.dtype or q_rope.shape != (*q.shape[:2], 64)
    ):
        return "q_rope does not match q"
    if kv_cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
        return "KV cache is not FP8"
    if (
        kv_cache.dim() != 4
        or kv_cache.shape[1:] != (64, 1, 656)
        or not kv_cache.is_contiguous()
    ):
        return "KV buffer is not contiguous V32 pages"
    if kv_cache.shape[0] * 64 > 1 << _GROUP_SHIFT:
        return "more KV rows than the token-group bits leave room for"
    if kv_cache.shape[0] * 64 * 1152 > max_bf16_bytes:
        return "BF16 copy of the gathered rows above the size limit"
    if indices.dim() != 2 or indices.dtype != torch.int32 or indices.stride(1) != 1:
        return "top-k ids are not contiguous int32 rows"
    if indices.shape[0] < q.shape[0]:
        return "fewer top-k rows than query tokens"
    if not 128 <= indices.shape[1] <= 2048 or indices.shape[1] % 64:
        return "top-k width is not a multiple of 64 in 128..2048"
    return None


def k31_prefill_reader(
    q: torch.Tensor,
    q_rope: Optional[torch.Tensor],
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    pack: bool = True,
    max_bf16_bytes: int = 1 << 30,
) -> Optional[torch.Tensor]:
    """Returns [T, 1, H, 512], or None when the caller should run the stock path."""
    global _DISABLED
    if _DISABLED is not None:
        return None
    reason = _decline_reason(q, q_rope, kv_cache, indices, max_bf16_bytes)
    if reason is not None:
        _note("declined", q, indices, reason)
        return None
    try:
        return _run(q, q_rope, kv_cache, indices[: q.shape[0]], sm_scale, pack)
    except RuntimeError as error:
        _DISABLED = error
        logger.error(
            "Kimi-K3.1 sparse prefill reader failed and is disabled for this process; the stock path runs: %r",
            error,
        )
        _note("declined", q, indices, "disabled after a runtime error")
        return None


def _run(q, q_rope, kv_cache, indices, sm_scale, pack):
    _, _, dequant = _load()
    capturing = q.is_cuda and torch.cuda.is_current_stream_capturing()
    rows = kv_cache.shape[0] * 64
    kv = torch.empty((rows, 1, 576), dtype=torch.bfloat16, device=kv_cache.device)
    dequant(kv_cache, kv)
    q_all = torch.cat([q, q_rope], dim=-1) if q_rope is not None else q.contiguous()
    T, h = q_all.shape[0], q_all.shape[1]
    topk_length = (indices >= 0).sum(dim=1, dtype=torch.int32)
    if (
        pack
        and indices.shape[1] * min(64 // h, _MAX_GROUP) <= 12288
        and _guard().want_packed(capturing)
    ):
        out = _forward_packed(q_all, kv, indices, sm_scale, topk_length, capturing)
        _note("packed", q, indices)
    else:
        _note("unpacked", q, indices)
        op, _, _ = _load()
        out = op(q_all, kv, indices.unsqueeze(1), float(sm_scale), 0, topk_length)
    return out.view(T, 1, h, 512)


def _reset_for_tests() -> None:
    global _OPS, _GUARD, _DISABLED, _WARNED_UNAVAILABLE
    _OPS = _GUARD = _DISABLED = None
    _WARNED_UNAVAILABLE = False
    _WS.clear()
    _DECLINE_REASONS.clear()
    for key in _CALLS:
        _CALLS[key] = 0
