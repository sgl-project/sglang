"""Env-gated, read-only probe that dumps DECODE-mode attention metadata.

Purpose
-------
Localize the Kimi-K2.6 2N-1P1D disagg non-MTP GSM8K drop (~0.88 vs 0.944
single-node). Prior analysis narrowed the cause to the *decode-mode* read of
the MORI-transferred prefix: both the triton and the aiter decode kernels are
fed the SAME scheduler-built metadata (``kv_indptr`` / ``kv_indices`` /
``seq_lens`` / ``num_kv_splits``) derived from ``req_to_token``, and that is
their only shared input. The verify/extend path uses a different metadata set
(``qo_indptr`` / ``custom_mask``), which is why MTP verify over the same KV is
correct. This probe dumps exactly that shared decode metadata plus the KV-pool
row norms at the attended slots, on REAL decode steps (not warmup), so a fixed
prompt can be diffed single-node vs disagg to find the first divergence.

Enable
------
  SGLANG_DEBUG_DISAGG_DECODE_META=1

Optional knobs:
  SGLANG_DEBUG_DISAGG_DECODE_META_LAYERS=0    comma list of layer_ids to dump
  SGLANG_DEBUG_DISAGG_DECODE_META_STEPS=8     max decode forwards to dump
  SGLANG_DEBUG_DISAGG_DECODE_META_MINLEN=16   skip reqs with seq_len below this
                                              (filters out the tiny warmup probe)
  SGLANG_DEBUG_DISAGG_DECODE_META_KVNORM=1    also dump KV-pool row norms at the
                                              attended slots (default on)

How to read the diff
--------------------
Run the SAME fixed prompt (bs=1) on single-node and on disagg, then compare:
  * seq_len at each step  -> must match; a mismatch is an off-by-one in the
    disagg decode seq_len bookkeeping.
  * KV-norm head/tail sequence for the prefix -> same prompt must give the same
    norms regardless of physical slots; a mismatch means the transferred KV
    values are wrong.
  * zero_rows -> must be 0; >0 means the attended prefix slots were never
    written (transfer/placement problem, on a real request this time).
  * idx==rtt -> sanity; kv_indices is a copy of req_to_token[req, :seq_len].

The probe is wrapped in try/except and skips CUDA/HIP graph capture, so it can
never break a run.
"""

from __future__ import annotations

import logging
import os
import threading

import torch

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() not in ("0", "", "false", "no")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


_ENABLED = _env_flag("SGLANG_DEBUG_DISAGG_DECODE_META")
_MAX_STEPS = _env_int("SGLANG_DEBUG_DISAGG_DECODE_META_STEPS", 8)
_MIN_SEQ_LEN = _env_int("SGLANG_DEBUG_DISAGG_DECODE_META_MINLEN", 16)
_KVNORM = _env_flag("SGLANG_DEBUG_DISAGG_DECODE_META_KVNORM", "1")


def _parse_layers() -> set:
    raw = os.environ.get("SGLANG_DEBUG_DISAGG_DECODE_META_LAYERS", "0")
    out = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if tok:
            try:
                out.add(int(tok))
            except ValueError:
                pass
    return out or {0}


_LAYERS = _parse_layers()
_FIRST_LAYER = min(_LAYERS)

_lock = threading.Lock()
_step = 0


def _ht(values, k: int = 6) -> str:
    """Compact head/tail view of a 1-D sequence."""
    vals = list(values)
    if len(vals) <= 2 * k:
        return str(vals)
    return f"{vals[:k]}...{vals[-k:]}"


def maybe_dump_decode_meta(tag: str, backend, layer, forward_batch) -> None:
    """Dump decode metadata for the first few real decode forwards.

    ``tag`` distinguishes the backend ("triton" / "aiter"). ``backend`` is the
    attention backend instance (must expose ``forward_metadata``,
    ``req_to_token_pool`` and ``token_to_kv_pool``). Safe no-op unless the
    ``SGLANG_DEBUG_DISAGG_DECODE_META`` env var is set.
    """
    if not _ENABLED:
        return
    try:
        _dump_impl(tag, backend, layer, forward_batch)
    except Exception as exc:  # never break the model forward
        logger.warning("[DDM] probe failed (%s): %r", tag, exc)


def _dump_impl(tag: str, backend, layer, forward_batch) -> None:
    global _step

    # Never run during CUDA/HIP graph capture: the .cpu()/.item() syncs below
    # are illegal while tracing a graph.
    try:
        from sglang.srt.model_executor.runner import get_is_capture_mode

        if get_is_capture_mode():
            return
    except Exception:
        pass

    layer_id = getattr(layer, "layer_id", 0)
    if layer_id not in _LAYERS:
        return

    fwd_mode = forward_batch.forward_mode
    is_decode = fwd_mode.is_decode() or fwd_mode.is_idle()
    is_verify = fwd_mode.is_target_verify()
    is_extend = (
        fwd_mode.is_extend() and not is_verify and not fwd_mode.is_draft_extend_v2()
    )
    # DECODE/VERIFY fire on the decode worker; EXTEND fires on the prefill
    # worker (its prompt extend) -> one disagg run yields both the
    # prefill-stored and decode-received latent norms for the same tokens.
    if not (is_decode or is_verify or is_extend):
        return

    seq_lens = getattr(forward_batch, "seq_lens_cpu", None)
    if seq_lens is None:
        seq_lens = forward_batch.seq_lens.detach().to("cpu")
    seq_lens = seq_lens.tolist()
    bs = len(seq_lens)

    # Skip the tiny warmup probe ("The capital of France is", seq_len ~6-9).
    if max(seq_lens, default=0) < _MIN_SEQ_LEN:
        return

    # One global step per qualifying decode forward; keyed on the first dumped
    # layer so all target layers of the same forward share a step id.
    with _lock:
        if layer_id == _FIRST_LAYER:
            _step += 1
        cur_step = _step
    if cur_step > _MAX_STEPS:
        return

    fmeta = getattr(backend, "forward_metadata", None)
    kv_indptr = getattr(fmeta, "kv_indptr", None)
    kv_indices = getattr(fmeta, "kv_indices", None)
    num_kv_splits = getattr(fmeta, "num_kv_splits", None)

    req_pool_indices = forward_batch.req_pool_indices.detach().to("cpu").tolist()
    req_to_token = backend.req_to_token_pool.req_to_token

    kv_indptr_cpu = (
        kv_indptr[: bs + 1].detach().to("cpu").tolist()
        if kv_indptr is not None
        else None
    )

    mode_name = "VERIFY" if is_verify else ("EXTEND" if is_extend else "DECODE")
    lines = []
    for b in range(bs):
        seq_len = int(seq_lens[b])
        if seq_len < _MIN_SEQ_LEN:
            continue
        rp = int(req_pool_indices[b])

        # req_to_token slot mapping for this request's tokens.
        rtt = req_to_token[rp, :seq_len].detach().to("cpu")

        # kv_indices slice for this request (CSR).
        idx_slice = None
        n_kv = None
        idx_str = "n/a"
        match_str = "n/a"
        if kv_indptr_cpu is not None and kv_indices is not None:
            lo, hi = int(kv_indptr_cpu[b]), int(kv_indptr_cpu[b + 1])
            n_kv = hi - lo
            idx_slice = kv_indices[lo:hi].detach().to("cpu")
            idx_str = _ht(idx_slice.tolist())
            # kv_indices == req_to_token[req, :seq_len] only for DECODE; EXTEND's
            # kv_indices covers the prefix (empty for a pure prefill), so the
            # equality check is decode-only to avoid a spurious LEN_MISMATCH.
            if not is_decode:
                match_str = "n/a(non-decode)"
            elif idx_slice.numel() == rtt.numel():
                match_str = str(bool(torch.equal(idx_slice.to(rtt.dtype), rtt)))
            else:
                match_str = f"LEN_MISMATCH({idx_slice.numel()} vs {rtt.numel()})"

        nsplit_str = "n/a"
        if num_kv_splits is not None:
            try:
                nsplit_str = str(int(num_kv_splits[b]))
            except Exception:
                nsplit_str = "?"

        block = (
            f"[DDM step={cur_step} tag={tag} L{layer_id} mode={mode_name}] "
            f"rp={rp} seq_len={seq_len} kv_indptr=[{kv_indptr_cpu[b] if kv_indptr_cpu else '?'},"
            f"{kv_indptr_cpu[b + 1] if kv_indptr_cpu else '?'}] n_kv={n_kv} "
            f"nsplit={nsplit_str} idx==rtt:{match_str}\n"
            f"    kv_idx  = {idx_str}\n"
            f"    rtt     = {_ht(rtt.tolist())}"
        )

        # KV-pool row norms at req_to_token[req, :seq_len] (real step, unlike the
        # warmup-only #30433 dump). This is the SAME quantity on both sides:
        # what the prefill worker stored (EXTEND) vs what the decode worker reads
        # (DECODE). Same prompt (shared 8-shot prefix) => identical norm sequence
        # iff the MORI transfer preserved the latent values.
        if _KVNORM and layer_id == _FIRST_LAYER:
            block += "\n" + _kv_norm_report(backend, layer_id, rtt)

        lines.append(block)

    if lines:
        logger.info("\n".join(lines))


def _kv_norm_report(backend, layer_id: int, slots: torch.Tensor) -> str:
    try:
        pool = backend.token_to_kv_pool
        kbuf = pool.get_key_buffer(layer_id)
        dev_slots = slots.to(device=kbuf.device, dtype=torch.long)
        rows = kbuf[dev_slots].reshape(dev_slots.shape[0], -1).float()
        norms = rows.norm(dim=-1)
        zero_rows = int((norms < 1e-6).sum().item())
        norms_list = [round(x, 3) for x in norms.detach().to("cpu").tolist()]
        return (
            f"    KV L{layer_id}: zero_rows={zero_rows}/{norms.numel()} "
            f"norm={_ht(norms_list)}"
        )
    except Exception as exc:
        return f"    KV L{layer_id}: norm probe failed: {exc!r}"
