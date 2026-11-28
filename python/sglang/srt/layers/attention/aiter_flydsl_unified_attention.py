"""AITER's FlyDSL BF16 unified attention in front of its Triton unified attention.

With SGLANG_USE_AITER_FLYDSL_UNIFIED_ATTN=1, every unified-attention call of the
AITER backend first goes to ``aiter.ops.flydsl.unified_attention_bf16``, which
covers gfx950, BF16 linear KV, Q/K head 192 with V head 128, 16 query heads per
KV head and page size 64 or 128 (MiMo-V2.6's geometry; run with --page-size 64
or 128). Calls it declines run the Triton kernel with the caller's arguments
unchanged.

FlyDSL takes an int32 ``seqused_k`` and ``block_table``. The Triton kernel keeps
the caller's ``seqused_k``, which may be int64.
"""

import logging

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Keyword arguments the Triton entry takes and the FlyDSL entry does not; they
# are hints for Triton's kernel choice, not part of the attention result.
_TRITON_ONLY_KWARGS = ("equal_seqlen_q", "attn_work")

# Every layer of a forward passes the same seqused_k tensor, so one int32 copy
# serves them all. seq_lens is updated in place between forwards, so the copy is
# dropped in init_forward_metadata. A copy made outside a graph capture must not
# be read inside one, hence the capture flag in the key.
_seqused_k_cache = None
_logged = set()


def reset_flydsl_unified_attention_cache():
    global _seqused_k_cache
    _seqused_k_cache = None


def _int32(t, cacheable):
    global _seqused_k_cache
    if t.dtype == torch.int32:
        return t
    if not cacheable:
        return t.to(torch.int32)
    capturing = torch.cuda.is_current_stream_capturing()
    c = _seqused_k_cache
    if c is None or c[0] is not t or c[1] != capturing:
        c = _seqused_k_cache = (t, capturing, t.to(torch.int32))
    return c[2]


def _log_once(key, msg, *args):
    if key not in _logged:
        _logged.add(key)
        logger.info(msg, *args)


def _load_flydsl():
    try:
        from aiter.ops.flydsl import unified_attention_bf16
    except ImportError as e:
        logger.warning(
            "SGLANG_USE_AITER_FLYDSL_UNIFIED_ATTN is set, but this AITER has no "
            "FlyDSL unified attention (%s); using Triton unified attention.",
            e,
        )
        return None
    logger.info("AITER FlyDSL unified attention enabled for the AITER backend.")
    return unified_attention_bf16


def with_flydsl_unified_attention(triton_unified_attention):
    """Return ``triton_unified_attention``, with FlyDSL in front when enabled."""
    if not envs.SGLANG_USE_AITER_FLYDSL_UNIFIED_ATTN.get():
        return triton_unified_attention
    flydsl = _load_flydsl()
    if flydsl is None:
        return triton_unified_attention
    declines = [0]

    def unified_attention(**kwargs):
        fly_kwargs = {k: v for k, v in kwargs.items() if k not in _TRITON_ONLY_KWARGS}
        fly_kwargs["seqused_k"] = _int32(kwargs["seqused_k"], cacheable=True)
        fly_kwargs["block_table"] = _int32(kwargs["block_table"], cacheable=False)
        if flydsl.unified_attention_flydsl(**fly_kwargs) is not None:
            _log_once(
                ("run", tuple(kwargs["window_size"] or (-1, -1))),
                "FlyDSL unified attention runs: window=%s q=%s k=%s",
                kwargs["window_size"],
                tuple(kwargs["q"].shape),
                tuple(kwargs["k"].shape),
            )
            return kwargs["out"]
        declines[0] += 1
        # Naming the reason repeats the host checks, so only sample it.
        if declines[0] <= 256 or declines[0] % 4096 == 0:
            reason = flydsl.flydsl_decline_reason(**fly_kwargs) or (
                "accepted, but no workspace for a captured call"
            )
            _log_once(
                ("decline", reason),
                "FlyDSL unified attention declined, running Triton: %s (q=%s k=%s)",
                reason,
                tuple(kwargs["q"].shape),
                tuple(kwargs["k"].shape),
            )
        return triton_unified_attention(**kwargs)

    return unified_attention
