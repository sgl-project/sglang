"""Host-only QPF/prestore decisions; graph replays do not execute this module.

BUCKET records describe actual CUDA capture, never its eager warmup. The first
QSA layer seen in each mode is the representative (qsa_layer=0); ``layer`` is
always the physical model layer ID. In Qwen3.8 the target's first QSA layer is 3,
while physical layer 0 is GDN. Distinct decisions for the same bucket are both
reported, so an earlier ACTIVE cannot hide a later FALLBACK.
"""

import atexit
from collections import Counter
import json
import logging

logger = logging.getLogger(__name__)
_representatives = {}
_buckets = set()
_fallbacks = {"QPF": Counter(), "KV_PRESTORE": Counter()}
_fallback_totals = {"QPF": 0, "KV_PRESTORE": 0}
_LOG_EVERY = 1024


def _mode(forward_mode):
    if forward_mode.is_target_verify():
        return "verify"
    if forward_mode.is_draft_extend_v2():
        return "draft"
    if forward_mode.is_decode():
        return "decode"
    return None


def flush_fallback_counts():
    """Cumulative eager calls, not CUDA graph replay counts; no device access."""
    for component, counts in _fallbacks.items():
        logger.info(
            "Q38_%s_FALLBACK_COUNT calls=%s scope=non_graph_layer_calls counts=%s",
            component, _fallback_totals[component], json.dumps(dict(sorted(counts.items()))),
        )


def observe(component, layer, forward_batch, rows, *, result, reason, graph,
            capturing, consumer="none"):
    """Only Python scalars / tensor metadata reach here; no CUDA operations."""
    mode = _mode(forward_batch.forward_mode)
    if mode is None:
        return
    bs = forward_batch.batch_size
    if graph and capturing:
        # Eager/autotune warmup may visit a different model before its graph.
        # It must not select the representative for the real capture.
        first_layer = _representatives.setdefault((component, mode), layer)
        if layer != first_layer:
            return
        key = (component, layer, bs, rows, mode, result, reason, consumer)
        if key not in _buckets:
            _buckets.add(key)
            logger.info(
                "Q38_%s_BUCKET bs=%s rows=%s mode=%s result=%s reason=%s "
                "layer=%s qsa_layer=0 capture=1 consumer=%s",
                component, bs, rows, mode, result, reason, layer, consumer,
            )
    elif not graph and result == "FALLBACK":
        counts = _fallbacks[component]
        key = f"layer={layer},bs={bs},rows={rows},mode={mode},reason={reason}"
        counts[key] += 1
        _fallback_totals[component] += 1
        # Sum only once per new count milestone; no per-call JSON or log I/O.
        if _fallback_totals[component] % _LOG_EVERY == 0:
            flush_fallback_counts()


atexit.register(flush_fallback_counts)
