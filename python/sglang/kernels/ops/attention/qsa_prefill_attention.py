"""QSA fallback for arbitrary physical-token selections.

The CANN 9.3 grouped prefill path lives in qsa_gbsa_prefill and receives
sequence metadata directly from the attention backend.
"""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _report_fallback():
    logger.warning(
        "QSA uses sparse attention fallback: native grouped prefill is "
        "disabled or this call needs arbitrary token-slot attention."
    )


def qsa_prefill_attention(q, k, v, slots, softmax_scale=None):
    """Preserve QSA masks, empty rows, and slot semantics on CANN 9.2.0."""
    from sgl_kernel_npu.operators.attention import qsa_sparse_attention

    _report_fallback()
    return qsa_sparse_attention(q, k, v, slots, softmax_scale)


def qsa_decode_attention(q, k, v, slots, softmax_scale=None):
    """The experimental decode GBSA switch also falls back on this runtime."""
    return qsa_prefill_attention(q, k, v, slots, softmax_scale)
