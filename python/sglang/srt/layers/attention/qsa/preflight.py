"""Early validation for QSA runtime backend dependencies."""

import logging

from sglang.srt.layers.attention.qsa.config import is_qwen_qsa

logger = logging.getLogger(__name__)


def _resolve_qsa_sparse_decode_backend():
    # Keep the large QSA implementation lazy for every non-QSA model.
    from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
        resolve_qsa_sparse_decode_backend,
    )

    return resolve_qsa_sparse_decode_backend()


def preflight_qsa_sparse_decode_backend(*, model_config, tp_rank: int) -> None:
    """Validate and report QSA sparse decode before model weight loading."""

    if not is_qwen_qsa(model_config.hf_text_config):
        return
    backend = _resolve_qsa_sparse_decode_backend()
    if tp_rank == 0:
        logger.info("QSA sparse decode backend: %s", backend.kind.value)


__all__ = ["preflight_qsa_sparse_decode_backend"]
