"""FlashInfer's batch-invariant quantizer classes under the deterministic mode.

Under ``--enable-deterministic-inference`` (and the batch-invariant mode it
turns on) the activation quantizers run FlashInfer's batch-invariant classes,
which compute TransformerEngine's quantization chains and carry a fingerprint
(``flashinfer/batch_invariant/README.md``, section "Quantization"). The
predicate below never reads the batch size: a provider that changed with M
would break batch invariance even though each provider is invariant on its own.
"""

from __future__ import annotations

import functools

import torch

from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
    is_batch_invariant_mode_enabled,
)
from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_context, get_exec
from sglang.srt.utils.device import is_flashinfer_available

# Compute capabilities of FlashInfer's batch-invariant quantizer classes.
_FLASHINFER_BI_QUANT_ARCHS = ((10, 0), (10, 3), (10, 7))


@functools.lru_cache(maxsize=None)
def flashinfer_bi_quant_supported(device_index: int) -> bool:
    if not is_flashinfer_available():
        return False
    try:
        # FlashInfer >= 0.7.4
        from flashinfer.batch_invariant import is_supported_device
    except ImportError:
        return False
    return is_supported_device(
        torch.device("cuda", device_index), _FLASHINFER_BI_QUANT_ARCHS
    )


def deterministic_quant_mode_enabled() -> bool:
    """The batch-invariant mode or ``--enable-deterministic-inference``."""
    if is_batch_invariant_mode_enabled():
        return True
    # Kernel tests and offline tools run without a published exec config.
    if not get_context().is_config_namespace_published("exec"):
        return False
    return get_exec().deterministic.enable_deterministic_inference


def use_flashinfer_bi_quantizers(device: torch.device) -> bool:
    """Whether activation quantization on ``device`` runs FlashInfer's
    batch-invariant classes: the deterministic mode is on, the opt-out
    ``SGLANG_DETERMINISTIC_FLASHINFER_QUANT=0`` is not set and the device is one
    the classes cover. Depends on the device only, never on M."""
    if not deterministic_quant_mode_enabled():
        return False
    if not envs.SGLANG_DETERMINISTIC_FLASHINFER_QUANT.get():
        return False
    if device.type != "cuda":
        return False
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return flashinfer_bi_quant_supported(index)
