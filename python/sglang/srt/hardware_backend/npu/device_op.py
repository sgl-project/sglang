"""Runtime-visible Ascend generation contracts.

Generation differences that fit behind one stable standalone operator belong in
``sgl-kernel-npu``, where the provider is picked while its wheel is built. What
lands here is the rest: dtype, layout, metadata and call-shape contracts that
SGLang runtime components have to agree on *before* a kernel is reached, and
which therefore cannot be hidden inside one operator (RFC #35709).

A device operator says how an already-selected feature runs on the current
generation. It does not decide whether that feature is enabled -- that stays
with the feature's own config -- and it does not select kernel providers.

The concrete object is resolved once from the SoC version; feature code reads
named contracts off it instead of testing the generation itself, which keeps
generation checks out of the inference hot path.
"""

from __future__ import annotations

import functools
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# torch_npu.npu.get_soc_version() code for the Ascend 950 (A5) series.
SOC_VERSION_ASCEND950 = 260


class NPUDeviceOperator:
    """Ascend 910 (A2/A3) contracts, and the fallback for an unknown SoC."""

    target = "Ascend910"

    # Whether a mixed prefill+decode batch has to be issued as two FIA calls.
    # 910 takes the whole batch in one npu_fused_infer_attention_score call.
    fia_splits_mixed_batch = False


class Ascend950DeviceOperator(NPUDeviceOperator):
    """Ascend 950 (A5) contracts."""

    target = "Ascend950"

    fia_splits_mixed_batch = True


_DEVICE_OPS_BY_SOC_VERSION = {
    SOC_VERSION_ASCEND950: Ascend950DeviceOperator,
}


def _soc_version() -> Optional[int]:
    try:
        import torch_npu

        return torch_npu.npu.get_soc_version()
    except (ImportError, AttributeError, RuntimeError) as error:
        logger.warning(
            "Failed to query the Ascend SoC version (%s); falling back to the "
            "%s contracts.",
            error,
            NPUDeviceOperator.target,
        )
        return None


@functools.cache
def get_npu_device_op() -> NPUDeviceOperator:
    """Return the device operator for the running Ascend generation."""
    device_op = _DEVICE_OPS_BY_SOC_VERSION.get(_soc_version(), NPUDeviceOperator)()
    logger.info("Ascend runtime contracts: %s", device_op.target)
    return device_op
