# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Optional Cake MLA value projection and sigmoid gate for the K3 decode.

``MlaValue`` (written by Lee Nau, @leejnau, in sgl-project/sgl-nvidia-mega#17 and
relocated in-tree by @YaoleWu) projects each decode row's BF16 attention latent by
W_UV, rounds the projection to BF16, multiplies it by the FP32 sigmoid of the gate
and writes zero rows for requests with length <= 0. The Cake kernel FlashInfer
ships as ``flashinfer.cake_mla_value_gate`` computes the same function from the
same tensors; SGLang reaches it through ``sglang.kernels.ops.attention``.

Adding ``k3_flashinfer_mla_value`` to ``SGLANG_NVIDIA_MEGAKERNEL_SLOTS`` selects it
wherever ``PreparedMla`` launches MlaValue: the default executor with no profile
token (``community``), ``k3_native_mla`` (DCP1 and DCP8) and ``k3_full_decode``
without DCP. Fused DCP (``k3_full_decode`` with a ``DcpBox``) keeps ``DcpValue``. A
bucket that cannot be prepared keeps MlaValue.
"""

import logging
from typing import Any

import torch

from sglang.kernels.ops import attention as attention_ops
from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

SLOT = "k3_flashinfer_mla_value"
# Device index -> an explicit capability refusal shared by all buckets. Runtime
# and load failures can be transient or bucket-local, so do not cache them here.
_REFUSED: dict[Any, str] = {}


def _refuse(device: Any, reason: str) -> None:
    _REFUSED[device] = reason
    logger.warning(f"Kimi K3 {SLOT}: {reason}; using MlaValue")


def prepare_value_gate(w_uv: torch.Tensor, *, rows: int, q: int) -> Any:
    """Cake runner bound to ``w_uv`` for ``rows`` token rows, ``q`` per request,
    or None to keep MlaValue. Prepare outside CUDA graph capture, after the weight's
    final storage is bound; a replaced storage needs a new runner."""
    if SLOT not in envs.SGLANG_NVIDIA_MEGAKERNEL_SLOTS.get():
        return None
    device = w_uv.device.index
    if device in _REFUSED:
        return None
    try:
        if not attention_ops.can_use_cake_mla_value_gate(w_uv):
            _refuse(device, "needs SM100/SM103 and flashinfer.cake_mla_value_gate")
            return None
        runner = attention_ops.prepare_cake_mla_value_gate(
            w_uv, max_rows=rows, rows_per_request=q
        )
    except (ImportError, NotImplementedError) as error:
        reason = f"FlashInfer cannot prepare ({type(error).__name__}: {error})"
        _refuse(device, reason)
        return None
    # Invalid layouts, capture refusals, OOM and unclassified JIT/load errors
    # affect this preparation only. A RuntimeError is not a capability verdict.
    except Exception as error:
        logger.warning_once(
            f"Kimi K3 {SLOT}: FlashInfer cannot prepare this bucket "
            f"({type(error).__name__}: {error}); using MlaValue"
        )
        return None
    logger.info_once(f"Kimi K3 {SLOT}: {runner.program} on {runner.arch}")
    return runner
