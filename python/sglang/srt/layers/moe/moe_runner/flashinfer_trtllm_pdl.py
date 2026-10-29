"""Re-enable PDL for FlashInfer's trtllm-gen fused MoE launch chain on SM107.

FlashInfer forces ``enable_pdl=False`` for every trtllm-gen MoE op on CC 10.7
through ``flashinfer.fused_moe.core._device_support_moe_pdl``, so the routing,
GEMM1, GEMM2 and finalize launches of each MoE layer run fully serialized.
``SGLANG_TRTLLM_MOE_SM107_PDL`` selects the ops that get the device default
(PDL on) back: ``off`` keeps FlashInfer's gate, ``all`` covers every trtllm-gen
MoE op, and a comma list of ``fp4``, ``fp8_block``, ``fp8_tensor``, ``bf16``,
``mxint4`` covers the named ones. The caller's ``enable_pdl`` (see
``trtllm_moe_enable_pdl``) still applies. Only the launch attribute and the
kernels' dependency-wait placement change; the math, and so the output for a
given tactic, is unchanged, and ``enable_pdl`` is not part of the autotune key.
"""

from __future__ import annotations

import logging
import sys

from sglang.srt.environ import envs
from sglang.srt.runtime_context import get_parallel

logger = logging.getLogger(__name__)

# FlashInfer op (the caller of the gate) -> mode token.
_GATE_CALLERS = {
    "trtllm_fp4_block_scale_moe_op": "fp4",
    "trtllm_fp8_block_scale_moe_op": "fp8_block",
    "trtllm_fp8_per_tensor_scale_moe_op": "fp8_tensor",
    "trtllm_fp8_per_tensor_scale_routed_moe_op": "fp8_tensor",
    "trtllm_bf16_moe_op": "bf16",
    "trtllm_mxint4_block_scale_moe_op": "mxint4",
}


def apply_trtllm_moe_sm107_pdl() -> None:
    mode = envs.SGLANG_TRTLLM_MOE_SM107_PDL.get().strip().lower()
    if mode in ("", "0", "off", "none", "false"):
        return
    if mode in ("1", "on", "true"):
        mode = "all"
    ops = None
    if mode != "all":
        ops = frozenset(t.strip() for t in mode.split(",") if t.strip())
        unknown = ops - set(_GATE_CALLERS.values())
        if unknown:
            logger.warning(
                "SGLANG_TRTLLM_MOE_SM107_PDL: ignoring unknown ops %s", sorted(unknown)
            )
        if not ops - unknown:
            return
    try:
        from flashinfer.fused_moe import core
    except ImportError:
        return
    stock_gate = getattr(core, "_device_support_moe_pdl", None)
    if stock_gate is None or getattr(stock_gate, "_sglang_sm107_pdl", False):
        return
    opened = set()

    def _device_support_moe_pdl(device) -> bool:
        if stock_gate(device):
            return True
        if get_parallel().tp_size != 1:
            return False
        if core.get_compute_capability(device) != (10, 7):
            return False
        if not core.device_support_pdl(device):
            return False
        caller = sys._getframe(1).f_code.co_name
        if ops is not None and _GATE_CALLERS.get(caller) not in ops:
            return False
        if caller not in opened:
            opened.add(caller)
            logger.info("trtllm-gen MoE PDL gate opened on SM107 for %s", caller)
        return True

    _device_support_moe_pdl._sglang_sm107_pdl = True
    core._device_support_moe_pdl = _device_support_moe_pdl
    logger.info("trtllm-gen MoE PDL re-enabled on SM107 for TP1 (ops=%s)", mode)
