"""Capability-selected fused add+RMSNorm+quantization for Ascend NPU boundaries.

Wraps the torch_npu fused operators that combine the residual add, RMSNorm and
the activation quantization MoE/linear consumers ask for:

- ``npu_add_rms_norm_dynamic_mx_quant``   -> MXFP8 (E4M3) / MXFP4 (E2M1) block32
- ``npu_rms_norm_dynamic_mx_quant``       -> same, without a residual add
- ``npu_add_rms_norm_dynamic_quant``      -> INT8 / INT4 per-token dynamic

Every path is selected by probing the installed CANN/torch_npu ABI (fail
closed) and by the complete set of consumer contracts at the boundary. The
fused operators do NOT return a floating-point normalized tensor: when any
consumer still needs one, callers keep the existing norm, never ``updated_residual``.
"""

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence, Tuple

import torch

from sglang.srt.hardware_backend.npu.moe.contracts import (
    INT4,
    INT8,
    MXFP4,
    MXFP8,
    ActivationContract,
    QuantizedOperand,
)
from sglang.srt.hardware_backend.npu.moe.tp_fusion import (
    supports_cann_symbol,
    supports_npu_op,
)

logger = logging.getLogger(__name__)

# E8M0 scale bytes may be materialized as uint8 or the explicit torch dtype
# depending on the torch_npu version.
E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)

# --online-quantization grammar: TARGET:SCHEME pairs.
ONLINE_QUANT_TARGETS = ("gate", "moe", "shared_expert", "qkv", "dense", "all")
ONLINE_QUANT_SCHEMES = {
    "w8a8_mxfp8": "W8A8_MXFP8",
    "w8a8_mxfp": "W8A8_MXFP8",
    "w4a8_mxfp": "W4A8_MXFP",
    "w4a4_mxfp4": "W4A4_MXFP4",
    "w8a8_dynamic": "W8A8_DYNAMIC",
    "w4a4_dynamic": "W4A4_DYNAMIC",
}


def parse_online_quantization(spec: Optional[str]) -> dict:
    """Parse 'TARGET:SCHEME[,TARGET:SCHEME...]' into {target: scheme}.

    Raises ValueError on malformed input; 'all' expands to every target.
    """
    if not spec:
        return {}
    parsed = {}
    for pair in spec.split(","):
        pair = pair.strip().lower()
        if not pair:
            continue
        target, sep, scheme = pair.partition(":")
        target, scheme = target.strip(), scheme.strip()
        if not sep or target not in ONLINE_QUANT_TARGETS:
            raise ValueError(
                f"--online-quantization expects TARGET:SCHEME pairs; got {pair!r}"
            )
        if scheme not in ONLINE_QUANT_SCHEMES:
            raise ValueError(
                f"--online-quantization scheme {scheme!r} is not one of "
                f"{sorted(ONLINE_QUANT_SCHEMES)}"
            )
        if target in parsed:
            raise ValueError(f"Duplicate --online-quantization target: {target}")
        parsed[target] = ONLINE_QUANT_SCHEMES[scheme]
    if "all" in parsed:
        scheme = parsed.pop("all")
        parsed = {
            **{target: scheme for target in ONLINE_QUANT_TARGETS if target != "all"},
            **parsed,
        }
    return parsed


@dataclass(frozen=True)
class NormQuantizedReadout:
    """A boundary readout: quantized norm operand plus the updated residual.

    ``normalized_float`` stays ``None`` on the fused paths (the operators do
    not produce it); consumers needing floating-point input must select a
    different readout at the boundary, not dequantize ``operand``.
    """

    operand: QuantizedOperand
    updated_residual: Optional[torch.Tensor]
    normalized_float: Optional[torch.Tensor] = None


def _float4_x2_dtype():
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        _get_float4_e2m1fn_x2_dtype,
    )

    return _get_float4_e2m1fn_x2_dtype()


@lru_cache(maxsize=None)
def native_norm_op():
    """Probe the installed MX add+norm+quant ABI once; fail closed."""
    if not (
        supports_npu_op(
            "npu_add_rms_norm_dynamic_mx_quant",
            ("epsilon", "scale_alg", "round_mode", "dst_type"),
        )
        and supports_cann_symbol("aclnnAddRmsNormDynamicMxQuant")
    ):
        raise RuntimeError(
            "npu_add_rms_norm_dynamic_mx_quant is unavailable in this CANN build"
        )
    op = torch.ops.npu.npu_add_rms_norm_dynamic_mx_quant
    if len(op.default._schema.returns) != 4:
        raise RuntimeError("Unsupported npu_add_rms_norm_dynamic_mx_quant ABI")
    return op


@lru_cache(maxsize=None)
def native_norm_no_residual_op():
    if not (
        supports_npu_op("npu_rms_norm_dynamic_mx_quant", ("epsilon", "dst_type"))
        and supports_cann_symbol("aclnnRmsNormDynamicMxQuant")
    ):
        raise RuntimeError(
            "npu_rms_norm_dynamic_mx_quant is unavailable in this CANN build"
        )
    return torch.ops.npu.npu_rms_norm_dynamic_mx_quant


@lru_cache(maxsize=None)
def native_int_norm_op():
    if not (
        supports_npu_op("npu_add_rms_norm_dynamic_quant", ("y_dtype", "output_mask"))
        and supports_cann_symbol("aclnnAddRmsNormDynamicQuantV2")
    ):
        raise RuntimeError(
            "npu_add_rms_norm_dynamic_quant is unavailable in this CANN build"
        )
    op = torch.ops.npu.npu_add_rms_norm_dynamic_quant
    if len(op.default._schema.returns) != 5:
        raise RuntimeError("Unsupported npu_add_rms_norm_dynamic_quant ABI")
    return op


@lru_cache(maxsize=1)
def supported_norm_quant_contracts() -> Tuple[ActivationContract, ...]:
    """Contracts this build's fused norm operators can emit, probed once.
    Fails closed (empty) on hosts without an NPU runtime."""
    contracts = []
    try:
        import torch_npu  # noqa: F401  (only reachable on NPU hosts)
    except (ImportError, ModuleNotFoundError):
        return ()

    if _probe_mx_contract(MXFP8):
        contracts.append(MXFP8)
    if _probe_mx_contract(MXFP4):
        contracts.append(MXFP4)
    if _probe_int_contract(torch.int8):
        contracts.append(INT8)
    if _probe_int_contract(torch.quint4x2):
        contracts.append(INT4)
    return tuple(contracts)


def _probe_mx_contract(contract: ActivationContract) -> bool:
    """Probe one MX dtype on a minimal device tensor; unsupported ABIs fail."""
    try:
        op = native_norm_op()
        if contract == MXFP4:
            dst = _float4_x2_dtype()
        else:
            dst = torch.float8_e4m3fn
        x = torch.zeros((2, 128), dtype=torch.bfloat16, device="npu")
        residual = torch.zeros_like(x)
        gamma = torch.ones(128, dtype=torch.bfloat16, device="npu")
        q, residual_out, scale, rstd = op(
            x,
            residual,
            gamma,
            beta=None,
            epsilon=1e-6,
            scale_alg=contract.scale_alg,
            round_mode=contract.round_mode,
            dst_type=dst,
        )
        width = 128 // (2 if contract == MXFP4 else 1)
        if tuple(q.shape) != (2, width) or residual_out.shape != x.shape:
            return False
        _mx_operand(q, scale, x, contract)
        return True
    except Exception:  # noqa: BLE001 - probing must fail closed on any ABI gap
        return False


def _probe_int_contract(dst) -> bool:
    try:
        op = native_int_norm_op()
        x = torch.zeros((2, 128), dtype=torch.bfloat16, device="npu")
        residual = torch.zeros_like(x)
        gamma = torch.ones(128, dtype=torch.bfloat16, device="npu")
        q, _, residual_out, scale, _ = op(
            x, residual, gamma, y_dtype=dst, epsilon=1e-6, output_mask=[True, False]
        )
        if residual_out.shape != x.shape:
            return False
        contract = INT8 if dst is torch.int8 else INT4
        _int_operand(q, scale, x, contract)
        return True
    except Exception:  # noqa: BLE001
        return False


def _mx_operand(q, scale, source, contract: ActivationContract) -> QuantizedOperand:
    if contract == MXFP4:
        # FP4 payloads are packed two values per byte (physical width H/2).
        rows, width = source.shape
        if q.dtype != torch.uint8 or tuple(q.shape) != (rows, width // 2):
            raise RuntimeError("Unexpected fused MXFP4 norm payload layout")
    elif q.dtype != torch.float8_e4m3fn or tuple(q.shape) != tuple(source.shape):
        raise RuntimeError("Unexpected fused MXFP8 norm payload layout")
    if scale.ndim == 2:
        # Flat 2D e8m0 block scale -> pair-split [M, H//64, 2] view.
        scale = scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)
    operand = QuantizedOperand(q, scale, contract, tuple(source.shape), source.dtype)
    operand.validate(contract)
    return operand


def _int_operand(q, scale, source, contract: ActivationContract) -> QuantizedOperand:
    rows, width = source.shape
    if contract == INT4:
        # Documented fused INT4 storage: int32 [M, H/8].
        if q.dtype != torch.int32 or tuple(q.shape) != (rows, width // 8):
            raise RuntimeError("Unexpected fused INT4 norm payload layout")
    elif q.dtype != torch.int8 or tuple(q.shape) != (rows, width):
        raise RuntimeError("Unexpected fused INT8 norm payload layout")
    scale = scale.reshape(rows) if scale.numel() == rows else scale
    operand = QuantizedOperand(q, scale, contract, tuple(source.shape), source.dtype)
    operand.validate(contract)
    return operand


def native_add_rmsnorm_quant(
    x: torch.Tensor,
    residual: Optional[torch.Tensor],
    gamma: torch.Tensor,
    eps: float,
    contract: ActivationContract,
) -> NormQuantizedReadout:
    """Fused (optional add +) RMSNorm + dynamic quantization on the NPU.

    ``updated_residual`` is the pre-norm sum ``x + residual`` — never a valid
    substitute for the normalized activation.
    """
    if x.dtype not in (torch.bfloat16, torch.float16) or x.ndim != 2:
        raise ValueError("Fused norm quantization requires FP16/BF16 [T, H] input")
    if contract.block_size and x.shape[1] % 64:
        raise ValueError("MX norm quantization requires H % 64 == 0")
    if x.shape[1] % contract.packing:
        raise ValueError("Norm width is incompatible with activation packing")
    if x.shape[0] == 0:
        rows, width = x.shape
        dtype = {
            INT8: torch.int8,
            INT4: torch.int32,
            MXFP8: torch.float8_e4m3fn,
            MXFP4: torch.uint8,
        }[contract]
        data = torch.empty(
            (rows, width // contract.packing), dtype=dtype, device=x.device
        )
        scale_shape = (rows, width // 64, 2) if contract.block_size else (rows,)
        scale = torch.empty(
            scale_shape,
            dtype=torch.uint8 if contract.block_size else torch.float32,
            device=x.device,
        )
        return NormQuantizedReadout(
            QuantizedOperand(data, scale, contract, tuple(x.shape), x.dtype), residual
        )
    if contract in (MXFP8, MXFP4):
        dst = torch.float8_e4m3fn if contract == MXFP8 else _float4_x2_dtype()
        kwargs = dict(
            beta=None,
            epsilon=eps,
            scale_alg=contract.scale_alg,
            round_mode="rint",
            dst_type=dst,
        )
        if residual is None:
            try:
                op = native_norm_no_residual_op()
            except RuntimeError:
                op = None
            if op is not None:
                q, scale, _ = op(x, gamma, **kwargs)
                return NormQuantizedReadout(_mx_operand(q, scale, x, contract), None)
        q, residual_out, scale, _ = native_norm_op()(
            x, torch.zeros_like(x) if residual is None else residual, gamma, **kwargs
        )
        return NormQuantizedReadout(
            _mx_operand(q, scale, x, contract),
            None if residual is None else residual_out,
        )
    if contract in (INT8, INT4):
        dst = torch.int8 if contract == INT8 else torch.quint4x2
        q, _, residual_out, scale, _ = native_int_norm_op()(
            x,
            torch.zeros_like(x) if residual is None else residual,
            gamma,
            y_dtype=dst,
            epsilon=eps,
            output_mask=[True, False],
        )
        return NormQuantizedReadout(
            _int_operand(q, scale, x, contract),
            None if residual is None else residual_out,
        )
    raise ValueError(f"Fused norm quantization cannot emit {contract}")


def can_fuse_norm_quant(
    consumer_contracts: Sequence[Optional[ActivationContract]],
    capture: bool = False,
) -> Tuple[Optional[ActivationContract], str]:
    """Select a fused norm+quant contract accepted by every live consumer.

    Returns ``(contract, reason)``; ``contract`` is ``None`` with a recorded
    fallback reason when fusion must be declined. ``None`` consumer contracts
    mean a consumer needs a floating-point normalized tensor, which the fused
    operators do not produce.
    """
    if not consumer_contracts:
        return None, "no quantized consumer at this boundary"
    wanted = set()
    for contract in consumer_contracts:
        if contract is None:
            return None, "a consumer requires a floating-point normalized tensor"
        wanted.add(contract)
    if len(wanted) != 1:
        kinds = sorted(contract.kind for contract in wanted)
        return None, f"consumers disagree on the activation contract: {kinds}"
    contract = wanted.pop()
    available = supported_norm_quant_contracts()
    if contract not in available:
        return None, f"CANN fused norm cannot emit {contract.kind} here"
    del capture  # Capability probes run during construction, before capture.
    return contract, f"fused {contract.kind} norm+quant selected"
