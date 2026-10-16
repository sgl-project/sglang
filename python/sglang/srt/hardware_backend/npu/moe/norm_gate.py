"""Checkpoint-selected router precision and opt-in load-time NPU conversion."""

import logging

import torch

from sglang.srt.hardware_backend.npu.moe.contracts import (
    INT8,
    MXFP8,
    ActivationContract,
    QuantizedOperand,
)
from sglang.srt.hardware_backend.npu.moe.tp_fusion import (
    mxfp8_input,
    supports_npu_op,
)
from sglang.srt.layers.quantization.base_config import QuantizeMethodBase

logger = logging.getLogger(__name__)

# Gate weight formats this module can prepare online. MXFP8 is the first
# choice (documented native FP32 matmul output on Ascend 950); INT8 matmuls
# return FP16/BF16 and the FP32 cast afterwards is NOT FP32 accumulation.
GATE_SCHEME_CONTRACTS = {
    "W8A8_MXFP8": MXFP8,
    "W8A8_DYNAMIC": INT8,
}
# Declined online gate schemes, with recorded reasons (fail closed):
GATE_SCHEME_DECLINED = {
    "W4A8_MXFP": "packed-FP4 gate weights need offline-validated packing ABI",
    "W4A4_MXFP4": "MXFP4 gate accuracy/packing is not validated",
    "W4A4_DYNAMIC": "INT4 gate packing ABI and top-k stability unproven",
}

_E8M0 = getattr(torch, "float8_e8m0fnu", torch.uint8)


def _e8m0_dtype():
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        _require_e8m0_dtype,
    )

    return _require_e8m0_dtype()


def gate_input_contract(gate) -> "ActivationContract | None":
    """The activation contract the configured gate consumes, if quantized."""
    adapter = getattr(gate, "_npu_online_gate", None)
    if adapter is not None:
        return adapter.contract
    if hasattr(gate, "quant_method"):
        from sglang.srt.hardware_backend.npu.moe.contracts import (
            linear_input_contract,
        )

        return linear_input_contract(gate)
    return None


def pair_split_scale(scale: torch.Tensor, rows: int, hidden_size: int):
    if scale.dtype not in (torch.uint8, _E8M0):
        raise ValueError(f"Expected E8M0 scale bytes, got {scale.dtype}")
    if scale.ndim == 2:
        scale = scale.reshape(scale.shape[0], scale.shape[1] // 2, 2)
    if tuple(scale.shape) != (rows, hidden_size // 64, 2):
        raise ValueError("MX gate scale is not token-major paired block32")
    return scale


class QuantizedMoEGateMethod(QuantizeMethodBase):
    """Online conversion of a loaded float MoE router-gate weight.

    Converts once in ``process_weights_after_loading`` (the standard loader
    hook runs it on fully loaded weights, outside graph capture) and keeps
    the original float weight so the reference gate remains available for
    fallback and A/B measurement.
    """

    def __init__(self, scheme: str):
        if scheme not in GATE_SCHEME_CONTRACTS:
            reason = GATE_SCHEME_DECLINED.get(scheme, f"unknown scheme {scheme}")
            raise ValueError(f"Online gate quantization declined: {reason}")
        self.scheme = scheme
        self.contract = GATE_SCHEME_CONTRACTS[scheme]
        self.output_dtype = torch.float32 if self.contract == MXFP8 else torch.bfloat16

    # The gate module owns its weight parameter already (custom gates and
    # ReplicatedLinear both create it); online conversion adds buffers only.
    # The conversion is deterministic from the current weight, so the
    # loader's sentinel pass (overlap loading) can safely re-run it.
    def create_weights(self, layer, *args, **kwargs):
        return None

    @torch.no_grad()
    def process_weights_after_loading(self, layer):
        arguments = ("pertoken_scale", "output_dtype")
        if self.contract == MXFP8:
            arguments += ("scale_dtype", "pertoken_scale_dtype", "group_sizes")
        if not supports_npu_op("npu_quant_matmul", arguments):
            raise ValueError(
                f"Online {self.scheme} gate requires npu_quant_matmul {arguments}"
            )
        weight = layer.weight
        if weight.ndim != 2 or weight.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(
                f"Online gate quantization requires a 2D FP16/BF16/FP32 weight; "
                f"got {weight.dtype} {tuple(weight.shape)}"
            )
        if self.contract == MXFP8:
            q, scale = torch.ops.npu.npu_dynamic_mx_quant(
                weight.to(torch.bfloat16) if weight.dtype == torch.float32 else weight,
                dst_type=torch.float8_e4m3fn,
                block_size=32,
                scale_alg=0,
                round_mode="rint",
            )
            if q.dtype != torch.float8_e4m3fn or q.shape != weight.shape:
                raise RuntimeError("Unsupported online gate MXFP8 quantization output")
            experts, hidden = weight.shape
            gate_weight = q.transpose(0, 1)
            gate_scale = pair_split_scale(scale, experts, hidden).transpose(0, 1)
        else:  # INT8
            # Quantize rows of [E, H]: per-row scales are per-output-channel,
            # which the quant matmul's weight_scale expects.
            q, scale = torch.ops.npu.npu_dynamic_quant(weight.to(torch.bfloat16))
            if q.dtype != torch.int8 or q.shape != weight.shape:
                raise RuntimeError("Unsupported online gate INT8 quantization output")
            gate_weight = q.transpose(0, 1)
            gate_scale = scale.flatten().contiguous()
        layer.register_buffer("gate_quant_weight", gate_weight, persistent=False)
        layer.register_buffer("gate_quant_scale", gate_scale, persistent=False)
        bias = getattr(layer, "bias", None)
        if bias is not None and self.contract == MXFP8:
            layer.register_buffer("gate_quant_bias", bias.float(), persistent=False)
        # Exercise the actual projection dimensions before serving/capture.
        sample = torch.zeros(
            (1, weight.shape[1]), dtype=torch.bfloat16, device=weight.device
        )
        result = self.apply(layer, sample)
        if not torch.isfinite(result).all().item():
            raise RuntimeError(
                f"Online {self.scheme} projection probe returned non-finite output"
            )
        logger.info(
            "Online %s router gate: %s weights converted after loading "
            "(original float weight retained; this changes routing numerics)",
            self.scheme,
            tuple(weight.shape),
        )

    def apply(self, layer, hidden_states, bias=None):
        if not hasattr(layer, "gate_quant_weight"):
            raise RuntimeError("Online gate weights were not prepared after loading")
        if isinstance(hidden_states, QuantizedOperand):
            operand = hidden_states
            operand.validate(self.contract)
            operand.record_stream()
        elif isinstance(hidden_states, tuple):
            data, scale = hidden_states
            stream = torch.get_device_module(data.device).current_stream()
            data.record_stream(stream)
            scale.record_stream(stream)
            operand = (data, scale)
        else:
            if self.contract == MXFP8:
                operand = mxfp8_input(hidden_states)
            else:
                data, scale = torch.ops.npu.npu_dynamic_quant(hidden_states)
                operand = (data, scale)
        if self.contract == MXFP8:
            q, scale = operand
            rows, hidden = q.shape
            logits = torch.ops.npu.npu_quant_matmul(
                q,
                layer.gate_quant_weight,
                layer.gate_quant_scale,
                pertoken_scale=pair_split_scale(scale, rows, hidden),
                scale_dtype=_e8m0_dtype(),
                pertoken_scale_dtype=_e8m0_dtype(),
                output_dtype=self.output_dtype,
                group_sizes=[1, 1, 32],
                bias=getattr(layer, "gate_quant_bias", bias)
                if bias is not None
                else None,
            )
        else:
            q, scale = operand
            logits = torch.ops.npu.npu_quant_matmul(
                q,
                layer.gate_quant_weight,
                layer.gate_quant_scale,
                pertoken_scale=scale.flatten(),
                bias=bias,
                output_dtype=self.output_dtype,
            )
        if logits.dtype != self.output_dtype:
            raise RuntimeError(
                f"Quantized gate matmul returned {logits.dtype}, "
                f"expected {self.output_dtype}"
            )
        # Downstream routing (correction bias, grouped top-k, renormalize)
        # consumes FP32 logits; the cast here is explicit, and for INT8 it is
        # NOT FP32 accumulation.
        return logits if isinstance(self, OnlineLinearMethod) else logits.float()


def offline_gate_config(quant_config, prefix):
    """Resolve this router's descriptor, including checkpoint prefix aliases."""
    if quant_config is None or quant_config.get_name() != "modelslim":
        return None
    scheme = quant_config.resolve_projection_scheme(prefix)
    if scheme in (None, "", "FLOAT"):
        return None
    if scheme not in quant_config.linear_scheme_types():
        raise ValueError(f"Offline router {prefix} uses unsupported ModelSlim {scheme}")
    return quant_config


def configure_online_gate(gate, layer_prefix: str = "", quant_config=None) -> bool:
    """Install the opt-in online gate quantization on ``gate``.

    Returns True when installed. Selection is per-module: only the routed
    router gate is touched (never the shared-expert sigmoid gate, SwiGLU
    projections or attention gates). Without the flag, the gate keeps its
    checkpoint precision. Unsupported requested conversions raise an error.
    """
    from sglang.srt.runtime_context import get_flags

    config = offline_gate_config(quant_config, layer_prefix)
    if config is not None:
        # Custom FP32 routers use the same checkpoint parameters and loaders
        # as ReplicatedLinear, without changing their correction-bias slots.
        from sglang.srt.layers.quantization.modelslim.modelslim import (
            ModelSlimLinearMethod,
        )
        from sglang.srt.model_loader.weight_utils import default_weight_loader

        output_size, input_size = gate.weight.shape
        gate.scheme = config.get_linear_scheme(gate, layer_prefix)
        gate.quant_method = ModelSlimLinearMethod(config)
        gate.quant_method.create_weights(
            gate,
            input_size,
            [output_size],
            input_size,
            output_size,
            torch.bfloat16,
            weight_loader=default_weight_loader,
        )
        gate._npu_gate_method = gate.quant_method
        return False
    schemes = get_flags().moe.online_quantization or {}
    scheme = schemes.get("gate")
    if scheme is None:
        return False
    method = QuantizedMoEGateMethod(scheme)
    gate.quant_method = method
    gate._npu_online_gate = method
    gate._npu_gate_method = method
    return True


def select_moe_boundary_quant_format(moe_block) -> str:
    """Select the fused norm+quant format for a MoE input boundary, or ''.

    Fusion requires the COMPLETE consumer set (router gate, W13, shared
    experts) to accept one typed activation contract: a floating-point gate
    without online quantization disables the fused boundary (the unfused
    reference norm + independent quantization stays). Packed contracts
    (MXFP4/INT4) additionally require installed routing payload support.
    """
    from sglang.srt.hardware_backend.npu.layernorm_quant import (
        can_fuse_norm_quant,
    )
    from sglang.srt.hardware_backend.npu.moe.contracts import (
        moe_input_contract,
    )
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.layers.moe.token_dispatcher.ascend_tp import AscendTPDispatcher
    from sglang.srt.runtime_context import get_parallel

    if (
        is_dp_attention_enabled()
        or get_parallel().attn_tp_size != get_parallel().tp_size
    ):
        return ""
    if getattr(moe_block, "is_hash", False) or getattr(
        moe_block, "_shared_expert_tp1", False
    ):
        return ""
    if getattr(moe_block, "shared_expert_gate", None) is not None:
        return ""
    parallel = get_parallel()
    if (
        parallel.attn_cp_size != 1
        or parallel.moe_dp_size != 1
        or parallel.enable_prefill_cp
    ):
        return ""
    experts = moe_block.experts
    if not isinstance(experts.dispatcher, AscendTPDispatcher):
        return ""
    if experts.moe_ep_size != 1:
        return ""
    consumers = [gate_input_contract(moe_block.gate)]
    w13 = moe_input_contract(getattr(experts, "w13_kernel", None))
    consumers.append(w13)
    shared = getattr(moe_block, "shared_experts", None)
    if shared is not None and not getattr(moe_block, "num_fused_shared_experts", 0):
        from sglang.srt.hardware_backend.npu.moe.contracts import (
            linear_input_contract,
        )

        consumers.append(linear_input_contract(shared.gate_up_proj))
    contract, reason = can_fuse_norm_quant(consumers)
    if contract is None:
        logger.info_once("NPU fused MoE norm+quant declined: %s", reason)
        return ""
    from sglang.srt.hardware_backend.npu.moe.init_routing import supports_packed_routing

    if contract.kind == "int4" or (
        contract.kind == "mxfp4" and not supports_packed_routing("mxfp4", True)
    ):
        # INT4 has no passthrough ABI; MXFP4 requires a matching device probe.
        logger.info(
            "NPU fused MoE norm+quant declined: %s passthrough unavailable",
            contract.kind,
        )
        return ""
    logger.info_once("NPU fused MoE norm+quant selected: %s", contract.kind)
    return f"npu_{contract.kind}"


def gate_forward(gate, hidden_states):
    """Forward a router gate: the quantized path when configured.

    ``hidden_states`` may be a BF16/FP32 tensor or a ``QuantizedOperand``
    produced by the fused norm readout; the operand's contract must match
    the configured gate scheme. Without an adapter the module's own forward
    runs (float gate, or a ModelSlim offline-quantized linear gate).
    """
    adapter = getattr(gate, "_npu_online_gate", None)
    if adapter is None:
        method = getattr(gate, "_npu_gate_method", None)
        if method is not None:
            return method.apply(gate, hidden_states).float()
        return gate(hidden_states)
    if (
        isinstance(hidden_states, QuantizedOperand)
        and hidden_states.contract != adapter.contract
    ):
        raise ValueError(
            f"Gate activation contract {hidden_states.contract} does not "
            f"match {adapter.contract}"
        )
    return adapter.apply(gate, hidden_states)


class OnlineLinearMethod(QuantizedMoEGateMethod):
    """Convert an explicitly selected floating-point projection after loading."""

    def __init__(self, scheme, params_dtype):
        super().__init__(scheme)
        self.output_dtype = params_dtype

    def create_weights(self, layer, *args, **kwargs):
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        UnquantizedLinearMethod().create_weights(layer, *args, **kwargs)


def online_linear_method(layer, prefix):
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.runtime_context import get_flags

    parts = prefix.split(".")
    name = parts[-1]
    if "shared_experts" in parts or "shared_expert" in parts:
        target = "shared_expert"
    elif name in (
        "qkv_proj",
        "q_proj",
        "k_proj",
        "v_proj",
        "q_a_proj",
        "q_b_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
    ):
        target = "qkv"
    elif (
        "mlp" in parts
        and "experts" not in parts
        and name in ("gate_up_proj", "gate_proj", "up_proj", "down_proj")
    ):
        target = "dense"
    else:
        return layer.quant_method
    scheme = (get_flags().moe.online_quantization or {}).get(target)
    if scheme is None:
        return layer.quant_method
    if not isinstance(layer.quant_method, UnquantizedLinearMethod):
        # Offline checkpoint storage cannot be interpreted as floating weights.
        return layer.quant_method
    return OnlineLinearMethod(scheme, layer.params_dtype)


def select_linear_boundary_quant_format(projections):
    from sglang.srt.hardware_backend.npu.layernorm_quant import can_fuse_norm_quant
    from sglang.srt.hardware_backend.npu.moe.contracts import linear_input_contract
    from sglang.srt.layers.dp_attention import is_dp_attention_enabled
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if (
        is_dp_attention_enabled()
        or parallel.attn_cp_size != 1
        or parallel.enable_prefill_cp
        or parallel.attn_tp_size != parallel.tp_size
    ):
        return ""
    contract, _ = can_fuse_norm_quant(
        [linear_input_contract(proj) for proj in projections]
    )
    return f"npu_{contract.kind}" if contract is not None else ""
