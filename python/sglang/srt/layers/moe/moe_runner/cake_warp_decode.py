"""Cake NVFP4 warp-decode MoE hook for the FlashInfer TRT-LLM NVFP4 MoE path.

Route ``moe_nvfp4_warp_decode`` (``SGLANG_CAKE_ROUTES``). When the ModelOpt
NVFP4 fused-MoE method runs the FlashInfer TRT-LLM runner
(``--moe-runner-backend flashinfer_trtllm[_routed]``), decode batches with
``1 <= num_tokens <= 32`` whose ``(activation, H, I_local, E, top_k)`` is in
``sglang.kernels.cake_kernels.moe_warp_decode.GEOMETRIES`` run FlashInfer's
``CakeWarpDecodeRunner`` instead of ``trtllm_fp4_block_scale[_routed]_moe``.
Every other call (more tokens, prefill, deferred finalize, EP shards, fused
shared experts, parameterised SwiGLU / SiTU, per-token activation scales, a
non-SM100/SM103 device, or the runner raising) takes the TRT-LLM path unchanged.

Weights: the Cake runner consumes the exact ``trtllm_fp4_routed`` physical view
that ``align_fp4_moe_weights_for_flashinfer_trtllm`` already materialised on the
layer (shuffled uint8 E2M1 weights, interleaved E4M3 block scales,
``g1_scale_c`` / ``g1_alphas`` / ``g2_alphas`` epilogue scalars), so no second
copy of the expert weights is made. The only extra device memory per layer is a
per-expert fp32 ``gemm1_alpha`` placeholder (``E * 4`` bytes) plus the static
routing buffers and the runner's per-(stream, num_tokens) workspaces.

Activations: the hook runs after the TRT-LLM path quantised the hidden states
with ``w13_input_scale_quant``; Cake and TRT-LLM consume identical bytes.

CUDA graphs: the FlashInfer runner must prepare one workspace per
(stream, num_tokens) and range-validate ``topk_ids`` on the host before a
capture. SGLang's graph runner performs two eager warmups per captured batch
size on the capture stream, which satisfies both. Routing is copied into static
inference-mode buffers (one pair per ``num_tokens``) so the runner's validation
receipt, which is bound to the tensor identity and version, matches on replay
and so eager serving stays under the runner's 64-receipt cap. If capture is
observed for a ``num_tokens`` that was never run eagerly the hook falls back
(logged once).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Set

from sglang.kernels.cake_kernels._routes import cake_route_enabled

logger = logging.getLogger(__name__)

ROUTE = "moe_nvfp4_warp_decode"
MAX_TOKENS = 32
_ACTIVATION_KEY = "swiglu"

_logged: Set[str] = set()


def _log_once(key: str, level: int, message: str, *args: Any) -> None:
    if key in _logged:
        return
    _logged.add(key)
    logger.log(level, message, *args)


def reset_logs_for_tests() -> None:
    _logged.clear()


def _capturing() -> bool:
    import torch

    return torch.cuda.is_current_stream_capturing()


def _activation_pack(hidden_states_q, hidden_states_scale, topk_ids, topk_weights):
    from flashinfer.fused_moe import MoEActivationPack, RoutingInputMode

    return MoEActivationPack(
        hidden_states_q,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )


def _weight_pack(view: Dict[str, Any]):
    from flashinfer.fused_moe import MoEWeightPack

    pack = MoEWeightPack()
    pack.prepare_for("cake", view)
    return pack


def _supports(**kwargs: Any) -> bool:
    from sglang.kernels.cake_kernels.moe_warp_decode import supports_warp_decode

    return supports_warp_decode(**kwargs)


def _build_runner(*, intermediate_size: int, num_experts: int, top_k: int, device):
    """Construct, check and JIT-build ``CakeWarpDecodeRunner`` for one geometry.

    ``hidden_size`` is not part of ``MoEConfig``; the runner validates it per
    call from ``hidden_states_q.shape[1] * 2`` against its geometry table.
    """
    from flashinfer.fused_moe import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantFormat,
        RoutingConfig,
        SwiGLU,
    )

    from sglang.kernels.ops.moe.cake import (
        cake_warp_decode_config,
        cake_warp_decode_runner,
    )

    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4),
        experts=ExpertConfig(intermediate_size=intermediate_size),
        activation=SwiGLU(),
        backend=BackendOptions((cake_warp_decode_config(),)),
        execution=ExecutionConfig(enable_pdl=True, tune_max_num_tokens=MAX_TOKENS),
    )
    runner = cake_warp_decode_runner(config, device)
    runner.check_support()
    runner.build()
    return runner


def _layer_rejection(layer: Any) -> Optional[str]:
    """Static per-layer admission; returns the first failing condition."""
    cfg = layer.moe_runner_config
    if cfg.activation != "silu" or not cfg.is_gated:
        return f"activation={cfg.activation!r} is_gated={cfg.is_gated}"
    if (
        cfg.gemm1_alpha is not None
        or cfg.gemm1_beta is not None
        or cfg.gemm1_clamp_limit is not None
        or cfg.swiglu_limit is not None
    ):
        return (
            "parameterised SwiGLU (alpha/beta/limit) is not a default-SwiGLU geometry"
        )
    if cfg.apply_router_weight_on_input:
        return "apply_router_weight_on_input"
    if cfg.routed_scaling_factor not in (None, 1.0):
        return f"routed_scaling_factor={cfg.routed_scaling_factor}"
    if cfg.num_fused_shared_experts:
        return f"num_fused_shared_experts={cfg.num_fused_shared_experts}"
    if layer.moe_ep_size != 1 or layer.num_local_experts != layer.num_experts:
        return (
            f"expert parallelism (ep_size={layer.moe_ep_size}, "
            f"local={layer.num_local_experts}/{layer.num_experts})"
        )
    for name in (
        "w13_weight",
        "w13_weight_scale",
        "w2_weight",
        "w2_weight_scale",
        "g1_scale_c",
        "g1_alphas",
        "g2_alphas",
    ):
        if getattr(layer, name, None) is None:
            return f"layer has no {name} (TRT-LLM weight layout not materialised)"
    return None


class CakeWarpDecodeMoE:
    """Per-layer Cake warp-decode state: runner, weight view, static routing buffers."""

    def __init__(
        self,
        *,
        layer_id: Optional[int],
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        device,
        runner: Any,
        weight_pack: Any,
    ) -> None:
        self.layer_id = layer_id
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.device = device
        self._runner = runner
        self._weight_pack = weight_pack
        # num_tokens -> (int32 ids, bf16 weights) inference tensors.
        self._routing: Dict[int, tuple] = {}
        # num_tokens values that completed an eager (non-capturing) run.
        self._warmed: Set[int] = set()
        self._disabled_reason: Optional[str] = None

    # ----- construction -----------------------------------------------------

    @classmethod
    def maybe_create(cls, layer: Any) -> Optional[CakeWarpDecodeMoE]:
        """Build the per-layer state when the route is enabled and admitted; never raises."""
        try:
            if not cake_route_enabled(ROUTE):
                return None
            reason = _layer_rejection(layer)
            if reason is not None:
                _log_once(
                    f"reject:{reason}",
                    logging.INFO,
                    "Cake %s: layer %s not admitted: %s",
                    ROUTE,
                    getattr(layer, "layer_id", None),
                    reason,
                )
                return None
            w2 = layer.w2_weight
            hidden_size = int(w2.shape[1])
            intermediate_size = int(w2.shape[2]) * 2
            num_experts = int(layer.num_experts)
            top_k = int(layer.top_k)
            device = w2.device
            if intermediate_size != int(layer.intermediate_size_per_partition):
                _log_once(
                    "reject:intermediate",
                    logging.INFO,
                    "Cake %s: w2 intermediate %d != intermediate_size_per_partition %d",
                    ROUTE,
                    intermediate_size,
                    layer.intermediate_size_per_partition,
                )
                return None
            if not _supports(
                activation=_ACTIVATION_KEY,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
                device=device,
            ):
                _log_once(
                    f"reject:geometry:{hidden_size},{intermediate_size},{num_experts},{top_k}",
                    logging.INFO,
                    "Cake %s: geometry (swiglu, H=%d, I=%d, E=%d, top_k=%d) on %s is "
                    "outside the calibrated table or the device is not SM100/SM103",
                    ROUTE,
                    hidden_size,
                    intermediate_size,
                    num_experts,
                    top_k,
                    device,
                )
                return None
            view = cls.weight_view(layer, num_experts, device)
            runner = _build_runner(
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
                device=device,
            )
            state = cls(
                layer_id=getattr(layer, "layer_id", None),
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
                device=device,
                runner=runner,
                weight_pack=_weight_pack(view),
            )
            logger.info(
                "Cake %s: layer %s armed for (swiglu, H=%d, I=%d, E=%d, top_k=%d), "
                "1 <= num_tokens <= %d",
                ROUTE,
                state.layer_id,
                hidden_size,
                intermediate_size,
                num_experts,
                top_k,
                MAX_TOKENS,
            )
            return state
        except Exception as exc:  # the route must never break model loading
            _log_once(
                f"create-failed:{type(exc).__name__}",
                logging.WARNING,
                "Cake %s: layer %s disabled, runner construction failed: %r",
                ROUTE,
                getattr(layer, "layer_id", None),
                exc,
            )
            return None

    @staticmethod
    def weight_view(layer: Any, num_experts: int, device) -> Dict[str, Any]:
        """``trtllm_fp4_routed`` view over the layer's existing TRT-LLM tensors.

        No weight bytes are copied. ``gemm1_alpha`` is the per-expert fp32 ones
        placeholder FlashInfer's own ``prepare_weights`` emits for default SwiGLU.
        """
        import torch

        w13_scale = layer.w13_weight_scale.data
        w2_scale = layer.w2_weight_scale.data
        if w13_scale.dtype != torch.float8_e4m3fn:
            w13_scale = w13_scale.view(torch.float8_e4m3fn)
        if w2_scale.dtype != torch.float8_e4m3fn:
            w2_scale = w2_scale.view(torch.float8_e4m3fn)
        return {
            "gemm1_weights": layer.w13_weight.data.view(torch.uint8),
            "gemm1_weights_scale": w13_scale,
            "gemm2_weights": layer.w2_weight.data.view(torch.uint8),
            "gemm2_weights_scale": w2_scale,
            "output1_scale_scalar": layer.g1_scale_c.data,
            "output1_scale_gate_scalar": layer.g1_alphas.data,
            "output2_scale_scalar": layer.g2_alphas.data,
            "gemm1_alpha": torch.ones(num_experts, dtype=torch.float32, device=device),
        }

    # ----- per-call -----------------------------------------------------------

    def _fallback(self, reason: str, *args: Any) -> bool:
        _log_once(
            f"fallback:{reason}",
            logging.INFO,
            "Cake %s: layer %s falling back to TRT-LLM: " + reason,
            ROUTE,
            self.layer_id,
            *args,
        )
        return False

    def _routing_buffers(self, num_tokens: int):
        import torch

        buffers = self._routing.get(num_tokens)
        if buffers is None:
            with torch.inference_mode():
                buffers = (
                    torch.empty(
                        (num_tokens, self.top_k), dtype=torch.int32, device=self.device
                    ),
                    torch.empty(
                        (num_tokens, self.top_k),
                        dtype=torch.bfloat16,
                        device=self.device,
                    ),
                )
            self._routing[num_tokens] = buffers
        return buffers

    @staticmethod
    def _materialize_routing(topk_output: Any, layer_id: Optional[int]):
        """``(topk_ids, topk_weights)`` from a standard or bypassed TopKOutput."""
        ids = getattr(topk_output, "topk_ids", None)
        weights = getattr(topk_output, "topk_weights", None)
        if ids is None or weights is None:
            to_standard = getattr(topk_output, "to_standard", None)
            if to_standard is None:
                return None
            standard = to_standard(layer_id)
            ids, weights = standard.topk_ids, standard.topk_weights
        return ids, weights

    def run(
        self,
        hidden_states_q,
        hidden_states_scale,
        topk_output: Any,
        output,
        *,
        per_token_scale=None,
    ) -> bool:
        """Run the Cake warp-decode MoE into ``output``; False means "take the TRT-LLM path"."""
        import torch

        if self._disabled_reason is not None:
            return False
        num_tokens = int(hidden_states_q.shape[0])
        if not 1 <= num_tokens <= MAX_TOKENS:
            return self._fallback("num_tokens=%d outside 1..%d", num_tokens, MAX_TOKENS)
        if per_token_scale is not None:
            return self._fallback("per-token activation scale")
        if hidden_states_q.dtype != torch.uint8 or hidden_states_q.ndim != 2:
            return self._fallback(
                "hidden_states_q %s %s is not packed NVFP4",
                hidden_states_q.dtype,
                tuple(hidden_states_q.shape),
            )
        if int(hidden_states_q.shape[1]) * 2 != self.hidden_size:
            return self._fallback(
                "hidden size %d != %d",
                int(hidden_states_q.shape[1]) * 2,
                self.hidden_size,
            )
        if (
            output.dtype != torch.bfloat16
            or tuple(output.shape) != (num_tokens, self.hidden_size)
            or not output.is_contiguous()
        ):
            return self._fallback(
                "output %s %s is not a contiguous bf16 [num_tokens, hidden]",
                output.dtype,
                tuple(output.shape),
            )
        capturing = _capturing()
        if capturing and num_tokens not in self._warmed:
            return self._fallback(
                "CUDA graph capture for num_tokens=%d before an eager warmup",
                num_tokens,
            )
        routing = self._materialize_routing(topk_output, self.layer_id)
        if routing is None:
            return self._fallback(
                "unsupported TopKOutput %s", type(topk_output).__name__
            )
        ids, weights = routing
        if tuple(ids.shape) != (num_tokens, self.top_k):
            return self._fallback(
                "topk_ids shape %s != (%d, %d)",
                tuple(ids.shape),
                num_tokens,
                self.top_k,
            )
        ids_buf, weights_buf = self._routing_buffers(num_tokens)
        try:
            with torch.inference_mode():
                # Padded rows carry id -1 (mask_topk_ids); the runner rejects any
                # id outside [0, E). Park them on expert 0 with weight 0.
                invalid = (ids < 0) | (ids >= self.num_experts)
                ids_buf.copy_(ids.masked_fill(invalid, 0))
                weights_buf.copy_(weights.masked_fill(invalid, 0))
            act = _activation_pack(
                hidden_states_q, hidden_states_scale, ids_buf, weights_buf
            )
            inputs = self._runner.pack_inputs(act, self._weight_pack)
            inputs[0] = output  # write into the engine's (symmetric) output buffer
            self._runner.forward(
                inputs, tactic=-1, **self._runner.launch_kwargs_for(inputs)
            )
        except Exception as exc:
            self._disabled_reason = repr(exc)
            logger.warning(
                "Cake %s: layer %s disabled after runner error (TRT-LLM path "
                "takes over): %r",
                ROUTE,
                self.layer_id,
                exc,
            )
            return False
        if not capturing:
            self._warmed.add(num_tokens)
        _log_once(
            "taken",
            logging.INFO,
            "Cake %s: first Cake launch (layer %s, num_tokens=%d, capturing=%s)",
            ROUTE,
            self.layer_id,
            num_tokens,
            capturing,
        )
        return True


def maybe_create_cake_warp_decode_moe(layer: Any) -> Optional[CakeWarpDecodeMoE]:
    """Hook for ``ModelOptNvFp4FusedMoEMethod.process_weights_after_loading``."""
    return CakeWarpDecodeMoE.maybe_create(layer)
