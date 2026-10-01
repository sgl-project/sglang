"""Opt-in caller for AITER's prepared exact-M4/M8 GLM MXFP4 operators.

Handles retain weights and private output/scratch for their layer and stream.
They are created only outside graph capture; missing factories or unsupported
contracts use the ordinary AITER runner. The caller's scoped target-verify flag
also enables canonical inactive routing in the existing HIP padding launch.
"""

import logging
import os
from functools import cache

logger = logging.getLogger(__name__)


def enabled():
    return os.environ.get("SGLANG_AITER_TINY_GLM_MOE", "0").lower() in (
        "1",
        "true",
        "yes",
    )


def warmup_stream(stream):
    # FullCudaGraphBackend otherwise warms on the caller stream and captures on
    # a different stream. Use the capture stream for opt-in eager warmups so its
    # private handles are allocated/JIT-compiled before capture starts.
    from contextlib import nullcontext

    if not enabled() or stream is None:
        return nullcontext()
    import torch

    return torch.cuda.stream(stream)


def model_supported(config, *, tp, ep, nextn):
    return (
        not nextn
        and tp == 8
        and ep == 1
        and "GlmMoeDsaForCausalLM" in (getattr(config, "architectures", None) or ())
        and getattr(config, "hidden_size", None) == 6144
        and getattr(config, "moe_intermediate_size", None) == 2048
        and getattr(config, "n_routed_experts", None) == 256
        and getattr(config, "num_experts_per_tok", None) == 8
        and getattr(config, "n_shared_experts", None) == 1
        and getattr(config, "hidden_act", None) == "silu"
        and not getattr(config, "swiglu_limit", None)
    )


def target_supported(batch):
    return (
        batch is not None
        and batch.forward_mode.is_target_verify()
        and getattr(batch.spec_info, "draft_token_num", None) == 4
        and os.environ.get("SGLANG_MORI_NO_PAD_MASK", "0").lower()
        not in ("1", "true", "yes")
        and os.environ.get("SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK", "0").lower()
        not in ("1", "true", "yes")
    )


@cache
def factories():
    try:
        from aiter.ops.flydsl.kernels.mxmoe_tiny_m4 import make_operator as m4
        from aiter.ops.flydsl.mxmoe_tiny_m8 import make_operator as m8
    except ImportError:
        logger.warning("AITER tiny GLM MoE factories unavailable; using native MoE")
        return None
    return {4: m4, 8: m8}


def runner_supported(config, quant, inputs):
    return (
        inputs.hidden_states.shape == (inputs.topk_ids.shape[0], 6144)
        and inputs.hidden_states.shape[0] in (4, 8)
        and inputs.topk_ids.shape
        == inputs.topk_weights.shape
        == (inputs.hidden_states.shape[0], 9)
        and inputs.quant_type.value == "per_1x32"
        and quant.quant_type.value == "per_1x32"
        and config.activation == "silu"
        and config.is_gated
        and not config.no_combine
        and not config.apply_router_weight_on_input
        and config.gemm1_alpha is None
        and config.gemm1_beta is None
        and config.gemm1_clamp_limit is None
        and not config.swiglu_limit
        and config.num_fused_shared_experts == 1
        and config.num_experts == config.num_local_experts == 257
        and not config.use_tp_all_gather_activation
        and not quant.doweight_stage1
        and quant.expert_mask is None
        and quant.b13 is None
        and quant.b2 is None
        and not quant.hidden_pad
        and not quant.intermediate_pad
        and not quant.swiglu_limit
        and quant.a13_scale is None
        and inputs.a1_scale is None
        and quant.a2_scale is None
        and inputs.output_dtype is None
        and inputs.num_local_tokens is None
        and (quant.fused_moe_kwargs or {}) == {"gate_mode": "separated"}
    )


def overlaps(a, b):
    """Metadata-only half-open byte ranges; views into scratch are forbidden."""
    return (
        a.data_ptr() < b.data_ptr() + b.numel() * b.element_size()
        and b.data_ptr() < a.data_ptr() + a.numel() * a.element_size()
    )


class PreparedTinyGlm:
    def __init__(self, layer):
        import torch

        self.layer_id = getattr(layer, "layer_id", None)
        self.logged = set()
        self.handles = {}
        self.outputs = {}
        self.weights = {}
        if not enabled() or not getattr(layer, "_aiter_tiny_glm_target", False):
            return
        if not layer.w13_weight.is_cuda or not all(
            hasattr(torch, name) for name in ("float4_e2m1fn_x2", "float8_e8m0fnu")
        ):
            return
        if (
            getattr(
                torch.cuda.get_device_properties(layer.w13_weight.device),
                "gcnArchName",
                "",
            ).split(":")[0]
            != "gfx950"
        ):
            return
        self.make = factories()
        if self.make is None:
            return
        self.weights = {
            "w1": layer.w13_weight.view(torch.float4_e2m1fn_x2),
            "w2": layer.w2_weight.view(torch.float4_e2m1fn_x2),
            "w1_scale": layer.w13_weight_scale.view(torch.float8_e8m0fnu),
            "w2_scale": layer.w2_weight_scale.view(torch.float8_e8m0fnu),
        }
        for name in ("w1", "w2"):
            original = layer.w13_weight if name == "w1" else layer.w2_weight
            if not getattr(original, "is_shuffled", False):
                self.weights = {}
                return
            self.weights[name].is_shuffled = True
        if tuple(self.weights["w1"].shape) != (257, 512, 3072) or tuple(
            self.weights["w2"].shape
        ) != (257, 6144, 128):
            self.weights = {}
            return
        if any(not value.is_contiguous() for value in self.weights.values()):
            self.weights = {}
            return
        self.prepare_stream()

    def prepare_stream(self):
        import torch

        if not self.weights or torch.cuda.is_current_stream_capturing():
            return
        stream = torch.cuda.current_stream(self.weights["w1"].device).cuda_stream
        for rows in (4, 8):
            key = (rows, stream)
            if key not in self.handles:
                op = self.make[rows](weights=self.weights, rows=rows)
                self.handles[key] = op
                logger.info(
                    "Prepared AITER GLM tiny MoE layer stream=%s M=%s", stream, rows
                )

    def run_if_supported(self, inputs, quant, config):
        import torch

        if not self.weights or not runner_supported(config, quant, inputs):
            return None
        x, ids, rw = inputs.hidden_states, inputs.topk_ids, inputs.topk_weights
        if (x.dtype, ids.dtype, rw.dtype) != (
            torch.bfloat16,
            torch.int32,
            torch.float32,
        ):
            return None
        if any(
            not value.is_contiguous() or value.device != self.weights["w1"].device
            for value in (x, ids, rw)
        ):
            return None
        if (
            quant.w13_weight.data_ptr(),
            quant.w2_weight.data_ptr(),
            quant.w13_scale.data_ptr(),
            quant.w2_scale.data_ptr(),
        ) != tuple(value.data_ptr() for value in self.weights.values()):
            return None
        self.prepare_stream()
        key = (x.shape[0], torch.cuda.current_stream(x.device).cuda_stream)
        op = self.handles.get(key)
        if op is None:
            self.record(x.shape[0], "fallback", "stream was not prewarmed")
            return None
        # Both APIs return reused output. Never feed it (or its view) back as
        # an input, including M4 whose public factory has no alias guard.
        if any(
            overlaps(value, out)
            for value in (x, ids, rw)
            for out in self.outputs.values()
        ):
            return None
        out = getattr(op, "out", None)
        scratch = getattr(op, "scratch", ())
        if out is not None and any(overlaps(value, out) for value in (x, ids, rw)):
            return None
        if any(overlaps(value, buf) for value in (x, ids, rw) for buf in scratch):
            return None
        output = op.run(x, ids, rw)
        self.outputs[key] = output
        self.record(x.shape[0], "candidate", "native supplied9")
        return output

    def record_routing_count(self, batch, rows):
        # Presence only: never inspect the scalar value or synchronize a GPU.
        kind = "none" if batch.moe_num_token_non_padded() is None else "gpu"
        self.record(rows, "routing", f"num_token_non_padded={kind}")

    def record(self, rows, event, reason):
        import torch

        capture = torch.cuda.is_current_stream_capturing()
        key = (rows, event, reason, capture)
        if key not in self.logged:
            self.logged.add(key)
            logger.info(
                "AITER tiny GLM MoE layer=%s M=%s phase=target_verify capture=%s event=%s reason=%s",
                self.layer_id,
                rows,
                capture,
                event,
                reason,
            )
