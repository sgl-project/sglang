# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash monokernels on gfx950: the spec and its glue (RFC #43949 section 4).

Two spans a layer, with SGLang's attention core and output projection between them:

- ``k1`` (layer_in -> core_in): the previous layer's pending mHC post, this layer's pre, norm and attention front
  (wqkv_a, q / kv norms, wq_b, RoPE, the SWA KV write); it hands its query to SGLang's attention core.
- ``ffn`` (attn_partial -> layer_out): the attention all-reduce (the span starts at a partial point), the FFN seam,
  the MoE with the shared expert, and the MoE all-reduce; the layer's last mHC post stays pending.

``k1`` serves the layers whose previous layer leaves its post pending: not the first layer, not an Engram layer, and
in a DSpark verify step not a layer whose input DSpark captures."""

from __future__ import annotations

import logging
import time
from typing import Optional

import torch

from sglang.experimental.monokernels.guards import SWITCH, span_selection
from sglang.experimental.monokernels.hip.common.plan import BLOCKS
from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.sites import WAIT_SITES
from sglang.experimental.monokernels.runtime import MonokernelError
from sglang.experimental.monokernels.spec import (
    DeviceLimits,
    Exactness,
    Feature,
    MonoSpec,
    Numerics,
    Point,
    Requirements,
    Span,
    SpanPlan,
    StepKind,
)
from sglang.srt.environ import envs
from sglang.srt.layers.utils.common import get_layer_id
from sglang.srt.utils.common import get_bool_env_var, is_gfx95_supported

logger = logging.getLogger(__name__)

SGLANG_READS = {
    "srt/environ.py": (
        "envs",
        "SGLANG_RAGGED_VERIFY_MODE",
        "SGLANG_DSV4_FP4_EXPERTS",
        "SGLANG_DSV4_FP4_DEQUANT",
        "SGLANG_USE_AITER_MOE_GU_ITLV",
        "SGLANG_DSV4_KV_LAYOUT",
    ),
    "srt/layers/utils/common.py": ("get_layer_id",),
    "srt/utils/common.py": ("get_bool_env_var", "is_gfx95_supported"),
    "srt/arg_groups/fields/parallel.py": ("tp_size",),
    "srt/arg_groups/fields/spec.py": ("speculative_algorithm",),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_token_to_kv_pool",
    ),
    "srt/layers/attention/deepseek_v4_backend_hip_radix.py": ("get_swa_out_cache_loc",),
    "srt/mem_cache/deepseek_v4_memory_pool.py": (
        "get_swa_raw_buffer",
        "get_swa_key_page_size",
        "request_window",
        "uniform_fp8",
    ),
    "srt/models/deepseek_v4.py": (
        "remap_weight_name_to_dpsk_hf_format",
        "dspark_layers_to_capture",
        "start_layer",
        "end_layer",
        "engram",
        "compressor",
        "indexer",
        "fuse_wqa_wkv",
        "q_head_norm",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "wqkv_a",
        "wq_b",
        "q_norm",
        "kv_norm",
        "freqs_cis",
        "n_local_heads",
    ),
    "srt/models/deepseek_v2.py": ("is_hash", "is_nextn", "forward_normal"),
    "srt/models/deepseek_common/amd/deepseek_v4_hip.py": (
        "hc_boundary_fused",
        "skip_head_pad",
    ),
    "srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py": (
        "hc_boundary",
        "_gfx95_dense_post_attention_norm",
    ),
    "srt/distributed/communication_op.py": ("tensor_model_parallel_all_reduce",),
    "srt/runtime_context.py": ("get_parallel",),
}

MAX_ROWS = 48
# DeepSeek-V4.1-Flash's shapes, routing and activation, which the kernels build in
_MODEL = dict(
    hidden_size=5120,
    moe_intermediate_size=2304,
    n_routed_experts=384,
    num_experts_per_tok=6,
    n_shared_experts=1,
    routed_scaling_factor=1.5,
    scoring_func="sqrtsoftplus",
    norm_topk_prob=True,
    topk_method="noaux_tc",
    hidden_act="silu",
    swiglu_limit=10.0,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rms_norm_eps=1e-20,
)
# n_group / topk_group (8 / 4 in SGLang's DeepseekV41Config) are not read: DS4.1 routes ungrouped
_SHARED = ("gate_proj", "up_proj", "down_proj")

K1 = Span("k1", Point.LAYER_IN, Point.CORE_IN)
FFN = Span("ffn", Point.ATTN_PARTIAL, Point.LAYER_OUT)

# max_abs = max |out - ref|, max_rel = max |out - ref| / max |ref|. Each bound is twice the largest value over every
# production width, TP2 and TP4, rounded up to a power of two; k1.kv and k1.residual are bitwise. ffn.residual is the
# same fp32 hc_post as SGLang's in another order: up to 1 bf16 ulp of the largest residual element, which reaches
# 2**-7 at the wider widths, so its max_abs bound is 2**-6. ffn.out is checked against an fp32 reduction of the
# golden partials and, at attach, against the normal path (SGLANG_TEST_MONO_CHECK_NUMERICS).
NUMERICS = Numerics(
    Exactness.TOLERANCE,
    {
        "k1.kv.max_abs": 0.0,
        "k1.kv.max_rel": 0.0,
        "k1.residual.max_abs": 0.0,
        "k1.residual.max_rel": 0.0,
        "k1.q.max_abs": 2**-5,
        "k1.q.max_rel": 2**-6,
        "k1.pre.max_abs": 2**-20,
        "k1.pre.max_rel": 2**-20,
        "k1.post.max_abs": 2**-19,
        "k1.post.max_rel": 2**-20,
        "k1.comb.max_abs": 2**-19,
        "k1.comb.max_rel": 2**-19,
        "ffn.residual.max_abs": 2**-6,
        "ffn.residual.max_rel": 2**-8,
        "ffn.out.max_abs": 1.0,
        "ffn.out.max_rel": 2**-5,
        "ffn.post.max_abs": 2**-14,
        "ffn.post.max_rel": 2**-15,
        "ffn.comb.max_abs": 2**-13,
        "ffn.comb.max_rel": 2**-13,
        "ffn.pre.max_abs": 2**-14,
        "ffn.pre.max_rel": 2**-14,
    },
)


def _text_config(hf_config):
    get = getattr(hf_config, "get_text_config", None)
    return get() if get is not None else hf_config


def unsupported_model(config) -> Optional[str]:
    """Why the kernels cannot serve this model config, or None."""
    for key, want in _MODEL.items():
        if getattr(config, key, None) != want:
            return f"needs DeepSeek-V4.1-Flash's {key} = {want!r}"
    return None


def config_disable_reasons(cfg, hf_config) -> list:
    """Before loading: the deployment and the checkpoint config."""
    reasons = []
    algo = (cfg.speculative_algorithm or "none").upper()
    if algo not in ("NONE", "DSPARK"):
        reasons.append("speculative decoding other than DSpark is not supported")
    if algo == "DSPARK" and envs.SGLANG_RAGGED_VERIFY_MODE.get() != "static":
        reasons.append(
            "DSpark needs SGLANG_RAGGED_VERIFY_MODE=static (uniform verify steps)"
        )
    if cfg.tp_size not in (2, 4):
        reasons.append(f"tensor parallel size {cfg.tp_size}; the kernels serve 2 and 4")
    if not is_gfx95_supported():
        reasons.append("needs gfx950")
    why = unsupported_model(_text_config(hf_config))
    if why is not None:
        reasons.append(why)
    # the MoE's MXFP4 experts in aiter's gate / up interleaved (16, 16) shuffle (Fp8MoEMethod)
    if (
        not envs.SGLANG_DSV4_FP4_EXPERTS.get()
        or envs.SGLANG_DSV4_FP4_DEQUANT.get()
        or not envs.SGLANG_USE_AITER_MOE_GU_ITLV.get()
        or get_bool_env_var("AITER_FORCE_A8W4")
    ):
        reasons.append(
            "needs the MXFP4 experts in aiter's gate / up interleaved layout"
        )
    selected = span_selection(SPEC)[0]
    if K1.name in selected and FFN.name not in selected:
        reasons.append(
            "SGLANG_TEST_MONO_SPANS: k1 hands its outputs to ffn, so k1 needs ffn"
        )
    if K1.name in selected and envs.SGLANG_DSV4_KV_LAYOUT.get().lower() != "v4":
        reasons.append("k1 needs the V4 KV layout (fp8 nope, bf16 rope, 584 B a token)")
    return reasons


def _ffn_layer(layer) -> bool:
    return layer.hc_boundary_fused and not layer.mlp.is_hash and not layer.mlp.is_nextn


def _k1_layer(layer) -> bool:
    a = layer.self_attn
    return a.compress_ratio in (1, 2) and a.compressor is None and a.indexer is None


def _k1_unsupported(attn) -> Optional[str]:
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
        is_unified_kv_triton,
    )
    from sglang.srt.models.deepseek_common.amd.deepseek_v4_hip import skip_head_pad

    if is_unified_kv_triton():
        return "k1 needs the paged SWA cache, not unified_kv"
    if not skip_head_pad(attn):
        # the forward pads q's heads for this attention backend; K1 writes the unpadded q
        return "k1 needs unpadded attention heads (aiter or Triton attention at TP > 1)"
    if attn.q_head_norm or not attn.fuse_wqa_wkv:
        return "k1 needs the fused wqkv_a and no q head norm"
    if attn.head_dim != 512 or attn.q_lora_rank != 1280 or attn.rope_head_dim != 64:
        return "k1 needs DeepSeek-V4.1-Flash's attention shapes"
    return None


def model_disable_reasons(owner) -> list:
    """After loading: the kernels' imports, the GPU, the loaded layers and their weights."""
    reasons = []
    try:
        from sglang.experimental.monokernels.hip.deepseek_v41.gfx950 import (  # noqa: F401
            runner,
        )
    except ImportError as err:
        return [f"needs FlyDSL and AITER's FlyDSL helpers ({err})"]
    cus = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    if cus < BLOCKS:
        # every workgroup of a launch must stay resident: a partitioned GPU would deadlock
        reasons.append(f"needs {BLOCKS} compute units, the GPU has {cus}")
    model = owner.model
    selected = span_selection(SPEC)[0]
    for i in range(model.start_layer, model.end_layer):
        layer = model.layers[i]
        if not _ffn_layer(layer):
            continue
        if not getattr(layer.mlp.experts.w13_weight, "is_shuffled", False):
            reasons.append(f"layer {i}: MoE weights not in aiter's shuffle")
        if K1.name in selected and _k1_layer(layer):
            why = _k1_unsupported(layer.self_attn)
            if why is not None:
                reasons.append(f"layer {i}: {why}")
    return sorted(set(reasons), key=reasons.index)


def _k1_reason(model, i: int, capture: set) -> Optional[str]:
    """Why layer i's previous layer applies its own post (so K1 cannot start there), or None."""
    if i == model.start_layer:
        return "the first layer has no pending post"
    if model.layers[i].engram is not None:
        return "Engram reads the residual stream before this layer"
    if i in capture:
        return "DSpark captures this layer's input"
    return None


def spans(owner, kind: StepKind) -> SpanPlan:
    model = owner.model
    capture = (
        set(model.dspark_layers_to_capture or ())
        if kind is StepKind.TARGET_VERIFY
        else set()
    )
    layers, left_out = {}, {}
    for i in range(model.start_layer, model.end_layer):
        layer = model.layers[i]
        if not _ffn_layer(layer):
            continue
        covered = [FFN]
        if _k1_layer(layer):
            why = _k1_reason(model, i, capture)
            if why is None:
                covered.insert(0, K1)
            else:
                left_out.setdefault(why, []).append(i)
        layers[i] = tuple(covered)
    notes = tuple(f"k1 not on layers {ids}: {why}" for why, ids in left_out.items())
    return SpanPlan(layers, notes)


def load_tensor(state, owner, name: str, tensor: torch.Tensor) -> None:
    """Keep this rank's shard of each FP8 shared-expert tensor of a decoder layer, before shared-expert fusion
    converts it to MXFP4 (the FFN launch runs the shared expert in FP8)."""
    if FFN.name not in span_selection(SPEC)[0]:
        return
    hf = owner.remap_weight_name_to_dpsk_hf_format(
        name, is_nextn=False, num_hidden_layers=owner.config.num_hidden_layers
    )
    if "mlp.shared_experts" not in hf:
        return
    i = get_layer_id(hf)
    model = owner.model
    # the checkpoint's mtp.* layers and layers of other pipeline ranks are not served
    if i is None or not model.start_layer <= i < model.end_layer:
        return
    proj = next((p for p in _SHARED if f".{p}." in hf), None)
    if proj is None:
        return
    kind = "weight" if hf.endswith(".weight") else "scale"
    # the TP-sharded intermediate dimension
    dim = 1 if proj == "down_proj" else 0
    from sglang.srt.runtime_context import get_parallel

    p = get_parallel()
    t = tensor.view(torch.uint8)
    n = t.shape[dim] // p.tp_size
    shard = t.narrow(dim, p.tp_rank * n, n).to(torch.cuda.current_device()).contiguous()
    state.data.setdefault("shared", {})[i, proj, kind] = state.keep(
        f"layer {i} shared {proj} {kind}", shard
    )


def loaded(state, owner) -> None:
    """Copy the K1 layers' row-major wqkv_a and wq_b before process_weights_after_loading may reorder them."""
    if K1.name not in span_selection(SPEC)[0]:
        return
    model, plain = owner.model, {}
    for i in range(model.start_layer, model.end_layer):
        layer = model.layers[i]
        if (
            _ffn_layer(layer)
            and _k1_layer(layer)
            and _k1_reason(model, i, set()) is None
        ):
            a = layer.self_attn
            plain[i] = (
                state.keep(f"layer {i} wqkv_a", a.wqkv_a.weight.detach().clone()),
                state.keep(f"layer {i} wq_b", a.wq_b.weight.detach().clone()),
            )
    state.data["k1_plain"] = plain


def _e8m0(scale: torch.Tensor) -> torch.Tensor:
    """A linear's f32 32 x 32 block scales as the E8M0 codes K1 reads: a power of two's exponent bits."""
    if scale.dtype != torch.float32:
        raise RuntimeError(f"K1 needs f32 block scales, got {scale.dtype}")
    bits = scale.contiguous().view(torch.int32)
    if not bool(((bits & 0x807FFFFF) == 0).all()):
        raise RuntimeError("K1 needs power-of-two (ue8m0) block scales")
    return (bits >> 23).to(torch.uint8)


def _ffn_weights(layer, shared: dict, state, tp: int):
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.runner import (
        MonoLayerWeights,
    )

    i, moe = layer.layer_id, layer.mlp
    sh = {(p, k): shared.get((i, p, k)) for p in _SHARED for k in ("weight", "scale")}
    missing = sorted(f"{p} {k}" for (p, k), t in sh.items() if t is None)
    if missing:
        raise RuntimeError(f"layer {i}: no shared-expert shards for {missing}")
    # n: the routed experts, before a fused shared one
    e, n = moe.experts, moe.gate.weight.shape[0]

    def u8(t):
        return t.view(torch.uint8)

    w = MonoLayerWeights(
        hc_ffn_fn=layer.hc_ffn_fn,
        hc_ffn_scale=layer.hc_ffn_scale,
        hc_ffn_base=layer.hc_ffn_base,
        ffn_norm=layer.post_attention_layernorm.weight,
        gate_w=moe.gate.weight,
        # SGLang routes with a bf16 copy of the f32 checkpoint bias; the kernels read f32
        bias=state.keep(
            f"layer {i} router bias f32", moe.gate.e_score_correction_bias.float()
        ),
        w13=e.w13_weight[:n],
        w13_s=u8(e.w13_weight_scale_inv)[: n * e.w13_weight.shape[1]],
        w2=e.w2_weight[:n],
        w2_s=u8(e.w2_weight_scale_inv)[: n * e.w2_weight.shape[1]],
        sgu=state.keep(
            f"layer {i} shared gate_up",
            torch.cat([sh["gate_proj", "weight"], sh["up_proj", "weight"]]).view(
                torch.float8_e4m3fn
            ),
        ),
        sgu_s=state.keep(
            f"layer {i} shared gate_up scale",
            torch.cat([sh["gate_proj", "scale"], sh["up_proj", "scale"]]),
        ),
        sw2=sh["down_proj", "weight"].view(torch.float8_e4m3fn),
        sw2_s=sh["down_proj", "scale"],
    )
    w.check(tp)
    return w


def _k1_weights(layer, plain: tuple, cos_sin: torch.Tensor) -> dict:
    a = layer.self_attn

    def fp8(lin, copy, shape):
        # the FP8 linear method may have rewritten the weight in place; K1 reads the row-major copy then
        shuffled = getattr(lin, "aiter_bpreshuffled", False) or getattr(
            lin, "mxfp8_native_ready", False
        )
        w = copy if shuffled else lin.weight
        s = _e8m0(lin.weight_scale_inv)
        if not (
            tuple(w.shape) == shape
            and w.dtype == torch.float8_e4m3fn
            and w.is_contiguous()
            and tuple(s.shape) == (shape[0] // 32, shape[1] // 32)
        ):
            raise RuntimeError(
                f"K1: weight {tuple(w.shape)} {w.dtype}, scales {tuple(s.shape)}, want {shape}"
            )
        return w, s

    wqkv, wqkv_s = fp8(a.wqkv_a, plain[0], (1792, 5120))
    wqb, wqb_s = fp8(a.wq_b, plain[1], (a.n_local_heads * 512, 1280))
    w = dict(
        hc_fn=layer.hc_attn_fn,
        hc_scale=layer.hc_attn_scale,
        hc_base=layer.hc_attn_base,
        norm=layer.input_layernorm.weight,
        wqkv=wqkv,
        wqkv_s=wqkv_s,
        q_norm=a.q_norm.weight,
        kv_norm=a.kv_norm.weight,
        wqb=wqb,
        wqb_s=wqb_s,
        cos_sin=cos_sin,
    )
    bad = [name for name, t in w.items() if not (t.is_contiguous() and t.is_cuda)]
    if (
        bad
        or w["hc_fn"].dtype != torch.float32
        or not (w["q_norm"].dtype == w["kv_norm"].dtype == torch.bfloat16)
    ):
        raise RuntimeError(f"K1: unexpected weights {bad or 'dtypes'}")
    if not (
        cos_sin.dtype == torch.float32 and cos_sin.dim() == 2 and cos_sin.shape[1] == 64
    ):
        raise RuntimeError(
            f"K1: rope table {tuple(cos_sin.shape)} {cos_sin.dtype}, want [positions, 64] f32"
        )
    return w


def _warm_normal_path(moe, device: torch.device) -> float:
    """Run the normal MoE path once at widths 1 to MAX_ROWS, so no later step compiles its kernels."""
    t0 = time.perf_counter()
    x = torch.randn(
        MAX_ROWS, _MODEL["hidden_size"], dtype=torch.bfloat16, device=device
    )
    for m in range(1, MAX_ROWS + 1):
        # per rank, no collective
        moe.forward_normal(x[:m], return_moe_output=True)
    torch.cuda.synchronize(device)
    return time.perf_counter() - t0


class _FfnSpan:
    def __init__(self, mono, layer_id: int):
        self.mono, self.layer_id = mono, layer_id

    def __call__(self, part, residual, post, comb, pre):
        """wo_b's unreduced output and the seam state -> (MoE output, residual, post, comb, pre)."""
        out, residual, post, comb, pre = self.mono.runner.ffn(
            self.mono.ffn_w[self.layer_id],
            part,
            residual,
            post,
            comb,
            pre,
            self.layer_id,
        )
        return out, residual, post.view(post.shape[0], -1), comb, pre


class _K1Span:
    def __init__(self, mono, layer_id: int, first: bool):
        self.mono, self.layer_id, self.first = mono, layer_id, first

    def __call__(self, positions, forward_batch, pending_post, prev_pre):
        """The previous layer's pending post and this layer's attention front -> (q, residual, post, comb, pre);
        the step's KV rows are written to the SWA cache."""
        from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.runner import (
            HC,
            check_rows,
        )
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            get_token_to_kv_pool,
        )

        if pending_post is None:
            raise MonokernelError(
                f"{SWITCH}: k1 on layer {self.layer_id} without the previous layer's pending post"
            )
        x, residual, post, comb = pending_post
        r, w, lid = self.mono.runner, self.mono.k1_w[self.layer_id], self.layer_id
        M = x.shape[0]
        slot = get_attn_backend().get_swa_out_cache_loc(forward_batch)
        # K1 writes through raw pointers: a layout it does not expect would corrupt memory, not fault
        if not (1 <= M <= MAX_ROWS and slot.shape[0] >= M and positions.shape[0] >= M):
            raise RuntimeError(
                f"K1: {M} rows, {tuple(slot.shape)} SWA slots, {tuple(positions.shape)} positions"
            )
        bf, f32 = torch.bfloat16, torch.float32
        hidden = _MODEL["hidden_size"]
        check_rows(
            "K1",
            M,
            r.device,
            {
                "x": (x, hidden, bf),
                "residual": (residual, HC * hidden, bf),
                "post": (post, HC, f32),
                "comb": (comb, HC * HC, f32),
                "pre": (prev_pre, HC, f32),
            },
        )
        if r.q.shape[1] != self.mono.heads:
            raise RuntimeError(
                f"K1: q buffer {tuple(r.q.shape)}, {self.mono.heads} heads"
            )
        # the plan's first K1 layer copies the step's positions and slots into the persistent buffers, so a
        # graph capture records the copy
        if self.first:
            r.pos64[:M].copy_(positions[:M])
            r.slot64[:M].copy_(slot[:M])
        pool = get_token_to_kv_pool()
        if pool.request_window is not None or pool.uniform_fp8:
            raise RuntimeError("K1 needs the paged V4 SWA pool")
        swa = pool.get_swa_raw_buffer(lid)
        block = pool.get_swa_key_page_size()
        if not (
            swa.dtype == torch.uint8
            and swa.dim() == 2
            and swa.stride(0) >= block * 584
            and swa.stride(1) == 1
        ):
            raise RuntimeError(
                f"K1: SWA cache {swa.dtype} {tuple(swa.shape)} {swa.stride()}, page {block} x 584 B"
            )
        r.kernel(M, "k1")(
            residual.data_ptr(),
            x.data_ptr(),
            post.data_ptr(),
            comb.data_ptr(),
            prev_pre.data_ptr(),
            w["hc_fn"].data_ptr(),
            w["hc_scale"].data_ptr(),
            w["hc_base"].data_ptr(),
            w["norm"].data_ptr(),
            r.res_mid.data_ptr(),
            r.post_a.data_ptr(),
            r.comb_a.data_ptr(),
            r.pre_a.data_ptr(),
            w["wqkv"].data_ptr(),
            w["wqkv_s"].data_ptr(),
            w["q_norm"].data_ptr(),
            w["kv_norm"].data_ptr(),
            w["wqb"].data_ptr(),
            w["wqb_s"].data_ptr(),
            w["cos_sin"].data_ptr(),
            r.pos64.data_ptr(),
            r.slot64.data_ptr(),
            swa.data_ptr(),
            swa.stride(0),
            block,
            r.q.data_ptr(),
            r.scratch(M).data_ptr(),
            r.epoch.data_ptr(),
            stream=torch.cuda.current_stream(),
        )
        return r.q[:M], r.res_mid[:M], r.post_a[:M], r.comb_a[:M], r.pre_a[:M]


class DSV41Mono:
    """DeepSeek-V4.1's attached monokernels on one rank: the runner, the bound weights, and the spans."""

    def __init__(self, ctx, owner, state):
        from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.runner import (
            DSV41MonoLayer,
        )

        model = owner.model
        plan = spans(owner, StepKind.DECODE).layers
        ffn_ids = [i for i, s in plan.items() if FFN in s and FFN.name in ctx.spans]
        k1_ids = [i for i, s in plan.items() if K1 in s and K1.name in ctx.spans]
        self.runner = DSV41MonoLayer(
            ctx.tp_size, ctx.tp_rank, ctx.group, ctx.device, ctx.deadline_s, ctx.fault
        )
        shared = state.data.get("shared", {})
        self.ffn_w = {
            i: _ffn_weights(model.layers[i], shared, state, ctx.tp_size)
            for i in ffn_ids
        }
        state.data.pop("shared", None)
        plain = state.data.pop("k1_plain", {})
        missing = [i for i in k1_ids if i not in plain]
        if missing:
            raise RuntimeError(
                f"no row-major wqkv_a / wq_b copies for K1 layers {missing}"
            )
        tables: list = []
        self.k1_w = {}
        for i in k1_ids:
            a = model.layers[i].self_attn
            cos_sin = next((t for f, t in tables if f is a.freqs_cis), None)
            if cos_sin is None:
                cos_sin = state.keep(
                    f"rope table {len(tables)}",
                    torch.cat([a.freqs_cis.real, a.freqs_cis.imag], -1)
                    .float()
                    .contiguous(),
                )
                tables.append((a.freqs_cis, cos_sin))
            self.k1_w[i] = _k1_weights(model.layers[i], plain[i], cos_sin)
        self.heads = model.layers[k1_ids[0]].self_attn.n_local_heads if k1_ids else 0
        self.warm_s = (
            _warm_normal_path(model.layers[ffn_ids[0]].mlp, ctx.device)
            if ffn_ids
            else 0.0
        )

    def allocate(self, widths) -> dict:
        scratch = self.runner.allocate(widths)
        persistent = self.runner.persistent_bytes()
        return {
            "persistent MiB": round(sum(persistent.values()) / 2**20, 1),
            "scratch MiB per width": {
                w: round(b / 2**20, 2) for w, b in scratch.items()
            },
            "normal-path warm-up s": round(self.warm_s, 1),
        }

    def build_widths(self, widths) -> dict:
        return self.runner.build(widths, k1=bool(self.k1_w))

    def check_numerics(self, owner, widths) -> dict:
        """The ffn span against SGLang's normal path on the loaded weights (RFC section 10): the same seam inputs
        through the attention all-reduce, the mHC boundary, the post-attention norm and the layer's MoE
        (forward_normal, with its all-reduce), at every width, on the first, a middle and the last ffn layer.
        Runs collectives in the same order on every rank. Returns {(width, key): value}."""
        from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.runner import HC
        from sglang.experimental.monokernels.spec import measure
        from sglang.srt.distributed.communication_op import (
            tensor_model_parallel_all_reduce,
        )
        from sglang.srt.models.deepseek_common.amd.deepseek_v4_fused_mhc import (
            _gfx95_dense_post_attention_norm,
            hc_boundary,
        )

        ids = sorted(self.ffn_w)
        layers = sorted({ids[0], ids[len(ids) // 2], ids[-1]}) if ids else []
        dev, hidden = self.runner.device, _MODEL["hidden_size"]
        rank, tp = self.runner.rank, self.runner.tp
        out: dict = {}
        for M in widths:
            g = torch.Generator(device=dev).manual_seed(11 + M)
            res = torch.randn(M, HC, hidden, generator=g, device=dev)
            res = (res * torch.rsqrt(res.pow(2).mean(-1, keepdim=True))).to(
                torch.bfloat16
            )
            comb = torch.rand(M, HC, HC, generator=g, device=dev) + 0.1
            for _ in range(20):
                comb = comb / comb.sum(-1, keepdim=True)
                comb = comb / comb.sum(-2, keepdim=True)
            post = torch.sigmoid(torch.randn(M, HC, generator=g, device=dev)) * 2
            pre = torch.sigmoid(torch.randn(M, HC, generator=g, device=dev)) + 1e-6
            gp = torch.Generator(device=dev).manual_seed(100 + 7 * rank + M)
            part = (torch.randn(M, hidden, generator=gp, device=dev) * 0.5 / tp).to(
                torch.bfloat16
            )
            for i in layers:
                layer = owner.model.layers[i]
                got = [
                    t.clone()
                    for t in self.runner.ffn(
                        self.ffn_w[i], part, res, post, comb.contiguous(), pre, i
                    )
                ]
                x = tensor_model_parallel_all_reduce(part.clone())
                res2, y, co = hc_boundary(
                    layer, x, res, post, comb.contiguous(), pre,
                    layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base,
                )  # fmt: skip
                normed = _gfx95_dense_post_attention_norm(layer, y, co)
                ref_out = layer.mlp.forward_normal(normed)
                pre_n, post_n, comb_n = co.tensors()
                refs = (ref_out, res2, post_n, comb_n, pre_n)
                for name, a, b in zip(
                    ("out", "residual", "post", "comb", "pre"), got, refs
                ):
                    m = measure(a.float().reshape(b.shape), b.float())
                    for metric, v in m.items():
                        key = (M, f"ffn.{name}.{metric}")
                        out[key] = max(out.get(key, 0.0), v)
        torch.cuda.synchronize(dev)
        return out

    def span(self, plan, layer_id: int, span: Span):
        if span == K1:
            return _K1Span(self, layer_id, plan.first.get(K1.name) == layer_id)
        if span == FFN:
            return _FfnSpan(self, layer_id)
        raise ValueError(f"unknown span {span}")


def build(ctx, owner, state) -> DSV41Mono:
    return DSV41Mono(ctx, owner, state)


SPEC = MonoSpec(
    name="deepseek_v41",
    devices={
        "gfx950": DeviceLimits(
            max_rows=MAX_ROWS, tp_sizes=(2, 4), resident_blocks=BLOCKS
        )
    },
    features=frozenset({Feature.SPECULATIVE}),
    requires=Requirements(
        dtypes=(
            "MXFP4 routed experts",
            "FP8 shared expert",
            "FP8 wqkv_a / wq_b with E8M0 scales",
        ),
        kv_cache="paged V4 SWA pool (fp8 nope, bf16 rope)",
        attention_backends=("deepseek_v4_backend_hip_radix",),
        moe_backends=("aiter MXFP4, gate / up interleaved",),
    ),
    span_names=(K1.name, FFN.name),
    spans=spans,
    numerics=NUMERICS,
    config_disable_reasons=config_disable_reasons,
    model_disable_reasons=model_disable_reasons,
    build=build,
    load_tensor=load_tensor,
    loaded=loaded,
    wait_sites=WAIT_SITES,
)
