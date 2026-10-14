# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash on gfx950: the monokernel spec and its glue (RFC #43949).

The HIP layer loop hands each layer an mHC state whose last mixing step may still
be pending (``deepseek_v4_hip.forward_layer_fused_boundary``). That state is this
spec's LAYER_IN and LAYER_OUT, and each span returns what the fused boundary
(``forward_hc_pre_from_prev_fused_boundary``) returns.

- ``layer`` (LAYER_IN -> LAYER_OUT), in two launches, on every backbone layer
  (compress ratio 0, 1 or 2). K1 runs the attention seam and front and writes
  the step's SWA KV records and K2's key table; K2 runs the attention core, wo_a,
  wo_b, the FFN seam, the MoE and both all-reduces. The core reads KV records that
  every workgroup of K1 wrote, and each gfx950 XCD has its own L2: a launch
  boundary is the cheapest device-wide release.
  - K1 folds the previous layer's pending post into the residual. Where none is
    owed (the first layer, the broadcast embedding; an Engram layer, whose
    Engram write SGLang's loop runs on the settled residual) it takes the
    residual as it is, collapsed with the previous pre (copy 0 on the first).
  - An index source (its compressor and indexer, ratio 1 or 2): K1 also writes
    its normed input and qr, SGLang's ``forward_low_ratio_sources`` runs on them
    between the launches, as SGLang runs it before the layer's attention, and K2
    builds the key table from the top-k it selected.
  - A window-only layer (ratio 0) reads no compressed cache.

A layer of another compress ratio keeps the normal path; DeepSeek-V4.1-Flash has
none. The FFN launch (``gfx950/layer.py``) serves no span: the kernel checks
use it as the reference for K2's FFN half.

Under a plan the covered layers take DSpark's captures, the residual each target
layer's boundary rebuilds, so the layers before them keep their post pending.

The kernels read SGLang's weights in place, except the copies logged at startup:
each covered layer's FP8 shared expert (this rank's shard, kept at load, because
``--enforce-shared-experts-fusion`` converts it to MXFP4), the E8M0 codes of the
f32 block scales, the f32 router bias, this rank's f32 attention sinks, the RoPE
table (one per table: layers 0 and 1 have their own), and wo_a as MXFP8 where
SGLang dequantized it to bf16 (K2 widens it back to bf16 in registers and keeps
the attention output in bf16, as the normal path's wo_a does). Pad rows write KV
slot 0, the reserved dummy slot, as the normal path does.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch

from sglang.experimental.monokernels.hip.common.plan import BLOCKS
from sglang.experimental.monokernels.hip.deepseek_v41 import HF_CONFIG
from sglang.experimental.monokernels.spec import (
    DeviceLimits,
    Feature,
    MonoSpec,
    Numerics,
    NumericsKind,
    Requirements,
    Span,
    SpanPoint,
)

MAX_ROWS = 48  # the rows a launch takes at most (``layer.MAX_TOKENS``)
LAYER = Span("layer", SpanPoint.LAYER_IN, SpanPoint.LAYER_OUT)  # K1, then K2

_SHARED = ("gate_proj", "up_proj", "down_proj")

# The SGLang names this glue reads, by file under python/sglang/ (``f(p)``: a
# parameter, ``"x"``: a string literal); test_mono_contract pins them.
SGLANG_READS = {
    "kernels/ops/attention/dsv4/unified_kv_kernels/env_gate.py": (
        "is_unified_kv_triton",
    ),
    "kernels/ops/layernorm/mhc_boundary_hip.py": ("tensors",),
    "srt/arg_groups/fields/*.py": (
        "speculative_algorithm",
        "enable_encoder_swa_bounded_replay",
        "fp8_gemm_runner_backend",
    ),
    "srt/environ.py": (
        "envs",
        "SGLANG_DSV4_FP4_EXPERTS",
        "SGLANG_DSV4_FP4_DEQUANT",
        "SGLANG_USE_AITER_MOE_GU_ITLV",
        "SGLANG_DSV4_KV_LAYOUT",
    ),
    # an index source's compressor and indexer, between K1 and K2
    "srt/layers/attention/deepseek_v4_backend.py": (
        "forward_low_ratio_sources(layer)",
        "forward_low_ratio_sources(x)",
        "forward_low_ratio_sources(q_lora)",
        "forward_low_ratio_sources(positions)",
        "forward_low_ratio_sources(forward_batch)",
    ),
    "srt/layers/attention/deepseek_v4_backend_hip_radix.py": (
        "forward_low_ratio_sources",
        "forward_metadata",
        "core_attn_metadata",
        "get_swa_out_cache_loc",
        "swa_page_indices",
        "swa_topk_lengths",
        "sparse_page_indices",
        "sparse_topk_lengths",
    ),
    "srt/layers/quantization/fp8.py": (
        "aiter_bpreshuffled",
        "w13_weight",
        "w2_weight",
        "w13_weight_scale_inv",
        '"w2_weight_scale_inv"',
    ),
    "srt/layers/quantization/fp8_hip.py": ("mxfp8_native_ready",),
    "srt/layers/quantization/mxfp4.py": ("is_shuffled",),
    "srt/layers/utils/common.py": ("get_layer_id",),
    "srt/mem_cache/deepseek_v4_memory_pool.py": (
        "get_swa_raw_buffer",
        "get_swa_key_page_size",
        "get_extra_key_buffer",
        "get_extra_key_page_size",
    ),
    "srt/model_executor/forward_context.py": (
        "get_attn_backend",
        "get_token_to_kv_pool",
    ),
    # the fused mHC boundary the spans stand in for, and its return protocol
    "srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py": (
        "forward_hc_pre_from_prev_fused_boundary(plan)",
        "forward_hc_pre_from_prev_fused_boundary(pending_post)",
        "forward_hc_pre_from_prev_fused_boundary(prev_pre)",
        "forward_hc_pre_from_prev_fused_boundary(defer_post)",
    ),
    "srt/models/deepseek_common/amd/deepseek_v4_hip.py": (
        "forward_layer_fused_boundary(plan)",
        "hc_boundary_fused",
    ),
    "srt/models/deepseek_v2.py": (
        "experts",
        "gate",
        "e_score_correction_bias",
        "is_nextn",
        "is_hash",
        "forward_normal",
    ),
    "srt/models/deepseek_v4.py": (
        "layers",
        "config",
        "dspark_layers_to_capture",
        "self_attn",
        "mlp",
        "engram",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_attn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "hc_ffn_base",
        "input_layernorm",
        "post_attention_layernorm",
        "hc_post",
        "compressor",
        "indexer",
        "q_head_norm",
        "fuse_wqa_wkv",
        "n_local_heads",
        "wqkv_a",
        "wq_b",
        "wo_a",
        "wo_b",
        "q_norm",
        "kv_norm",
        "freqs_cis",
        "attn_sink",
        "compress_ratio",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_rank"', '"tp_size"', '"tp_group"'),
    "srt/utils/common.py": ("get_bool_env_var",),
}

# Per launch output against the normal path on the same GPU (the kernel tests)
NUMERICS = Numerics(
    NumericsKind.TOLERANCE,
    {
        # K1: the seam's mixes and the SWA KV records as the normal path writes them
        "k1.kv.max_abs": 0.0,
        "k1.mix.max_abs": 1e-6,
        # wq_b summed in another order; bf16 rounding flips only
        "k1.q.cosine": 0.99999,
        "k1.q.rel": 5e-3,
        # an index source's normed input as the normal path's norm writes it, and
        # its qr (q_norm's sum of squares in another order: bf16 flips)
        "k1.normed.max_abs": 0.0,
        "k1.qr.rel": 4e-3,
        # K2: the attention core output, wo_a's MXFP8 output and wo_b's partial
        "k2.attn_out.cosine": 0.9998,
        "k2.wo_a.rel": 5e-3,
        "k2.wo_b.rel": 1e-3,
        # the FFN launch (K2 runs the same stages)
        "ffn.mix.rel": 1e-4,
        "ffn.normed.cosine": 0.9999,
        "ffn.normed.rel": 0.012,
        "ffn.moe.cosine": 0.9999,
        "ffn.moe.rel": 0.015,
    },
)


def _layer_spans(config) -> dict[int, tuple[Span, ...]]:
    first = getattr(config, "num_hash_layers", 0) or 0  # hash-routed MoE: normal path
    return {
        i: (LAYER,)
        for i in range(first, config.num_hidden_layers)
        if config.compress_ratios[i] in (0, 1, 2)
    }


def _config_disable_reasons(cfg, hf_config) -> list[str]:
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.env_gate import (
        is_unified_kv_triton,
    )
    from sglang.srt.environ import envs
    from sglang.srt.utils import get_bool_env_var

    reasons = [
        f"needs DeepSeek-V4.1-Flash's {key} = {want!r}"
        for key, want in HF_CONFIG.items()
        if getattr(hf_config, key, None) != want
    ]
    if getattr(hf_config, "model_type", None) != "deepseek_v41":
        reasons.append(
            f"model type {getattr(hf_config, 'model_type', None)!r}, needs deepseek_v41"
        )
    backend = getattr(cfg, "fp8_gemm_runner_backend", None)
    if backend != "aiter":
        # the other FP8 GEMM backends reorder wqkv_a, wq_b and wo_b in place
        reasons.append(
            f"--fp8-gemm-backend {backend!r}: needs aiter, which keeps the attention "
            "weights row-major for K1 and K2"
        )
    algo = (cfg.speculative_algorithm or "NONE").upper()
    if algo not in ("NONE", "DSPARK"):
        # DSpark's target steps are uniform verify steps; other drafts differ
        reasons.append(f"speculative algorithm {algo}: only DSpark")
    if (
        not envs.SGLANG_DSV4_FP4_EXPERTS.get()
        or envs.SGLANG_DSV4_FP4_DEQUANT.get()
        or not envs.SGLANG_USE_AITER_MOE_GU_ITLV.get()
        or get_bool_env_var("AITER_FORCE_A8W4")
    ):
        reasons.append(
            "needs the MXFP4 experts in aiter's gate / up interleaved layout"
        )
    if is_unified_kv_triton() or getattr(
        cfg, "enable_encoder_swa_bounded_replay", False
    ):
        reasons.append("needs the paged SWA cache, not unified_kv or a request window")
    if envs.SGLANG_DSV4_KV_LAYOUT.get().lower() != "v4":
        reasons.append("needs the V4 KV layout (fp8 nope, bf16 rope, 584 B a token)")
    if importlib.util.find_spec("flydsl") is None:
        reasons.append("needs FlyDSL")
    return reasons


def _model_disable_reasons(model, hf_config, copies: dict) -> list[str]:
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.attention.plan import (
        Dims,
    )
    from sglang.srt.runtime_context import get_parallel

    heads = Dims(get_parallel().tp_size).heads
    reasons, unseen = [], []
    for i in _layer_spans(hf_config):
        layer = model.layers[i]
        a, moe = layer.self_attn, layer.mlp
        if not layer.hc_boundary_fused or moe.is_nextn or moe.is_hash:
            reasons.append(f"layer {i}: needs the fused mHC boundary and a routed MoE")
        if not getattr(moe.experts.w13_weight, "is_shuffled", False):
            reasons.append(f"layer {i}: MoE weights not in aiter's shuffle")
        if any(
            ("shared expert fp8", i, p, k) not in copies
            for p in _SHARED
            for k in ("weight", "scale")
        ):
            unseen.append(i)
        if a.q_head_norm or not a.fuse_wqa_wkv or a.n_local_heads != heads:
            reasons.append(
                f"layer {i}: K1 needs the fused wqkv_a, {heads} local heads and no "
                "q head norm"
            )
        if (a.compressor is not None or a.indexer is not None) and (
            a.compress_ratio not in (1, 2)
        ):
            reasons.append(
                f"layer {i}: a compressor or indexer at compress ratio "
                f"{a.compress_ratio}; SGLang's low-ratio sources serve ratio 1 or 2"
            )
        if any(
            getattr(lin, "aiter_bpreshuffled", False)
            or getattr(lin, "mxfp8_native_ready", False)
            for lin in (a.wqkv_a, a.wq_b, a.wo_a, a.wo_b)
        ):
            reasons.append(
                f"layer {i}: an FP8 linear method reordered an attention weight in "
                "place; K1 and K2 read them row-major"
            )
    if unseen:
        reasons.append(
            "the shared expert's FP8 checkpoint tensors were not seen at load "
            f"(layers {unseen})"
        )
    return reasons


def _load_tensor(copies: dict, model, name: str, tensor: torch.Tensor) -> None:
    """Keep this rank's shard of each covered layer's FP8 shared expert."""
    from sglang.srt.layers.utils import get_layer_id
    from sglang.srt.runtime_context import get_parallel

    proj = next((p for p in _SHARED if f"mlp.shared_experts.{p}." in name), None)
    i = None if proj is None else get_layer_id(name)
    if i is None or i not in _layer_spans(model.config):
        return  # the checkpoint's mtp.* layers too
    p = get_parallel()
    dim = 1 if proj == "down_proj" else 0  # the TP-sharded intermediate dimension
    t = tensor.view(torch.uint8)
    n = t.shape[dim] // p.tp_size
    shard = t.narrow(dim, p.tp_rank * n, n).to(torch.cuda.current_device())
    kind = "weight" if name.endswith(".weight") else "scale"
    copies["shared expert fp8", i, proj, kind] = shard.contiguous()


@dataclass
class _Layer:
    """One covered layer: its module and weights, and its caches (bound at the
    layer's first launch)."""

    i: int
    module: Any
    ffn: Any  # FFNWeights
    k1: Any = None  # K1Weights
    k2: Any = None  # K2Weights
    ratio: int = 0  # the compress ratio: which sparse key table K1 reads (0: none)
    # an index source: SGLang's compressor and indexer run between K1 and K2
    index: bool = False
    swa: Any = None  # the SWA cache, uint8 [pages, page bytes]
    swa_block: int = 0
    comp: Any = None  # the compressed cache, the same layout
    comp_block: int = 0


def _keep(copies: dict, t: torch.Tensor, *key) -> torch.Tensor:
    copies[key] = t
    return t


def _codes(copies: dict, i: int, name: str, lin) -> torch.Tensor:
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.weights import e8m0

    return _keep(
        copies,
        e8m0(lin.weight_scale_inv, f"layer {i} {name}"),
        "e8m0 scale codes",
        i,
        name,
    )


def _weights(copies: dict, tables: list, layer, i: int) -> _Layer:
    """Layer i's tensors as the launches read them."""
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.attention.plan import (
        Dims,
    )
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.runner import (
        FFNWeights,
        K1Weights,
        K2Weights,
    )
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.weights import (
        mxfp8_exact,
        rope_table,
    )
    from sglang.srt.runtime_context import get_parallel

    p, a, moe = get_parallel(), layer.self_attn, layer.mlp
    d, e, u8, e4m3 = Dims(p.tp_size), moe.experts, torch.uint8, torch.float8_e4m3fn
    n = moe.gate.weight.shape[0]  # the routed experts, before a fused shared one

    def gate_up(kind):  # the shared expert's gate and up projections, one tensor
        parts = [copies.pop(("shared expert fp8", i, q, kind)) for q in _SHARED[:2]]
        return _keep(copies, torch.cat(parts), "shared expert fp8", i, "gate_up", kind)

    def down(kind):
        return copies["shared expert fp8", i, "down_proj", kind]

    ffn = FFNWeights(
        hc_ffn_fn=layer.hc_ffn_fn,
        hc_ffn_scale=layer.hc_ffn_scale,
        hc_ffn_base=layer.hc_ffn_base,
        ffn_norm=layer.post_attention_layernorm.weight,
        gate_w=moe.gate.weight,
        bias=_keep(
            copies, moe.gate.e_score_correction_bias.float(), "f32 router bias", i
        ),
        w13=e.w13_weight[:n],
        w13_s=e.w13_weight_scale_inv.view(u8)[: n * e.w13_weight.shape[1]],
        w2=e.w2_weight[:n],
        w2_s=e.w2_weight_scale_inv.view(u8)[: n * e.w2_weight.shape[1]],
        sgu=gate_up("weight").view(e4m3),
        sgu_s=gate_up("scale"),
        sw2=down("weight").view(e4m3),
        sw2_s=down("scale"),
    )
    ffn.check(p.tp_size)
    L = _Layer(i, layer, ffn)
    cos_sin = rope_table(a.freqs_cis, tables)
    L.k1 = K1Weights(
        hc_fn=layer.hc_attn_fn,
        hc_scale=layer.hc_attn_scale,
        hc_base=layer.hc_attn_base,
        norm=layer.input_layernorm.weight,
        wqkv=a.wqkv_a.weight,
        wqkv_s=_codes(copies, i, "wqkv_a", a.wqkv_a),
        q_norm=a.q_norm.weight,
        kv_norm=a.kv_norm.weight,
        wqb=a.wq_b.weight,
        wqb_s=_codes(copies, i, "wq_b", a.wq_b),
        cos_sin=cos_sin,
    )
    L.k1.check(d)
    if a.wo_a.weight.dtype == e4m3:
        woa, woa_s = a.wo_a.weight, _codes(copies, i, "wo_a", a.wo_a)
    else:  # SGLang dequantized the checkpoint's FP8 wo_a to bf16 at load
        woa, woa_s = mxfp8_exact(
            a.wo_a.weight.reshape(d.groups * 1024, d.group_k), f"layer {i} wo_a"
        )
        _keep(copies, woa, "wo_a mxfp8", i, "weight")
        _keep(copies, woa_s, "wo_a mxfp8", i, "scale")
    sink = a.attn_sink.detach()[p.tp_rank * d.heads : (p.tp_rank + 1) * d.heads]
    L.k2 = K2Weights(
        woa=woa,
        woa_s=woa_s,
        wob=a.wo_b.weight,
        wob_s=_codes(copies, i, "wo_b", a.wo_b),
        sink=_keep(copies, sink.float().contiguous(), "f32 attention sinks", i),
        cos_sin=cos_sin,
    )
    L.k2.check(d)
    L.ratio = a.compress_ratio
    L.index = a.compressor is not None or a.indexer is not None
    return L


def _boundary_out(layer, defer_post, x, residual, post, comb, pre):
    """What the fused boundary returns: (hidden_states, pre, pending post); with
    ``defer_post`` this layer's post stays pending for the next layer."""
    post = post.view(post.shape[0], -1)
    if defer_post:
        return None, pre, (x, residual, post, comb)
    return layer.hc_post(x, residual, post, comb), pre, None


class _Built:
    """One rank's monokernels: the runner and each covered layer's span."""

    def __init__(self, model, runner, layers: dict):
        self.model, self.runner, self._layers = model, runner, layers
        self._normal_moe = model.layers[min(layers)].mlp
        self._targets_checked = False
        self._kernels = {
            (i, SpanPoint.LAYER_IN): partial(self._layer, L) for i, L in layers.items()
        }

    def kernel(self, layer_id: int, point: SpanPoint, key):
        return self._kernels.get((layer_id, point))

    def _layer(
        self, L, plan, positions, forward_batch, pending_post, prev_pre, defer_post
    ):
        """LAYER_IN -> LAYER_OUT: the previous layer's pending post (x, residual,
        post, comb) and pre in; the fused boundary's outputs out. With no post
        owed (x None) the residual arrives settled: the first layer's broadcast
        embedding (no pre: copy 0), or an Engram layer's residual."""
        from sglang.srt.model_executor.forward_context import (
            get_attn_backend,
            get_token_to_kv_pool,
        )

        self._warm_normal_moe()
        x, residual, post, comb = pending_post
        if L.swa is None:  # the layer's caches, at its first launch
            pool = get_token_to_kv_pool()
            L.swa, L.swa_block = (
                pool.get_swa_raw_buffer(L.i),
                pool.get_swa_key_page_size(),
            )
            if L.ratio:
                # SGLang types the compressed cache as e4m3; K2 addresses its records' bytes
                L.comp = pool.get_extra_key_buffer(L.i).view(torch.uint8)
                L.comp_block = pool.get_extra_key_page_size(L.i)
        backend, M = get_attn_backend(), residual.shape[0]
        if "k1" not in plan.state:
            # the forward's first K1 converts them; a graph records the copies
            slot = backend.get_swa_out_cache_loc(forward_batch)
            plan.state["k1"] = (
                positions[:M].long().contiguous(),
                slot[:M].long().contiguous(),
            )
        pos, slot = plan.state["k1"]
        md = backend.forward_metadata.core_attn_metadata
        if L.ratio:  # the top-k of the layer's index source (its own, after K1)
            topk = (md.sparse_page_indices(L.ratio), md.sparse_topk_lengths(L.ratio))
        else:  # the window only
            topk = (self.runner.no_topk, self.runner.no_topk_lens)
        keys = (md.swa_page_indices, md.swa_topk_lengths, *topk)
        if x is None:
            residual = residual.contiguous()  # the broadcast embedding may be a view
            if prev_pre is None:
                prev_pre = self.runner.first_pre[:M]
        _, res_mid, *_ = self.runner.k1(
            L.k1, x, residual, post, comb, prev_pre, pos, slot, L.swa, L.swa_block,
            keys, index=L.index,
        )  # fmt: skip
        self._capture(plan, L, res_mid)
        if L.index:
            # the compressor and indexer on K1's normed input and qr, as SGLang's
            # low-ratio path runs them before the layer's attention
            backend.forward_low_ratio_sources(
                layer=L.module.self_attn,
                x=self.runner.normed[:M],
                q_lora=self.runner.qr[:M],
                positions=positions,
                forward_batch=forward_batch,
            )
        # K2 reads K1's outputs from the runner's buffers
        out = self.runner.k2(
            L.k2, L.ffn, M, pos, L.swa, L.swa_block, L.comp, L.comp_block,
            slot=slot, keys=keys, index=L.index,
        )  # fmt: skip
        return _boundary_out(L.module, defer_post, *out)

    def _capture(self, plan, L, residual) -> None:
        """DSpark reads the mean of each target layer's input residual; under a plan
        the model loop leaves these captures to the spans."""
        targets = self.model.dspark_layers_to_capture
        if targets is None:
            return
        if not self._targets_checked:  # an uncovered target would lose its capture
            missing = sorted(set(targets) - set(self._layers))
            if missing:
                raise RuntimeError(f"DSpark target layers {missing} have no span")
            self._targets_checked = True
        if L.i in targets:
            plan.state["dspark_aux"].append(residual.mean(dim=1))

    def _warm_normal_moe(self) -> None:
        """At the first eager span, run the normal MoE at 1 to MAX_ROWS rows: the
        decode graphs no longer run it, so no later extend step compiles its
        kernels (per rank, no collective)."""
        moe = self._normal_moe
        if moe is None or torch.cuda.is_current_stream_capturing():
            return
        self._normal_moe = None
        x = torch.randn(
            MAX_ROWS,
            HF_CONFIG["hidden_size"],
            dtype=torch.bfloat16,
            device=self.runner.device,
        )
        for m in range(1, MAX_ROWS + 1):
            moe.forward_normal(x[:m], return_moe_output=True)


def _build(model, hf_config, ctx) -> _Built:
    from sglang.experimental.monokernels.hip.deepseek_v41.gfx950.runner import (
        DSV41MonoRunner,
    )
    from sglang.srt.runtime_context import get_parallel

    tables: list = []
    layers = {
        i: _weights(ctx.copies, tables, model.layers[i], i)
        for i in _layer_spans(hf_config)
    }
    for n, (_, t) in enumerate(tables):
        ctx.copies["rope table", n] = t
    # the peer handshake is collective: every rank gets here (attach agreed first)
    p = get_parallel()
    device = model.layers[0].input_layernorm.weight.device
    runner = DSV41MonoRunner(p.tp_size, p.tp_rank, p.tp_group.cpu_group, device)
    runner.set_deadline(ctx.deadline_s)
    # K1 with and without the owed post (the residual arrives settled on the
    # first layer and after an Engram write), K2 with and without top-k rows
    k1 = {(fold, L.index) for L in layers.values() for fold in (True, False)}
    k2 = {(L.ratio > 0, L.index) for L in layers.values()}
    # Every rank builds every width. FlyDSL's disk cache hands a rank the kernels
    # another rank built (a lock per kernel), so each rank starts at its own share.
    widths = list(ctx.widths)
    start = len(widths) * p.tp_rank // p.tp_size
    for m in widths[start:] + widths[:start]:
        runner.precompile(m, k1=k1, k2=k2)
    return _Built(model, runner, layers)


SPEC = MonoSpec(
    name="deepseek_v41_hip",
    devices={
        "gfx950": DeviceLimits(
            widths=tuple(range(1, MAX_ROWS + 1)),
            tp_sizes=(2, 4),
            resident_blocks=BLOCKS,
        )
    },
    features=frozenset({Feature.SPECULATIVE_DECODING}),
    requires=Requirements(
        dtypes=frozenset({"bfloat16"}),
        kv_cache_dtypes=frozenset({"fp8_e4m3"}),
        attention_backends=frozenset({"dsv4"}),
        moe_runner_backends=frozenset({"auto", "aiter"}),
    ),
    spans=_layer_spans,
    numerics=NUMERICS,
    config_disable_reasons=_config_disable_reasons,
    model_disable_reasons=_model_disable_reasons,
    build=_build,
    load_tensor=_load_tensor,
)
