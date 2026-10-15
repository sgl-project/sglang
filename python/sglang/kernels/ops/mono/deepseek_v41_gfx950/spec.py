"""DeepSeek-V4.1-Flash mono spec (gfx950): the whole-layer launches (K1 + K2) on every decoder layer that can take
them; any other layer runs SGLang's module path.

K2's FFN half runs the TP reduce, the FFN mHC seam, router + top-6, the shared expert, the routed MXFP4 experts and the
MoE reduce. A layer's inputs and outputs are exactly the fused mHC boundary's pending post ``(x, residual, post, comb)``
plus the next pre, so it slots between two SGLang boundaries unchanged.

Weights are views of what SGLang loaded (routed experts in aiter's (16, 16) / shuffle_scale order), except: the shared
expert, which the kernels read row-major while SGLang's path keeps aiter's (16, 16) preshuffle (it still serves steps
above 48 rows), gets a row-major copy (~8.8 MB / layer at TP4); its float32 power-of-two block scales become E8M0 bytes
and the bf16 routing bias f32.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Any

import torch

from sglang.kernels.ops.mono.spec import (
    DeviceLimits,
    Exactness,
    Feature,
    MonoContext,
    MonoSpec,
    MonoStep,
    Numerics,
    Point,
    Requirements,
    Span,
)

# the SGLang names this module reads (pinned by test_mono_contract)
SGLANG_READS = (
    "sglang.srt.environ.envs",
    "sglang.srt.layers.quantization.fp8_utils.unshuffle_aiter_fp8_weight",
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.resolve_compressed_kv_layout",
    "sglang.srt.mem_cache.deepseek_v4_memory_pool.select_dsv4_kv_layout",
    "sglang.srt.model_executor.forward_context.get_attn_backend",
    "sglang.srt.runtime_context.get_server_args",
    "sglang.srt.runtime_context.get_spec",
)

MAX_ROWS = 48  # 8 requests x (1 + 5 DSpark draft tokens)
HIDDEN = 5120


def _text_config(hf_config):
    return getattr(hf_config, "text_config", None) or hf_config


K1 = Span("k1", Point.LAYER_IN, Point.CORE_IN)  # hands its query straight to K2
K2 = Span("k2", Point.CORE_IN, Point.LAYER_OUT)
WHOLE = (K1, K2)


def spans(hf_config) -> dict[int, tuple[Span, ...]]:
    return {i: WHOLE for i in range(_text_config(hf_config).num_hidden_layers)}


def config_disable_reasons(cfg, hf_config) -> list[str]:
    t = _text_config(hf_config)
    want = dict(
        hidden_size=HIDDEN,
        n_routed_experts=384,
        num_experts_per_tok=6,
        moe_intermediate_size=2304,
        n_shared_experts=1,
        hc_mult=4,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
    )
    reasons = [
        f"config {k}={getattr(t, k, None)!r}, the kernels are built for {v!r}"
        for k, v in want.items()
        if getattr(t, k, None) != v
    ]
    if getattr(t, "model_type", None) != "deepseek_v41":
        reasons.append(
            f"model_type {getattr(t, 'model_type', None)!r}: the kernels are DeepSeek-V4.1-Flash only"
        )
    if not getattr(t, "hc_pre_from_prev_sublayer", False):
        reasons.append(
            "the mono launches need the pre-from-previous-sublayer mHC boundary (hc_pre_from_prev_sublayer)"
        )
    if getattr(cfg, "enforce_shared_experts_fusion", False):
        reasons.append(
            "--enforce-shared-experts-fusion: the kernels run the shared expert separately (MXFP8); "
            "drop the flag or pass --disable-shared-experts-fusion"
        )
    if os.environ.get("SGLANG_USE_AITER", "0").lower() not in ("1", "true"):
        reasons.append(
            "SGLANG_USE_AITER=1 is required (the kernels read aiter's MXFP4 MoE layout)"
        )
    if os.environ.get("SGLANG_USE_AITER_MOE_GU_ITLV", "1").lower() in ("0", "false"):
        reasons.append(
            "SGLANG_USE_AITER_MOE_GU_ITLV=0: the kernels read gate/up interleaved (GUGU)"
        )
    main = os.environ.get("SGLANG_DSV4_KV_LAYOUT", "auto").lower()
    comp = os.environ.get("SGLANG_DSV4_COMPRESSED_KV_LAYOUT", "auto").lower()
    flashmla = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "auto").lower()
    v41 = main == "v41" or (main == "auto" and flashmla in ("auto", "aiter_sparse"))
    if not v41 and comp not in ("auto", "fp8"):
        reasons.append(
            "V4 main cache: SGLANG_DSV4_COMPRESSED_KV_LAYOUT must be auto or fp8"
        )
    return reasons


def _moe(layer):
    return layer.mlp


def model_disable_reasons(model) -> list[str]:
    reasons = []
    for i, layer in enumerate(model.layers):
        mlp = _moe(layer)
        if getattr(mlp, "shared_experts", None) is None:
            reasons.append(
                f"layer {i}: no separate shared expert (shared-experts fusion is on)"
            )
            continue
        if not getattr(mlp.experts.w13_weight, "is_shuffled", False):
            reasons.append(f"layer {i}: routed experts not in aiter's (16, 16) shuffle")
        if i >= 1 and reasons:
            break  # one layer's reasons say it for all
    return reasons


def _e8m0_bytes(scale: torch.Tensor) -> torch.Tensor:
    """Block scales as E8M0 bytes: uint8 / float8_e8m0fnu as stored, or float32 powers of two converted."""
    if scale.dtype == torch.uint8:
        return scale.contiguous()
    if scale.dtype == getattr(torch, "float8_e8m0fnu", None):
        return scale.contiguous().view(torch.uint8)
    assert scale.dtype == torch.float32, scale.dtype
    bits = scale.contiguous().view(torch.int32)
    if bool((bits & 0x7FFFFF).any()):
        raise ValueError("shared-expert block scales are not powers of two (not ue8m0)")
    return ((bits >> 23) & 0xFF).to(torch.uint8)


def _row_major(proj) -> torch.Tensor:
    """The linear's FP8 weight row-major: a view, or an unshuffled copy when aiter preshuffled it in place."""
    if not getattr(proj, "aiter_bpreshuffled", False):
        return proj.weight.data
    from sglang.srt.layers.quantization.fp8_utils import unshuffle_aiter_fp8_weight

    return unshuffle_aiter_fp8_weight(proj.weight.data)


def _mxfp8_blocks(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A bf16 weight back to e4m3 + [N/32, K/32] E8M0 bytes. SGLang dequantizes wo_a to bf16 at load when the
    checkpoint's blocks are 32 x 32 (no 128 x 128 fp8 absorb GEMM); those values are e4m3 x 2^s exactly, and the
    amax-derived scale never exceeds the checkpoint's, so this round trip is exact -- checked."""
    n, k = w.shape
    blocks = w.float().reshape(n // 32, 32, k // 32, 32)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp_min(2.0**-126)
    code = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    q = (blocks / torch.exp2(code)).to(torch.float8_e4m3fn)
    back = (q.float() * torch.exp2(code)).reshape(n, k)
    if not torch.equal(back.to(w.dtype), w):
        raise ValueError("wo_a bf16 -> MXFP8 re-quantization is not exact")
    return q.reshape(n, k).contiguous(), (code.reshape(n // 32, k // 32) + 127).to(
        torch.uint8
    ).contiguous()


def _wo_a(attn) -> tuple[torch.Tensor, torch.Tensor]:
    w = attn.wo_a.weight.data
    if w.dtype == torch.bfloat16:
        return _mxfp8_blocks(w)
    return w.contiguous(), _e8m0_bytes(attn.wo_a.weight_scale_inv.data).reshape(
        w.shape[0] // 32, -1
    )


def k2a_mode() -> int:
    """With ``--enable-monokernel``: 1 (default) = the K2i launch (ATOM's FP4 scorer) feeds SGLang's top-k and
    candidate kernels on the indexer layers; ``SGLANG_TEST_MONO_DSV41_K2A=0`` (A/B only) keeps SGLang's indexer logits.
    0 whenever mono is off."""
    from sglang.kernels.ops.mono.guards import mono_enabled
    from sglang.srt.runtime_context import get_server_args

    if not mono_enabled(get_server_args()):
        return 0
    from sglang.srt.environ import envs

    mode = envs.SGLANG_TEST_MONO_DSV41_K2A.get()
    assert mode in (0, 1), f"SGLANG_TEST_MONO_DSV41_K2A={mode}: only 0 / 1 exist"
    return mode


def _has_indexer_side(layer) -> bool:
    attn = layer.self_attn
    return (
        getattr(attn, "compressor", None) is not None
        or getattr(attn, "indexer", None) is not None
    )


def _whole_layer_ok(layer_id: int, layer) -> bool:
    """Whole-layer launch (K1 + K2) on every layer of ratio 0 / 1 / 2 with the fused wqa/wkv projection: compressor /
    indexer layers run their indexer + compressor between the launches; layer 0 takes a one-hot pre (SGLang's copy-0
    collapse); Engram layers run Engram before the layer. Any other layer takes SGLang's module path."""
    attn = layer.self_attn
    return getattr(attn, "compress_ratio", 0) in (0, 1, 2) and getattr(
        attn, "fuse_wqa_wkv", False
    )


def _cos_sin(attn) -> torch.Tensor:
    """The layer's RoPE table as the kernels read it: [positions, 64] f32 = cos (32) | sin (32)."""
    f = attn.freqs_cis
    return torch.cat([f.real, f.imag], dim=-1).float().contiguous()


def _wqkv(attn) -> tuple[torch.Tensor, torch.Tensor]:
    """[wq_a; wkv] row-major + E8M0 blocks: the fused projection as loaded, or (DSpark draft stages, unfused) a copy."""
    if getattr(attn, "fuse_wqa_wkv", False):
        return _row_major(attn.wqkv_a), _e8m0_bytes(attn.wqkv_a.weight_scale_inv.data)
    w = torch.cat([_row_major(attn.wq_a), _row_major(attn.wkv)], 0).contiguous()
    s = torch.cat(
        [
            _e8m0_bytes(attn.wq_a.weight_scale_inv.data),
            _e8m0_bytes(attn.wkv.weight_scale_inv.data),
        ],
        0,
    )
    return w, s.contiguous()


def _attn_weights(layer_id: int, layer, runner, tp_rank: int):
    attn = layer.self_attn
    h = attn.n_local_heads
    wo_a, wo_a_scale = _wo_a(attn)
    wqkv, wqkv_scale = _wqkv(attn)
    return runner.AttnWeights(
        layer_id=layer_id,
        wqkv=wqkv,
        wqkv_scale=wqkv_scale,
        q_norm=attn.q_norm.weight.contiguous(),
        kv_norm=attn.kv_norm.weight.contiguous(),
        wq_b=_row_major(attn.wq_b),
        wq_b_scale=_e8m0_bytes(attn.wq_b.weight_scale_inv.data),
        wo_a=wo_a,
        wo_a_scale=wo_a_scale,
        wo_b=_row_major(attn.wo_b),
        wo_b_scale=_e8m0_bytes(attn.wo_b.weight_scale_inv.data),
        attn_sink=attn.attn_sink.data[tp_rank * h : (tp_rank + 1) * h]
        .float()
        .contiguous(),
        cos_sin=_cos_sin(attn),
        ratio=int(getattr(attn, "compress_ratio", 0) or 0),
    )


def _layer_weights(layer, runner):
    mlp = _moe(layer)
    se, ex = mlp.shared_experts, mlp.experts
    E = ex.w13_weight.shape[0]
    w = runner.MonoLayerWeights(
        attn=None,
        hc_attn_fn=layer.hc_attn_fn,
        hc_attn_scale=layer.hc_attn_scale,
        hc_attn_base=layer.hc_attn_base,
        attn_norm=layer.input_layernorm.weight,
        hc_ffn_fn=layer.hc_ffn_fn.contiguous(),
        hc_ffn_scale=layer.hc_ffn_scale.contiguous(),
        hc_ffn_base=layer.hc_ffn_base.contiguous(),
        ffn_norm=layer.post_attention_layernorm.weight.contiguous(),
        gate_w=mlp.gate.weight.contiguous(),
        bias=mlp.gate.e_score_correction_bias.detach().float().contiguous(),
        w13=ex.w13_weight.data,
        w13_s=_e8m0_bytes(ex.w13_weight_scale_inv.data).reshape(
            E * ex.w13_weight.shape[1], -1
        ),
        w2=ex.w2_weight.data,
        w2_s=_e8m0_bytes(ex.w2_weight_scale_inv.data).reshape(
            E * ex.w2_weight.shape[1], -1
        ),
        sgu=_row_major(se.gate_up_proj),
        sgu_s=_e8m0_bytes(se.gate_up_proj.weight_scale_inv.data),
        sw2=_row_major(se.down_proj),
        sw2_s=_e8m0_bytes(se.down_proj.weight_scale_inv.data),
    )
    return w


@dataclass
class Packed:
    runner: Any  # kernels.runner.DSV41MonoLayer
    weights: dict[int, Any]
    spans: dict[int, tuple[Span, ...]]
    rows_per_request: tuple[int, ...]
    between: frozenset = (
        frozenset()
    )  # whole layers whose indexer / compressor run between K1 and K2

    def kernel(self, rows: int) -> MonoKernel:
        return MonoKernel(self, rows)

    def span(self, route):
        from sglang.kernels.ops.mono.deepseek_v41_gfx950 import glue

        return functools.partial(glue.run_layer, route)


def _deadline(ctx: MonoContext) -> int:
    from sglang.kernels.ops.mono.common_hip.guard import deadline_ticks

    return deadline_ticks(getattr(ctx.server_args, "watchdog_timeout", None))


def build(model, ctx: MonoContext) -> Packed:
    from sglang.kernels.ops.mono.deepseek_v41_gfx950 import runner

    weights, layer_spans, between = {}, {}, set()
    for i, layer in enumerate(model.layers):
        if not _whole_layer_ok(i, layer):
            continue
        w = _layer_weights(layer, runner)
        try:
            w.check(ctx.tp_size)
            w.attn = _attn_weights(i, layer, runner, ctx.tp_rank)
            w.attn.check(runner.Dims(ctx.tp_size))
        except AssertionError as e:
            from sglang.kernels.ops.mono.guards import MonoRefused

            raise MonoRefused(
                f"--enable-monokernel refused (deepseek_v41, model): layer {i}: {e}"
            ) from None
        layer_spans[i] = WHOLE
        if _has_indexer_side(layer):
            between.add(i)
        weights[i] = w
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
        resolve_compressed_kv_layout,
        select_dsv4_kv_layout,
    )

    layout, option = select_dsv4_kv_layout()
    kv41 = layout.value == "v41"
    # ratio 1 and 2 resolve alike (auto -> V41_FP4 next to V41, "lossless for the e2m1 latents")
    comp = {resolve_compressed_kv_layout(layout, r_, option).value for r_ in (1, 2)}
    assert len(comp) == 1, comp
    cfp4 = comp == {"v41_fp4"}
    r = runner.DSV41MonoLayer(
        ctx.tp_size,
        ctx.tp_rank,
        ctx.cpu_group,
        ctx.device,
        kv41=kv41,
        cfp4=cfp4,
        deadline_ticks=_deadline(ctx),
    )
    from sglang.srt.runtime_context import get_spec

    spec = get_spec()
    q = (spec.speculative_num_draft_tokens or 1) if spec.speculative_algorithm else 1
    return Packed(
        runner=r,
        weights=weights,
        spans=layer_spans,
        rows_per_request=(q,),
        between=frozenset(between),
    )


class MonoKernel:
    """One width's launches, shared by every layer: K1 + K2 per compress ratio of the whole-layer layers."""

    def __init__(self, packed: Packed, rows: int) -> None:
        self.packed, self.rows = packed, rows
        for ratio in sorted(
            {w.attn.ratio for w in packed.weights.values() if w.attn is not None}
        ):
            packed.runner.kernels(rows, ratio)
        packed.runner.scratch(rows)  # the width's scratch, outside any capture

    def launch_whole(
        self,
        step: MonoStep,
        layer_id: int,
        x,
        residual,
        post,
        comb,
        pre,
        positions,
        meta: dict,
        between=None,
    ):
        """The whole layer from the attention seam: (x, residual, post, comb) = the previous layer's pending FFN
        post, ``pre`` its next pre; ``meta`` = the attention inputs (``whole_layer_meta``)."""
        w = self.packed.weights[layer_id]
        ratio = w.attn.ratio
        return self.packed.runner.forward(
            w,
            x,
            residual,
            post,
            comb,
            pre,
            positions,
            meta["slot_mapping"],
            meta["swa_cache"],
            meta["swa_indices"],
            meta["swa_lens"],
            meta["token_to_req"],
            topk_indices=meta["topk"][ratio] if ratio else None,
            comp_cache=meta["comp_cache"] if ratio else None,
            comp_block_table=meta["comp_block_table"] if ratio else None,
            between=between,
        )


# KV record bytes a token -- a page holds its data rows, then its scale words:
# V4 (fp8_ds_mla) 576 + 8 (448 e4m3 + 64 bf16 RoPE | 7 UE8M0 + pad); V41 512 + 16 (512 e4m3 | 16 UE8M0)


def _record_view(buf: torch.Tensor, block: int, record: int) -> torch.Tensor:
    buf = buf if buf.dtype == torch.uint8 else buf.view(torch.uint8)
    assert buf.stride(0) >= block * record, (tuple(buf.shape), buf.stride(), block)
    return torch.as_strided(
        buf, (buf.shape[0], block, record), (buf.stride(0), record, 1)
    )


def whole_layer_meta(
    layer_id: int,
    ratio: int,
    forward_batch,
    rows: int,
    cache: dict,
    record: int,
    comp_record: int | None = None,
    swa_width: int = 128,
) -> dict:
    """The whole-layer launch's attention inputs from SGLang's DSV4 HIP backend for this step (graph-safe: views and
    device ops on the backend's persistent metadata). ``cache``: per-step memo shared by the step's layers."""
    from sglang.srt.model_executor.forward_context import get_attn_backend

    backend = get_attn_backend()
    pool = backend.token_to_kv_pool
    core = backend.forward_metadata.core_attn_metadata
    if "slot_mapping" not in cache:
        cache["slot_mapping"] = backend.get_swa_out_cache_loc(forward_batch)[:rows].to(
            torch.int64
        )
        idx = core.swa_page_indices
        assert idx.shape[-1] == swa_width, (
            f"SWA window indices width {idx.shape[-1]}, the kernels read {swa_width}"
        )
        cache["swa_indices"] = (
            idx.reshape(-1, swa_width)[:rows].contiguous().view(rows, 1, swa_width)
        )
        cache["swa_lens"] = (
            core.swa_topk_lengths.reshape(-1)[:rows].to(torch.int32).contiguous()
        )
        cache["token_to_req"] = torch.arange(
            rows, dtype=torch.int32, device=cache["slot_mapping"].device
        )
        cache["comp_block_table"] = core.page_table.to(torch.int32).contiguous()
    m = dict(cache)
    assert pool.get_swa_key_bytes_per_token() == record, (
        f"SWA record {pool.get_swa_key_bytes_per_token()} B, the kernels read {record} B (layer {layer_id})"
    )
    m["swa_cache"] = _record_view(
        pool.get_swa_key_buffer_radix(layer_id), pool.swa_page_size, record
    )
    if ratio:
        # a view of the live buffer, re-read every layer: an indexer layer earlier in the step rewrites it in place
        t = core.sparse_raw_indices(ratio)[:rows]
        assert t.dtype == torch.int32 and t.shape[-1] == 512 and t.is_contiguous(), (
            t.dtype,
            tuple(t.shape),
        )
        m["topk"] = {ratio: t}
        crec = comp_record or record
        assert pool.get_extra_key_bytes_per_token(layer_id) == crec, (
            pool.get_extra_key_bytes_per_token(layer_id)
        )
        m["comp_cache"] = _record_view(
            pool.get_extra_key_buffer(layer_id),
            pool.get_extra_key_page_size(layer_id),
            crec,
        )
    return m


def eligible(ctx: MonoContext, step: MonoStep):
    return None


GFX950 = DeviceLimits(
    widths=tuple(range(1, MAX_ROWS + 1)), tp_sizes=(2, 4), resident_blocks=256
)
# the MoE runs on aiter (SGLANG_USE_AITER=1, checked in config_disable_reasons) under "auto"
REQUIRES = Requirements(
    dtypes=frozenset({"auto", "bfloat16"}),
    kv_cache=frozenset({"fp8_e4m3"}),
    attention_backends=frozenset({"dsv4"}),
    moe_backends=frozenset({"auto", "aiter"}),
)
# max |diff| over max |ref| of a layer's hidden output: <= 15% at one check step, median 2.5% over the 40 layers
NUMERICS = Numerics(Exactness.TOLERANCE, {"k2.hidden.max_rel": 0.2})


def skip_metadata(name, forward_batch, key, source) -> bool:
    """K2i scores a ratio's indexer logits itself: SGLang's aiter decode schedule for it goes unread."""
    if name != "low_ratio_decode_schedule":
        return False
    from sglang.kernels.ops.mono.deepseek_v41_gfx950.indexer_hook import (
        skips_schedule,
    )

    return skips_schedule(forward_batch, key, source)


MONO_SPEC = MonoSpec(
    name="deepseek_v41",
    devices={"gfx950": GFX950},
    features=frozenset({Feature.SPEC_VERIFY, Feature.EAGER}),
    requires=REQUIRES,
    spans=spans,
    numerics=NUMERICS,
    config_disable_reasons=config_disable_reasons,
    model_disable_reasons=model_disable_reasons,
    build=build,
    eligible=eligible,
    skip_metadata=skip_metadata,
)
