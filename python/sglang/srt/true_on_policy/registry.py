"""TOP op registry: the ops, the serving topology each op supports, and the families composed of ops.

A layout is admitted iff the model type belongs to an admitted family and every op of that family
admits it. The trainer keys its cells by the same op names.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import msgspec

from sglang.srt.true_on_policy.contracts import data_parallel_layout

# A cell refused because the layout is not yet verified against the trainer, not for a known mechanism.
NOT_QUALIFIED = "not qualified"


class Cell(msgspec.Struct, frozen=True, kw_only=True):
    """What one op supports on one axis; every degree it refuses carries the reason."""

    sizes: Optional[tuple[int, ...]] = None
    at_least: int = 1
    refused: str = ""

    def __post_init__(self):
        restricts = self.sizes is not None or self.at_least > 1
        if restricts and not self.refused:
            raise ValueError("a restricting TOP cell needs a reason")

    def admits(self, value: int, layout: msgspec.Struct) -> bool:
        if self.sizes is not None and value not in self.sizes:
            return False
        return value >= self.at_least


EVERY = Cell()


def only(*sizes: int, why: str) -> Cell:
    return Cell(sizes=sizes, refused=why)


def at_least(minimum: int, *, why: str) -> Cell:
    return Cell(at_least=minimum, refused=why)


def refused(why: str) -> Cell:
    return Cell(sizes=(1,), refused=why)


# (steps, topk, draft tokens); None in a pattern slot matches any value.
SpecRecipe = tuple[int, int, int]
SpecPattern = tuple[Optional[int], Optional[int], Optional[int]]
_ANY_RECIPE: SpecPattern = (None, None, None)


class Spec(msgspec.Struct, frozen=True, kw_only=True):
    """The speculative recipes an op verifies; serving without speculation always passes."""

    recipes: tuple[SpecPattern, ...] = ()
    refused: str = ""

    def __post_init__(self):
        if _ANY_RECIPE not in self.recipes and not self.refused:
            raise ValueError("a restricting TOP spec cell needs a reason")

    def admits(self, value: Optional[SpecRecipe], layout: msgspec.Struct) -> bool:
        return value is None or any(
            all(want is None or want == got for want, got in zip(pattern, value))
            for pattern in self.recipes
        )


EVERY_SPEC = Spec(recipes=(_ANY_RECIPE,))


class ServingEnvelope(msgspec.Struct, frozen=True, kw_only=True):
    ep: Cell
    dp_attention: Cell
    pp: Cell
    spec: Spec


class ServingLayout(msgspec.Struct, frozen=True, kw_only=True):
    ep: int
    dp_attention: int
    pp: int
    spec: Optional[SpecRecipe]

    @classmethod
    def from_server_args(cls, server_args: Any) -> ServingLayout:
        spec = None
        if server_args.speculative_algorithm is not None:
            spec = (
                server_args.speculative_num_steps,
                server_args.speculative_eagle_topk,
                server_args.speculative_num_draft_tokens,
            )
        return cls(
            ep=server_args.ep_size,
            dp_attention=data_parallel_layout(server_args)[0],
            pp=server_args.pp_size,
            spec=spec,
        )


class Op(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    serving: tuple[ServingEnvelope, ...]
    # False for serving-only ops, which have no trainer cells.
    in_trainer: bool = True


class Family(msgspec.Struct, frozen=True, kw_only=True):
    name: str
    # Trainer and serving spellings of the checkpoint's model_type.
    model_types: tuple[str, ...]
    ops: tuple[str, ...]
    # Ops a block-FP8 checkpoint runs instead of ``ops``: precision selects programs, never cells.
    # None: the family has no FP8 program, and a block-FP8 checkpoint is refused.
    fp8_ops: Optional[tuple[str, ...]] = None
    # Ops a run with images or video adds to the text program. None: the family has no vision program.
    vision_ops: Optional[tuple[str, ...]] = None

    def ops_for(self, *, fp8: bool, vision: bool = False) -> tuple[str, ...]:
        if fp8 and self.fp8_ops is None:
            raise NotImplementedError(
                f"[top] {self.name} has no block-FP8 program; use a BF16 checkpoint"
            )
        ops = self.fp8_ops if fp8 else self.ops
        if not vision:
            return ops
        if self.vision_ops is None:
            raise NotImplementedError(
                f"[top] {self.name} has no vision program; send text-only samples"
            )
        return ops + self.vision_ops


def _serving(**cells) -> ServingEnvelope:
    return ServingEnvelope(
        **{"ep": EVERY, "dp_attention": EVERY, "pp": EVERY, "spec": EVERY_SPEC, **cells}
    )


def _op(name: str, *serving: ServingEnvelope, in_trainer: bool = True) -> Op:
    return Op(name=name, serving=serving or (_serving(),), in_trainer=in_trainer)


# Every op until it is verified under more: no serving stages, no speculation.
_STRICT_SERVING = _serving(
    pp=refused(NOT_QUALIFIED),
    spec=Spec(refused=NOT_QUALIFIED),
)
# Ops qualified under speculation: verify scores and draws each row as non-spec decode would. Any recipe here; the
# family's draft op restricts them, and a family admits speculation only once every one of its ops does.
_SPEC_SERVING = msgspec.structs.replace(_STRICT_SERVING, spec=EVERY_SPEC)
_EP_SLOT_TRANSPORT = (
    "the experts run behind the DeepEP slot transport, which needs EP>=2"
)

OPS: tuple[Op, ...] = (
    # Every projection, the LM head included: serving's BI DeepGEMM over the full K, with no fallback.
    _op("bi_linear", _SPEC_SERVING),
    # Stock fused RMSNorm in BF16 with matching input and weight dtypes.
    _op("rms_norm", _SPEC_SERVING),
    # Residual add fused into the next norm; the trainer hands [branch, residual] between layers unsummed.
    _op("packed_residual", _SPEC_SERVING),
    # Plain RoPE (NeoX or interleaved) on serving's fused kernel: FP32 cos/sin from a CPU-derived table, one BF16
    # rounding. M-RoPE runs eager torch with its per-product rounding.
    _op("rope", _SPEC_SERVING),
    # Dense SwiGLU: SiLU rounded to BF16 before the multiply, as the trainer's eager activation.
    _op("swiglu", _SPEC_SERVING),
    # Causal packed-GQA FA4 in batch-invariant mode with one key split, and the fused BF16 Q/K norm.
    _op(
        "fa4_attention",
        msgspec.structs.replace(_STRICT_SERVING, dp_attention=refused(NOT_QUALIFIED)),
    ),
    # BF16 logits divided by the FP32 temperature, then the vocabulary-tiled log-softmax.
    _op("lm_head", _SPEC_SERVING),
    # Seeded exponential race over the full log-probs; a draw depends on its seed and position only.
    _op("sampling", _SPEC_SERVING, in_trainer=False),
    # Qwen softmax routing on the fused router kernel: FP32 softmax, top-k ties by expert id, FP32 renormalize.
    _op("qwen_router", _SPEC_SERVING),
    # Slot-level DeepEP transport: one row per (token, k) slot out and back, device-side receipts, owner-side combine.
    _op("ep_slot_transport", _SPEC_SERVING),
    # Qwen3MoeSparseMoeBlock experts: DeepGEMM BF16 GEMMs, the accurate SiLU, FP32 slot-order combine.
    _op(
        "qwen3_moe_experts",
        msgspec.structs.replace(
            _STRICT_SERVING,
            ep=only(1, why=NOT_QUALIFIED),
        ),
    ),
    # Qwen3.5 GDN: fixed-tile chunk prefill, and decode that follows it bit for bit (the row engine), so the
    # trainer runs the same chunk prefill over each sample.
    _op(
        "gdn",
        msgspec.structs.replace(
            _STRICT_SERVING,
            # Verify runs the root and its drafts as one multi-row row-engine step, so only a chain.
            spec=Spec(
                recipes=((None, 1, None),), refused="GDN verifies a chain (top-k 1)"
            ),
        ),
    ),
    # GLM-5.3-Flash KDA: chunk_kda with the safe gate, the fused intra kernel and one pinned config per kernel
    # (kda_top), and decode that follows it bit for bit (kda_rows); the trainer runs the same prefill per sample.
    _op(
        "kda",
        msgspec.structs.replace(
            _STRICT_SERVING,
            # Verify runs the root and its drafts as consecutive row-engine steps, so only a chain.
            spec=Spec(
                recipes=((None, 1, None),), refused="KDA verifies a chain (top-k 1)"
            ),
        ),
    ),
    # GLM-5.3-Flash stream contraction: the plain BF16 mean over the 4 hyper-connection streams (hc_contract), under the
    # batch-invariant mean.
    _op("hc_mean_head", _SPEC_SERVING),
    # GLM-5.3-Flash routed experts: glm_experts' DeepGEMM BF16 GEMMs and accurate SiLU, gate and up clamped at the
    # SwiGLU limit in BF16 (masked rows inside the masked SiLU, contiguous rows by moe.top_clamp_swiglu_input).
    _op("glm_clamped_experts", _SPEC_SERVING),
    # GLM-5.3-Flash dense and shared-expert MLP: serving's elementwise silu_and_mul_clamp between BI GEMMs.
    _op("clamped_swiglu_bf16", _SPEC_SERVING),
    # Qwen3.5 full attention: FA4 at head 256; the stock fused Q/K Gemma norm + partial NeoX RoPE (FP32 cos/sin) +
    # gate split kernel, which the trainer's forward calls too; serving's fused BF16 output gate.
    _op("gdn_full_attention", _SPEC_SERVING),
    # GDN-family decode: keys split at fixed positions, FA4 per piece, FA's combine; the trainer splits alike.
    _op("decode_kv_split", _SPEC_SERVING),
    # Qwen3.5 MoE shared expert: the dense SwiGLU MLP, scaled by sigmoid of a one-column gate on narrow fixed BI tiles.
    _op("shared_expert_gate", _SPEC_SERVING),
    # Qwen2MoeSparseMoeBlock experts (Qwen3.5/3.6 MoE): DeepGEMM BF16 GEMMs, the accurate SiLU, slot-order combine.
    _op(
        "qwen2_moe_experts",
        msgspec.structs.replace(
            _SPEC_SERVING,
            ep=at_least(2, why=_EP_SLOT_TRANSPORT),
        ),
    ),
    # GLM DSA: serving's FP8 indexer logits, FlashInfer's deterministic top-k in ascending slot order, the BF16 head
    # gate on BI tiles, interleaved fused RoPE, and BF16 sparse FlashMLA over the selected latent rows.
    _op("dsa", _SPEC_SERVING),
    # GLM routing: BF16 gate logits, FP32 sigmoid plus the FP32 correction bias, deterministic top-k by expert id,
    # FP32 renormalize of the unbiased scores.
    _op("glm_router", _SPEC_SERVING),
    # DeepseekV2MoE experts (GLM-5, BF16): DeepGEMM BF16 GEMMs, the accurate SiLU, FP32 slot-order combine with the
    # routed scaling folded into the slot weights, the shared expert added after.
    _op(
        "glm_experts",
        msgspec.structs.replace(
            _SPEC_SERVING,
            ep=at_least(2, why=_EP_SLOT_TRANSPORT),
        ),
    ),
    # Block-FP8 projections (E4M3 payloads, 128x128 UE8M0 weight scales): the row's activations quantized per 128
    # channels, then DeepGEMM's FP8 GEMM over the full K; the trainer runs the same op on the checkpoint's bytes.
    _op("fp8_linear", _SPEC_SERVING),
    # DeepseekV2MoE experts on a block-FP8 checkpoint behind the DeepEP slot transport: rows quantized per 128
    # channels with amax floored at 1e-4 and sent FP8 (at the source in prefill, DeepEP's fused cast in decode),
    # DeepGEMM's FP8 grouped GEMMs with the SwiGLU fused into the down projection's FP8 quantizer, FP32 slot-order
    # combine.
    _op(
        "glm_fp8_experts",
        msgspec.structs.replace(
            _SPEC_SERVING,
            ep=at_least(2, why=_EP_SLOT_TRANSPORT),
        ),
    ),
    # GLM's checkpoint MTP layer as the EAGLE draft. Verify makes every token draft-independent, so the draft runs
    # serving's own program; (5, 1, 6) needs a block-FP8 checkpoint (true_on_policy.mtp).
    _op(
        "glm_mtp_draft",
        msgspec.structs.replace(
            _STRICT_SERVING,
            spec=Spec(recipes=((1, 1, 2), (5, 1, 6)), refused=NOT_QUALIFIED),
        ),
        in_trainer=False,
    ),
    # DeepSeek-V4 dense and shared-expert MLP on a block-FP8 checkpoint: FP8 gate_up GEMM, the routed experts' clamped
    # FP32 SwiGLU fused into the down projection's per-128 FP8 quantizer, FP8 down GEMM over the full K.
    _op("clamped_swiglu", _SPEC_SERVING),
    # DeepSeek-V4 routing: FP32 gate logits on BI tiles; hash layers take tid2eid[token], the rest the fused
    # bias-ranked sqrtsoftplus top-k; unscaled renormalized FP32 weights, routed scaling applied in the combine.
    _op("dsv4_router", _SPEC_SERVING),
    # DeepseekV2MoE experts on DeepSeek-V4's block-FP8 checkpoint behind the DeepEP slot transport: the input sent FP8
    # as glm_fp8_experts', the FP8 grouped GEMMs with SwiGLU clamped at swiglu_limit before the down projection's FP8
    # quantizer, FP32 slot-order combine.
    _op(
        "dsv4_experts",
        msgspec.structs.replace(
            _SPEC_SERVING,
            ep=at_least(2, why=_EP_SLOT_TRANSPORT),
        ),
    ),
    # DeepSeek-V4 manifold hyper-connections: TileLang mhc_pre/mhc_post with the sublayer RMSNorm fused into pre, and
    # the pre-norm DeepGEMM at a pinned split count, so every token count runs one program.
    _op("mhc", _SPEC_SERVING),
    # DeepSeek-V4 attention: sparse attention over the FP8 SWA and compressed pools as one BF16 program (selected rows
    # dequantized, attention sink), the C4 indexer's deterministic top-k in ascending logical order, wo_a as the FP8
    # GEMM; page-aligned radix hits equal a cold prefill.
    _op("dsv4_attention", _SPEC_SERVING),
    # DeepSeek-V4's checkpoint MTP layer as the EAGLE draft, its head on the target's fused mHC head kernel. Verify makes
    # every token draft-independent; (3, 1, 4) is the cookbook low-latency recipe for DS4-Flash.
    _op(
        "dsv4_mtp_draft",
        msgspec.structs.replace(
            _STRICT_SERVING,
            spec=Spec(recipes=((1, 1, 2), (3, 1, 4), (5, 1, 6)), refused=NOT_QUALIFIED),
        ),
        in_trainer=False,
    ),
    # DeepSeek-V4.1 projections on its 32x32 block-FP8 checkpoint (E4M3, UE8M0 scales): rows quantized per 32 channels
    # with power-of-two scales, then SM90 DeepGEMM's BLOCK32 GEMM over the full K, with no BF16 weight cache or MXFP8
    # view; the trainer runs the same op on the checkpoint's bytes.
    _op("fp8_b32_linear", _SPEC_SERVING),
    # DeepSeek-V4.1 shared expert: 32x32 FP8 gate_up GEMM, SwiGLU clamped at 10 in FP32 fused into the down input's
    # per-32 UE8M0 FP8 quantizer (one token per block, no BF16 rounding), 32x32 FP8 down GEMM over the full K.
    _op("clamped_swiglu_b32", _SPEC_SERVING),
    # DeepSeek-V4.1 mHC, each hyper-connection collapsing with the previous one's pre: hc_mix_stats_sinkhorn (split-K
    # at a K-fixed slice count), hc_combine to BF16 then the sublayer RMSNorm, mhc_post_split_h, at every row count;
    # the head collapses the last streams the same way. No fused boundary, no stats side stream.
    _op("mhc_prev", _SPEC_SERVING),
    # DeepSeek-V4.1 Engram: n-gram hash ids from SGLang's integer hash kernels (the trainer hashes packed requests from
    # position 0), exact E4M3 x UE8M0 table rows, the 32x32 FP8 wkv, the fused FP32 gate (one program per token and
    # stream copy) added into every stream before the layer.
    _op("engram", _SPEC_SERVING),
    # DeepSeek-V4.1 attention: 32x32 FP8 q/kv projections, fused q RoPE and kv norm + RoPE + FP8 SWA store; ratio-1/2
    # KV sources pool on the batch-invariant FP32 projection (torch pair pooling, FP32 RMSNorm, fused RoPE + FP4);
    # index layers score serving's paged FP4 logits per row, rank candidate blocks and positions by deterministic
    # top-k (ties to the smaller index, ascending); the DS4 sparse core over the shared FP8 pools; BI BF16 wo_a.
    # Page-aligned radix hits equal a cold prefill: a page holds whole ratio-2 pairs (validate_dsv41_radix).
    _op("dsv41_attention", _SPEC_SERVING),
    # DeepSeek-V4.1 routing: dsv4_router's program (FP32 gate logits on BI tiles, fused bias-ranked top-k, unscaled
    # renormalized FP32 weights) with sqrtsoftplus through log1p (NaNs ranked first); no hash layers.
    _op("dsv41_router", _SPEC_SERVING),
    # DeepSeek-V4.1's bundled DSpark block drafter (3 stages, block 5) under static verify of the anchor and 5 drafts;
    # verify accepts drafts while they equal TOP's race targets (true_on_policy.dsv41_dspark), so every emitted token
    # is non-speculative decode's and the draft runs serving's own program.
    _op(
        "dsv41_dspark_draft",
        msgspec.structs.replace(
            _STRICT_SERVING, spec=Spec(recipes=((1, 1, 6),), refused=NOT_QUALIFIED)
        ),
        in_trainer=False,
    ),
    # Qwen3.8-Flash-Next gated residual (4 hyper-connection branches): per-branch offset RMSNorm, the low-rank read
    # gate (split-K down projection, fixed-tile up projection) and the inject + combine write, one Triton program at
    # every row count with BF16 rounding at the module's boundaries.
    _op("hyper_connection", _SPEC_SERVING),
    # Qwen3.8-Flash-Next PLE n-gram increment: BI key/value projections, the FP32 gate chain, and the dilated short
    # convolution over BF16 inputs read from the request's conv state, one program for prefill, decode and verify:
    # a verify row reads the request's root and earlier drafts as its history, and the state after each draft step
    # is kept for the commit.
    _op(
        "ple",
        msgspec.structs.replace(
            _STRICT_SERVING,
            spec=Spec(
                recipes=((None, 1, None),), refused="PLE verifies a chain (top-k 1)"
            ),
        ),
    ),
    # Qwen3.8-Flash-Next full attention: the fused Q/K norm + partial RoPE + gate and the output gate as
    # gdn_full_attention, with QSA in place of FA4: the indexer's BI projection, fused norm + RoPE and compressed keys,
    # FlashInfer deterministic block top-k with ties to the lower block id, and FP32 online-softmax attention over the selected rows in fixed
    # tiles, read from the KV pool; a verify row takes the decode route with its own position and request, its completed
    # groups are compressed from the keys its own decode would read, and only accepted rows enter the pending key ring.
    # The indexer rotates each lane at its MRoPE axis, so image positions run the text program.
    _op(
        "qsa",
        msgspec.structs.replace(
            _STRICT_SERVING,
            spec=Spec(
                recipes=((None, 1, 2), (None, 1, 3), (None, 1, 4)),
                refused="QSA verifies a chain of at most 4 tokens (its pending key ring holds one group)",
            ),
        ),
    ),
    # Qwen3.8-Flash-Next's checkpoint MTP layer as the EAGLE draft. Verify makes every token draft-independent, so the
    # draft runs serving's own program; (3, 1, 4) is the cookbook NEXTN recipe.
    _op(
        "qwen4_mtp_draft",
        msgspec.structs.replace(
            _STRICT_SERVING, spec=Spec(recipes=((3, 1, 4),), refused=NOT_QUALIFIED)
        ),
        in_trainer=False,
    ),
    # Qwen3.5's frozen vision tower (true_on_policy.vision), forward only: every projection the BI DeepGEMM with the
    # bias added in BF16, the per-image position interpolation, FP32 2D rotary tables through eager rotary,
    # noncausal FA4 with one key split, torch LayerNorm and GELU. The tower runs unsharded on the rank that prefills
    # the request (TOP engines widen as attention DP only), and only at prefill, so speculation never reaches it.
    _op("qwen_vision_tower", _SPEC_SERVING),
    # DeepSeek-V4.1's frozen ViT and 3x3 aligner (true_on_policy.vision), forward only: every projection the BI
    # DeepGEMM with the bias added in BF16, FP32-weight RMSNorm, the release's FP32 half-split 2D rotary, noncausal FA4
    # with one key split, the image span between start/newline/end rows. Image rows route by the VL bias through the
    # fused gate (dsv41_router's kernel, every row of a vision engine) and keep their streams through the engram gate.
    _op("dsv41_vision_tower", _SPEC_SERVING),
    # GLM-5.3-Flash DSA (glm5_next, NoPE MLA, pooled indexer; true_on_policy.glm5_next_dsa): the FP32-output BI head
    # gate, complete 4-token key groups compressed by the two-pass softmax kernel in extend and decode alike, DeepGEMM
    # FP8 MQA logits, FlashInfer's deterministic top-512 groups in ascending order expanded plus the always-selected
    # tail, then TileLang sparse MLA over the BF16 512-wide latent, 64 heads per block, the selection padded to 2112.
    # A verify row scores and selects as the decode of its position: groups its window completes are compressed by the
    # same two-pass softmax, and its tail is the request's keys up to the row.
    _op(
        "glm5_next_dsa",
        msgspec.structs.replace(
            _STRICT_SERVING,
            spec=Spec(
                recipes=((None, 1, None),),
                refused="GLM-5.3 DSA verifies a chain (top-k 1): a row's key tail is its request's sequence",
            ),
        ),
    ),
    # GLM-5.3-Flash's checkpoint MTP layer as the EAGLE draft. Verify makes every token draft-independent, so the draft
    # runs serving's own program; (5, 1, 6) is the cookbook low-latency recipe.
    _op(
        "glm5_next_mtp_draft",
        msgspec.structs.replace(
            _STRICT_SERVING,
            spec=Spec(recipes=((1, 1, 2), (5, 1, 6)), refused=NOT_QUALIFIED),
        ),
        in_trainer=False,
    ),
)

_DENSE = ("bi_linear", "rms_norm", "rope", "swiglu", "lm_head", "sampling")
_GDN = ("gdn", "decode_kv_split")
_GLM = _DENSE + (
    "packed_residual",
    "dsa",
    "glm_router",
    "ep_slot_transport",
    "glm_mtp_draft",
)
_DSV41 = (
    ("bi_linear", "rms_norm", "lm_head", "sampling")
    + ("fp8_b32_linear", "clamped_swiglu_b32", "mhc_prev", "engram")
    + (
        "dsv41_attention",
        "dsv41_router",
        "dsv4_experts",
        "ep_slot_transport",
        "dsv41_dspark_draft",
    )
)
_DSV4 = (
    ("bi_linear", "rms_norm", "rope", "clamped_swiglu", "lm_head", "sampling")
    + ("dsv4_attention", "mhc", "dsv4_router", "dsv4_experts", "fp8_linear")
    + ("ep_slot_transport", "dsv4_mtp_draft")
)
FAMILIES: tuple[Family, ...] = (
    Family(
        name="Qwen3 dense",
        model_types=("qwen3",),
        ops=_DENSE + ("packed_residual", "fa4_attention"),
    ),
    Family(
        name="Qwen3 MoE",
        model_types=("qwen3_moe",),
        ops=_DENSE
        + ("packed_residual", "fa4_attention")
        + ("qwen_router", "qwen3_moe_experts", "ep_slot_transport"),
    ),
    Family(
        name="Qwen GDN dense",
        model_types=("qwen3_5", "qwen3_5_text"),
        ops=_DENSE + ("packed_residual", "gdn_full_attention") + _GDN,
        vision_ops=("qwen_vision_tower",),
    ),
    Family(
        name="Qwen GDN MoE",
        model_types=("qwen3_5_moe", "qwen3_5_moe_text"),
        ops=_DENSE
        + ("packed_residual", "gdn_full_attention")
        + _GDN
        + (
            "qwen_router",
            "qwen2_moe_experts",
            "shared_expert_gate",
            "ep_slot_transport",
        ),
        vision_ops=("qwen_vision_tower",),
    ),
    Family(
        name="GLM",
        model_types=("glm_moe_dsa",),
        ops=_GLM + ("glm_experts",),
        fp8_ops=_GLM + ("glm_fp8_experts", "fp8_linear"),
    ),
    # BF16 text model (gdn_contract.validate_qwen4_exp_top). The full-attention layers are QSA, so there is no FA4
    # decode split, and hyper-connections replace the residual norms.
    Family(
        name="Qwen3.8-Flash-Next",
        model_types=("qwen4_exp", "qwen4_exp_text"),
        ops=_DENSE
        + ("gdn", "hyper_connection", "ple", "qsa")
        + (
            "qwen_router",
            "qwen2_moe_experts",
            "shared_expert_gate",
            "ep_slot_transport",
            "qwen4_mtp_draft",
        ),
        vision_ops=("qwen_vision_tower",),
    ),
    # GLM-5.3-Flash, BF16 (glm5_next.validate_top_glm5_next): KDA and DSA layers under mHC, the GLM gate with clamped
    # BF16 experts and a clamped shared expert, a clamped dense first layer and the stream-mean head.
    Family(
        name="GLM-5.3-Flash",
        model_types=("glm5_next", "glm5_next_text"),
        ops=("bi_linear", "rms_norm", "lm_head", "sampling")
        + ("mhc", "hc_mean_head", "kda", "glm5_next_dsa")
        + (
            "glm_router",
            "glm_clamped_experts",
            "clamped_swiglu_bf16",
            "ep_slot_transport",
            "glm5_next_mtp_draft",
        ),
    ),
    # Block-FP8 only (validate_dsv4); every op is the FP8 program, declared as both the default and the FP8 ops.
    Family(
        name="DeepSeek-V4",
        model_types=("deepseek_v4",),
        ops=_DSV4,
        fp8_ops=_DSV4,
    ),
    # Release checkpoint on Hopper (validate_dsv41): 32x32 FP8 dense, MXFP4 experts cast to 128x128 FP8 at load and
    # run by DS4's expert program. FP8 only, so its ops are declared as both the default and the FP8 ops.
    Family(
        name="DeepSeek-V4.1",
        model_types=("deepseek_v41",),
        ops=_DSV41,
        fp8_ops=_DSV41,
        vision_ops=("dsv41_vision_tower",),
    ),
)


def index_ops(ops: Sequence[Op], families: Sequence[Family]) -> dict[str, Op]:
    by_name = {op.name: op for op in ops}
    if len(by_name) != len(ops):
        raise ValueError("TOP op names must be unique")
    for family in families:
        missing = [
            name
            for name in family.ops + (family.fp8_ops or ()) + (family.vision_ops or ())
            if name not in by_name
        ]
        if missing:
            raise ValueError(f"TOP family {family.name!r} names unknown ops {missing}")
    return by_name


OPS_BY_NAME = index_ops(OPS, FAMILIES)


def envelope_refusals(
    envelopes: Sequence[msgspec.Struct], layout: msgspec.Struct
) -> list[str]:
    """Nothing when some envelope admits the layout, else the failing axes of the closest one."""
    closest = None
    for envelope in envelopes:
        failing = [
            f"{axis}={getattr(layout, axis)} refused: {getattr(envelope, axis).refused}"
            for axis in envelope.__struct_fields__
            if not getattr(envelope, axis).admits(getattr(layout, axis), layout)
        ]
        if not failing:
            return []
        if closest is None or len(failing) < len(closest):
            closest = failing
    return closest if closest is not None else ["no envelope declared"]


def find_family(model_type: str, families: Sequence[Family] = FAMILIES) -> Family:
    for family in families:
        if model_type in family.model_types:
            return family
    admitted = ", ".join(family.name for family in families) or "none"
    raise NotImplementedError(
        f"[top] no TOP family admits model_type {model_type!r}; admitted families: {admitted}"
    )


def admit(
    model_type: str,
    layout: msgspec.Struct,
    *,
    fp8: bool,
    side: str,
    envelopes_of: Callable[[str], Sequence[msgspec.Struct]],
    families: Sequence[Family] = FAMILIES,
    vision: bool = False,
) -> Family:
    """Admit ``layout`` on one side, or refuse naming every refusing op and axis."""
    family = find_family(model_type, families)
    refusals = [
        f"op {name}: {line}"
        for name in family.ops_for(fp8=fp8, vision=vision)
        for line in envelope_refusals(envelopes_of(name), layout)
    ]
    if refusals:
        raise NotImplementedError(
            f"[top] {family.name} ({model_type}) refuses this {side} layout:\n  "
            + "\n  ".join(refusals)
        )
    return family


def admit_serving(
    server_args: Any, *, model_type: str, fp8: bool, vision: bool = False
) -> Family:
    return admit(
        model_type,
        ServingLayout.from_server_args(server_args),
        fp8=fp8,
        side="serving",
        envelopes_of=lambda name: OPS_BY_NAME[name].serving,
        vision=vision,
    )
